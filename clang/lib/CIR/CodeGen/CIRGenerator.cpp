//===--- CIRGenerator.cpp - Emit CIR from ASTs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This builds an AST and converts it to CIR.
//
//===----------------------------------------------------------------------===//

#include "CIRGenModule.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace cir;
using namespace clang;

void CIRGenerator::anchor() {}

CIRGenerator::CIRGenerator(clang::DiagnosticsEngine &diags,
                           llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs,
                           const CodeGenOptions &CGO)
    : Diags(diags), fs(std::move(vfs)), codeGenOpts{CGO},
      HandlingTopLevelDecls(0) {}
CIRGenerator::~CIRGenerator() {
  // There should normally not be any leftover inline method definitions.
  assert(DeferredInlineMemberFuncDefs.empty() || Diags.hasErrorOccurred());
}

static void setMLIRDataLayout(mlir::ModuleOp &mod, const llvm::DataLayout &dl) {
  auto *context = mod.getContext();
  mod->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
               mlir::StringAttr::get(context, dl.getStringRepresentation()));
  mlir::DataLayoutSpecInterface dlSpec = mlir::translateDataLayout(dl, context);
  mod->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, dlSpec);
}

void CIRGenerator::Initialize(ASTContext &astCtx) {
  using namespace llvm;

  this->astCtx = &astCtx;

  mlirCtx = std::make_unique<mlir::MLIRContext>();
  mlirCtx->getOrLoadDialect<mlir::DLTIDialect>();
  mlirCtx->getOrLoadDialect<mlir::func::FuncDialect>();
  mlirCtx->getOrLoadDialect<mlir::cir::CIRDialect>();
  mlirCtx->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  mlirCtx->getOrLoadDialect<mlir::memref::MemRefDialect>();
  mlirCtx->getOrLoadDialect<mlir::omp::OpenMPDialect>();
  CGM = std::make_unique<CIRGenModule>(*mlirCtx.get(), astCtx, codeGenOpts,
                                       Diags);
  auto mod = CGM->getModule();
  auto layout = llvm::DataLayout(astCtx.getTargetInfo().getDataLayoutString());
  setMLIRDataLayout(mod, layout);
}

bool CIRGenerator::verifyModule() { return CGM->verifyModule(); }

bool CIRGenerator::EmitFunction(const FunctionDecl *FD) {
  llvm_unreachable("NYI");
}

mlir::ModuleOp CIRGenerator::getModule() { return CGM->getModule(); }

bool CIRGenerator::HandleTopLevelDecl(DeclGroupRef D) {
  if (Diags.hasErrorOccurred())
    return true;

  HandlingTopLevelDeclRAII HandlingDecl(*this);

  for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I) {
    CGM->buildTopLevelDecl(*I);
  }

  return true;
}

void CIRGenerator::HandleTranslationUnit(ASTContext &C) {
  // Release the Builder when there is no error.
  if (!Diags.hasErrorOccurred() && CGM)
    CGM->Release();

  // If there are errors before or when releasing the CGM, reset the module to
  // stop here before invoking the backend.
  if (Diags.hasErrorOccurred()) {
    if (CGM)
      // TODO: CGM->clear();
      // TODO: M.reset();
      return;
  }
}

void CIRGenerator::HandleInlineFunctionDefinition(FunctionDecl *D) {
  if (Diags.hasErrorOccurred())
    return;

  assert(D->doesThisDeclarationHaveABody());

  // We may want to emit this definition. However, that decision might be
  // based on computing the linkage, and we have to defer that in case we are
  // inside of something that will chagne the method's final linkage, e.g.
  //   typedef struct {
  //     void bar();
  //     void foo() { bar(); }
  //   } A;
  DeferredInlineMemberFuncDefs.push_back(D);

  // Provide some coverage mapping even for methods that aren't emitted.
  // Don't do this for templated classes though, as they may not be
  // instantiable.
  if (!D->getLexicalDeclContext()->isDependentContext())
    CGM->AddDeferredUnusedCoverageMapping(D);
}

void CIRGenerator::buildDefaultMethods() { CGM->buildDefaultMethods(); }

void CIRGenerator::buildDeferredDecls() {
  if (DeferredInlineMemberFuncDefs.empty())
    return;

  // Emit any deferred inline method definitions. Note that more deferred
  // methods may be added during this loop, since ASTConsumer callbacks can be
  // invoked if AST inspection results in declarations being added.
  HandlingTopLevelDeclRAII HandlingDecls(*this);
  for (unsigned I = 0; I != DeferredInlineMemberFuncDefs.size(); ++I)
    CGM->buildTopLevelDecl(DeferredInlineMemberFuncDefs[I]);
  DeferredInlineMemberFuncDefs.clear();
}

/// HandleTagDeclDefinition - This callback is invoked each time a TagDecl to
/// (e.g. struct, union, enum, class) is completed. This allows the client hack
/// on the type, which can occur at any point in the file (because these can be
/// defined in declspecs).
void CIRGenerator::HandleTagDeclDefinition(TagDecl *D) {
  if (Diags.hasErrorOccurred())
    return;

  // Don't allow re-entrant calls to CIRGen triggered by PCH deserialization to
  // emit deferred decls.
  HandlingTopLevelDeclRAII HandlingDecl(*this, /*EmitDeferred=*/false);

  CGM->UpdateCompletedType(D);

  // For MSVC compatibility, treat declarations of static data members with
  // inline initializers as definitions.
  if (astCtx->getTargetInfo().getCXXABI().isMicrosoft()) {
    llvm_unreachable("NYI");
  }
  // For OpenMP emit declare reduction functions, if required.
  if (astCtx->getLangOpts().OpenMP) {
    llvm_unreachable("NYI");
  }
}

void CIRGenerator::HandleTagDeclRequiredDefinition(const TagDecl *D) {
  if (Diags.hasErrorOccurred())
    return;

  // Don't allow re-entrant calls to CIRGen triggered by PCH deserialization to
  // emit deferred decls.
  HandlingTopLevelDeclRAII HandlingDecl(*this, /*EmitDeferred=*/false);

  if (CGM->getModuleDebugInfo())
    llvm_unreachable("NYI");
}

void CIRGenerator::HandleCXXStaticMemberVarInstantiation(VarDecl *D) {
  if (Diags.hasErrorOccurred())
    return;

  CGM->HandleCXXStaticMemberVarInstantiation(D);
}

void CIRGenerator::CompleteTentativeDefinition(VarDecl *D) {
  if (Diags.hasErrorOccurred())
    return;

  CGM->buildTentativeDefinition(D);
}
