//===- CIRGenModule.cpp - Per-Module state for CIR generation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for CIR translation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenModule.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

using namespace clang;
using namespace clang::CIRGen;

CIRGenModule::CIRGenModule(mlir::MLIRContext &mlirContext,
                           clang::ASTContext &astContext,
                           const clang::CodeGenOptions &cgo,
                           DiagnosticsEngine &diags)
    : builder(mlirContext, *this), astContext(astContext),
      langOpts(astContext.getLangOpts()),
      theModule{mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlirContext))},
      diags(diags), target(astContext.getTargetInfo()), genTypes(*this) {

  // Initialize cached types
  VoidTy = cir::VoidType::get(&getMLIRContext());
  SInt8Ty = cir::IntType::get(&getMLIRContext(), 8, /*isSigned=*/true);
  SInt16Ty = cir::IntType::get(&getMLIRContext(), 16, /*isSigned=*/true);
  SInt32Ty = cir::IntType::get(&getMLIRContext(), 32, /*isSigned=*/true);
  SInt64Ty = cir::IntType::get(&getMLIRContext(), 64, /*isSigned=*/true);
  SInt128Ty = cir::IntType::get(&getMLIRContext(), 128, /*isSigned=*/true);
  UInt8Ty = cir::IntType::get(&getMLIRContext(), 8, /*isSigned=*/false);
  UInt16Ty = cir::IntType::get(&getMLIRContext(), 16, /*isSigned=*/false);
  UInt32Ty = cir::IntType::get(&getMLIRContext(), 32, /*isSigned=*/false);
  UInt64Ty = cir::IntType::get(&getMLIRContext(), 64, /*isSigned=*/false);
  UInt128Ty = cir::IntType::get(&getMLIRContext(), 128, /*isSigned=*/false);
  FP16Ty = cir::FP16Type::get(&getMLIRContext());
  BFloat16Ty = cir::BF16Type::get(&getMLIRContext());
  FloatTy = cir::SingleType::get(&getMLIRContext());
  DoubleTy = cir::DoubleType::get(&getMLIRContext());
  FP80Ty = cir::FP80Type::get(&getMLIRContext());
  FP128Ty = cir::FP128Type::get(&getMLIRContext());
}

mlir::Location CIRGenModule::getLoc(SourceLocation cLoc) {
  assert(cLoc.isValid() && "expected valid source location");
  const SourceManager &sm = astContext.getSourceManager();
  PresumedLoc pLoc = sm.getPresumedLoc(cLoc);
  StringRef filename = pLoc.getFilename();
  return mlir::FileLineColLoc::get(builder.getStringAttr(filename),
                                   pLoc.getLine(), pLoc.getColumn());
}

mlir::Location CIRGenModule::getLoc(SourceRange cRange) {
  assert(cRange.isValid() && "expected a valid source range");
  mlir::Location begin = getLoc(cRange.getBegin());
  mlir::Location end = getLoc(cRange.getEnd());
  mlir::Attribute metadata;
  return mlir::FusedLoc::get({begin, end}, metadata, builder.getContext());
}

void CIRGenModule::emitGlobal(clang::GlobalDecl gd) {
  const auto *global = cast<ValueDecl>(gd.getDecl());

  if (const auto *fd = dyn_cast<FunctionDecl>(global)) {
    // Update deferred annotations with the latest declaration if the function
    // was already used or defined.
    if (fd->hasAttr<AnnotateAttr>())
      errorNYI(fd->getSourceRange(), "deferredAnnotations");
    if (!fd->doesThisDeclarationHaveABody()) {
      if (!fd->doesDeclarationForceExternallyVisibleDefinition())
        return;

      errorNYI(fd->getSourceRange(),
               "function declaration that forces code gen");
      return;
    }
  } else {
    assert(cast<VarDecl>(global)->isFileVarDecl() &&
           "Cannot emit local var decl as global");
  }

  // TODO(CIR): Defer emitting some global definitions until later
  emitGlobalDefinition(gd);
}

void CIRGenModule::emitGlobalFunctionDefinition(clang::GlobalDecl gd,
                                                mlir::Operation *op) {
  auto const *funcDecl = cast<FunctionDecl>(gd.getDecl());
  if (clang::IdentifierInfo *identifier = funcDecl->getIdentifier()) {
    auto funcOp = builder.create<cir::FuncOp>(
        getLoc(funcDecl->getSourceRange()), identifier->getName());
    theModule.push_back(funcOp);
  } else {
    errorNYI(funcDecl->getSourceRange().getBegin(),
             "function definition with a non-identifier for a name");
  }
}

void CIRGenModule::emitGlobalVarDefinition(const clang::VarDecl *vd,
                                           bool isTentative) {
  mlir::Type type = getTypes().convertType(vd->getType());
  if (clang::IdentifierInfo *identifier = vd->getIdentifier()) {
    auto varOp = builder.create<cir::GlobalOp>(getLoc(vd->getSourceRange()),
                                               identifier->getName(), type);
    // TODO(CIR): This code for processing initial values is a placeholder
    // until class ConstantEmitter is upstreamed and the code for processing
    // constant expressions is filled out.  Only the most basic handling of
    // certain constant expressions is implemented for now.
    const VarDecl *initDecl;
    const Expr *initExpr = vd->getAnyInitializer(initDecl);
    if (initExpr) {
      mlir::Attribute initializer;
      if (APValue *value = initDecl->evaluateValue()) {
        switch (value->getKind()) {
        case APValue::Int: {
          initializer = builder.getAttr<cir::IntAttr>(type, value->getInt());
          break;
        }
        case APValue::Float: {
          initializer = builder.getAttr<cir::FPAttr>(type, value->getFloat());
          break;
        }
        case APValue::LValue: {
          if (value->getLValueBase()) {
            errorNYI(initExpr->getSourceRange(),
                     "non-null pointer initialization");
          } else {
            if (auto ptrType = mlir::dyn_cast<cir::PointerType>(type)) {
              initializer = builder.getConstPtrAttr(
                  ptrType, value->getLValueOffset().getQuantity());
            } else {
              llvm_unreachable(
                  "non-pointer variable initialized with a pointer");
            }
          }
          break;
        }
        default:
          errorNYI(initExpr->getSourceRange(), "unsupported initializer kind");
          break;
        }
      } else {
        errorNYI(initExpr->getSourceRange(), "non-constant initializer");
      }
      varOp.setInitialValueAttr(initializer);
    }
    theModule.push_back(varOp);
  } else {
    errorNYI(vd->getSourceRange().getBegin(),
             "variable definition with a non-identifier for a name");
  }
}

void CIRGenModule::emitGlobalDefinition(clang::GlobalDecl gd,
                                        mlir::Operation *op) {
  const auto *decl = cast<ValueDecl>(gd.getDecl());
  if (const auto *fd = dyn_cast<FunctionDecl>(decl)) {
    // TODO(CIR): Skip generation of CIR for functions with available_externally
    // linkage at -O0.

    if (const auto *method = dyn_cast<CXXMethodDecl>(decl)) {
      // Make sure to emit the definition(s) before we emit the thunks. This is
      // necessary for the generation of certain thunks.
      (void)method;
      errorNYI(method->getSourceRange(), "member function");
      return;
    }

    if (fd->isMultiVersion())
      errorNYI(fd->getSourceRange(), "multiversion functions");
    emitGlobalFunctionDefinition(gd, op);
    return;
  }

  if (const auto *vd = dyn_cast<VarDecl>(decl))
    return emitGlobalVarDefinition(vd, !vd->hasDefinition());

  llvm_unreachable("Invalid argument to CIRGenModule::emitGlobalDefinition");
}

// Emit code for a single top level declaration.
void CIRGenModule::emitTopLevelDecl(Decl *decl) {

  // Ignore dependent declarations.
  if (decl->isTemplated())
    return;

  switch (decl->getKind()) {
  default:
    errorNYI(decl->getBeginLoc(), "declaration of kind",
             decl->getDeclKindName());
    break;

  case Decl::Function: {
    auto *fd = cast<FunctionDecl>(decl);
    // Consteval functions shouldn't be emitted.
    if (!fd->isConsteval())
      emitGlobal(fd);
    break;
  }

  case Decl::Var: {
    auto *vd = cast<VarDecl>(decl);
    emitGlobal(vd);
    break;
  }
  }
}

DiagnosticBuilder CIRGenModule::errorNYI(SourceLocation loc,
                                         llvm::StringRef feature) {
  unsigned diagID = diags.getCustomDiagID(
      DiagnosticsEngine::Error, "ClangIR code gen Not Yet Implemented: %0");
  return diags.Report(loc, diagID) << feature;
}

DiagnosticBuilder CIRGenModule::errorNYI(SourceRange loc,
                                         llvm::StringRef feature) {
  return errorNYI(loc.getBegin(), feature) << loc;
}
