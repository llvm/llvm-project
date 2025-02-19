//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal per-function state used for AST-to-ClangIR code gen
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"

#include "clang/AST/GlobalDecl.h"

#include <cassert>

namespace clang::CIRGen {

CIRGenFunction::CIRGenFunction(CIRGenModule &cgm, CIRGenBuilderTy &builder,
                               bool suppressNewContext)
    : CIRGenTypeCache(cgm), cgm{cgm}, builder(builder) {}

CIRGenFunction::~CIRGenFunction() {}

// This is copied from clang/lib/CodeGen/CodeGenFunction.cpp
cir::TypeEvaluationKind CIRGenFunction::getEvaluationKind(QualType type) {
  type = type.getCanonicalType();
  while (true) {
    switch (type->getTypeClass()) {
#define TYPE(name, parent)
#define ABSTRACT_TYPE(name, parent)
#define NON_CANONICAL_TYPE(name, parent) case Type::name:
#define DEPENDENT_TYPE(name, parent) case Type::name:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(name, parent) case Type::name:
#include "clang/AST/TypeNodes.inc"
      llvm_unreachable("non-canonical or dependent type in IR-generation");

    case Type::ArrayParameter:
    case Type::HLSLAttributedResource:
      llvm_unreachable("NYI");

    case Type::Auto:
    case Type::DeducedTemplateSpecialization:
      llvm_unreachable("undeduced type in IR-generation");

    // Various scalar types.
    case Type::Builtin:
    case Type::Pointer:
    case Type::BlockPointer:
    case Type::LValueReference:
    case Type::RValueReference:
    case Type::MemberPointer:
    case Type::Vector:
    case Type::ExtVector:
    case Type::ConstantMatrix:
    case Type::FunctionProto:
    case Type::FunctionNoProto:
    case Type::Enum:
    case Type::ObjCObjectPointer:
    case Type::Pipe:
    case Type::BitInt:
      return cir::TEK_Scalar;

    // Complexes.
    case Type::Complex:
      return cir::TEK_Complex;

    // Arrays, records, and Objective-C objects.
    case Type::ConstantArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
    case Type::Record:
    case Type::ObjCObject:
    case Type::ObjCInterface:
      return cir::TEK_Aggregate;

    // We operate on atomic values according to their underlying type.
    case Type::Atomic:
      type = cast<AtomicType>(type)->getValueType();
      continue;
    }
    llvm_unreachable("unknown type kind!");
  }
}

mlir::Type CIRGenFunction::convertTypeForMem(QualType t) {
  return cgm.getTypes().convertTypeForMem(t);
}

mlir::Type CIRGenFunction::convertType(QualType t) {
  return cgm.getTypes().convertType(t);
}

mlir::Location CIRGenFunction::getLoc(SourceLocation srcLoc) {
  // Some AST nodes might contain invalid source locations (e.g.
  // CXXDefaultArgExpr), workaround that to still get something out.
  if (srcLoc.isValid()) {
    const SourceManager &sm = getContext().getSourceManager();
    PresumedLoc pLoc = sm.getPresumedLoc(srcLoc);
    StringRef filename = pLoc.getFilename();
    return mlir::FileLineColLoc::get(builder.getStringAttr(filename),
                                     pLoc.getLine(), pLoc.getColumn());
  }
  // Do our best...
  assert(currSrcLoc && "expected to inherit some source location");
  return *currSrcLoc;
}

mlir::Location CIRGenFunction::getLoc(SourceRange srcLoc) {
  // Some AST nodes might contain invalid source locations (e.g.
  // CXXDefaultArgExpr), workaround that to still get something out.
  if (srcLoc.isValid()) {
    mlir::Location beg = getLoc(srcLoc.getBegin());
    mlir::Location end = getLoc(srcLoc.getEnd());
    SmallVector<mlir::Location, 2> locs = {beg, end};
    mlir::Attribute metadata;
    return mlir::FusedLoc::get(locs, metadata, &getMLIRContext());
  }
  if (currSrcLoc) {
    return *currSrcLoc;
  }
  // We're brave, but time to give up.
  return builder.getUnknownLoc();
}

mlir::Location CIRGenFunction::getLoc(mlir::Location lhs, mlir::Location rhs) {
  SmallVector<mlir::Location, 2> locs = {lhs, rhs};
  mlir::Attribute metadata;
  return mlir::FusedLoc::get(locs, metadata, &getMLIRContext());
}

void CIRGenFunction::startFunction(GlobalDecl gd, QualType returnType,
                                   cir::FuncOp fn, cir::FuncType funcType,
                                   SourceLocation loc,
                                   SourceLocation startLoc) {
  assert(!curFn &&
         "CIRGenFunction can only be used for one function at a time");

  fnRetTy = returnType;
  curFn = fn;

  mlir::Block *entryBB = &fn.getBlocks().front();
  builder.setInsertionPointToStart(entryBB);
}

void CIRGenFunction::finishFunction(SourceLocation endLoc) {}

mlir::LogicalResult CIRGenFunction::emitFunctionBody(const clang::Stmt *body) {
  auto result = mlir::LogicalResult::success();
  if (const CompoundStmt *block = dyn_cast<CompoundStmt>(body))
    emitCompoundStmtWithoutScope(*block);
  else
    result = emitStmt(body, /*useCurrentScope=*/true);
  return result;
}

cir::FuncOp CIRGenFunction::generateCode(clang::GlobalDecl gd, cir::FuncOp fn,
                                         cir::FuncType funcType) {
  const auto funcDecl = cast<FunctionDecl>(gd.getDecl());
  SourceLocation loc = funcDecl->getLocation();
  Stmt *body = funcDecl->getBody();
  SourceRange bodyRange =
      body ? body->getSourceRange() : funcDecl->getLocation();

  SourceLocRAIIObject fnLoc{*this, loc.isValid() ? getLoc(loc)
                                                 : builder.getUnknownLoc()};

  // This will be used once more code is upstreamed.
  [[maybe_unused]] mlir::Block *entryBB = fn.addEntryBlock();

  startFunction(gd, funcDecl->getReturnType(), fn, funcType, loc,
                bodyRange.getBegin());

  if (isa<CXXDestructorDecl>(funcDecl))
    getCIRGenModule().errorNYI(bodyRange, "C++ destructor definition");
  else if (isa<CXXConstructorDecl>(funcDecl))
    getCIRGenModule().errorNYI(bodyRange, "C++ constructor definition");
  else if (getLangOpts().CUDA && !getLangOpts().CUDAIsDevice &&
           funcDecl->hasAttr<CUDAGlobalAttr>())
    getCIRGenModule().errorNYI(bodyRange, "CUDA kernel");
  else if (isa<CXXMethodDecl>(funcDecl) &&
           cast<CXXMethodDecl>(funcDecl)->isLambdaStaticInvoker())
    getCIRGenModule().errorNYI(bodyRange, "Lambda static invoker");
  else if (funcDecl->isDefaulted() && isa<CXXMethodDecl>(funcDecl) &&
           (cast<CXXMethodDecl>(funcDecl)->isCopyAssignmentOperator() ||
            cast<CXXMethodDecl>(funcDecl)->isMoveAssignmentOperator()))
    getCIRGenModule().errorNYI(bodyRange, "Default assignment operator");
  else if (body) {
    if (mlir::failed(emitFunctionBody(body))) {
      fn.erase();
      return nullptr;
    }
  } else
    llvm_unreachable("no definition for normal function");

  // This code to insert a cir.return or cir.trap at the end of the function is
  // temporary until the function return code, including
  // CIRGenFunction::LexicalScope::emitImplicitReturn(), is upstreamed.
  mlir::Block &lastBlock = fn.getRegion().back();
  if (lastBlock.empty() || !lastBlock.mightHaveTerminator() ||
      !lastBlock.getTerminator()->hasTrait<mlir::OpTrait::IsTerminator>()) {
    builder.setInsertionPointToEnd(&lastBlock);
    if (mlir::isa<cir::VoidType>(funcType.getReturnType())) {
      builder.create<cir::ReturnOp>(getLoc(bodyRange.getEnd()));
    } else {
      builder.create<cir::TrapOp>(getLoc(bodyRange.getEnd()));
    }
  }

  if (mlir::failed(fn.verifyBody()))
    return nullptr;

  finishFunction(bodyRange.getEnd());

  return fn;
}

} // namespace clang::CIRGen
