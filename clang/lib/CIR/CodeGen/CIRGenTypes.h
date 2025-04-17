//===--- CIRGenTypes.h - Type translation for CIR CodeGen -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the code that handles AST -> CIR type lowering.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CODEGENTYPES_H
#define LLVM_CLANG_LIB_CODEGEN_CODEGENTYPES_H

#include "ABIInfo.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenRecordLayout.h"

#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "llvm/ADT/SmallPtrSet.h"

namespace clang {
class ASTContext;
class FunctionType;
class GlobalDecl;
class QualType;
class Type;
} // namespace clang

namespace mlir {
class Type;
}

namespace clang::CIRGen {

class CallArgList;
class CIRGenBuilderTy;
class CIRGenModule;

/// This class organizes the cross-module state that is used while lowering
/// AST types to CIR types.
class CIRGenTypes {
  CIRGenModule &cgm;
  clang::ASTContext &astContext;
  CIRGenBuilderTy &builder;

  const ABIInfo &theABIInfo;

  /// Contains the CIR type for any converted RecordDecl.
  llvm::DenseMap<const clang::Type *, std::unique_ptr<CIRGenRecordLayout>>
      cirGenRecordLayouts;

  /// Contains the CIR type for any converted RecordDecl
  llvm::DenseMap<const clang::Type *, cir::RecordType> recordDeclTypes;

  /// Hold memoized CIRGenFunctionInfo results
  llvm::FoldingSet<CIRGenFunctionInfo> functionInfos;

  /// This set keeps track of records that we're currently converting to a CIR
  /// type. For example, when converting:
  /// struct A { struct B { int x; } } when processing 'x', the 'A' and 'B'
  /// types will be in this set.
  llvm::SmallPtrSet<const clang::Type *, 4> recordsBeingLaidOut;

  llvm::SmallPtrSet<const CIRGenFunctionInfo *, 4> functionsBeingProcessed;
  /// Heper for convertType.
  mlir::Type convertFunctionTypeInternal(clang::QualType ft);

public:
  CIRGenTypes(CIRGenModule &cgm);
  ~CIRGenTypes();

  CIRGenBuilderTy &getBuilder() const { return builder; }
  CIRGenModule &getCGModule() const { return cgm; }

  /// Utility to check whether a function type can be converted to a CIR type
  /// (i.e. doesn't depend on an incomplete tag type).
  bool isFuncTypeConvertible(const clang::FunctionType *ft);
  bool isFuncParamTypeConvertible(clang::QualType type);

  /// This map of clang::Type to mlir::Type (which includes CIR type) is a
  /// cache of types that have already been processed.
  using TypeCacheTy = llvm::DenseMap<const clang::Type *, mlir::Type>;
  TypeCacheTy typeCache;

  mlir::MLIRContext &getMLIRContext() const;
  clang::ASTContext &getASTContext() const { return astContext; }

  bool noRecordsBeingLaidOut() const { return recordsBeingLaidOut.empty(); }

  const ABIInfo &getABIInfo() const { return theABIInfo; }

  /// Convert a Clang type into a mlir::Type.
  mlir::Type convertType(clang::QualType type);

  mlir::Type convertRecordDeclType(const clang::RecordDecl *recordDecl);

  std::unique_ptr<CIRGenRecordLayout>
  computeRecordLayout(const clang::RecordDecl *rd, cir::RecordType *ty);

  std::string getRecordTypeName(const clang::RecordDecl *,
                                llvm::StringRef suffix);

  /// Convert type T into an mlir::Type. This differs from convertType in that
  /// it is used to convert to the memory representation for a type. For
  /// example, the scalar representation for bool is i1, but the memory
  /// representation is usually i8 or i32, depending on the target.
  // TODO: convert this comment to account for MLIR's equivalence
  mlir::Type convertTypeForMem(clang::QualType, bool forBitField = false);

  /// Return whether a type can be zero-initialized (in the C++ sense) with an
  /// LLVM zeroinitializer.
  bool isZeroInitializable(clang::QualType ty);

  const CIRGenFunctionInfo &arrangeFreeFunctionCall(const FunctionType *fnType);

  const CIRGenFunctionInfo &arrangeCIRFunctionInfo(CanQualType returnType);
};

} // namespace clang::CIRGen

#endif
