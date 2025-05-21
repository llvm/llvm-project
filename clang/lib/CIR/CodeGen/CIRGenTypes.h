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

#include "clang/AST/DeclCXX.h"
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
class CIRGenCXXABI;
class CIRGenModule;

/// This class organizes the cross-module state that is used while lowering
/// AST types to CIR types.
class CIRGenTypes {
  CIRGenModule &cgm;
  clang::ASTContext &astContext;
  CIRGenBuilderTy &builder;
  CIRGenCXXABI &theCXXABI;

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

  llvm::SmallVector<const clang::RecordDecl *, 8> deferredRecords;

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

  /// Derives the 'this' type for CIRGen purposes, i.e. ignoring method CVR
  /// qualification.
  clang::CanQualType deriveThisType(const clang::CXXRecordDecl *rd,
                                    const clang::CXXMethodDecl *md);

  /// This map of clang::Type to mlir::Type (which includes CIR type) is a
  /// cache of types that have already been processed.
  using TypeCacheTy = llvm::DenseMap<const clang::Type *, mlir::Type>;
  TypeCacheTy typeCache;

  mlir::MLIRContext &getMLIRContext() const;
  clang::ASTContext &getASTContext() const { return astContext; }

  bool isRecordLayoutComplete(const clang::Type *ty) const;
  bool noRecordsBeingLaidOut() const { return recordsBeingLaidOut.empty(); }
  bool isRecordBeingLaidOut(const clang::Type *ty) const {
    return recordsBeingLaidOut.count(ty);
  }

  const ABIInfo &getABIInfo() const { return theABIInfo; }

  /// Convert a Clang type into a mlir::Type.
  mlir::Type convertType(clang::QualType type);

  mlir::Type convertRecordDeclType(const clang::RecordDecl *recordDecl);

  std::unique_ptr<CIRGenRecordLayout>
  computeRecordLayout(const clang::RecordDecl *rd, cir::RecordType *ty);

  std::string getRecordTypeName(const clang::RecordDecl *,
                                llvm::StringRef suffix);

  const CIRGenRecordLayout &getCIRGenRecordLayout(const clang::RecordDecl *rd);

  /// Convert type T into an mlir::Type. This differs from convertType in that
  /// it is used to convert to the memory representation for a type. For
  /// example, the scalar representation for bool is i1, but the memory
  /// representation is usually i8 or i32, depending on the target.
  // TODO: convert this comment to account for MLIR's equivalence
  mlir::Type convertTypeForMem(clang::QualType, bool forBitField = false);

  /// Get the CIR function type for \arg Info.
  cir::FuncType getFunctionType(const CIRGenFunctionInfo &info);

  // The arrangement methods are split into three families:
  //   - those meant to drive the signature and prologue/epilogue
  //     of a function declaration or definition,
  //   - those meant for the computation of the CIR type for an abstract
  //     appearance of a function, and
  //   - those meant for performing the CIR-generation of a call.
  // They differ mainly in how they deal with optional (i.e. variadic)
  // arguments, as well as unprototyped functions.
  //
  // Key points:
  // - The CIRGenFunctionInfo for emitting a specific call site must include
  //   entries for the optional arguments.
  // - The function type used at the call site must reflect the formal
  // signature
  //   of the declaration being called, or else the call will go away.
  // - For the most part, unprototyped functions are called by casting to a
  //   formal signature inferred from the specific argument types used at the
  //   call-site. However, some targets (e.g. x86-64) screw with this for
  //   compatability reasons.

  const CIRGenFunctionInfo &arrangeGlobalDeclaration(GlobalDecl gd);

  /// Free functions are functions that are compatible with an ordinary C
  /// function pointer type.
  const CIRGenFunctionInfo &
  arrangeFunctionDeclaration(const clang::FunctionDecl *fd);

  /// Return whether a type can be zero-initialized (in the C++ sense) with an
  /// LLVM zeroinitializer.
  bool isZeroInitializable(clang::QualType ty);
  bool isZeroInitializable(const RecordDecl *rd);

  const CIRGenFunctionInfo &
  arrangeCXXMethodCall(const CallArgList &args,
                       const clang::FunctionProtoType *type,
                       RequiredArgs required, unsigned numPrefixArgs);

  /// C++ methods have some special rules and also have implicit parameters.
  const CIRGenFunctionInfo &
  arrangeCXXMethodDeclaration(const clang::CXXMethodDecl *md);

  const CIRGenFunctionInfo &
  arrangeCXXMethodType(const clang::CXXRecordDecl *rd,
                       const clang::FunctionProtoType *ftp,
                       const clang::CXXMethodDecl *md);

  const CIRGenFunctionInfo &arrangeFreeFunctionCall(const CallArgList &args,
                                                    const FunctionType *fnType);

  const CIRGenFunctionInfo &
  arrangeCIRFunctionInfo(CanQualType returnType,
                         llvm::ArrayRef<CanQualType> argTypes,
                         RequiredArgs required);

  const CIRGenFunctionInfo &
  arrangeFreeFunctionType(CanQual<FunctionProtoType> fpt);
  const CIRGenFunctionInfo &
  arrangeFreeFunctionType(CanQual<FunctionNoProtoType> fnpt);
};

} // namespace clang::CIRGen

#endif
