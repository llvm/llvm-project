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
#include "CIRGenCall.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenRecordLayout.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/ABI.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/IR/MLIRContext.h"

#include <utility>

namespace llvm {
class FunctionType;
class DataLayout;
class Type;
class LLVMContext;
class StructType;
} // namespace llvm

namespace clang {
class ASTContext;
template <typename> class CanQual;
class CXXConstructorDecl;
class CXXDestructorDecl;
class CXXMethodDecl;
class CodeGenOptions;
class FieldDecl;
class FunctionProtoType;
class ObjCInterfaceDecl;
class ObjCIvarDecl;
class PointerType;
class QualType;
class RecordDecl;
class TagDecl;
class TargetInfo;
class Type;
typedef CanQual<Type> CanQualType;
class GlobalDecl;

} // end namespace clang

namespace mlir {
class Type;
namespace cir {
class StructType;
} // namespace cir
} // namespace mlir

namespace cir {
class CallArgList;
class CIRGenCXXABI;
class CIRGenModule;
class CIRGenFunctionInfo;
class CIRGenBuilderTy;

/// This class organizes the cross-module state that is used while lowering
/// AST types to CIR types.
class CIRGenTypes {
  clang::ASTContext &Context;
  cir::CIRGenBuilderTy &Builder;
  CIRGenModule &CGM;
  const clang::TargetInfo &Target;
  CIRGenCXXABI &TheCXXABI;

  // This should not be moved earlier, since its initialization depends on some
  // of the previous reference members being already initialized
  const ABIInfo &TheABIInfo;

  /// Contains the CIR type for any converted RecordDecl.
  llvm::DenseMap<const clang::Type *, std::unique_ptr<CIRGenRecordLayout>>
      CIRGenRecordLayouts;

  /// Contains the CIR type for any converted RecordDecl
  llvm::DenseMap<const clang::Type *, mlir::cir::StructType> recordDeclTypes;

  /// Hold memoized CIRGenFunctionInfo results
  llvm::FoldingSet<CIRGenFunctionInfo> FunctionInfos;

  /// This set keeps track of records that we're currently converting to a CIR
  /// type. For example, when converting:
  /// struct A { struct B { int x; } } when processing 'x', the 'A' and 'B'
  /// types will be in this set.
  llvm::SmallPtrSet<const clang::Type *, 4> RecordsBeingLaidOut;

  llvm::SmallPtrSet<const CIRGenFunctionInfo *, 4> FunctionsBeingProcessed;

  /// True if we didn't layout a function due to being inside a recursive struct
  /// conversion, set this to true.
  bool SkippedLayout;

  llvm::SmallVector<const clang::RecordDecl *, 8> DeferredRecords;

  /// Heper for ConvertType.
  mlir::Type ConvertFunctionTypeInternal(clang::QualType FT);

public:
  CIRGenTypes(CIRGenModule &cgm);
  ~CIRGenTypes();

  cir::CIRGenBuilderTy &getBuilder() const { return Builder; }
  CIRGenModule &getModule() const { return CGM; }

  /// Utility to check whether a function type can be converted to a CIR type
  /// (i.e. doesn't depend on an incomplete tag type).
  bool isFuncTypeConvertible(const clang::FunctionType *FT);
  bool isFuncParamTypeConvertible(clang::QualType Ty);

  /// Convert clang calling convention to LLVM calling convention.
  unsigned ClangCallConvToCIRCallConv(clang::CallingConv CC);

  /// Derives the 'this' type for CIRGen purposes, i.e. ignoring method CVR
  /// qualification.
  clang::CanQualType DeriveThisType(const clang::CXXRecordDecl *RD,
                                    const clang::CXXMethodDecl *MD);

  /// This map keeps cache of llvm::Types and maps clang::Type to
  /// corresponding llvm::Type.
  using TypeCacheTy = llvm::DenseMap<const clang::Type *, mlir::Type>;
  TypeCacheTy TypeCache;

  clang::ASTContext &getContext() const { return Context; }
  mlir::MLIRContext &getMLIRContext() const;

  bool isRecordLayoutComplete(const clang::Type *Ty) const;
  bool noRecordsBeingLaidOut() const { return RecordsBeingLaidOut.empty(); }
  bool isRecordBeingLaidOut(const clang::Type *Ty) const {
    return RecordsBeingLaidOut.count(Ty);
  }

  /// Return whether a type can be zero-initialized (in the C++ sense) with an
  /// LLVM zeroinitializer.
  bool isZeroInitializable(clang::QualType T);

  /// Check if the pointer type can be zero-initialized (in the C++ sense)
  /// with an LLVM zeroinitializer.
  bool isPointerZeroInitializable(clang::QualType T);

  /// Return whether a record type can be zero-initialized (in the C++ sense)
  /// with an LLVM zeroinitializer.
  bool isZeroInitializable(const clang::RecordDecl *RD);

  const ABIInfo &getABIInfo() const { return TheABIInfo; }
  CIRGenCXXABI &getCXXABI() const { return TheCXXABI; }

  /// Convert type T into a mlir::Type.
  mlir::Type ConvertType(clang::QualType T);

  mlir::Type convertRecordDeclType(const clang::RecordDecl *recordDecl);

  std::unique_ptr<CIRGenRecordLayout>
  computeRecordLayout(const clang::RecordDecl *D, mlir::cir::StructType *Ty);

  std::string getRecordTypeName(const clang::RecordDecl *,
                                llvm::StringRef suffix);

  /// Determine if a C++ inheriting constructor should have parameters matching
  /// those of its inherited constructor.
  bool inheritingCtorHasParams(const clang::InheritedConstructor &Inherited,
                               clang::CXXCtorType Type);

  const CIRGenRecordLayout &getCIRGenRecordLayout(const clang::RecordDecl *RD);

  /// Convert type T into an mlir::Type. This differs from
  /// convertType in that it is used to convert to the memory representation
  /// for a type. For example, the scalar representation for _Bool is i1, but
  /// the memory representation is usually i8 or i32, depending on the target.
  // TODO: convert this comment to account for MLIR's equivalence
  mlir::Type convertTypeForMem(clang::QualType, bool forBitField = false);

  /// Get the CIR function type for \arg Info.
  mlir::cir::FuncType GetFunctionType(const CIRGenFunctionInfo &Info);

  mlir::cir::FuncType GetFunctionType(clang::GlobalDecl GD);

  /// Get the LLVM function type for use in a vtable, given a CXXMethodDecl. If
  /// the method to has an incomplete return type, and/or incomplete argument
  /// types, this will return the opaque type.
  mlir::cir::FuncType GetFunctionTypeForVTable(clang::GlobalDecl GD);

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

  const CIRGenFunctionInfo &arrangeGlobalDeclaration(clang::GlobalDecl GD);

  /// UpdateCompletedType - when we find the full definition for a TagDecl,
  /// replace the 'opaque' type we previously made for it if applicable.
  void UpdateCompletedType(const clang::TagDecl *TD);

  /// Free functions are functions that are compatible with an ordinary C
  /// function pointer type.
  const CIRGenFunctionInfo &
  arrangeFunctionDeclaration(const clang::FunctionDecl *FD);

  const CIRGenFunctionInfo &
  arrangeBuiltinFunctionCall(clang::QualType resultType,
                             const CallArgList &args);

  const CIRGenFunctionInfo &arrangeCXXConstructorCall(
      const CallArgList &Args, const clang::CXXConstructorDecl *D,
      clang::CXXCtorType CtorKind, unsigned ExtraPrefixArgs,
      unsigned ExtraSuffixArgs, bool PassProtoArgs = true);

  const CIRGenFunctionInfo &
  arrangeCXXMethodCall(const CallArgList &args,
                       const clang::FunctionProtoType *type,
                       RequiredArgs required, unsigned numPrefixArgs);

  /// C++ methods have some special rules and also have implicit parameters.
  const CIRGenFunctionInfo &
  arrangeCXXMethodDeclaration(const clang::CXXMethodDecl *MD);
  const CIRGenFunctionInfo &arrangeCXXStructorDeclaration(clang::GlobalDecl GD);

  const CIRGenFunctionInfo &
  arrangeCXXMethodType(const clang::CXXRecordDecl *RD,
                       const clang::FunctionProtoType *FTP,
                       const clang::CXXMethodDecl *MD);

  const CIRGenFunctionInfo &
  arrangeFreeFunctionCall(const CallArgList &Args,
                          const clang::FunctionType *Ty, bool ChainCall);

  const CIRGenFunctionInfo &
  arrangeFreeFunctionType(clang::CanQual<clang::FunctionProtoType> Ty);

  const CIRGenFunctionInfo &
  arrangeFreeFunctionType(clang::CanQual<clang::FunctionNoProtoType> FTNP);

  /// "Arrange" the LLVM information for a call or type with the given
  /// signature.  This is largely an internal method; other clients
  /// should use one of the above routines, which ultimately defer to
  /// this.
  ///
  /// \param argTypes - must all actually be canonical as params
  const CIRGenFunctionInfo &arrangeCIRFunctionInfo(
      clang::CanQualType returnType, FnInfoOpts opts,
      llvm::ArrayRef<clang::CanQualType> argTypes,
      clang::FunctionType::ExtInfo info,
      llvm::ArrayRef<clang::FunctionProtoType::ExtParameterInfo> paramInfos,
      RequiredArgs args);
};
} // namespace cir

#endif
