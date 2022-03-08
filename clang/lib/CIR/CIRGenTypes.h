//===--- CIRGenTypes.h - Type translation for LLVM CodeGen -----*- C++ -*-===//
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

#include "mlir/Dialect/CIR/IR/CIRTypes.h"
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
class OpBuilder;
namespace cir {
class StructType;
} // namespace cir
} // namespace mlir

namespace cir {
class CIRGenCXXABI;
class CIRGenModule;

/// This class organizes the cross-module state that is used while lowering
/// AST types to CIR types.
class CIRGenTypes {
  clang::ASTContext &Context;
  mlir::OpBuilder &Builder;
  CIRGenModule &CGM;
  CIRGenCXXABI &TheCXXABI;

  llvm::DenseMap<const clang::Type *, mlir::cir::StructType> recordDeclTypes;

public:
  CIRGenTypes(CIRGenModule &cgm);
  ~CIRGenTypes();

  /// This map keeps cache of llvm::Types and maps clang::Type to
  /// corresponding llvm::Type.
  using TypeCacheTy = llvm::DenseMap<const clang::Type *, mlir::Type>;
  TypeCacheTy TypeCache;

  clang::ASTContext &getContext() const { return Context; }
  mlir::MLIRContext &getMLIRContext() const;

  CIRGenCXXABI &getCXXABI() const { return TheCXXABI; }
  /// ConvertType - Convert type T into a mlir::Type.
  mlir::Type ConvertType(clang::QualType T);

  mlir::Type convertRecordDeclType(const clang::RecordDecl *recordDecl);

  mlir::cir::StructType computeRecordLayout(const clang::RecordDecl *);

  std::string getRecordTypeName(const clang::RecordDecl *,
                                llvm::StringRef suffix);

  /// convertTypeForMem - Convert type T into an mlir::Type. This differs from
  /// convertType in that it is used to convert to the memory representation for
  /// a type. For example, the scalar representation for _Bool is i1, but the
  /// memory representation is usually i8 or i32, depending on the target.
  // TODO: convert this comment to account for MLIR's equivalence
  mlir::Type convertTypeForMem(clang::QualType, bool forBitField = false);
};
} // namespace cir

#endif
