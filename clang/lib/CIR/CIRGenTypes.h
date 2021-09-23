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

#include "clang/CodeGen/CGFunctionInfo.h"
#include "llvm/ADT/DenseMap.h"

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

namespace CodeGen {
class ABIInfo;
class CGCXXABI;
class CGRecordLayout;
class CodeGenModule;
class RequiredArgs;
} // end namespace CodeGen
} // end namespace clang

namespace mlir {
class Type;
class OpBuilder;
} // namespace mlir

/// This class organizes the cross-module state that is used while lowering
/// AST types to CIR types.
namespace cir {
class CIRGenTypes {
  clang::ASTContext &Context;
  mlir::OpBuilder &Builder;

public:
  CIRGenTypes(clang::ASTContext &Ctx, mlir::OpBuilder &B);
  ~CIRGenTypes();

  /// This map keeps cache of llvm::Types and maps clang::Type to
  /// corresponding llvm::Type.
  using TypeCacheTy = llvm::DenseMap<const clang::Type *, mlir::Type>;
  TypeCacheTy TypeCache;

  clang::ASTContext &getContext() const { return Context; }

  /// ConvertType - Convert type T into a mlir::Type.
  mlir::Type ConvertType(clang::QualType T);
};
} // namespace cir

#endif
