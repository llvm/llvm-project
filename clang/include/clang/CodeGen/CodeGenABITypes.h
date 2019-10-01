//==---- CodeGenABITypes.h - Convert Clang types to LLVM types for ABI -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CodeGenABITypes is a simple interface for getting LLVM types for
// the parameters and the return value of a function given the Clang
// types.
//
// The class is implemented as a public wrapper around the private
// CodeGenTypes class in lib/CodeGen.
//
// It allows other clients, like LLDB, to determine the LLVM types that are
// actually used in function calls, which makes it possible to then determine
// the actual ABI locations (e.g. registers, stack locations, etc.) that
// these parameters are stored in.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGEN_CODEGENABITYPES_H
#define LLVM_CLANG_CODEGEN_CODEGENABITYPES_H

#include "clang/AST/CanonicalType.h"
#include "clang/AST/Type.h"
#include "clang/CodeGen/CGFunctionInfo.h"

namespace llvm {
  class DataLayout;
  class Module;
  class Function;
  class FunctionType;
  class Type;
}

namespace clang {
class ASTContext;
class CXXRecordDecl;
class CXXMethodDecl;
class CodeGenOptions;
class CoverageSourceInfo;
class DiagnosticsEngine;
class HeaderSearchOptions;
class ObjCMethodDecl;
class PreprocessorOptions;

namespace CodeGen {
class CGFunctionInfo;
class CodeGenModule;

const CGFunctionInfo &arrangeObjCMessageSendSignature(CodeGenModule &CGM,
                                                      const ObjCMethodDecl *MD,
                                                      QualType receiverType);

const CGFunctionInfo &arrangeFreeFunctionType(CodeGenModule &CGM,
                                              CanQual<FunctionProtoType> Ty);

const CGFunctionInfo &arrangeFreeFunctionType(CodeGenModule &CGM,
                                              CanQual<FunctionNoProtoType> Ty);

const CGFunctionInfo &arrangeCXXMethodType(CodeGenModule &CGM,
                                           const CXXRecordDecl *RD,
                                           const FunctionProtoType *FTP,
                                           const CXXMethodDecl *MD);

const CGFunctionInfo &arrangeFreeFunctionCall(CodeGenModule &CGM,
                                              CanQualType returnType,
                                              ArrayRef<CanQualType> argTypes,
                                              FunctionType::ExtInfo info,
                                              RequiredArgs args);

/// Returns null if the function type is incomplete and can't be lowered.
llvm::FunctionType *convertFreeFunctionType(CodeGenModule &CGM,
                                            const FunctionDecl *FD);

llvm::Type *convertTypeForMemory(CodeGenModule &CGM, QualType T);

/// Given a non-bitfield struct field, return its index within the elements of
/// the struct's converted type.  The returned index refers to a field number in
/// the complete object type which is returned by convertTypeForMemory.  FD must
/// be a field in RD directly (i.e. not an inherited field).
unsigned getLLVMFieldNumber(CodeGenModule &CGM,
                            const RecordDecl *RD, const FieldDecl *FD);

/// Returns the default constructor for a C struct with non-trivially copyable
/// fields, generating it if necessary. The returned function uses the `cdecl`
/// calling convention, returns void, and takes a single argument that is a
/// pointer to the address of the struct.
llvm::Function *getNonTrivialCStructDefaultConstructor(CodeGenModule &GCM,
                                                       CharUnits DstAlignment,
                                                       bool IsVolatile,
                                                       QualType QT);

/// Returns the copy constructor for a C struct with non-trivially copyable
/// fields, generating it if necessary. The returned function uses the `cdecl`
/// calling convention, returns void, and takes two arguments: pointers to the
/// addresses of the destination and source structs, respectively.
llvm::Function *getNonTrivialCStructCopyConstructor(CodeGenModule &CGM,
                                                    CharUnits DstAlignment,
                                                    CharUnits SrcAlignment,
                                                    bool IsVolatile,
                                                    QualType QT);

/// Returns the move constructor for a C struct with non-trivially copyable
/// fields, generating it if necessary. The returned function uses the `cdecl`
/// calling convention, returns void, and takes two arguments: pointers to the
/// addresses of the destination and source structs, respectively.
llvm::Function *getNonTrivialCStructMoveConstructor(CodeGenModule &CGM,
                                                    CharUnits DstAlignment,
                                                    CharUnits SrcAlignment,
                                                    bool IsVolatile,
                                                    QualType QT);

/// Returns the copy assignment operator for a C struct with non-trivially
/// copyable fields, generating it if necessary. The returned function uses the
/// `cdecl` calling convention, returns void, and takes two arguments: pointers
/// to the addresses of the destination and source structs, respectively.
llvm::Function *getNonTrivialCStructCopyAssignmentOperator(
    CodeGenModule &CGM, CharUnits DstAlignment, CharUnits SrcAlignment,
    bool IsVolatile, QualType QT);

/// Return the move assignment operator for a C struct with non-trivially
/// copyable fields, generating it if necessary. The returned function uses the
/// `cdecl` calling convention, returns void, and takes two arguments: pointers
/// to the addresses of the destination and source structs, respectively.
llvm::Function *getNonTrivialCStructMoveAssignmentOperator(
    CodeGenModule &CGM, CharUnits DstAlignment, CharUnits SrcAlignment,
    bool IsVolatile, QualType QT);

/// Returns the destructor for a C struct with non-trivially copyable fields,
/// generating it if necessary. The returned function uses the `cdecl` calling
/// convention, returns void, and takes a single argument that is a pointer to
/// the address of the struct.
llvm::Function *getNonTrivialCStructDestructor(CodeGenModule &CGM,
                                               CharUnits DstAlignment,
                                               bool IsVolatile, QualType QT);

}  // end namespace CodeGen
}  // end namespace clang

#endif
