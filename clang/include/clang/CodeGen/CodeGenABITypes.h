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
#include "clang/Basic/ABI.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/Support/Compiler.h"
#include "llvm/IR/BasicBlock.h"

namespace llvm {
class AttrBuilder;
class Constant;
class Function;
class FunctionType;
class Type;
}

namespace clang {
class CXXConstructorDecl;
class CXXDestructorDecl;
class CXXRecordDecl;
class CXXMethodDecl;
class GlobalDecl;
class ObjCMethodDecl;
class ObjCProtocolDecl;

namespace CodeGen {
class CGFunctionInfo;
class CodeGenModule;

/// Additional implicit arguments to add to a constructor argument list.
struct ImplicitCXXConstructorArgs {
  /// Implicit arguments to add before the explicit arguments, but after the
  /// `*this` argument (which always comes first).
  SmallVector<llvm::Value *, 1> Prefix;

  /// Implicit arguments to add after the explicit arguments.
  SmallVector<llvm::Value *, 1> Suffix;
};

CLANG_ABI const CGFunctionInfo &arrangeObjCMessageSendSignature(CodeGenModule &CGM,
                                                      const ObjCMethodDecl *MD,
                                                      QualType receiverType);

CLANG_ABI const CGFunctionInfo &arrangeFreeFunctionType(CodeGenModule &CGM,
                                              CanQual<FunctionProtoType> Ty);

CLANG_ABI const CGFunctionInfo &arrangeFreeFunctionType(CodeGenModule &CGM,
                                              CanQual<FunctionNoProtoType> Ty);

CLANG_ABI const CGFunctionInfo &arrangeCXXMethodType(CodeGenModule &CGM,
                                           const CXXRecordDecl *RD,
                                           const FunctionProtoType *FTP,
                                           const CXXMethodDecl *MD);

CLANG_ABI const CGFunctionInfo &arrangeFreeFunctionCall(CodeGenModule &CGM,
                                              CanQualType returnType,
                                              ArrayRef<CanQualType> argTypes,
                                              FunctionType::ExtInfo info,
                                              RequiredArgs args);

/// Returns the implicit arguments to add to a complete, non-delegating C++
/// constructor call.
CLANG_ABI ImplicitCXXConstructorArgs
getImplicitCXXConstructorArgs(CodeGenModule &CGM, const CXXConstructorDecl *D);

CLANG_ABI llvm::Value *
getCXXDestructorImplicitParam(CodeGenModule &CGM, llvm::BasicBlock *InsertBlock,
                              llvm::BasicBlock::iterator InsertPoint,
                              const CXXDestructorDecl *D, CXXDtorType Type,
                              bool ForVirtualBase, bool Delegating);

/// Returns null if the function type is incomplete and can't be lowered.
CLANG_ABI llvm::FunctionType *convertFreeFunctionType(CodeGenModule &CGM,
                                            const FunctionDecl *FD);

CLANG_ABI llvm::Type *convertTypeForMemory(CodeGenModule &CGM, QualType T);

/// Given a non-bitfield struct field, return its index within the elements of
/// the struct's converted type.  The returned index refers to a field number in
/// the complete object type which is returned by convertTypeForMemory.  FD must
/// be a field in RD directly (i.e. not an inherited field).
CLANG_ABI unsigned getLLVMFieldNumber(CodeGenModule &CGM,
                            const RecordDecl *RD, const FieldDecl *FD);

/// Return a declaration discriminator for the given global decl.
CLANG_ABI uint16_t getPointerAuthDeclDiscriminator(CodeGenModule &CGM, GlobalDecl GD);

/// Return a type discriminator for the given function type.
CLANG_ABI uint16_t getPointerAuthTypeDiscriminator(CodeGenModule &CGM,
                                         QualType FunctionType);

/// Given the language and code-generation options that Clang was configured
/// with, set the default LLVM IR attributes for a function definition.
/// The attributes set here are mostly global target-configuration and
/// pipeline-configuration options like the target CPU, variant stack
/// rules, whether to optimize for size, and so on.  This is useful for
/// frontends (such as Swift) that generally intend to interoperate with
/// C code and rely on Clang's target configuration logic.
///
/// As a general rule, this function assumes that meaningful attributes
/// haven't already been added to the builder.  It won't intentionally
/// displace any existing attributes, but it also won't check to avoid
/// overwriting them.  Callers should generally apply customizations after
/// making this call.
///
/// This function assumes that the caller is not defining a function that
/// requires special no-builtin treatment.
CLANG_ABI void addDefaultFunctionDefinitionAttributes(CodeGenModule &CGM,
                                            llvm::AttrBuilder &attrs);

/// Returns the default constructor for a C struct with non-trivially copyable
/// fields, generating it if necessary. The returned function uses the `cdecl`
/// calling convention, returns void, and takes a single argument that is a
/// pointer to the address of the struct.
CLANG_ABI llvm::Function *getNonTrivialCStructDefaultConstructor(CodeGenModule &GCM,
                                                       CharUnits DstAlignment,
                                                       bool IsVolatile,
                                                       QualType QT);

/// Returns the copy constructor for a C struct with non-trivially copyable
/// fields, generating it if necessary. The returned function uses the `cdecl`
/// calling convention, returns void, and takes two arguments: pointers to the
/// addresses of the destination and source structs, respectively.
CLANG_ABI llvm::Function *getNonTrivialCStructCopyConstructor(CodeGenModule &CGM,
                                                    CharUnits DstAlignment,
                                                    CharUnits SrcAlignment,
                                                    bool IsVolatile,
                                                    QualType QT);

/// Returns the move constructor for a C struct with non-trivially copyable
/// fields, generating it if necessary. The returned function uses the `cdecl`
/// calling convention, returns void, and takes two arguments: pointers to the
/// addresses of the destination and source structs, respectively.
CLANG_ABI llvm::Function *getNonTrivialCStructMoveConstructor(CodeGenModule &CGM,
                                                    CharUnits DstAlignment,
                                                    CharUnits SrcAlignment,
                                                    bool IsVolatile,
                                                    QualType QT);

/// Returns the copy assignment operator for a C struct with non-trivially
/// copyable fields, generating it if necessary. The returned function uses the
/// `cdecl` calling convention, returns void, and takes two arguments: pointers
/// to the addresses of the destination and source structs, respectively.
CLANG_ABI llvm::Function *getNonTrivialCStructCopyAssignmentOperator(
    CodeGenModule &CGM, CharUnits DstAlignment, CharUnits SrcAlignment,
    bool IsVolatile, QualType QT);

/// Return the move assignment operator for a C struct with non-trivially
/// copyable fields, generating it if necessary. The returned function uses the
/// `cdecl` calling convention, returns void, and takes two arguments: pointers
/// to the addresses of the destination and source structs, respectively.
CLANG_ABI llvm::Function *getNonTrivialCStructMoveAssignmentOperator(
    CodeGenModule &CGM, CharUnits DstAlignment, CharUnits SrcAlignment,
    bool IsVolatile, QualType QT);

/// Returns the destructor for a C struct with non-trivially copyable fields,
/// generating it if necessary. The returned function uses the `cdecl` calling
/// convention, returns void, and takes a single argument that is a pointer to
/// the address of the struct.
CLANG_ABI llvm::Function *getNonTrivialCStructDestructor(CodeGenModule &CGM,
                                               CharUnits DstAlignment,
                                               bool IsVolatile, QualType QT);

/// Get a pointer to a protocol object for the given declaration, emitting it if
/// it hasn't already been emitted in this translation unit. Note that the ABI
/// for emitting a protocol reference in code (e.g. for a protocol expression)
/// in most runtimes is not as simple as just materializing a pointer to this
/// object.
CLANG_ABI llvm::Constant *emitObjCProtocolObject(CodeGenModule &CGM,
                                       const ObjCProtocolDecl *p);
}  // end namespace CodeGen
}  // end namespace clang

#endif
