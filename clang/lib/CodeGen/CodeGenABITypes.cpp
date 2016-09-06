//==--- CodeGenABITypes.cpp - Convert Clang types to LLVM types for ABI ----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/CodeGenABITypes.h"
#include "CodeGenModule.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/PreprocessorOptions.h"

using namespace clang;
using namespace CodeGen;

const CGFunctionInfo &
CodeGen::arrangeObjCMessageSendSignature(CodeGenModule &CGM,
                                         const ObjCMethodDecl *MD,
                                         QualType receiverType) {
  return CGM.getTypes().arrangeObjCMessageSendSignature(MD, receiverType);
}

const CGFunctionInfo &
CodeGen::arrangeFreeFunctionType(CodeGenModule &CGM,
                                 CanQual<FunctionProtoType> Ty,
                                 const FunctionDecl *FD) {
  return CGM.getTypes().arrangeFreeFunctionType(Ty, FD);
}

const CGFunctionInfo &
CodeGen::arrangeFreeFunctionType(CodeGenModule &CGM,
                                 CanQual<FunctionNoProtoType> Ty) {
  return CGM.getTypes().arrangeFreeFunctionType(Ty);
}

const CGFunctionInfo &
CodeGen::arrangeCXXMethodType(CodeGenModule &CGM,
                              const CXXRecordDecl *RD,
                              const FunctionProtoType *FTP,
                              const CXXMethodDecl *MD) {
  return CGM.getTypes().arrangeCXXMethodType(RD, FTP, MD);
}

const CGFunctionInfo &
CodeGen::arrangeFreeFunctionCall(CodeGenModule &CGM,
                                 CanQualType returnType,
                                 ArrayRef<CanQualType> argTypes,
                                 FunctionType::ExtInfo info,
                                 RequiredArgs args) {
  return CGM.getTypes().arrangeLLVMFunctionInfo(
      returnType, /*IsInstanceMethod=*/false, /*IsChainCall=*/false, argTypes,
      info, {}, args);
}
