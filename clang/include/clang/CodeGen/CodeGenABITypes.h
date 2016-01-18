//==---- CodeGenABITypes.h - Convert Clang types to LLVM types for ABI -----==//
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
// It allows other clients, like LLDB, to determine the LLVM types that are
// actually used in function calls, which makes it possible to then determine
// the acutal ABI locations (e.g. registers, stack locations, etc.) that
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
}

namespace clang {
class ASTContext;
class CXXRecordDecl;
class CodeGenOptions;
class CoverageSourceInfo;
class DiagnosticsEngine;
class HeaderSearchOptions;
class ObjCMethodDecl;
class PreprocessorOptions;

namespace CodeGen {
class CGFunctionInfo;
class CodeGenModule;

class CodeGenABITypes
{
public:
  CodeGenABITypes(ASTContext &C, llvm::Module &M,
                  CoverageSourceInfo *CoverageInfo = nullptr);
  ~CodeGenABITypes();

  /// These methods all forward to methods in the private implementation class
  /// CodeGenTypes.

  const CGFunctionInfo &arrangeObjCMessageSendSignature(
                                                     const ObjCMethodDecl *MD,
                                                     QualType receiverType);
  const CGFunctionInfo &arrangeFreeFunctionType(
                                               CanQual<FunctionProtoType> Ty);
  const CGFunctionInfo &arrangeFreeFunctionType(
                                             CanQual<FunctionNoProtoType> Ty);
  const CGFunctionInfo &arrangeCXXMethodType(const CXXRecordDecl *RD,
                                             const FunctionProtoType *FTP);
  const CGFunctionInfo &arrangeFreeFunctionCall(CanQualType returnType,
                                                ArrayRef<CanQualType> argTypes,
                                                FunctionType::ExtInfo info,
                                                RequiredArgs args);

private:
  /// Default CodeGenOptions object used to initialize the
  /// CodeGenModule and otherwise not used. More specifically, it is
  /// not used in ABI type generation, so none of the options matter.
  std::unique_ptr<CodeGenOptions> CGO;
  std::unique_ptr<HeaderSearchOptions> HSO;
  std::unique_ptr<PreprocessorOptions> PPO;

  /// The CodeGenModule we use get to the CodeGenTypes object.
  std::unique_ptr<CodeGen::CodeGenModule> CGM;
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
