//=== unittests/CodeGen/TestCompiler.h - Match on the LLVM IR ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_UNITTESTS_CODEGEN_TESTCOMPILER_H
#define CLANG_UNITTESTS_CODEGEN_TESTCOMPILER_H


#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Parse/ParseAST.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Host.h"

namespace llvm {

struct TestCompiler {
  LLVMContext Context;
  clang::CompilerInstance compiler;
  std::unique_ptr<clang::CodeGenerator> CG;
  llvm::Module *M = nullptr;
  unsigned PtrSize = 0;

  TestCompiler(clang::LangOptions LO,
               clang::CodeGenOptions CGO = clang::CodeGenOptions()) {
    compiler.getLangOpts() = LO;
    compiler.getCodeGenOpts() = CGO;
    compiler.createDiagnostics();

    std::string TrStr = llvm::Triple::normalize(llvm::sys::getProcessTriple());
    llvm::Triple Tr(TrStr);
    Tr.setOS(Triple::Linux);
    Tr.setVendor(Triple::VendorType::UnknownVendor);
    Tr.setEnvironment(Triple::EnvironmentType::UnknownEnvironment);
    compiler.getTargetOpts().Triple = Tr.getTriple();
    compiler.setTarget(clang::TargetInfo::CreateTargetInfo(
        compiler.getDiagnostics(),
        std::make_shared<clang::TargetOptions>(compiler.getTargetOpts())));

    const clang::TargetInfo &TInfo = compiler.getTarget();
    PtrSize = TInfo.getPointerWidth(0) / 8;

    compiler.createFileManager();
    compiler.createSourceManager(compiler.getFileManager());
    compiler.createPreprocessor(clang::TU_Prefix);

    compiler.createASTContext();

    CG.reset(CreateLLVMCodeGen(compiler.getDiagnostics(),
                               "main-module",
                               compiler.getHeaderSearchOpts(),
                               compiler.getPreprocessorOpts(),
                               compiler.getCodeGenOpts(),
                               Context));
  }

  void init(const char *TestProgram,
            std::unique_ptr<clang::ASTConsumer> Consumer = nullptr) {
    if (!Consumer)
      Consumer = std::move(CG);

    compiler.setASTConsumer(std::move(Consumer));

    compiler.createSema(clang::TU_Prefix, nullptr);

    clang::SourceManager &sm = compiler.getSourceManager();
    sm.setMainFileID(sm.createFileID(
        llvm::MemoryBuffer::getMemBuffer(TestProgram), clang::SrcMgr::C_User));
  }

  const BasicBlock *compile() {
    clang::ParseAST(compiler.getSema(), false, false);
    M =
      static_cast<clang::CodeGenerator&>(compiler.getASTConsumer()).GetModule();

    // Do not expect more than one function definition.
    auto FuncPtr = M->begin();
    for (; FuncPtr != M->end(); ++FuncPtr)
      if (!FuncPtr->isDeclaration())
        break;
    assert(FuncPtr != M->end());
    const llvm::Function &Func = *FuncPtr;
    ++FuncPtr;
    for (; FuncPtr != M->end(); ++FuncPtr)
      if (!FuncPtr->isDeclaration())
        break;
    assert(FuncPtr == M->end());

    // The function must consist of single basic block.
    auto BBPtr = Func.begin();
    assert(Func.begin() != Func.end());
    const BasicBlock &BB = *BBPtr;
    ++BBPtr;
    assert(BBPtr == Func.end());

    return &BB;
  }
};

} // namespace llvm
#endif // CLANG_UNITTESTS_CODEGEN_TESTCOMPILER_H
