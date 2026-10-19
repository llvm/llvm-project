//===--- ModuleLinker.cpp - Shared bitcode link helpers -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/ModuleLinker.h"

#include "clang/Basic/CodeGenOptions.h"
#include "clang/CodeGenUtils/BackendDiagnosticHandler.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/IPO/Internalize.h"

using namespace clang;

bool clang::loadLinkModules(CompilerInstance &CI, llvm::LLVMContext &Ctx,
                            llvm::SmallVectorImpl<LinkModule> &LinkModules) {
  if (!LinkModules.empty())
    return false;

  for (const CodeGenOptions::BitcodeFileToLink &F :
       CI.getCodeGenOpts().LinkBitcodeFiles) {
    auto BCBuf = CI.getFileManager().getBufferForFile(F.Filename);
    if (!BCBuf) {
      CI.getDiagnostics().Report(diag::err_cannot_open_file)
          << F.Filename << BCBuf.getError().message();
      LinkModules.clear();
      return true;
    }

    llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
        llvm::getOwningLazyBitcodeModule(std::move(*BCBuf), Ctx);
    if (!ModuleOrErr) {
      llvm::handleAllErrors(
          ModuleOrErr.takeError(), [&](llvm::ErrorInfoBase &EIB) {
            CI.getDiagnostics().Report(diag::err_cannot_open_file)
                << F.Filename << EIB.message();
          });
      LinkModules.clear();
      return true;
    }

    LinkModules.push_back({std::move(ModuleOrErr.get()), F.PropagateAttrs,
                           F.Internalize, F.LinkFlags});
  }
  return false;
}

bool clang::linkInModules(llvm::Module &M,
                          llvm::SmallVectorImpl<LinkModule> &LinkModules,
                          const CodeGenOptions &CodeGenOpts,
                          const LangOptions &LangOpts,
                          const TargetOptions &TargetOpts,
                          BackendDiagnosticConsumer &DiagConsumer) {
  for (LinkModule &LM : LinkModules) {
    assert(LM.Module && "LinkModule does not actually have a module");

    if (LM.PropagateAttrs)
      for (llvm::Function &F : *LM.Module) {
        // Skip intrinsics. Keep consistent with how intrinsics are created
        // in LLVM IR.
        if (F.isIntrinsic())
          continue;
        CodeGen::mergeDefaultFunctionDefinitionAttributes(
            F, CodeGenOpts, LangOpts, TargetOpts, LM.Internalize);
      }

    DiagConsumer.setCurLinkModule(LM.Module.get());
    bool Err;

    if (LM.Internalize) {
      Err = llvm::Linker::linkModules(
          M, std::move(LM.Module), LM.LinkFlags,
          [](llvm::Module &M, const llvm::StringSet<> &GVS) {
            llvm::internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
              return !GV.hasName() || (GVS.count(GV.getName()) == 0);
            });
          });
    } else
      Err = llvm::Linker::linkModules(M, std::move(LM.Module), LM.LinkFlags);

    if (Err)
      return true;
  }

  LinkModules.clear();
  return false; // success
}
