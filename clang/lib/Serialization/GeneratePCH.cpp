//===--- GeneratePCH.cpp - Sema Consumer for PCH Generation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PCHGenerator, which as a SemaConsumer that generates
//  a PCH file.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/Serialization/ASTWriter.h"
#include "llvm/Bitstream/BitstreamWriter.h"

using namespace clang;

PCHGenerator::PCHGenerator(
    const Preprocessor &PP, InMemoryModuleCache &ModuleCache,
    StringRef OutputFile, StringRef isysroot, std::shared_ptr<PCHBuffer> Buffer,
    ArrayRef<std::shared_ptr<ModuleFileExtension>> Extensions,
    bool AllowASTWithErrors, bool IncludeTimestamps,
    bool BuildingImplicitModule, bool ShouldCacheASTInMemory, bool IsForBMI)
    : PP(PP), OutputFile(OutputFile), isysroot(isysroot.str()),
      SemaPtr(nullptr), Buffer(std::move(Buffer)), Stream(this->Buffer->Data),
      Writer(Stream, this->Buffer->Data, ModuleCache, Extensions,
             IncludeTimestamps, BuildingImplicitModule),
      AllowASTWithErrors(AllowASTWithErrors),
      ShouldCacheASTInMemory(ShouldCacheASTInMemory), IsForBMI(IsForBMI) {
  this->Buffer->IsComplete = false;
}

PCHGenerator::~PCHGenerator() {
}

void PCHGenerator::HandleTranslationUnit(ASTContext &Ctx) {
  // Don't create a PCH if there were fatal failures during module loading.
  if (PP.getModuleLoader().HadFatalFailure)
    return;

  bool hasErrors = PP.getDiagnostics().hasErrorOccurred();
  if (hasErrors && !AllowASTWithErrors)
    return;

  Module *Module = nullptr;
  if (PP.getLangOpts().isCompilingModule() || IsForBMI) {
    Module = PP.getHeaderSearchInfo().lookupModule(
        PP.getLangOpts().CurrentModule, SourceLocation(),
        /*AllowSearch*/ false);
    if (!Module) {
      // If we have errors, then that might have prevented the creation of the
      // module - otherwise, for the case we are compiling a module, it must be
      // present.
      // Conversely, IsForBMI output is speculative and only produced for TUs
      // in which module interfaces are discovered, thus it is not an error to
      // find that there is no module in this case.
      assert((hasErrors || IsForBMI) &&
             "emitting module but current module doesn't exist");
      return;
    }
  } // else, non-modular PCH.

  // Errors that do not prevent the PCH from being written should not cause the
  // overall compilation to fail either.
  if (AllowASTWithErrors)
    PP.getDiagnostics().getClient()->clear();

  assert(SemaPtr && "No Sema?");

  // A module implementation implicitly pulls in its interface module.
  // Since it has the same name as the implementation, it will be found
  // by the lookup above.  Fortunately, Sema records the difference in
  // the ModuleScopes; We do not need to output the BMI in that case.
  if (IsForBMI && SemaPtr->isModuleImplementation())
    return;

  if (IsForBMI) {

    assert(Module && !Module->IsFromModuleFile &&
           "trying to re-write a module?");

    // Here we would ideally use a P1184 server to find the module name.
    // However, in the short-term we are going to (ab-)use the name/file pairs
    // that can be specified with -fmodule-file=Name=Path.  If there is no
    // entry there, then we fall back to the default CMI name, based on the
    // source file name.
    HeaderSearch &HS = PP.getHeaderSearchInfo();
    const HeaderSearchOptions &HSOpts = HS.getHeaderSearchOpts();
    std::string ModuleFilename;
    if (!HSOpts.PrebuiltModuleFiles.empty() ||
        !HSOpts.PrebuiltModulePaths.empty())
      ModuleFilename = HS.getPrebuiltModuleFileName(Module->Name);

    if (!ModuleFilename.empty())
      OutputFile = ModuleFilename;

    // So now attach that name to the buffer we are about to create.
    Buffer->PresumedFileName = OutputFile;
  }

  Buffer->Signature = Writer.WriteAST(*SemaPtr, OutputFile, Module, isysroot,
                                      ShouldCacheASTInMemory);

  Buffer->IsComplete = true;
}

ASTMutationListener *PCHGenerator::GetASTMutationListener() {
  return &Writer;
}

ASTDeserializationListener *PCHGenerator::GetASTDeserializationListener() {
  return &Writer;
}
