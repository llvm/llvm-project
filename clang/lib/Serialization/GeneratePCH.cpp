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
    bool ShouldCacheASTInMemory)
    : PP(PP), OutputFile(OutputFile), isysroot(isysroot.str()),
      SemaPtr(nullptr), Buffer(std::move(Buffer)), Stream(this->Buffer->Data),
      Writer(Stream, this->Buffer->Data, ModuleCache, Extensions,
             IncludeTimestamps),
      AllowASTWithErrors(AllowASTWithErrors),
      ShouldCacheASTInMemory(ShouldCacheASTInMemory) {
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
  if (PP.getLangOpts().isCompilingModule()) {
    Module = PP.getHeaderSearchInfo().lookupModule(
        PP.getLangOpts().CurrentModule, /*AllowSearch*/ false);
    if (!Module) {
      assert(hasErrors && "emitting module but current module doesn't exist");
      return;
    }
  }

  // Emit the PCH file to the Buffer.
  assert(SemaPtr && "No Sema?");
  Buffer->Signature =
      Writer.WriteAST(*SemaPtr, OutputFile, Module, isysroot,
                      // For serialization we are lenient if the errors were
                      // only warn-as-error kind.
                      PP.getDiagnostics().hasUncompilableErrorOccurred(),
                      ShouldCacheASTInMemory);

  Buffer->IsComplete = true;
}

ASTMutationListener *PCHGenerator::GetASTMutationListener() {
  return &Writer;
}

ASTDeserializationListener *PCHGenerator::GetASTDeserializationListener() {
  return &Writer;
}
