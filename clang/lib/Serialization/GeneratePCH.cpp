//===--- GeneratePCH.cpp - Sema Consumer for PCH Generation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PCHGenerator, which as a SemaConsumer that generates
//  a PCH file.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ASTWriter.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/FileManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/SemaConsumer.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include <string>

using namespace clang;

PCHGenerator::PCHGenerator(
  const Preprocessor &PP, StringRef OutputFile,
  clang::Module *Module, StringRef isysroot,
  std::shared_ptr<PCHBuffer> Buffer,
  ArrayRef<llvm::IntrusiveRefCntPtr<ModuleFileExtension>> Extensions,
  bool AllowASTWithErrors, bool IncludeTimestamps)
    : PP(PP), OutputFile(OutputFile), Module(Module), isysroot(isysroot.str()),
      SemaPtr(nullptr), Buffer(Buffer), Stream(Buffer->Data),
      Writer(Stream, Extensions, IncludeTimestamps),
      AllowASTWithErrors(AllowASTWithErrors) {
  Buffer->IsComplete = false;
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

  // Emit the PCH file to the Buffer.
  assert(SemaPtr && "No Sema?");
  Buffer->Signature =
      Writer.WriteAST(*SemaPtr, OutputFile, Module, isysroot, hasErrors);

  Buffer->IsComplete = true;
}

ASTMutationListener *PCHGenerator::GetASTMutationListener() {
  return &Writer;
}

ASTDeserializationListener *PCHGenerator::GetASTDeserializationListener() {
  return &Writer;
}
