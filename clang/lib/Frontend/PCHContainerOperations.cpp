//===--- Frontend/PCHContainerOperations.cpp - PCH Containers ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines PCHContainerOperations and RawPCHContainerOperation.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/AST/ASTConsumer.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Lex/ModuleLoader.h"

using namespace clang;

namespace {

/// \brief A PCHContainerGenerator that writes out the PCH to a flat file.
class RawPCHContainerGenerator : public ASTConsumer {
  std::shared_ptr<PCHBuffer> Buffer;
  raw_pwrite_stream *OS;

public:
  RawPCHContainerGenerator(llvm::raw_pwrite_stream *OS,
                           std::shared_ptr<PCHBuffer> Buffer)
      : Buffer(Buffer), OS(OS) {}

  ~RawPCHContainerGenerator() override = default;

  void HandleTranslationUnit(ASTContext &Ctx) override {
    if (Buffer->IsComplete) {
      // Make sure it hits disk now.
      *OS << Buffer->Data;
      OS->flush();
    }
    // Free the space of the temporary buffer.
    llvm::SmallVector<char, 0> Empty;
    Buffer->Data = std::move(Empty);
  }
};

} // anonymous namespace

std::unique_ptr<ASTConsumer> RawPCHContainerWriter::CreatePCHContainerGenerator(
    CompilerInstance &CI, const std::string &MainFileName,
    const std::string &OutputFileName, llvm::raw_pwrite_stream *OS,
    std::shared_ptr<PCHBuffer> Buffer) const {
  return llvm::make_unique<RawPCHContainerGenerator>(OS, Buffer);
}

void RawPCHContainerReader::ExtractPCH(
    llvm::MemoryBufferRef Buffer, llvm::BitstreamReader &StreamFile) const {
  StreamFile.init((const unsigned char *)Buffer.getBufferStart(),
                  (const unsigned char *)Buffer.getBufferEnd());
}

PCHContainerOperations::PCHContainerOperations() {
  registerWriter(llvm::make_unique<RawPCHContainerWriter>());
  registerReader(llvm::make_unique<RawPCHContainerReader>());
}
