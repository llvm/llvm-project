//=== Serialization/PCHContainerOperations.cpp - PCH Containers -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines PCHContainerOperations and RawPCHContainerOperation.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/PCHContainerOperations.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/ModuleLoader.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace clang;

PCHContainerWriter::~PCHContainerWriter() {}
PCHContainerReader::~PCHContainerReader() {}

namespace {

/// A PCHContainerGenerator that writes out the PCH to a flat file.
class RawPCHContainerGenerator : public ASTConsumer {
  std::shared_ptr<PCHBuffer> Buffer;
  std::unique_ptr<raw_pwrite_stream> OS;

public:
  RawPCHContainerGenerator(std::unique_ptr<llvm::raw_pwrite_stream> OS,
                           std::shared_ptr<PCHBuffer> Buffer)
      : Buffer(std::move(Buffer)), OS(std::move(OS)) {}

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

/// A PCHContainerGenerator that writes out the PCH to a flat file if the
/// action is needed (and the filename is determined at the time the output
/// is done).
class RawPCHDeferredContainerGenerator : public ASTConsumer {
  std::shared_ptr<PCHBuffer> Buffer;

public:
  RawPCHDeferredContainerGenerator(std::shared_ptr<PCHBuffer> Buffer)
      : Buffer(std::move(Buffer)) {}

  ~RawPCHDeferredContainerGenerator() override = default;

  void HandleTranslationUnit(ASTContext &Ctx) override {
    if (Buffer->IsComplete && !Buffer->PresumedFileName.empty()) {
      std::error_code EC;
      StringRef Parent = llvm::sys::path::parent_path(Buffer->PresumedFileName);
      if (!Parent.empty())
        EC = llvm::sys::fs::create_directory(Parent);
      if (!EC) {
        int FD;
        EC = llvm::sys::fs::openFileForWrite(Buffer->PresumedFileName, FD);
        if (!EC) {
          std::unique_ptr<raw_pwrite_stream> OS;
          OS.reset(new llvm::raw_fd_ostream(FD, /*shouldClose=*/true));
          *OS << Buffer->Data;
          OS->flush(); // Make sure it hits disk now.
          // Here we would notify P1184 servers that the module is created
        } else
          llvm::dbgs() << " Problem creating : " << Buffer->PresumedFileName
                       << "\n";
      } else
        llvm::dbgs() << " Problem creating dir : " << Parent << "\n";
    }

    // Free the space of the temporary buffer.
    llvm::SmallVector<char, 0> Empty;
    Buffer->Data = std::move(Empty);
  }
};

} // anonymous namespace

std::unique_ptr<ASTConsumer> RawPCHContainerWriter::CreatePCHContainerGenerator(
    CompilerInstance &CI, const std::string &MainFileName,
    const std::string &OutputFileName, std::unique_ptr<llvm::raw_pwrite_stream> OS,
    std::shared_ptr<PCHBuffer> Buffer) const {
  return std::make_unique<RawPCHContainerGenerator>(std::move(OS), Buffer);
}

ArrayRef<llvm::StringRef> RawPCHContainerReader::getFormats() const {
  static StringRef Raw("raw");
  return ArrayRef(Raw);
}

std::unique_ptr<ASTConsumer>
RawPCHContainerWriter::CreatePCHDeferredContainerGenerator(
    CompilerInstance &CI, const std::string &MainFileName,
    const std::string &OutputFileName,
    std::unique_ptr<llvm::raw_pwrite_stream> OS,
    std::shared_ptr<PCHBuffer> Buffer) const {
  return std::make_unique<RawPCHDeferredContainerGenerator>(Buffer);
}

StringRef
RawPCHContainerReader::ExtractPCH(llvm::MemoryBufferRef Buffer) const {
  return Buffer.getBuffer();
}

PCHContainerOperations::PCHContainerOperations() {
  registerWriter(std::make_unique<RawPCHContainerWriter>());
  registerReader(std::make_unique<RawPCHContainerReader>());
}
