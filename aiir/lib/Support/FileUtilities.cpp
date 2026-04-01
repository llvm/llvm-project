//===- FileUtilities.cpp - utilities for working with files ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of common utilities for working with files.
//
//===----------------------------------------------------------------------===//

#include "aiir/Support/FileUtilities.h"
#include "aiir/Support/LLVM.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace aiir;

static std::unique_ptr<llvm::MemoryBuffer>
openInputFileImpl(StringRef inputFilename, std::string *errorMessage,
                  std::optional<llvm::Align> alignment) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(
      inputFilename, /*IsText=*/false, /*RequiresNullTerminator=*/true,
      alignment);
  if (std::error_code error = fileOrErr.getError()) {
    if (errorMessage)
      *errorMessage = "cannot open input file '" + inputFilename.str() +
                      "': " + error.message();
    return nullptr;
  }

  return std::move(*fileOrErr);
}
std::unique_ptr<llvm::MemoryBuffer>
aiir::openInputFile(StringRef inputFilename, std::string *errorMessage) {
  return openInputFileImpl(inputFilename, errorMessage,
                           /*alignment=*/std::nullopt);
}
std::unique_ptr<llvm::MemoryBuffer>
aiir::openInputFile(llvm::StringRef inputFilename, llvm::Align alignment,
                    std::string *errorMessage) {
  return openInputFileImpl(inputFilename, errorMessage, alignment);
}

std::unique_ptr<llvm::ToolOutputFile>
aiir::openOutputFile(StringRef outputFilename, std::string *errorMessage) {
  std::error_code error;
  auto result = std::make_unique<llvm::ToolOutputFile>(outputFilename, error,
                                                       llvm::sys::fs::OF_None);
  if (error) {
    if (errorMessage)
      *errorMessage = "cannot open output file '" + outputFilename.str() +
                      "': " + error.message();
    return nullptr;
  }

  return result;
}
