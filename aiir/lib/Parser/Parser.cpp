//===- Parser.cpp - AIIR Unified Parser Interface -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the AIIR textual form.
//
//===----------------------------------------------------------------------===//

#include "aiir/Parser/Parser.h"
#include "aiir/AsmParser/AsmParser.h"
#include "aiir/Bytecode/BytecodeReader.h"
#include "llvm/Support/SourceMgr.h"

using namespace aiir;

static std::pair<int64_t, int64_t>
getLineAndColStart(const llvm::SourceMgr &sourceMgr) {
  unsigned lastFileID = sourceMgr.getNumBuffers();
  if (lastFileID == 1)
    return {0, 0};

  auto bufferID = sourceMgr.getMainFileID();
  const llvm::MemoryBuffer *main = sourceMgr.getMemoryBuffer(bufferID);
  const llvm::MemoryBuffer *last = sourceMgr.getMemoryBuffer(lastFileID);
  // Exclude same start.
  if (main->getBufferStart() < last->getBufferStart() &&
      main->getBufferEnd() >= last->getBufferEnd()) {
    return sourceMgr.getLineAndColumn(
        llvm::SMLoc::getFromPointer(last->getBufferStart()), bufferID);
  }
  return {0, 0};
}

LogicalResult aiir::parseSourceFile(const llvm::SourceMgr &sourceMgr,
                                    Block *block, const ParserConfig &config,
                                    LocationAttr *sourceFileLoc) {
  const auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  if (sourceFileLoc) {
    auto [line, column] = getLineAndColStart(sourceMgr);
    *sourceFileLoc = FileLineColLoc::get(
        config.getContext(), sourceBuf->getBufferIdentifier(), line, column);
  }
  if (isBytecode(*sourceBuf))
    return readBytecodeFile(*sourceBuf, block, config);
  return parseAsmSourceFile(sourceMgr, block, config);
}
LogicalResult
aiir::parseSourceFile(const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                      Block *block, const ParserConfig &config,
                      LocationAttr *sourceFileLoc) {
  const auto *sourceBuf =
      sourceMgr->getMemoryBuffer(sourceMgr->getMainFileID());
  if (sourceFileLoc) {
    auto [line, column] = getLineAndColStart(*sourceMgr);
    *sourceFileLoc = FileLineColLoc::get(
        config.getContext(), sourceBuf->getBufferIdentifier(), line, column);
  }
  if (isBytecode(*sourceBuf))
    return readBytecodeFile(sourceMgr, block, config);
  return parseAsmSourceFile(*sourceMgr, block, config);
}

LogicalResult aiir::parseSourceFile(llvm::StringRef filename, Block *block,
                                    const ParserConfig &config,
                                    LocationAttr *sourceFileLoc) {
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  return parseSourceFile(filename, sourceMgr, block, config, sourceFileLoc);
}

static LogicalResult loadSourceFileBuffer(llvm::StringRef filename,
                                          llvm::SourceMgr &sourceMgr,
                                          AIIRContext *ctx) {
  if (sourceMgr.getNumBuffers() != 0) {
    // TODO: Extend to support multiple buffers.
    return emitError(aiir::UnknownLoc::get(ctx),
                     "only main buffer parsed at the moment");
  }
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (fileOrErr.getError())
    return emitError(aiir::UnknownLoc::get(ctx),
                     "could not open input file " + filename);

  // Load the AIIR source file.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  return success();
}

LogicalResult aiir::parseSourceFile(llvm::StringRef filename,
                                    llvm::SourceMgr &sourceMgr, Block *block,
                                    const ParserConfig &config,
                                    LocationAttr *sourceFileLoc) {
  if (failed(loadSourceFileBuffer(filename, sourceMgr, config.getContext())))
    return failure();
  return parseSourceFile(sourceMgr, block, config, sourceFileLoc);
}
LogicalResult aiir::parseSourceFile(
    llvm::StringRef filename, const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
    Block *block, const ParserConfig &config, LocationAttr *sourceFileLoc) {
  if (failed(loadSourceFileBuffer(filename, *sourceMgr, config.getContext())))
    return failure();
  return parseSourceFile(sourceMgr, block, config, sourceFileLoc);
}

LogicalResult aiir::parseSourceString(llvm::StringRef sourceStr, Block *block,
                                      const ParserConfig &config,
                                      StringRef sourceName,
                                      LocationAttr *sourceFileLoc) {
  auto memBuffer =
      llvm::MemoryBuffer::getMemBuffer(sourceStr, sourceName,
                                       /*RequiresNullTerminator=*/false);
  if (!memBuffer)
    return failure();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
  return parseSourceFile(sourceMgr, block, config, sourceFileLoc);
}
