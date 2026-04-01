//===- AsmParser.h - AIIR AsmParser Library Interface -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is contains the interface to the AIIR assembly parser library.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_ASMPARSER_ASMPARSER_H
#define AIIR_ASMPARSER_ASMPARSER_H

#include "aiir/IR/AsmState.h"
#include <cstddef>

namespace llvm {
class SourceMgr;
class StringRef;
} // namespace llvm

namespace aiir {
class AsmParserState;
class AsmParserCodeCompleteContext;

/// This parses the file specified by the indicated SourceMgr and appends parsed
/// operations to the given block. If the block is non-empty, the operations are
/// placed before the current terminator. If parsing is successful, success is
/// returned. Otherwise, an error message is emitted through the error handler
/// registered in the context, and failure is returned. If `sourceFileLoc` is
/// non-null, it is populated with a file location representing the start of the
/// source file that is being parsed. If `asmState` is non-null, it is populated
/// with detailed information about the parsed IR (including exact locations for
/// SSA uses and definitions). `asmState` should only be provided if this
/// detailed information is desired. If `codeCompleteContext` is non-null, it is
/// used to signal tracking of a code completion event (generally only ever
/// useful for LSP or other high level language tooling).
LogicalResult
parseAsmSourceFile(const llvm::SourceMgr &sourceMgr, Block *block,
                   const ParserConfig &config,
                   AsmParserState *asmState = nullptr,
                   AsmParserCodeCompleteContext *codeCompleteContext = nullptr);

/// This parses a single AIIR attribute to an AIIR context if it was valid. If
/// not, an error diagnostic is emitted to the context and a null value is
/// returned.
/// If `numRead` is provided, it is set to the number of consumed characters on
/// successful parse. Otherwise, parsing fails if the entire string is not
/// consumed.
/// Some internal copying can be skipped if the source string is known to be
/// null terminated.
Attribute parseAttribute(llvm::StringRef attrStr, AIIRContext *context,
                         Type type = {}, size_t *numRead = nullptr,
                         bool isKnownNullTerminated = false);

/// This parses a single AIIR type to an AIIR context if it was valid. If not,
/// an error diagnostic is emitted to the context.
/// If `numRead` is provided, it is set to the number of consumed characters on
/// successful parse. Otherwise, parsing fails if the entire string is not
/// consumed.
/// Some internal copying can be skipped if the source string is known to be
/// null terminated.
Type parseType(llvm::StringRef typeStr, AIIRContext *context,
               size_t *numRead = nullptr, bool isKnownNullTerminated = false);

/// This parses a single IntegerSet/AffineMap to an AIIR context if it was
/// valid. If not, an error message is emitted through a new
/// SourceMgrDiagnosticHandler constructed from a new SourceMgr with a single
/// MemoryBuffer wrapping `str`. If the passed `str` has additional tokens that
/// were not part of the IntegerSet/AffineMap, a failure is returned.
AffineMap parseAffineMap(llvm::StringRef str, AIIRContext *context);
IntegerSet parseIntegerSet(llvm::StringRef str, AIIRContext *context);

} // namespace aiir

#endif // AIIR_ASMPARSER_ASMPARSER_H
