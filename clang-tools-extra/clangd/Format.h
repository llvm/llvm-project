//===--- Format.h - automatic code formatting ---------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Clangd uses clang-format for formatting operations.
// This file adapts it to support new scenarios like format-on-type.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FORMAT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FORMAT_H

#include "Protocol.h"
#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace clangd {

/// Applies limited formatting around new \p InsertedText.
/// The \p Code already contains the updated text before \p Cursor, and may have
/// had additional / characters (such as indentation) inserted by the editor.
///
/// Example breaking a line (^ is the cursor):
/// === before newline is typed ===
/// if(1){^}
/// === after newline is typed and editor indents ===
/// if(1){
///   ^}
/// === after formatIncremental(InsertedText="\n") ===
/// if (1) {
///   ^
/// }
///
/// We return sorted vector<tooling::Replacement>, not tooling::Replacements!
/// We may insert text both before and after the cursor. tooling::Replacements
/// would merge these, and thus lose information about cursor position.
std::vector<tooling::Replacement>
formatIncremental(llvm::StringRef Code, unsigned Cursor,
                  llvm::StringRef InsertedText, format::FormatStyle Style);

/// Determine the new cursor position after applying \p Replacements.
/// Analogue of tooling::Replacements::getShiftedCodePosition().
unsigned
transformCursorPosition(unsigned Offset,
                        const std::vector<tooling::Replacement> &Replacements);

} // namespace clangd
} // namespace clang

#endif

