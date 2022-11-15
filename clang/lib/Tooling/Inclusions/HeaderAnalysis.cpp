//===--- HeaderAnalysis.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Inclusions/HeaderAnalysis.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/HeaderSearch.h"

namespace clang::tooling {
namespace {

// Is Line an #if or #ifdef directive?
// FIXME: This makes headers with #ifdef LINUX/WINDOWS/MACOS marked as non
// self-contained and is probably not what we want.
bool isIf(llvm::StringRef Line) {
  Line = Line.ltrim();
  if (!Line.consume_front("#"))
    return false;
  Line = Line.ltrim();
  return Line.startswith("if");
}

// Is Line an #error directive mentioning includes?
bool isErrorAboutInclude(llvm::StringRef Line) {
  Line = Line.ltrim();
  if (!Line.consume_front("#"))
    return false;
  Line = Line.ltrim();
  if (!Line.startswith("error"))
    return false;
  return Line.contains_insensitive(
      "includ"); // Matches "include" or "including".
}

// Heuristically headers that only want to be included via an umbrella.
bool isDontIncludeMeHeader(llvm::MemoryBufferRef Buffer) {
  StringRef Content = Buffer.getBuffer();
  llvm::StringRef Line;
  // Only sniff up to 100 lines or 10KB.
  Content = Content.take_front(100 * 100);
  for (unsigned I = 0; I < 100 && !Content.empty(); ++I) {
    std::tie(Line, Content) = Content.split('\n');
    if (isIf(Line) && isErrorAboutInclude(Content.split('\n').first))
      return true;
  }
  return false;
}

} // namespace

bool isSelfContainedHeader(const FileEntry *FE, const SourceManager &SM,
                           HeaderSearch &HeaderInfo) {
  assert(FE);
  if (!HeaderInfo.isFileMultipleIncludeGuarded(FE) &&
      !HeaderInfo.hasFileBeenImported(FE))
    return false;
  // This pattern indicates that a header can't be used without
  // particular preprocessor state, usually set up by another header.
  return !isDontIncludeMeHeader(
      const_cast<SourceManager &>(SM).getMemoryBufferForFileOrNone(FE).value_or(
          llvm::MemoryBufferRef()));
}
} // namespace clang::tooling
