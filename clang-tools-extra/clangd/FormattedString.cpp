//===--- FormattedString.cpp --------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "FormattedString.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <string>

namespace clang {
namespace clangd {

namespace {
/// Escape a markdown text block. Ensures the punctuation will not introduce
/// any of the markdown constructs.
static std::string renderText(llvm::StringRef Input) {
  // Escaping ASCII punctiation ensures we can't start a markdown construct.
  constexpr llvm::StringLiteral Punctuation =
      R"txt(!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)txt";

  std::string R;
  for (size_t From = 0; From < Input.size();) {
    size_t Next = Input.find_first_of(Punctuation, From);
    R += Input.substr(From, Next - From);
    if (Next == llvm::StringRef::npos)
      break;
    R += "\\";
    R += Input[Next];

    From = Next + 1;
  }
  return R;
}

/// Renders \p Input as an inline block of code in markdown. The returned value
/// is surrounded by backticks and the inner contents are properly escaped.
static std::string renderInlineBlock(llvm::StringRef Input) {
  std::string R;
  // Double all backticks to make sure we don't close the inline block early.
  for (size_t From = 0; From < Input.size();) {
    size_t Next = Input.find("`", From);
    R += Input.substr(From, Next - From);
    if (Next == llvm::StringRef::npos)
      break;
    R += "``"; // double the found backtick.

    From = Next + 1;
  }
  // If results starts with a backtick, add spaces on both sides. The spaces
  // are ignored by markdown renderers.
  if (llvm::StringRef(R).startswith("`") || llvm::StringRef(R).endswith("`"))
    return "` " + std::move(R) + " `";
  // Markdown render should ignore first and last space if both are there. We
  // add an extra pair of spaces in that case to make sure we render what the
  // user intended.
  if (llvm::StringRef(R).startswith(" ") && llvm::StringRef(R).endswith(" "))
    return "` " + std::move(R) + " `";
  return "`" + std::move(R) + "`";
}
/// Render \p Input as markdown code block with a specified \p Language. The
/// result is surrounded by >= 3 backticks. Although markdown also allows to use
/// '~' for code blocks, they are never used.
static std::string renderCodeBlock(llvm::StringRef Input,
                                   llvm::StringRef Language) {
  // Count the maximum number of consecutive backticks in \p Input. We need to
  // start and end the code block with more.
  unsigned MaxBackticks = 0;
  unsigned Backticks = 0;
  for (char C : Input) {
    if (C == '`') {
      ++Backticks;
      continue;
    }
    MaxBackticks = std::max(MaxBackticks, Backticks);
    Backticks = 0;
  }
  MaxBackticks = std::max(Backticks, MaxBackticks);
  // Use the corresponding number of backticks to start and end a code block.
  std::string BlockMarker(/*Repeat=*/std::max(3u, MaxBackticks + 1), '`');
  return BlockMarker + Language.str() + "\n" + Input.str() + "\n" + BlockMarker;
}

} // namespace

void FormattedString::appendText(std::string Text) {
  // We merge consecutive blocks of text to simplify the overall structure.
  if (Chunks.empty() || Chunks.back().Kind != ChunkKind::PlainText) {
    Chunk C;
    C.Kind = ChunkKind::PlainText;
    Chunks.push_back(C);
  }
  // FIXME: ensure there is a whitespace between the chunks.
  Chunks.back().Contents += Text;
}

void FormattedString::appendCodeBlock(std::string Code, std::string Language) {
  Chunk C;
  C.Kind = ChunkKind::CodeBlock;
  C.Contents = std::move(Code);
  C.Language = std::move(Language);
  Chunks.push_back(std::move(C));
}

void FormattedString::appendInlineCode(std::string Code) {
  Chunk C;
  C.Kind = ChunkKind::InlineCodeBlock;
  C.Contents = std::move(Code);
  Chunks.push_back(std::move(C));
}

std::string FormattedString::renderAsMarkdown() const {
  std::string R;
  for (const auto &C : Chunks) {
    switch (C.Kind) {
    case ChunkKind::PlainText:
      R += renderText(C.Contents);
      continue;
    case ChunkKind::InlineCodeBlock:
      // Make sure we don't glue two backticks together.
      if (llvm::StringRef(R).endswith("`"))
        R += " ";
      R += renderInlineBlock(C.Contents);
      continue;
    case ChunkKind::CodeBlock:
      if (!R.empty() && !llvm::StringRef(R).endswith("\n"))
        R += "\n";
      R += renderCodeBlock(C.Contents, C.Language);
      R += "\n";
      continue;
    }
    llvm_unreachable("unhanlded ChunkKind");
  }
  return R;
}

std::string FormattedString::renderAsPlainText() const {
  std::string R;
  auto EnsureWhitespace = [&]() {
    if (R.empty() || isWhitespace(R.back()))
      return;
    R += " ";
  };
  for (const auto &C : Chunks) {
    switch (C.Kind) {
    case ChunkKind::PlainText:
      EnsureWhitespace();
      R += C.Contents;
      continue;
    case ChunkKind::InlineCodeBlock:
      EnsureWhitespace();
      R += C.Contents;
      continue;
    case ChunkKind::CodeBlock:
      if (!R.empty())
        R += "\n\n";
      R += C.Contents;
      if (!llvm::StringRef(C.Contents).endswith("\n"))
        R += "\n";
      continue;
    }
    llvm_unreachable("unhanlded ChunkKind");
  }
  while (!R.empty() && isWhitespace(R.back()))
    R.pop_back();
  return R;
}
} // namespace clangd
} // namespace clang
