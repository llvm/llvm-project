//===--- build_confusable_table.cpp - clang-tidy---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

int main(int argc, char *argv[]) {
  auto ErrorOrBuffer = MemoryBuffer::getFile(argv[1], true);
  if (!ErrorOrBuffer)
    return 1;
  std::unique_ptr<MemoryBuffer> Buffer = std::move(ErrorOrBuffer.get());
  StringRef Content = Buffer->getBuffer();
  Content = Content.drop_until([](char c) { return c == '#'; });
  SmallVector<StringRef> Lines;
  SplitString(Content, Lines, "\r\n");

  std::vector<std::pair<llvm::UTF32, SmallVector<llvm::UTF32>>> Entries;
  for (StringRef Line : Lines) {
    if (Line.startswith("#"))
      continue;

    SmallVector<StringRef> Values;
    Line.split(Values, ';');
    if (Values.size() < 2) {
      errs() << "Failed to parse: " << Line << "\n";
      return 2;
    }

    llvm::StringRef From = Values[0].trim();
    llvm::UTF32 CodePoint;
    From.getAsInteger(16, CodePoint);

    SmallVector<llvm::UTF32> To;
    SmallVector<StringRef> ToN;
    Values[1].split(ToN, ' ', -1, false);
    for (StringRef To_ : ToN) {
      llvm::UTF32 ToCodePoint;
      To_.trim().getAsInteger(16, ToCodePoint);
      To.push_back(ToCodePoint);
    }
    while (To.size() < 32)
      To.push_back(0);

    Entries.emplace_back(CodePoint, To);
  }
  std::sort(Entries.begin(), Entries.end());
  errs() << "Parsed " << Entries.size() << " Entries\n";

  std::error_code ec;
  llvm::raw_fd_ostream os(argv[2], ec);
  os << "struct {llvm::UTF32 codepoint; llvm::UTF32 values[32];} "
        "ConfusableEntries[] = {\n";
  for (auto const &Values : Entries) {
    os << "  { ";
    os << Values.first;
    os << ", {";
    for (auto CP : Values.second) {
      os << CP << ", ";
    }
    os << "}},\n";
  }
  os << "};\n";
  return 0;
}
