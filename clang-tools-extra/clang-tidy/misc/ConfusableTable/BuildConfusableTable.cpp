//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
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
  Content = Content.drop_until([](char C) { return C == '#'; });
  SmallVector<StringRef> Lines;
  SplitString(Content, Lines, "\r\n");

  std::vector<std::pair<llvm::UTF32, SmallVector<llvm::UTF32>>> Entries;
  SmallVector<StringRef> Values;
  for (StringRef Line : Lines) {
    if (Line.starts_with("#"))
      continue;

    Values.clear();
    Line.split(Values, ';');
    if (Values.size() < 2) {
      errs() << "Failed to parse: " << Line << "\n";
      return 2;
    }

    llvm::StringRef From = Values[0].trim();
    llvm::UTF32 CodePoint = 0;
    From.getAsInteger(16, CodePoint);

    SmallVector<llvm::UTF32> To;
    SmallVector<StringRef> ToN;
    Values[1].split(ToN, ' ', -1, false);
    for (StringRef ToI : ToN) {
      llvm::UTF32 ToCodePoint = 0;
      ToI.trim().getAsInteger(16, ToCodePoint);
      To.push_back(ToCodePoint);
    }
    // Sentinel
    To.push_back(0);

    Entries.emplace_back(CodePoint, To);
  }
  llvm::sort(Entries);

  unsigned LargestValue =
      llvm::max_element(Entries, [](const auto &Entry0, const auto &Entry1) {
        return Entry0.second.size() < Entry1.second.size();
      })->second.size();

  std::error_code Ec;
  llvm::raw_fd_ostream Os(argv[2], Ec);

  // FIXME: If memory consumption and/or lookup time becomes a constraint, it
  // maybe worth using a more elaborate data structure.
  Os << "struct {llvm::UTF32 codepoint; llvm::UTF32 values[" << LargestValue
     << "];} "
        "ConfusableEntries[] = {\n";
  for (const auto &Values : Entries) {
    Os << "  { ";
    Os << Values.first;
    Os << ", {";
    for (auto CP : Values.second)
      Os << CP << ", ";

    Os << "}},\n";
  }
  Os << "};\n";
  return 0;
}
