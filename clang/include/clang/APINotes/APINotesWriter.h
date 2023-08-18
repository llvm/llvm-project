//===-- APINotesWriter.h - API Notes Writer ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_APINOTES_WRITER_H
#define LLVM_CLANG_APINOTES_WRITER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace clang {
class FileEntry;

namespace api_notes {
class APINotesWriter {
  class Implementation;
  std::unique_ptr<Implementation> Implementation;

public:
  APINotesWriter(llvm::StringRef ModuleName, const FileEntry *SF);
  ~APINotesWriter();

  APINotesWriter(const APINotesWriter &) = delete;
  APINotesWriter &operator=(const APINotesWriter &) = delete;

  void writeToStream(llvm::raw_ostream &OS);
};
} // namespace api_notes
} // namespace clang

#endif
