//===-- include/flang/Parser/source.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_SOURCE_H_
#define FORTRAN_PARSER_SOURCE_H_

// Source file content is lightly normalized when the file is read.
//  - Line ending markers are converted to single newline characters
//  - A newline character is added to the last line of the file if one is needed
//  - A Unicode byte order mark is recognized if present.

#include "characters.h"
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace Fortran::parser {

std::string DirectoryName(std::string path);
std::string LocateSourceFile(
    std::string name, const std::vector<std::string> &searchPath);

class SourceFile;

struct SourcePosition {
  const SourceFile &file;
  int line, column;
};

class SourceFile {
public:
  explicit SourceFile(Encoding e) : encoding_{e} {}
  ~SourceFile();
  std::string path() const { return path_; }
  const char *content() const { return content_; }
  std::size_t bytes() const { return bytes_; }
  std::size_t lines() const { return lineStart_.size(); }
  Encoding encoding() const { return encoding_; }

  bool Open(std::string path, std::stringstream *error);
  bool ReadStandardInput(std::stringstream *error);
  void Close();
  SourcePosition FindOffsetLineAndColumn(std::size_t) const;
  std::size_t GetLineStartOffset(int lineNumber) const {
    return lineStart_.at(lineNumber - 1);
  }

private:
  bool ReadFile(std::string errorPath, std::stringstream *error);
  void IdentifyPayload();
  void RecordLineStarts();

  std::string path_;
  int fileDescriptor_{-1};
  bool isMemoryMapped_{false};
  const char *address_{nullptr};  // raw content
  std::size_t size_{0};
  const char *content_{nullptr};  // usable content
  std::size_t bytes_{0};
  std::vector<std::size_t> lineStart_;
  std::string normalized_;
  Encoding encoding_{Encoding::UTF_8};
};
}
#endif  // FORTRAN_PARSER_SOURCE_H_
