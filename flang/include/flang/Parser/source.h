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
#include "flang/Common/reference.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace Fortran::parser {

std::string DirectoryName(std::string path);
std::optional<std::string> LocateSourceFile(
    std::string name, const std::list<std::string> &searchPath);
std::vector<std::string> LocateSourceFileAll(
    std::string name, const std::vector<std::string> &searchPath);

class SourceFile;

struct SourcePosition {
  common::Reference<const SourceFile> sourceFile;
  common::Reference<const std::string>
      path; // may not be sourceFile.path() when #line present
  int line, column;
  int trueLineNumber;
};

class SourceFile {
public:
  explicit SourceFile(Encoding e) : encoding_{e} {}
  ~SourceFile();
  const std::string &path() const { return path_; }
  llvm::ArrayRef<char> content() const {
    return buf_->getBuffer().slice(bom_end_, buf_end_ - bom_end_);
  }
  std::size_t bytes() const { return content().size(); }
  std::size_t lines() const { return lineStart_.size(); }
  Encoding encoding() const { return encoding_; }

  bool Open(std::string path, llvm::raw_ostream &error);
  bool ReadStandardInput(llvm::raw_ostream &error);
  void Close();
  SourcePosition GetSourcePosition(std::size_t) const;
  std::size_t GetLineStartOffset(int lineNumber) const {
    return lineStart_.at(lineNumber - 1);
  }
  const std::string &SavePath(std::string &&);
  void LineDirective(int trueLineNumber, const std::string &, int);
  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;

private:
  struct SourcePositionOrigin {
    const std::string &path;
    int line;
  };

  void ReadFile();
  void IdentifyPayload();
  void RecordLineStarts();

  std::string path_;
  std::unique_ptr<llvm::WritableMemoryBuffer> buf_;
  std::vector<std::size_t> lineStart_;
  std::size_t bom_end_{0};
  std::size_t buf_end_;
  Encoding encoding_;
  std::set<std::string> distinctPaths_;
  std::map<std::size_t, SourcePositionOrigin> origins_;
};
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_SOURCE_H_
