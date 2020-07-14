//===-- lib/Parser/source.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/source.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/char-buffer.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace Fortran::parser {

SourceFile::~SourceFile() { Close(); }

static std::vector<std::size_t> FindLineStarts(llvm::StringRef source) {
  std::vector<std::size_t> result;
  if (source.size() > 0) {
    CHECK(source.back() == '\n' && "missing ultimate newline");
    std::size_t at{0};
    do {
      result.push_back(at);
      at = source.find('\n', at) + 1;
    } while (at < source.size());
    result.shrink_to_fit();
  }
  return result;
}

void SourceFile::RecordLineStarts() {
  lineStart_ = FindLineStarts({content().data(), bytes()});
}

// Check for a Unicode byte order mark (BOM).
// Module files all have one; so can source files.
void SourceFile::IdentifyPayload() {
  llvm::StringRef content{buf_->getBufferStart(), buf_->getBufferSize()};
  constexpr llvm::StringLiteral UTF8_BOM{"\xef\xbb\xbf"};
  if (content.startswith(UTF8_BOM)) {
    bom_end_ = UTF8_BOM.size();
    encoding_ = Encoding::UTF_8;
  }
}

std::string DirectoryName(std::string path) {
  auto lastSlash{path.rfind("/")};
  return lastSlash == std::string::npos ? path : path.substr(0, lastSlash);
}

std::string LocateSourceFile(
    std::string name, const std::vector<std::string> &searchPath) {
  if (name.empty() || name == "-" || name[0] == '/') {
    return name;
  }
  for (const std::string &dir : searchPath) {
    std::string path{dir + '/' + name};
    bool isDir{false};
    auto er = llvm::sys::fs::is_directory(path, isDir);
    if (!er && !isDir) {
      return path;
    }
  }
  return name;
}

std::size_t RemoveCarriageReturns(llvm::MutableArrayRef<char> buf) {
  std::size_t wrote{0};
  char *buffer{buf.data()};
  char *p{buf.data()};
  std::size_t bytes = buf.size();
  while (bytes > 0) {
    void *vp{static_cast<void *>(p)};
    void *crvp{std::memchr(vp, '\r', bytes)};
    char *crcp{static_cast<char *>(crvp)};
    if (!crcp) {
      std::memmove(buffer + wrote, p, bytes);
      wrote += bytes;
      break;
    }
    std::size_t chunk = crcp - p;
    auto advance{chunk + 1};
    if (chunk + 1 >= bytes || crcp[1] == '\n') {
      // CR followed by LF or EOF: omit
    } else if ((chunk == 0 && p == buf.data()) || crcp[-1] == '\n') {
      // CR preceded by LF or BOF: omit
    } else {
      // CR in line: retain
      ++chunk;
    }
    std::memmove(buffer + wrote, p, chunk);
    wrote += chunk;
    p += advance;
    bytes -= advance;
  }
  return wrote;
}

bool SourceFile::Open(std::string path, llvm::raw_ostream &error) {
  Close();
  path_ = path;
  std::string errorPath{"'"s + path_ + "'"};
  auto bufOr{llvm::WritableMemoryBuffer::getFile(path)};
  if (!bufOr) {
    auto err = bufOr.getError();
    error << "Could not open " << errorPath << ": " << err.message();
    return false;
  }
  buf_ = std::move(bufOr.get());
  ReadFile();
  return true;
}

bool SourceFile::ReadStandardInput(llvm::raw_ostream &error) {
  Close();
  path_ = "standard input";

  auto buf_or = llvm::MemoryBuffer::getSTDIN();
  if (!buf_or) {
    auto err = buf_or.getError();
    error << err.message();
    return false;
  }
  auto inbuf = std::move(buf_or.get());
  buf_ =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(inbuf->getBufferSize());
  llvm::copy(inbuf->getBuffer(), buf_->getBufferStart());
  ReadFile();
  return true;
}

void SourceFile::ReadFile() {
  buf_end_ = RemoveCarriageReturns(buf_->getBuffer());
  if (content().size() == 0 || content().back() != '\n') {
    // Don't bother to copy if we have spare memory
    if (content().size() >= buf_->getBufferSize()) {
      auto tmp_buf{llvm::WritableMemoryBuffer::getNewUninitMemBuffer(
          content().size() + 1)};
      llvm::copy(content(), tmp_buf->getBufferStart());
      Close();
      buf_ = std::move(tmp_buf);
    }
    buf_end_++;
    buf_->getBuffer()[buf_end_ - 1] = '\n';
  }
  IdentifyPayload();
  RecordLineStarts();
}

void SourceFile::Close() {
  path_.clear();
  buf_.reset();
}

SourcePosition SourceFile::FindOffsetLineAndColumn(std::size_t at) const {
  CHECK(at < bytes());

  auto it = llvm::upper_bound(lineStart_, at);
  auto low = std::distance(lineStart_.begin(), it - 1);
  return {*this, static_cast<int>(low + 1),
      static_cast<int>(at - lineStart_[low] + 1)};
}
} // namespace Fortran::parser
