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
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

// TODO: Port to Windows &c.

namespace Fortran::parser {

static constexpr bool useMMap{true};
static constexpr int minMapFileBytes{1};  // i.e., no minimum requirement
static constexpr int maxMapOpenFileDescriptors{100};
static int openFileDescriptors{0};

SourceFile::~SourceFile() { Close(); }

static std::vector<std::size_t> FindLineStarts(
    const char *source, std::size_t bytes) {
  std::vector<std::size_t> result;
  if (bytes > 0) {
    CHECK(source[bytes - 1] == '\n' && "missing ultimate newline");
    std::size_t at{0};
    do {
      result.push_back(at);
      const void *vp{static_cast<const void *>(&source[at])};
      const void *vnl{std::memchr(vp, '\n', bytes - at)};
      const char *nl{static_cast<const char *>(vnl)};
      at = nl + 1 - source;
    } while (at < bytes);
    result.shrink_to_fit();
  }
  return result;
}

void SourceFile::RecordLineStarts() {
  lineStart_ = FindLineStarts(content_, bytes_);
}

// Check for a Unicode byte order mark (BOM).
// Module files all have one; so can source files.
void SourceFile::IdentifyPayload() {
  content_ = address_;
  bytes_ = size_;
  if (content_) {
    static constexpr int BOMBytes{3};
    static const char UTF8_BOM[]{"\xef\xbb\xbf"};
    if (bytes_ >= BOMBytes && std::memcmp(content_, UTF8_BOM, BOMBytes) == 0) {
      content_ += BOMBytes;
      bytes_ -= BOMBytes;
      encoding_ = Encoding::UTF_8;
    }
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
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) == 0 && !S_ISDIR(statbuf.st_mode)) {
      return path;
    }
  }
  return name;
}

static std::size_t RemoveCarriageReturns(char *buffer, std::size_t bytes) {
  std::size_t wrote{0};
  char *p{buffer};
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
    std::memmove(buffer + wrote, p, chunk);
    wrote += chunk;
    p += chunk + 1;
    bytes -= chunk + 1;
  }
  return wrote;
}

bool SourceFile::Open(std::string path, std::stringstream *error) {
  Close();
  path_ = path;
  std::string errorPath{"'"s + path + "'"};
  errno = 0;
  fileDescriptor_ = open(path.c_str(), O_RDONLY);
  if (fileDescriptor_ < 0) {
    *error << "Could not open " << errorPath << ": " << std::strerror(errno);
    return false;
  }
  ++openFileDescriptors;
  return ReadFile(errorPath, error);
}

bool SourceFile::ReadStandardInput(std::stringstream *error) {
  Close();
  path_ = "standard input";
  fileDescriptor_ = 0;
  return ReadFile(path_, error);
}

bool SourceFile::ReadFile(std::string errorPath, std::stringstream *error) {
  struct stat statbuf;
  if (fstat(fileDescriptor_, &statbuf) != 0) {
    *error << "fstat failed on " << errorPath << ": " << std::strerror(errno);
    Close();
    return false;
  }
  if (S_ISDIR(statbuf.st_mode)) {
    *error << errorPath << " is a directory";
    Close();
    return false;
  }

  // Try to map a large source file into the process' address space.
  // Don't bother with small ones.  This also helps keep the number
  // of open file descriptors from getting out of hand.
  if (useMMap && S_ISREG(statbuf.st_mode)) {
    size_ = static_cast<std::size_t>(statbuf.st_size);
    if (size_ >= minMapFileBytes &&
        openFileDescriptors <= maxMapOpenFileDescriptors) {
      void *vp = mmap(0, size_, PROT_READ, MAP_SHARED, fileDescriptor_, 0);
      if (vp != MAP_FAILED) {
        address_ = static_cast<const char *>(const_cast<const void *>(vp));
        IdentifyPayload();
        if (bytes_ > 0 && content_[bytes_ - 1] == '\n' &&
            std::memchr(static_cast<const void *>(content_), '\r', bytes_) ==
                nullptr) {
          isMemoryMapped_ = true;
          RecordLineStarts();
          return true;
        }
        // The file needs to have its line endings normalized to simple
        // newlines.  Remap it for a private rewrite in place.
        vp = mmap(
            vp, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileDescriptor_, 0);
        if (vp != MAP_FAILED) {
          address_ = static_cast<const char *>(const_cast<const void *>(vp));
          IdentifyPayload();
          auto mutableContent{const_cast<char *>(content_)};
          bytes_ = RemoveCarriageReturns(mutableContent, bytes_);
          if (bytes_ > 0) {
            if (mutableContent[bytes_ - 1] == '\n' ||
                (bytes_ & 0xfff) != 0 /* don't cross into next page */) {
              if (mutableContent[bytes_ - 1] != '\n') {
                // Append a final newline.
                mutableContent[bytes_++] = '\n';
              }
              bool isNowReadOnly{mprotect(vp, bytes_, PROT_READ) == 0};
              CHECK(isNowReadOnly);
              content_ = mutableContent;
              isMemoryMapped_ = true;
              RecordLineStarts();
              return true;
            }
          }
        }
        munmap(vp, size_);
        address_ = content_ = nullptr;
        size_ = bytes_ = 0;
      }
    }
  }

  // Read it into an expandable buffer, then marshal its content into a single
  // contiguous block.
  CharBuffer buffer;
  while (true) {
    std::size_t count;
    char *to{buffer.FreeSpace(&count)};
    ssize_t got{read(fileDescriptor_, to, count)};
    if (got < 0) {
      *error << "could not read " << errorPath << ": " << std::strerror(errno);
      Close();
      return false;
    }
    if (got == 0) {
      break;
    }
    buffer.Claim(got);
  }
  if (fileDescriptor_ > 0) {
    close(fileDescriptor_);
    --openFileDescriptors;
  }
  fileDescriptor_ = -1;
  normalized_ = buffer.MarshalNormalized();
  address_ = normalized_.c_str();
  size_ = normalized_.size();
  IdentifyPayload();
  RecordLineStarts();
  return true;
}

void SourceFile::Close() {
  if (useMMap && isMemoryMapped_) {
    munmap(reinterpret_cast<void *>(const_cast<char *>(address_)), size_);
    isMemoryMapped_ = false;
  } else if (!normalized_.empty()) {
    normalized_.clear();
  } else if (address_) {
    delete[] address_;
  }
  address_ = content_ = nullptr;
  size_ = bytes_ = 0;
  if (fileDescriptor_ > 0) {
    close(fileDescriptor_);
    --openFileDescriptors;
  }
  fileDescriptor_ = -1;
  path_.clear();
}

SourcePosition SourceFile::FindOffsetLineAndColumn(std::size_t at) const {
  CHECK(at < bytes_);
  if (lineStart_.empty()) {
    return {*this, 1, static_cast<int>(at + 1)};
  }
  std::size_t low{0}, count{lineStart_.size()};
  while (count > 1) {
    std::size_t mid{low + (count >> 1)};
    if (lineStart_[mid] > at) {
      count = mid - low;
    } else {
      count -= mid - low;
      low = mid;
    }
  }
  return {*this, static_cast<int>(low + 1),
      static_cast<int>(at - lineStart_[low] + 1)};
}
}
