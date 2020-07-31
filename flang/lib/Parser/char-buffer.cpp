//===-- lib/Parser/char-buffer.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/char-buffer.h"
#include "flang/Common/idioms.h"
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace Fortran::parser {

char *CharBuffer::FreeSpace(std::size_t &n) {
  int offset{LastBlockOffset()};
  if (blocks_.empty()) {
    blocks_.emplace_front();
    last_ = blocks_.begin();
    lastBlockEmpty_ = true;
  } else if (offset == 0 && !lastBlockEmpty_) {
    last_ = blocks_.emplace_after(last_);
    lastBlockEmpty_ = true;
  }
  n = Block::capacity - offset;
  return last_->data + offset;
}

void CharBuffer::Claim(std::size_t n) {
  if (n > 0) {
    bytes_ += n;
    lastBlockEmpty_ = false;
  }
}

std::size_t CharBuffer::Put(const char *data, std::size_t n) {
  std::size_t chunk;
  for (std::size_t at{0}; at < n; at += chunk) {
    char *to{FreeSpace(chunk)};
    chunk = std::min(n - at, chunk);
    Claim(chunk);
    std::memcpy(to, data + at, chunk);
  }
  return bytes_ - n;
}

std::size_t CharBuffer::Put(const std::string &str) {
  return Put(str.data(), str.size());
}

std::string CharBuffer::Marshal() const {
  std::string result;
  std::size_t bytes{bytes_};
  result.reserve(bytes);
  for (const Block &block : blocks_) {
    std::size_t chunk{std::min(bytes, Block::capacity)};
    for (std::size_t j{0}; j < chunk; ++j) {
      result += block.data[j];
    }
    bytes -= chunk;
  }
  result.shrink_to_fit();
  CHECK(result.size() == bytes_);
  return result;
}
} // namespace Fortran::parser
