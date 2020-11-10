//===-- include/flang/Parser/char-buffer.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_CHAR_BUFFER_H_
#define FORTRAN_PARSER_CHAR_BUFFER_H_

// Defines a simple expandable buffer suitable for efficiently accumulating
// a stream of bytes.

#include <cstddef>
#include <list>
#include <string>
#include <utility>
#include <vector>

namespace Fortran::parser {

class CharBuffer {
public:
  CharBuffer() {}
  CharBuffer(CharBuffer &&that)
      : blocks_(std::move(that.blocks_)), bytes_{that.bytes_},
        lastBlockEmpty_{that.lastBlockEmpty_} {
    that.clear();
  }
  CharBuffer &operator=(CharBuffer &&that) {
    blocks_ = std::move(that.blocks_);
    bytes_ = that.bytes_;
    lastBlockEmpty_ = that.lastBlockEmpty_;
    that.clear();
    return *this;
  }

  bool empty() const { return bytes_ == 0; }
  std::size_t bytes() const { return bytes_; }

  void clear() {
    blocks_.clear();
    bytes_ = 0;
    lastBlockEmpty_ = false;
  }

  char *FreeSpace(std::size_t &);
  void Claim(std::size_t);

  // The return value is the byte offset of the new data,
  // i.e. the value of size() before the call.
  std::size_t Put(const char *data, std::size_t n);
  std::size_t Put(const std::string &);
  std::size_t Put(char x) { return Put(&x, 1); }

  std::string Marshal() const;

private:
  struct Block {
    static constexpr std::size_t capacity{1 << 20};
    char data[capacity];
  };

  int LastBlockOffset() const { return bytes_ % Block::capacity; }
  std::list<Block> blocks_;
  std::size_t bytes_{0};
  bool lastBlockEmpty_{false};
};
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_CHAR_BUFFER_H_
