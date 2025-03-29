//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_STREAM_TYPE_H
#define TEST_SUPPORT_STREAM_TYPE_H

#include <streambuf>
#include <string>
#include <utility>

template <class CharT>
class non_buffering_streambuf : public std::basic_streambuf<CharT> {
  using char_type   = CharT;
  using traits_type = std::char_traits<CharT>;
  using int_type    = typename traits_type::int_type;

public:
  non_buffering_streambuf(std::basic_string<char_type> underlying_data)
      : underlying_data_(std::move(underlying_data)), index_(0) {}

protected:
  int_type underflow() override {
    if (index_ != underlying_data_.size())
      return underlying_data_[index_];
    return traits_type::eof();
  }

  int_type uflow() override {
    if (index_ != underlying_data_.size())
      return underlying_data_[index_++];
    return traits_type::eof();
  }

private:
  std::basic_string<char_type> underlying_data_;
  size_t index_;
};

#endif // TEST_SUPPORT_STREAM_TYPE_H
