//===-- A simple implementation of string stream class ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_STRINGSTREAM_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_STRINGSTREAM_H

#include "ArrayRef.h"
#include "StringView.h"
#include "TypeTraits.h"

#include "src/__support/integer_to_string.h"

namespace __llvm_libc {
namespace cpp {

// This class is to be used to write simple strings into a user provided buffer
// without any dynamic memory allocation. There is no requirement to mimic the
// C++ standard library class std::stringstream.
class StringStream {
  MutableArrayRef<char> data;
  size_t write_ptr = 0; // The current write pointer
  bool err = false;     // If an error occurs while writing

  void write(const char *bytes, size_t size) {
    size_t i = 0;
    const size_t data_size = data.size();
    for (; write_ptr < data_size && i < size; ++i, ++write_ptr)
      data[write_ptr] = bytes[i];
    if (i < size) {
      // If some of the characters couldn't be written, set error.
      err = true;
    }
  }

public:
  static constexpr char ENDS = '\0';

  // Create a string stream which will write into |buf|.
  constexpr StringStream(const MutableArrayRef<char> &buf) : data(buf) {}

  // Return a StringView to the current characters in the stream. If a
  // null terminator was not explicitly written, then the return value
  // will not include one. In order to produce a StringView to a null
  // terminated string, write ENDS explicitly.
  StringView str() const { return StringView(data.data(), write_ptr); }

  // Write the characters from |str| to the stream.
  StringStream &operator<<(StringView str) {
    write(str.data(), str.size());
    return *this;
  }

  // Write the |val| as string.
  template <typename T, EnableIfType<IsIntegral<T>::Value, int> = 0>
  StringStream &operator<<(T val) {
    const auto int_to_str = integer_to_string(val);
    return operator<<(int_to_str.str());
  }

  template <typename T, EnableIfType<IsFloatingPointType<T>::Value, int> = 0>
  StringStream &operator<<(T val) {
    // If this specialization gets activated, then the static_assert will
    // trigger a compile error about missing floating point number support.
    static_assert(!IsFloatingPointType<T>::Value,
                  "Writing floating point numbers is not yet supported");
    return *this;
  }

  // Write a null-terminated string. The terminating null character is not
  // written to allow stremaing to continue.
  StringStream &operator<<(const char *str) {
    return operator<<(StringView(str));
  }

  // Write a single character.
  StringStream &operator<<(char a) {
    write(&a, 1);
    return *this;
  }

  // Return true if any write operation(s) failed due to insufficient size.
  bool overflow() const { return err; }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_STRINGSTREAM_H
