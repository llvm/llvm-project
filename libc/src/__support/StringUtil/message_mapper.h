//===-- A class for number to string mappings -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_MESSAGE_MAPPER_H
#define LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_MESSAGE_MAPPER_H

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/string_view.h"
#include <stddef.h>

namespace __llvm_libc {
namespace internal {

struct MsgMapping {
  int num;
  cpp::string_view msg;

  constexpr MsgMapping() : num(0), msg() { ; }

  constexpr MsgMapping(int init_num, const char *init_msg)
      : num(init_num), msg(init_msg) {
    ;
  }
};

constexpr size_t total_str_len(const MsgMapping *array, size_t len) {
  size_t total = 0;
  for (size_t i = 0; i < len; ++i) {
    // add 1 for the null terminator.
    total += array[i].msg.size() + 1;
  }
  return total;
}

constexpr size_t max_key_val(const MsgMapping *array, size_t len) {
  int max = 0;
  for (size_t i = 0; i < len; ++i) {
    if (array[i].num > max) {
      max = array[i].num;
    }
  }
  // max will never be negative since the starting value is 0. This is good,
  // since it's used as a length.
  return static_cast<size_t>(max);
}

template <size_t ARR_SIZE, size_t TOTAL_STR_LEN> class MessageMapper {
  int msg_offsets[ARR_SIZE] = {-1};
  char string_array[TOTAL_STR_LEN] = {'\0'};

public:
  constexpr MessageMapper(const MsgMapping raw_array[], size_t raw_array_len) {
    cpp::string_view string_mappings[ARR_SIZE] = {""};
    for (size_t i = 0; i < raw_array_len; ++i)
      string_mappings[raw_array[i].num] = raw_array[i].msg;

    int string_array_index = 0;
    for (size_t cur_num = 0; cur_num < ARR_SIZE; ++cur_num) {
      if (string_mappings[cur_num].size() != 0) {
        msg_offsets[cur_num] = string_array_index;
        // No need to replace with proper strcpy, this is evaluated at compile
        // time.
        for (size_t i = 0; i < string_mappings[cur_num].size() + 1;
             ++i, ++string_array_index) {
          string_array[string_array_index] = string_mappings[cur_num][i];
        }
      } else {
        msg_offsets[cur_num] = -1;
      }
    }
  }

  cpp::optional<cpp::string_view> get_str(int num) const {
    if (num >= 0 && static_cast<size_t>(num) < ARR_SIZE &&
        msg_offsets[num] != -1) {
      return {string_array + msg_offsets[num]};
    } else {
      return cpp::optional<cpp::string_view>();
    }
  }
};

template <size_t N> using MsgTable = cpp::array<MsgMapping, N>;

template <size_t N1, size_t N2>
constexpr MsgTable<N1 + N2> operator+(const MsgTable<N1> &t1,
                                      const MsgTable<N2> &t2) {
  MsgTable<N1 + N2> res{};
  for (size_t i = 0; i < N1; ++i)
    res[i] = t1[i];
  for (size_t i = 0; i < N2; ++i)
    res[N1 + i] = t2[i];
  return res;
}

} // namespace internal
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_MESSAGE_MAPPER_H
