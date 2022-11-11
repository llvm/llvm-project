//===-- A class for number to string mappings -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_MESSAGE_MAPPER
#define LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_MESSAGE_MAPPER

#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/string_view.h"
#include <stddef.h>

namespace __llvm_libc {
namespace internal {

struct MsgMapping {
  int num;
  cpp::string_view msg;

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

template <size_t ARR_SIZE, size_t TOTAL_STR_LEN> class MessageMapper {
  int msg_offsets[ARR_SIZE] = {-1};
  char string_array[TOTAL_STR_LEN] = {'\0'};

public:
  constexpr MessageMapper(const MsgMapping raw_array[], size_t raw_array_len) {
    cpp::string_view string_mappings[ARR_SIZE] = {""};
    for (size_t i = 0; i < raw_array_len; ++i)
      string_mappings[raw_array[i].num] = raw_array[i].msg;

    size_t string_array_index = 0;
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

} // namespace internal
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_MESSAGE_MAPPER
