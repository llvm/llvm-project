//===-- Implementation of a class for mapping errors to strings -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "error_to_string.h"

#include <stddef.h>

#include "platform_errors.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/stringstream.h"
#include "src/__support/StringUtil/message_mapper.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace {

constexpr size_t max_buff_size() {
  constexpr size_t unknown_str_len = sizeof("Unknown error");
  // the buffer should be able to hold "Unknown error" + ' ' + num_str
  return (unknown_str_len + 1 + IntegerToString<int>::buffer_size()) *
         sizeof(char);
}

// This is to hold error strings that have to be custom built. It may be
// rewritten on every call to strerror (or other error to string function).
constexpr size_t ERR_BUFFER_SIZE = max_buff_size();
LIBC_THREAD_LOCAL char error_buffer[ERR_BUFFER_SIZE];

constexpr size_t TOTAL_STR_LEN = total_str_len(PLATFORM_ERRORS);

// Since the StringMappings array is a map from error numbers to their
// corresponding strings, we have to have an array large enough we can use the
// error numbers as indexes. The current linux configuration has 132 values with
// the maximum value being 133 (41 and 58 are skipped). If other platforms use
// negative numbers or discontiguous ranges, then the array should be turned
// into a proper hashmap.
constexpr size_t ERR_ARRAY_SIZE = max_key_val(PLATFORM_ERRORS) + 1;

constexpr MessageMapper<ERR_ARRAY_SIZE, TOTAL_STR_LEN>
    ERROR_MAPPER(PLATFORM_ERRORS);

constexpr MessageMapper<ERR_ARRAY_SIZE, TOTAL_STR_LEN>
    ERRNO_NAME_MAPPER(PLATFORM_ERRNO_NAMES);

// Writes the text of an error that has no description of its own into the
// buffer, without a terminator. The buffer is large enough for the text and a
// terminator, which a caller that wants a C string adds.
cpp::string_view build_error_string(int err_num, cpp::span<char> buffer) {
  cpp::StringStream buffer_stream(buffer);
  buffer_stream << "Unknown error" << ' ' << err_num;
  return buffer_stream.str();
}

bool can_build_error_string(cpp::span<char> buffer) {
  return buffer.size() >= max_buff_size();
}

} // namespace

cpp::string_view get_error_string(int err_num) {
  return get_error_string(err_num, {error_buffer, ERR_BUFFER_SIZE});
}

cpp::string_view get_error_string(int err_num, cpp::span<char> buffer) {
  auto opt_str = ERROR_MAPPER.get_str(err_num);
  if (opt_str)
    return *opt_str;
  // If the buffer can't hold "Unknown error" + ' ' + num_str, then just
  // return "Unknown error".
  if (!can_build_error_string(buffer))
    return "Unknown error";
  return build_error_string(err_num, buffer);
}

const char *get_error_cstring(int err_num) {
  return get_error_cstring(err_num, {error_buffer, ERR_BUFFER_SIZE});
}

const char *get_error_cstring(int err_num, cpp::span<char> buffer) {
  // The descriptions of known errors are stored with their terminator.
  auto opt_str = ERROR_MAPPER.get_str(err_num);
  if (opt_str)
    return opt_str->data();
  if (!can_build_error_string(buffer))
    return "Unknown error";
  cpp::string_view text = build_error_string(err_num, buffer);
  buffer[text.size()] = '\0';
  return buffer.data();
}

cpp::optional<cpp::string_view> try_get_errno_name(int err_num) {
  return ERRNO_NAME_MAPPER.get_str(err_num);
}

} // namespace LIBC_NAMESPACE_DECL
