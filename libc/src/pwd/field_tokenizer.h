//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// In-place field tokenizer for colon-separated flat database files.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PWD_FIELD_TOKENIZER_H
#define LLVM_LIBC_SRC_PWD_FIELD_TOKENIZER_H

#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace pwd {

// In-place field tokenizer for delimited database records.
class FieldTokenizer {
  char *data_ptr;
  size_t data_len;
  char separator;

public:
  LIBC_INLINE constexpr explicit FieldTokenizer(cpp::span<char> buf,
                                                char sep = ':')
      : data_ptr(buf.data()), data_len(buf.size()), separator(sep) {}

  // Extracts the next null-terminated field.
  LIBC_INLINE cpp::optional<cpp::span<char>> next_field() {
    if (data_len == 0)
      return cpp::nullopt;

    cpp::string_view sv(data_ptr, data_len);
    size_t pos = sv.find_first_of(separator);

    // If a delimiter was found, replace it with a null terminator and return
    // the field.
    if (pos != cpp::string_view::npos) {
      data_ptr[pos] = '\0';
      auto field = cpp::span<char>(data_ptr, pos + 1);
      data_ptr += pos + 1;
      data_len -= pos + 1;
      return field;
    }

    // If null-terminated without delimiters, return the remaining span as the
    // final field.
    if (cpp::span<char>(data_ptr, data_len).back() == '\0') {
      auto field = cpp::span<char>(data_ptr, data_len);
      data_ptr += data_len;
      data_len = 0;
      return field;
    }

    // Otherwise, no more fields remain.
    return cpp::nullopt;
  }
};

} // namespace pwd
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PWD_FIELD_TOKENIZER_H
