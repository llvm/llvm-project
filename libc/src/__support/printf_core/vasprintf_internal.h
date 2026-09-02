//===-- Internal Implementation of asprintf ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_VASPRINTF_INTERNAL_H
#define LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_VASPRINTF_INTERNAL_H

#include "hdr/func/malloc.h"
#include "src/__support/arg_list.h"
#include "src/__support/error_or.h"
#include "src/__support/printf_core/core_structs.h"
#include "src/__support/printf_core/make_resizing_writer.h"
#include "src/__support/printf_core/printf_main.h"

namespace LIBC_NAMESPACE_DECL {
namespace printf_core {

constexpr size_t DEFAULT_BUFFER_SIZE = 200;

template <bool use_modular = false>
LIBC_INLINE ErrorOr<size_t> vasprintf_internal(char **ret,
                                               const char *__restrict format,
                                               internal::ArgList args) {
  char init_buff_on_stack[DEFAULT_BUFFER_SIZE];
  Writer writer = make_resizing_writer(init_buff_on_stack, DEFAULT_BUFFER_SIZE,
                                       /* buffer_on_stack = */ true);

  auto ret_val = [&] {
    if constexpr (use_modular)
      return printf_core::printf_main_modular(&writer, format, args);
    else
      return printf_core::printf_main(&writer, format, args);
  }();
  if (!ret_val.has_value()) {
    *ret = nullptr;
    return ret_val;
  }
  char *final_buff = writer.get_write_buffer().buff;
  if (final_buff == init_buff_on_stack) {
    *ret = static_cast<char *>(malloc(ret_val.value() + 1));
    if (ret == nullptr)
      return Error(ALLOCATION_ERROR);
    inline_memcpy(*ret, final_buff, ret_val.value());
  } else {
    *ret = final_buff;
  }
  (*ret)[ret_val.value()] = '\0';
  return ret_val;
}
} // namespace printf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_VASPRINTF_INTERNAL_H
