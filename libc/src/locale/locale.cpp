//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of static locale instances and locale category data.
///
//===----------------------------------------------------------------------===//

#include "src/locale/locale.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/locale/locale_data.h"

namespace LIBC_NAMESPACE_DECL {

__locale_t c_locale(&C_CTYPE_DATA, &C_NUMERIC_DATA, &C_TIME_DATA, nullptr,
                    &C_MONETARY_DATA, &C_MESSAGES_DATA);

__locale_t utf8_locale(&UTF8_CTYPE_DATA, &C_NUMERIC_DATA, &C_TIME_DATA, nullptr,
                       &C_MONETARY_DATA, &C_MESSAGES_DATA);

locale_t global_locale = DEFAULT_LOCALE_IS_UTF8 ? &utf8_locale : &c_locale;

[[maybe_unused]] static LIBC_THREAD_LOCAL locale_t thread_locale = nullptr;

locale_t get_thread_locale() {
  if constexpr (DISABLE_RUNTIME_LOCALE)
    return nullptr;
  return thread_locale;
}

void set_thread_locale(locale_t loc) {
  if constexpr (!DISABLE_RUNTIME_LOCALE)
    thread_locale = loc;
}

} // namespace LIBC_NAMESPACE_DECL
