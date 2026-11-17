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

#ifdef LIBC_CONF_ENABLE_RUNTIME_LOCALE
static locale_t global_locale =
    DEFAULT_LOCALE_IS_UTF8 ? &utf8_locale : &c_locale;

static LIBC_THREAD_LOCAL locale_t thread_locale = nullptr;
#endif

locale_t get_global_locale() {
#ifdef LIBC_CONF_ENABLE_RUNTIME_LOCALE
  return global_locale;
#else
  return DEFAULT_LOCALE_IS_UTF8 ? &utf8_locale : &c_locale;
#endif
}

void set_global_locale([[maybe_unused]] locale_t loc) {
#ifdef LIBC_CONF_ENABLE_RUNTIME_LOCALE
  global_locale = loc;
#endif
}

locale_t get_thread_locale() {
#ifdef LIBC_CONF_ENABLE_RUNTIME_LOCALE
  return thread_locale;
#else
  return nullptr;
#endif
}

void set_thread_locale([[maybe_unused]] locale_t loc) {
#ifdef LIBC_CONF_ENABLE_RUNTIME_LOCALE
  thread_locale = loc;
#endif
}

} // namespace LIBC_NAMESPACE_DECL
