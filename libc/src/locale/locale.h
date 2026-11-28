//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for locale state and structures.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LOCALE_LOCALE_H
#define LLVM_LIBC_SRC_LOCALE_LOCALE_H

#include "hdr/types/locale_t.h"
#include "src/__support/macros/config.h"
#include "src/locale/locale_data.h"

// Defined in the global namespace to match the forward declaration in
// locale_t.h.
struct __locale_t
    : LIBC_NAMESPACE::LocaleStorage<LIBC_NAMESPACE::ENABLE_RUNTIME_LOCALE> {
  using LIBC_NAMESPACE::LocaleStorage<
      LIBC_NAMESPACE::ENABLE_RUNTIME_LOCALE>::LocaleStorage;
};

namespace LIBC_NAMESPACE_DECL {

// The static "C" locale instance.
extern __locale_t c_locale;

// The static UTF-8 locale instance.
extern __locale_t utf8_locale;

// Global locale accessor functions.
locale_t get_global_locale();
void set_global_locale(locale_t loc);

// Thread-local locale accessor functions.
locale_t get_thread_locale();
void set_thread_locale(locale_t loc);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LOCALE_LOCALE_H
