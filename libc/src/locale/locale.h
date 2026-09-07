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

#ifndef LLVM_LIBC_SRC_LOCALE_LOCALECONV_H
#define LLVM_LIBC_SRC_LOCALE_LOCALECONV_H

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include "hdr/locale_macros.h"
#include "hdr/types/locale_t.h"

#include <stddef.h>

#ifndef LC_GLOBAL_LOCALE
#define LC_GLOBAL_LOCALE ((locale_t)(-1))
#endif

static constexpr size_t MAX_LOCALE_NAME_SIZE = 16;

struct __locale_data {
  char name[MAX_LOCALE_NAME_SIZE];
};

namespace LIBC_NAMESPACE_DECL {

// The pointer to the default "C" locale.
extern __locale_t c_locale;

// The pointer to the static UTF-8 locale.
extern __locale_t utf8_locale;

// The global locale instance.
extern locale_t global_locale;

// Thread-local locale accessor functions.
locale_t get_thread_locale();
void set_thread_locale(locale_t loc);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LOCALE_LOCALECONV_H
