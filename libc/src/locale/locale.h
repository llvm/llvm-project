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
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/locale/locale_data.h"

namespace LIBC_NAMESPACE_DECL {

template <bool DisableRuntimeLocale> struct LocaleStorage {
  const LcCtypeData *ctype_ptr;
  const LcNumericData *numeric_ptr;
  const LcTimeData *time_ptr;
  const void *collate_ptr;
  const LcMonetaryData *monetary_ptr;
  const LcMessagesData *messages_ptr;

  LIBC_INLINE constexpr LocaleStorage(const LcCtypeData *c,
                                      const LcNumericData *num,
                                      const LcTimeData *t, const void *col,
                                      const LcMonetaryData *mon,
                                      const LcMessagesData *msg)
      : ctype_ptr(c), numeric_ptr(num), time_ptr(t), collate_ptr(col),
        monetary_ptr(mon), messages_ptr(msg) {}

  LIBC_INLINE constexpr const LcCtypeData *ctype() const { return ctype_ptr; }
  LIBC_INLINE constexpr const LcNumericData *numeric() const {
    return numeric_ptr;
  }
  LIBC_INLINE constexpr const LcTimeData *time() const { return time_ptr; }
  LIBC_INLINE constexpr const LcMonetaryData *monetary() const {
    return monetary_ptr;
  }
  LIBC_INLINE constexpr const LcMessagesData *messages() const {
    return messages_ptr;
  }
};

template <> struct LocaleStorage<true> {
  LIBC_INLINE constexpr LocaleStorage(const LcCtypeData *,
                                      const LcNumericData *, const LcTimeData *,
                                      const void *, const LcMonetaryData *,
                                      const LcMessagesData *) {}

  LIBC_INLINE constexpr const LcCtypeData *ctype() const { return nullptr; }
  LIBC_INLINE constexpr const LcNumericData *numeric() const { return nullptr; }
  LIBC_INLINE constexpr const LcTimeData *time() const { return nullptr; }
  LIBC_INLINE constexpr const LcMonetaryData *monetary() const {
    return nullptr;
  }
  LIBC_INLINE constexpr const LcMessagesData *messages() const {
    return nullptr;
  }
};

} // namespace LIBC_NAMESPACE_DECL

struct __locale_t
    : LIBC_NAMESPACE::LocaleStorage<LIBC_NAMESPACE::DISABLE_RUNTIME_LOCALE> {
  using LIBC_NAMESPACE::LocaleStorage<
      LIBC_NAMESPACE::DISABLE_RUNTIME_LOCALE>::LocaleStorage;
};

namespace LIBC_NAMESPACE_DECL {

// The static "C" locale instance.
extern __locale_t c_locale;

// The static UTF-8 locale instance.
extern __locale_t utf8_locale;

// The global locale instance.
extern locale_t global_locale;

// Thread-local locale accessor functions.
locale_t get_thread_locale();
void set_thread_locale(locale_t loc);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LOCALE_LOCALE_H
