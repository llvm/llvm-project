//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of uselocale.
///
//===----------------------------------------------------------------------===//

#include "src/locale/uselocale.h"
#include "hdr/locale_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/locale/locale.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(locale_t, uselocale, ([[maybe_unused]] locale_t newloc)) {
  if constexpr (DISABLE_RUNTIME_LOCALE)
    return LC_GLOBAL_LOCALE;

  locale_t oldloc = get_thread_locale();
  if (newloc != nullptr)
    set_thread_locale(newloc == LC_GLOBAL_LOCALE ? nullptr : newloc);

  return (oldloc == nullptr) ? LC_GLOBAL_LOCALE : oldloc;
}

} // namespace LIBC_NAMESPACE_DECL
