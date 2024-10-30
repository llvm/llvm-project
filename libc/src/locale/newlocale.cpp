//===-- Implementation of newlocale ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/locale/newlocale.h"
#include "include/llvm-libc-macros/locale-macros.h"
#include "src/locale/locale.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(locale_t, newlocale,
                   (int category_mask, const char *locale_name, locale_t)) {
  cpp::string_view name(locale_name);
  if (category_mask > LC_ALL || (!name.empty() && name != "C"))
    return nullptr;

  return &c_locale;
}

} // namespace LIBC_NAMESPACE_DECL
