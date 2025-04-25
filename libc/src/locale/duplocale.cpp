//===-- Implementation of duplocale ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/locale/duplocale.h"
#include "include/llvm-libc-macros/locale-macros.h"
#include "src/locale/locale.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(locale_t, duplocale, (locale_t loc)) { return loc; }

} // namespace LIBC_NAMESPACE_DECL
