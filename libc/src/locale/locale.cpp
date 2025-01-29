//===-- Implementation of locale ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/locale/locale.h"

#include "include/llvm-libc-macros/locale-macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

__locale_t c_locale = {nullptr};

locale_t locale = nullptr;

} // namespace LIBC_NAMESPACE_DECL
