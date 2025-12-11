//===-- sanitizer_printf.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer.
//
// Declares the internal vsnprintf function, used inside run-time libraries.
// `internal_snprintf` is declared in sanitizer_libc.
//===----------------------------------------------------------------------===//

#include <stdarg.h>

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

int internal_vsnprintf(char* buff, int buff_length, const char* format,
                       va_list args);

}  // namespace __sanitizer
