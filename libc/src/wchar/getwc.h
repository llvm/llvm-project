//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the prototype function for the getwc function, which
/// reads a single character from the provided stream.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_WCHAR_GETWC_H
#define LLVM_LIBC_SRC_WCHAR_GETWC_H

#include "hdr/types/FILE.h"
#include "hdr/types/wint_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

wint_t getwc(::FILE *stream);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_WCHAR_GETWC_H
