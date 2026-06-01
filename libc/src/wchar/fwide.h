//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the prototype of the fwide function, which sets and
/// gets the orientation of a stream.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_WCHAR_FWIDE_H
#define LLVM_LIBC_SRC_WCHAR_FWIDE_H

#include "hdr/types/FILE.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int fwide(::FILE *stream, int mode);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_WCHAR_FWIDE_H
