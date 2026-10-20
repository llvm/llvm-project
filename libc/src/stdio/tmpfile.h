//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header of tmpfile.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_TMPFILE_H_
#define LLVM_LIBC_SRC_STDIO_TMPFILE_H_

#include "hdr/types/FILE.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

::FILE *tmpfile();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_TMPFILE_H_
