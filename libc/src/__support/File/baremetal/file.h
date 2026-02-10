//===--- A platform independent file data structure -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FILE_BAREMETAL_FILE_H
#define LLVM_LIBC_SRC___SUPPORT_FILE_BAREMETAL_FILE_H

#include "hdr/types/FILE.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// For internal libc uses it is preferable to use these declarations over the
// public header ones in hdr/stdio_macros.h to ensure these have the internal
// visibility and avoid the GOT relocations.

// TODO: We should eventually consolidate this header with File/file.h.

extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FILE_BAREMETAL_FILE_H
