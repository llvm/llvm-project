//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The part of the unwind ABI a backtrace needs
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_EXECINFO_UNWIND_H
#define LLVM_LIBC_SRC_EXECINFO_UNWIND_H

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace execinfo {

// Walking the stack is the unwinder's job, not the C library's, and the
// unwinder is a separate library which a program may or may not have loaded.
// These are declared here rather than taken from <unwind.h> so that libc does
// not have to be built against it, and weak so that a program without one
// still links; backtrace then reports no frames rather than failing to start.
//
// glibc does the same thing by a different route, loading libgcc_s when it is
// first asked for a backtrace.

enum UnwindReasonCode {
  UNWIND_NO_REASON = 0,
  UNWIND_END_OF_STACK = 5,
};

struct UnwindContext;

using UnwindTraceFn = UnwindReasonCode (*)(UnwindContext *, void *);

} // namespace execinfo
} // namespace LIBC_NAMESPACE_DECL

extern "C" {

[[gnu::weak]] LIBC_NAMESPACE::execinfo::UnwindReasonCode
_Unwind_Backtrace(LIBC_NAMESPACE::execinfo::UnwindTraceFn, void *);

[[gnu::weak]] uintptr_t
_Unwind_GetIP(LIBC_NAMESPACE::execinfo::UnwindContext *);

} // extern "C"

#endif // LLVM_LIBC_SRC_EXECINFO_UNWIND_H
