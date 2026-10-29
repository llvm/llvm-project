//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of backtrace
///
//===----------------------------------------------------------------------===//

#include "src/execinfo/backtrace.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/callonce.h"
#include "src/dlfcn/dlopen.h"
#include "src/dlfcn/dlsym.h"
#include "src/execinfo/unwind.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

struct Walk {
  void **buffer;
  int size;
  int count;
  // The first frame the unwinder reports is backtrace itself, which the
  // caller did not ask about.
  int skip;
  uintptr_t (*ip)(execinfo::UnwindContext *);
};

execinfo::UnwindReasonCode record(execinfo::UnwindContext *context,
                                  void *argument) {
  Walk *walk = static_cast<Walk *>(argument);
  if (walk->skip > 0) {
    --walk->skip;
    return execinfo::UNWIND_NO_REASON;
  }
  if (walk->count >= walk->size)
    return execinfo::UNWIND_END_OF_STACK;
  const uintptr_t ip = walk->ip(context);
  if (ip == 0)
    return execinfo::UNWIND_END_OF_STACK;
  walk->buffer[walk->count++] = reinterpret_cast<void *>(ip);
  return execinfo::UNWIND_NO_REASON;
}

// The unwinder a program has not linked against is looked for at the names
// it goes by, once, the first time a backtrace is asked for. glibc does the
// same with libgcc_s. Because this loads an object it is not safe to call for
// the first time from a signal handler; calling it once beforehand settles
// it.
struct Unwinder {
  execinfo::UnwindReasonCode (*walk)(execinfo::UnwindTraceFn, void *) = nullptr;
  uintptr_t (*ip)(execinfo::UnwindContext *) = nullptr;
};

LIBC_CONSTINIT CallOnceFlag looked_for_unwinder = callonce_impl::NOT_CALLED;
Unwinder unwinder;

void find_unwinder() {
  Unwinder found;
  if (_Unwind_Backtrace != nullptr && _Unwind_GetIP != nullptr) {
    found.walk = _Unwind_Backtrace;
    found.ip = _Unwind_GetIP;
    unwinder = found;
    return;
  }

  // RTLD_LAZY, which <dlfcn.h> states as a value rather than in a header
  // this can include.
  constexpr int LAZY = 0x00001;
  static const char *const NAMES[] = {"libunwind.so.1", "libgcc_s.so.1"};
  for (const char *name : NAMES) {
    void *handle = internal::dlopen_impl(name, LAZY);
    if (handle == nullptr)
      continue;
    void *walk = internal::dlsym_impl(handle, "_Unwind_Backtrace");
    void *ip = internal::dlsym_impl(handle, "_Unwind_GetIP");
    if (walk == nullptr || ip == nullptr)
      continue;
    found.walk = reinterpret_cast<decltype(found.walk)>(walk);
    found.ip = reinterpret_cast<decltype(found.ip)>(ip);
    break;
  }
  unwinder = found;
}

} // anonymous namespace

// The return addresses on the stack, innermost first. Nothing is allocated,
// so this is safe to call from a signal handler, which is what it is mostly
// for.
LLVM_LIBC_FUNCTION(int, backtrace, (void **buffer, int size)) {
  if (buffer == nullptr || size <= 0)
    return 0;
  // Without an unwinder anywhere there is no stack to walk. Reporting no
  // frames is the honest answer, and is what a program which prints them can
  // carry on from.
  callonce(&looked_for_unwinder, find_unwinder);
  if (unwinder.walk == nullptr)
    return 0;

  Walk walk = {buffer, size, 0, 1, unwinder.ip};
  unwinder.walk(&record, &walk);
  return walk.count;
}

} // namespace LIBC_NAMESPACE_DECL
