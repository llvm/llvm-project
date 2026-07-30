//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// Tests unwinding where the signal handler is a VAPI function.
//
/// `exit` is used as the signal handler and the unwinding is done in an atexit
/// handler.
/// `__builtin_debugtrap` is used because `raise` is also a VAPI function.

// REQUIRES: target={{.+}}-aix{{.*}}
// REQUIRES: has-filecheck

// ADDITIONAL_COMPILE_FLAGS: -fno-inline -fno-exceptions

// RUN: %{build}
// RUN: %{exec} %t.exe 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK \
// RUN:       %if libunwind-assertions-enabled %{ --check-prefix=DEBUG %} \
// RUN:       %s

#include <errno.h>
#include <libunwind.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

/// VAPI glue addresses
constexpr uintptr_t vapi_glue_addr_ext_32 = 0x8b80;
constexpr uintptr_t vapi_addr_64 = 0x8e00;
constexpr size_t vapi_size_64 = 0x0200;
constexpr uintptr_t vapi_glue_addr_begin = vapi_glue_addr_ext_32;
constexpr uintptr_t vapi_glue_addr_end = vapi_addr_64 + vapi_size_64;

struct FunctionDescriptor {
  uintptr_t entry;
  uintptr_t toc;
  uintptr_t env;
};

void my_atexit_handler(void) {
  FunctionDescriptor *fd = reinterpret_cast<FunctionDescriptor *>(exit);
  bool llu_enabled =
      vapi_glue_addr_begin <= fd->entry && fd->entry < vapi_glue_addr_end;

  fprintf(stderr, "Retrieve a cursor to `main` by stepping up.\n");
  // CHECK-LABEL: Retrieve a cursor to `main`
  unw_context_t context;
  unw_cursor_t cursor;
  unw_getcontext(&context);
  unw_init_local(&cursor, &context);
  /// Step from `my_atexit_handler` up to `exit`.
  unw_step(&cursor);
  if (!llu_enabled)
    fprintf(stderr,
            "libunwind: the next return address=VAPI_NOT_ENABLED from VAPI\n");
  /// Note:
  /// The synthetic VAPI_NOT_ENABLED output would appear _after_ the trace
  /// output indicating that the "next is a signal handler frame". Use DEBUG-DAG
  /// to allow for that.
  // DEBUG-LABEL: libunwind: stepWithTBTable: Look up traceback table of func=_Z17my_atexit_handlerv
  // DEBUG-DAG: libunwind: the next return address={{[^ ]*}} from VAPI
  // DEBUG-DAG: libunwind: The next is a signal handler frame
  /// Step from `exit` (signal handler, VAPI) up to `trapper`.
  unw_step(&cursor);
  if (!llu_enabled)
    fprintf(stderr,
            "libunwind: The return address in stack VAPI_NOT_ENABLED is within "
            "the range of VAPI address; set isKnownVapiNotActive to true\n");
  /// Note:
  /// The synthetic VAPI_NOT_ENABLED output would appear after _all_ of the trace
  /// output from the `unw_step` call. Use DEBUG instead of DEBUG-NEXT to allow
  /// for that.
  // DEBUG-LABEL: libunwind: stepWithTBTable: Look up traceback table of func=exit
  // DEBUG-NEXT: libunwind: Possible signal handler frame
  // DEBUG: libunwind: {{.*}} is within the range of VAPI address; set isKnownVapiNotActive
  /// Step from `trapper` up to `main`.
  unw_step(&cursor);
  // DEBUG-LABEL: libunwind: stepWithTBTable: Look up traceback table of func=_Z7trapperv
  // DEBUG-NOT: VAPI

  fprintf(stderr, "Resume `main` at the call to `trapper`.\n");
  // CHECK-LABEL: Resume `main`
  if (!llu_enabled)
    fprintf(stderr,
            "libunwind: VAPI: executing return glue VAPI_NOT_ENABLED\n");
  unw_resume(&cursor);
  // DEBUG: libunwind: VAPI: executing return glue
}

void trapper(void) {
  __builtin_debugtrap();
  _Exit(EXIT_FAILURE);
}

int main(void) {
  if (setenv("LIBUNWIND_PRINT_UNWINDING", "1", true) != 0) {
    perror("setenv");
    abort();
  }
  if (atexit(my_atexit_handler) != 0) {
    perror("atexit");
    abort();
  }
  if (signal(SIGTRAP, exit) == SIG_ERR) {
    perror("signal");
    abort();
  }
  trapper();
  _Exit(0);
}
