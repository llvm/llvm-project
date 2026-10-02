//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{.+}}-aix{{.*}}
// REQUIRES: has-filecheck

// ADDITIONAL_COMPILE_FLAGS: -fno-inline -fno-exceptions

// RUN: %{build}
// RUN: %{exec} %t.exe 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK \
// RUN:       %if libunwind-assertions-enabled %{ --check-prefix=DEBUG %} \
// RUN:       %s

// Tests use of the libunwind C API to step up from a context where the VAPI is
// active and to resume contexts where
// - the VAPI is active (and thus VAPI return glue is not called) and
// - where the VAPI is not active (and thus VAPI return glue _is_ called).
//
// In the latter case, which applies not just to the caller of the Virtual API
// but also to its ancestors, the return glue should always be called (i.e.,
// each time, without regard for whether the VAPI is currently active).

#include <assert.h>
#include <libunwind.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __64BIT__
#define R3 UNW_PPC64_R3
#else
#define R3 UNW_PPC_R3
#endif

extern "C" void *returns_twice_bsearch
    [[gnu::returns_twice]] (const void *, const void *, size_t, size_t,
                            int (*)(const void *,
                                    const void *)) __asm__("bsearch");

static bool llu_enabled;
static unw_cursor_t bsearch_cursor;
static unw_cursor_t bsearch_caller_cursor;
static unw_cursor_t main_cursor;

extern "C" int cmp(const void *pa, const void *pb) {
  (void)pa;
  (void)pb;

  fprintf(
      stderr,
      "Populate global cursors for `bsearch`, `bsearch_caller`, and `main`.\n");
  // CHECK-LABEL: Populate global cursors
  unw_context_t context;
  unw_cursor_t cursor;
  unw_getcontext(&context);
  unw_init_local(&cursor, &context);
  // Step from `cmp` up to `bsearch`.
  unw_step(&cursor);
  if (!llu_enabled)
    fprintf(stderr,
            "libunwind: the next return address=VAPI_NOT_ENABLED from VAPI\n");
  // DEBUG-LABEL: libunwind: stepWithTBTable: Look up traceback table of func=cmp
  // DEBUG: libunwind: the next return address=[[VAPI_RA:[^ ]*]] from VAPI
  bsearch_cursor = cursor;
  // Step from `bsearch` up to `bsearch_caller`.
  unw_step(&cursor);
  if (!llu_enabled)
    fprintf(stderr, "libunwind: return address=VAPI_NOT_ENABLED from VAPI\n");
  // DEBUG-LABEL: libunwind: stepWithTBTable: Look up traceback table of func=bsearch
  // DEBUG: libunwind: return address=[[VAPI_RA]] from VAPI
  bsearch_caller_cursor = cursor;
  // Step from `bsearch_caller` up to `main`.
  unw_step(&cursor);
  // DEBUG-LABEL: libunwind: stepWithTBTable: Look up traceback table of func=_Z14bsearch_callerv
  // DEBUG-NOT: VAPI
  main_cursor = cursor;

  // Test resuming context where VAPI is active.
  fprintf(stderr,
          "Return to `bsearch` with r3 set to 0 using the global cursor.\n");
  unw_set_reg(&bsearch_cursor, R3, (unw_word_t)0);
  unw_resume(&bsearch_cursor);
  __builtin_unreachable();
  // CHECK-LABEL: Return to `bsearch`
  // DEBUG-NOT: VAPI: executing return glue
}

char bsearch_caller_ret;
void *bsearch_caller(void) {
  volatile int state = 0;

  char c;
  char buf[3];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized-const-pointer"
  void *result = returns_twice_bsearch(&c, buf, 1, 1, cmp);
#pragma clang diagnostic pop
  assert(result == &buf[state]);

  // Test resuming context where VAPI is not active. The VAPI return glue should
  // be used each time without regard for whether the VAPI is currently active.

  // `state` is volatile to observe changes in the value between the multiple
  // returns in `setjmp`/`longjmp`-like usage. We need to spell out the
  // `volatile` accesses very explicitly thanks to changes in the C++ standard.
  if ((state = state + 1, state) < 3) {
    fprintf(stderr,
            "Return to `bsearch_caller` at the invocation of "
            "`returns_twice_bsearch` (really `bsearch`) with r3 set to "
            "&buf[%d] using the global cursor.\n",
            state);
    if (!llu_enabled)
      fprintf(stderr,
              "libunwind: VAPI: executing return glue VAPI_NOT_ENABLED\n");
    unw_set_reg(&bsearch_caller_cursor, R3, (unw_word_t)&buf[state]);
    unw_resume(&bsearch_caller_cursor);
    __builtin_unreachable();
    // CHECK-LABEL: Return to `bsearch_caller` {{.*}}&buf[1]
    // DEBUG: libunwind: VAPI: executing return glue
    // CHECK-LABEL: Return to `bsearch_caller` {{.*}}&buf[2]
    // DEBUG: libunwind: VAPI: executing return glue
  }

  // Test resuming context where VAPI is not active, one frame up from the VAPI
  // caller.
  fprintf(stderr, "Return to `main` at the invocation of `bsearch_caller` with "
                  "r3 set to `&bsearch_caller_ret` using the global cursor.\n");
  if (!llu_enabled)
    fprintf(stderr,
            "libunwind: VAPI: executing return glue VAPI_NOT_ENABLED\n");
  unw_set_reg(&main_cursor, R3, (unw_word_t)&bsearch_caller_ret);
  unw_resume(&main_cursor);
  __builtin_unreachable();
  // CHECK-LABEL: Return to `main`
  // DEBUG: libunwind: VAPI: executing return glue
}

// VAPI glue addresses
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

int main(void) {
  if (setenv("LIBUNWIND_PRINT_UNWINDING", "1", true) != 0) {
    perror("setenv");
    abort();
  }

  FunctionDescriptor *fd = reinterpret_cast<FunctionDescriptor *>(bsearch);
  llu_enabled =
      vapi_glue_addr_begin <= fd->entry && fd->entry < vapi_glue_addr_end;

  assert(bsearch_caller() == &bsearch_caller_ret);
}
