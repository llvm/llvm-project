// Test the behavior of malloc/calloc/realloc/new when the allocation size
// exceeds the configured max_allocation_size_mb flag.
// By default (allocator_may_return_null=0) the process should crash. With
// allocator_may_return_null=1 the allocator should return nullptr and set errno
// to the appropriate error code.
//
// RUN: %clangxx -O0 %s -o %t
// RUN: %run %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-NOTNULL
// RUN: %env_tool_opts=max_allocation_size_mb=3 %run %t malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOTNULL
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=0 \
// RUN:   not %run %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mCRASH
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=1 \
// RUN:   %run %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=0 \
// RUN:   not %run %t calloc 2>&1 | FileCheck %s --check-prefix=CHECK-cCRASH
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=1 \
// RUN:   %run %t calloc 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=0 \
// RUN:   not %run %t realloc 2>&1 | FileCheck %s --check-prefix=CHECK-rCRASH
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=1 \
// RUN:   %run %t realloc 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=0 \
// RUN:   not %run %t realloc-after-malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-mrCRASH
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=1 \
// RUN:   %run %t realloc-after-malloc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=0 \
// RUN:   not %run %t new 2>&1 | FileCheck %s --check-prefix=CHECK-nCRASH
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=1 \
// RUN:   not %run %t new 2>&1 | FileCheck %s --check-prefix=CHECK-nCRASH-OOM
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=0 \
// RUN:   not %run %t new-nothrow 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-nnCRASH
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=1 \
// RUN:   %run %t new-nothrow 2>&1 | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=0 \
// RUN:   not %run %t strndup 2>&1 | FileCheck %s --check-prefix=CHECK-sCRASH
// RUN: %env_tool_opts=max_allocation_size_mb=2:allocator_may_return_null=1 \
// RUN:   %run %t strndup 2>&1 | FileCheck %s --check-prefix=CHECK-NULL

// win32 is disabled due to failing errno tests.
// UNSUPPORTED: ubsan, target={{.*windows-msvc.*}}

// Symbolizer needs to allocated memory when reporting.
// UNSUPPORTED: internal_symbolizer

#include <assert.h>
#include <errno.h>
#include <limits>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t MaxAllocationSize = size_t{2} << 20;

static void *allocate(const char *Action, size_t Size) {
  if (!strcmp(Action, "malloc"))
    return malloc(Size);
  if (!strcmp(Action, "calloc"))
    return calloc((Size + 3) / 4, 4);
  if (!strcmp(Action, "realloc"))
    return realloc(nullptr, Size);
  if (!strcmp(Action, "realloc-after-malloc")) {
    void *P = malloc(100);
    if (void *Ret = realloc(P, Size))
      return Ret;
    free(P);
    return nullptr;
  }
  if (!strcmp(Action, "new"))
    return ::operator new(Size);
  if (!strcmp(Action, "new-nothrow"))
    return ::operator new(Size, std::nothrow);
  if (!strcmp(Action, "strndup")) {
    static char pstr[MaxAllocationSize + 1] = {'a'};
    for (size_t i = 0; i < MaxAllocationSize + 1; i++)
      pstr[i] = 'a';
    if (Size == MaxAllocationSize)
      pstr[MaxAllocationSize - 1] = '\0';
    return strndup(pstr, Size);
  }
  assert(0);
}

static void deallocate(const char *Action, void *Ptr) {
  if (!strcmp(Action, "malloc") || !strcmp(Action, "calloc") ||
      !strcmp(Action, "realloc") || !strcmp(Action, "realloc-after-malloc") ||
      !strcmp(Action, "strndup"))
    return free(Ptr);
  if (!strcmp(Action, "new"))
    return ::operator delete(Ptr);
  if (!strcmp(Action, "new-nothrow"))
    return ::operator delete(Ptr, std::nothrow);
  assert(0);
}

int main(int Argc, char **Argv) {
  assert(Argc == 2);
  const char *Action = Argv[1];
  fprintf(stderr, "%s:\n", Action);

  // Should succeed when max_allocation_size_mb is set.
  void *volatile P = allocate(Action, MaxAllocationSize);
  assert(P);
  deallocate(Action, P);

  // Should fail when max_allocation_size_mb is set.
  P = allocate(Action, MaxAllocationSize + 1);
  // The NULL pointer is printed differently on different systems, while (long)0
  // is always the same.
  fprintf(stderr, "errno: %d, P: %lx\n", errno, (long)P);
  deallocate(Action, P);

  // Should succeed when max_allocation_size_mb is set.
  P = allocate(Action, MaxAllocationSize);
  assert(P);
  deallocate(Action, P);

  return 0;
}

// CHECK-mCRASH: malloc:
// CHECK-mCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}
// CHECK-cCRASH: calloc:
// CHECK-cCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}
// CHECK-rCRASH: realloc:
// CHECK-rCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}
// CHECK-mrCRASH: realloc-after-malloc:
// CHECK-mrCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}
// CHECK-nCRASH: new:
// CHECK-nCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}
// CHECK-nCRASH-OOM: new:
// CHECK-nCRASH-OOM: {{SUMMARY: .*Sanitizer: out-of-memory}}
// CHECK-nnCRASH: new-nothrow:
// CHECK-nnCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}
// CHECK-sCRASH: strndup:
// CHECK-sCRASH: {{SUMMARY: .*Sanitizer: allocation-size-too-big}}

// CHECK-NULL: {{malloc|calloc|calloc-overflow|realloc|realloc-after-malloc|new-nothrow|strndup}}
// CHECK-NULL: errno: 12, P: 0
//
// CHECK-NOTNULL-NOT: P: 0
