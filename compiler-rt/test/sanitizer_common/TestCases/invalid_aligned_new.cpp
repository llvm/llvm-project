// Invalid alignment is a caller error, not a failure to obtain storage. The
// sanitizer must report it, even with allocator_may_return_null=1, rather than
// return nullptr or call std::new_handler.

// RUN: %clangxx -O0 -std=c++17 %s -o %t
// RUN: %env_tool_opts=allocator_may_return_null=1 not %run %t new 0 2>&1 | FileCheck %s -DALIGN=0
// RUN: %env_tool_opts=allocator_may_return_null=1 not %run %t new-array 0 2>&1 | FileCheck %s -DALIGN=0
// RUN: %env_tool_opts=allocator_may_return_null=1 not %run %t new-nothrow 0 2>&1 | FileCheck %s -DALIGN=0
// RUN: %env_tool_opts=allocator_may_return_null=1 not %run %t new-array-nothrow 0 2>&1 | FileCheck %s -DALIGN=0
// RUN: %env_tool_opts=allocator_may_return_null=1 not %run %t new 3 2>&1 | FileCheck %s -DALIGN=3
// RUN: %env_tool_opts=allocator_may_return_null=1 not %run %t new-array 3 2>&1 | FileCheck %s -DALIGN=3
// RUN: %env_tool_opts=allocator_may_return_null=1 not %run %t new-nothrow 3 2>&1 | FileCheck %s -DALIGN=3
// RUN: %env_tool_opts=allocator_may_return_null=1 not %run %t new-array-nothrow 3 2>&1 | FileCheck %s -DALIGN=3

// ubsan does not replace operator new. ASan and LSan do not override aligned
// operator new on Darwin, nor ASan on Windows (MSVC).
// UNSUPPORTED: ubsan, (asan || lsan) && darwin, target={{.*windows-msvc.*}}
// REQUIRES: stable-runtime

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

static void NewHandler() {
  std::fputs("new_handler called\n", stderr);
  std::abort();
}

int main(int argc, char **argv) {
  assert(argc == 3);
  std::set_new_handler(NewHandler);
  const auto alignment =
      static_cast<std::align_val_t>(std::strtoull(argv[2], nullptr, 0));

  void *p;
  if (std::strcmp(argv[1], "new") == 0)
    p = ::operator new(16, alignment);
  else if (std::strcmp(argv[1], "new-array") == 0)
    p = ::operator new[](16, alignment);
  else if (std::strcmp(argv[1], "new-nothrow") == 0)
    p = ::operator new(16, alignment, std::nothrow);
  else if (std::strcmp(argv[1], "new-array-nothrow") == 0)
    p = ::operator new[](16, alignment, std::nothrow);
  else
    assert(false);

  std::fprintf(stderr, "allocation unexpectedly returned %p\n", p);
  return 1;
}

// CHECK-NOT: new_handler called
// CHECK: ERROR: {{.*}}Sanitizer: invalid allocation alignment: [[ALIGN]],
// CHECK: SUMMARY: {{.*}}Sanitizer: invalid-allocation-alignment
