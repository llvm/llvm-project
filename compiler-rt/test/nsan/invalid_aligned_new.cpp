// Invalid alignment is a caller error, not a failure to obtain storage. NSan
// must report it, even with allocator_may_return_null=1, rather than return
// nullptr.

// RUN: %clangxx_nsan -O0 %s -o %t
// RUN: env NSAN_OPTIONS=allocator_may_return_null=1 not %run %t new 0 2>&1 | FileCheck %s -DALIGN=0
// RUN: env NSAN_OPTIONS=allocator_may_return_null=1 not %run %t new-nothrow 0 2>&1 | FileCheck %s -DALIGN=0
// RUN: env NSAN_OPTIONS=allocator_may_return_null=1 not %run %t new 3 2>&1 | FileCheck %s -DALIGN=3
// RUN: env NSAN_OPTIONS=allocator_may_return_null=1 not %run %t new-nothrow 3 2>&1 | FileCheck %s -DALIGN=3

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

int main(int argc, char **argv) {
  assert(argc == 3);
  const auto alignment =
      static_cast<std::align_val_t>(std::strtoull(argv[2], nullptr, 0));

  void *p;
  if (std::strcmp(argv[1], "new") == 0)
    p = ::operator new(16, alignment);
  else if (std::strcmp(argv[1], "new-nothrow") == 0)
    p = ::operator new(16, alignment, std::nothrow);
  else
    assert(false);

  std::fprintf(stderr, "allocation unexpectedly returned %p\n", p);
  return 1;
}

// CHECK: ERROR: NumericalStabilitySanitizer: invalid allocation alignment: [[ALIGN]],
