// Invalid alignment is a caller error, not a failure to obtain storage. DFSan
// must report it, even with allocator_may_return_null=1, rather than return
// nullptr. DFSan's operator new overrides serve uninstrumented code, so the
// allocation happens in an uninstrumented object linked into the program.

// RUN: %clangxx_dfsan -fno-sanitize=dataflow -std=c++17 -DLIB -c %s -o %t-lib.o
// RUN: echo 'fun:AllocAligned=uninstrumented' > %t.abilist
// RUN: echo 'fun:AllocAligned=discard' >> %t.abilist
// RUN: %clangxx_dfsan -std=c++17 -fsanitize-ignorelist=%t.abilist %s %t-lib.o -o %t
// RUN: env DFSAN_OPTIONS=allocator_may_return_null=1 not %run %t new 0 2>&1 | FileCheck %s -DALIGN=0
// RUN: env DFSAN_OPTIONS=allocator_may_return_null=1 not %run %t new-nothrow 0 2>&1 | FileCheck %s -DALIGN=0
// RUN: env DFSAN_OPTIONS=allocator_may_return_null=1 not %run %t new 3 2>&1 | FileCheck %s -DALIGN=3
// RUN: env DFSAN_OPTIONS=allocator_may_return_null=1 not %run %t new-nothrow 3 2>&1 | FileCheck %s -DALIGN=3

#include <cstddef>

extern "C" void *AllocAligned(bool nothrow, std::size_t alignment);

#ifdef LIB
#  include <new>

extern "C" void *AllocAligned(bool nothrow, std::size_t alignment) {
  const auto align = static_cast<std::align_val_t>(alignment);
  return nothrow ? ::operator new(16, align, std::nothrow)
                 : ::operator new(16, align);
}
#else
#  include <cassert>
#  include <cstdio>
#  include <cstdlib>
#  include <cstring>

int main(int argc, char **argv) {
  assert(argc == 3);
  const bool nothrow = std::strcmp(argv[1], "new-nothrow") == 0;
  assert(nothrow || std::strcmp(argv[1], "new") == 0);
  void *p = AllocAligned(nothrow, std::strtoull(argv[2], nullptr, 0));
  std::fprintf(stderr, "allocation unexpectedly returned %p\n", p);
  return 1;
}
#endif

// CHECK: ERROR: DataflowSanitizer: invalid allocation alignment: [[ALIGN]],
