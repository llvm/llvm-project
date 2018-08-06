// Test for ASAN_OPTIONS=start_deactivated=1 mode.
// Main executable is uninstrumented, but linked to ASan runtime. The shared
// library is instrumented. Memory errors before dlopen are not detected.

// RUN: %clangxx_asan -O0 -DSHARED_LIB %s -std=c++11 -fPIC -shared -o %t-so.so
// RUN: %clangxx -O0 %s -std=c++11 -c -o %t.o
// RUN: %clangxx_asan -O0 %t.o %libdl -o %t
// RUN: %env_asan_opts=start_deactivated=1,allocator_may_return_null=0 \
// RUN:   ASAN_ACTIVATION_OPTIONS=allocator_may_return_null=1 not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=start_deactivated=1 \
// RUN:   ASAN_ACTIVATION_OPTIONS=help=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-HELP
// RUN: %env_asan_opts=start_deactivated=1,verbosity=1 \
// RUN:   ASAN_ACTIVATION_OPTIONS=help=1,handle_segv=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-UNSUPPORTED
// RUN: %env_asan_opts=start_deactivated=1 \
// RUN:   ASAN_ACTIVATION_OPTIONS=help=1,handle_segv=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-UNSUPPORTED-V0

// Check that verbosity=1 in activation flags affects reporting of unrecognized activation flags.
// RUN: %env_asan_opts=start_deactivated=1 \
// RUN:   ASAN_ACTIVATION_OPTIONS=help=1,handle_segv=0,verbosity=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-UNSUPPORTED

// UNSUPPORTED: ios

// END.

#if !defined(SHARED_LIB)

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <string>

#include "sanitizer/asan_interface.h"

void test_malloc_shadow(char *p, size_t sz, bool expect_redzones) {
  // Last byte of the left redzone, if present.
  assert((char *)__asan_region_is_poisoned(p - 1, sz + 1) ==
         (expect_redzones ? p - 1 : nullptr));
  // The user memory.
  assert((char *)__asan_region_is_poisoned(p, sz) == nullptr);
  // First byte of the right redzone, if present.
  assert((char *)__asan_region_is_poisoned(p, sz + 1) ==
         (expect_redzones ? p + sz : nullptr));
}

typedef void (*Fn)();

int main(int argc, char *argv[]) {
  constexpr unsigned nPtrs = 200;
  char *ptrs[nPtrs];

  // Before activation: no redzones.
  for (size_t sz = 1; sz < nPtrs; ++sz) {
    ptrs[sz] = (char *)malloc(sz);
    test_malloc_shadow(ptrs[sz], sz, false);
  }

  // Create a honey pot for the future, instrumented, allocations. Since the
  // quarantine is disabled, chunks are going to be recycled right away and
  // reused for the new allocations. New allocations must get the proper
  // redzones anyway, whether it's a fresh or reused allocation.
  constexpr size_t HoneyPotBlockSize = 4096;
  constexpr int HoneyPotSize = 200;
  char *honeyPot[HoneyPotSize];
  for (int i = 1; i < HoneyPotSize; ++i) {
    honeyPot[i] = (char *)malloc(HoneyPotBlockSize);
    test_malloc_shadow(honeyPot[i], HoneyPotBlockSize, false);
  }
  for (int i = 1; i < HoneyPotSize; ++i)
    free(honeyPot[i]);

  std::string path = std::string(argv[0]) + "-so.so";
  void *dso = dlopen(path.c_str(), RTLD_NOW);
  if (!dso) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }

  // After this line ASan is activated and starts detecting errors.
  void *fn = dlsym(dso, "do_another_bad_thing");
  if (!fn) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    return 1;
  }

  // After activation: redzones.
  for (int i = 1; i < HoneyPotSize; ++i) {
    honeyPot[i] = (char *)malloc(HoneyPotBlockSize);
    test_malloc_shadow(honeyPot[i], HoneyPotBlockSize, true);
  }
  {
    char *p = (char *)malloc(HoneyPotBlockSize);
    test_malloc_shadow(p, HoneyPotBlockSize, true);
    free(p);
  }
  for (int i = 1; i < HoneyPotSize; ++i)
    free(honeyPot[i]);

  // Pre-existing allocations got redzones, too.
  for (size_t sz = 1; sz < nPtrs; ++sz) {
    test_malloc_shadow(ptrs[sz], sz, true);
    free(ptrs[sz]);
  }

  // Test that ASAN_ACTIVATION_OPTIONS=allocator_may_return_null=1 has effect.
  void *p = malloc((unsigned long)-2);
  assert(!p);
  // CHECK: WARNING: AddressSanitizer failed to allocate 0xfff{{.*}} bytes

  ((Fn)fn)();
  // CHECK: AddressSanitizer: heap-buffer-overflow
  // CHECK: READ of size 1
  // CHECK: {{#0 .* in do_another_bad_thing}}
  // CHECK: is located 5 bytes to the right of 100-byte region
  // CHECK: in do_another_bad_thing

  return 0;
}

#else  // SHARED_LIB

#include <stdio.h>
#include <stdlib.h>

extern "C" void do_another_bad_thing() {
  char *volatile p = (char *)malloc(100);
  printf("%hhx\n", p[105]);
}

#endif  // SHARED_LIB

// help=1 in activation flags lists only flags are are supported at activation
// CHECK-HELP: Available flags for {{.*}}Sanitizer:
// CHECK-HELP-NOT: handle_segv
// CHECK-HELP: max_redzone
// CHECK-HELP-NOT: handle_segv

// unsupported activation flags produce a warning ...
// CHECK-UNSUPPORTED: WARNING: found 1 unrecognized
// CHECK-UNSUPPORTED:   handle_segv

// ... but not at verbosity=0
// CHECK-UNSUPPORTED-V0-NOT: WARNING: found {{.*}} unrecognized
