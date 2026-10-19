// RUN: rm -rf %t && mkdir -p %t
// RUN: %clangxx -DBUILD_HSA -fPIC -shared %s -o %t/libhsa-runtime64.so.1 \
// RUN:   -Wl,-soname,libhsa-runtime64.so.1
// RUN: %clangxx -DBUILD_DSO -fsanitize=undefined -shared-libsan -fPIC -shared \
// RUN:   %s -o %t/libdso.so %t/libhsa-runtime64.so.1 -Wl,-rpath,%t
// RUN: %clangxx %s -o %t/early %t/libdso.so -Wl,-rpath,%t
// RUN: %run %t/early 2>&1 | FileCheck %s --check-prefix=EARLY
// RUN: %clangxx %s -o %t/late %t/libhsa-runtime64.so.1 %t/libdso.so \
// RUN:   -Wl,-rpath,%t
// RUN: %run %t/late 2>&1 | FileCheck %s --check-prefix=LATE

// The shared runtime must warn if HSA is loaded ahead of it, since the
// interceptors are then bypassed.

// REQUIRES: target={{.*linux.*}}, ubsan-standalone, ubsan-offload

#include <stdio.h>

#if defined(BUILD_HSA)
extern "C" int hsa_init() { return 42; }
#elif defined(BUILD_DSO)
extern "C" int dso() { return 0; }
#else
extern "C" int dso();
int main() {
  fprintf(stderr, "DONE\n");
  return dso();
}
#endif

// EARLY-NOT: WARNING
// EARLY: DONE

// LATE: WARNING: UndefinedBehaviorSanitizer: the runtime is loaded too late
// LATE: DONE
