// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -DBUILD_HSA -fPIC -shared %s -o %t/libhsa-runtime64.so.1 \
// RUN:   -Wl,-soname,libhsa-runtime64.so.1
// RUN: %clang_dasan -DBUILD_DSO -shared-libsan -fPIC -shared %s \
// RUN:   -o %t/libdso.so %t/libhsa-runtime64.so.1 -Wl,-rpath,%t
// RUN: %clang %s -o %t/early %t/libdso.so -Wl,-rpath,%t
// RUN: %run %t/early 2>&1 | FileCheck %s --check-prefix=EARLY
// RUN: %clang %s -o %t/late %t/libhsa-runtime64.so.1 %t/libdso.so \
// RUN:   -Wl,-rpath,%t
// RUN: %run %t/late 2>&1 | FileCheck %s --check-prefix=LATE

// The shared runtime must warn if HSA is loaded ahead of it, since the
// interceptors are then bypassed.

#include <stdio.h>

#if defined(BUILD_HSA)
int hsa_init(void) { return 42; }
#elif defined(BUILD_DSO)
int dso(void) { return 0; }
#else
int dso(void);
int main(void) {
  fprintf(stderr, "DONE\n");
  return dso();
}
#endif

// EARLY-NOT: WARNING
// EARLY: DONE

// LATE: WARNING: DeviceAddressSanitizer: the runtime is loaded too late
// LATE: DONE
