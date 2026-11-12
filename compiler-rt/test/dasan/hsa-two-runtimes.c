// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -DBUILD_HSA -fPIC -shared %s -o %t/libhsa-runtime64.so.1 \
// RUN:   -Wl,-soname,libhsa-runtime64.so.1
// RUN: %clang_dasan -DBUILD_DSO -shared-libsan -fPIC -shared %s \
// RUN:   -o %t/libhip.so %t/libhsa-runtime64.so.1 \
// RUN:   -Wl,--disable-new-dtags -Wl,-rpath,%t
// The executable's static runtime and the shared runtime loaded by the
// sanitized library are the mix ASan rejects.
// RUN: %clang_dasan %s -o %t/two %t/libhip.so \
// RUN:   -Wl,--disable-new-dtags -Wl,-rpath,%t
// RUN: not %run %t/two 2>&1 | FileCheck %s --check-prefix=MIX
// One shared runtime is the supported link.
// RUN: %clang -DSHARED %s -o %t/shared %t/libhip.so \
// RUN:   -Wl,--disable-new-dtags -Wl,-rpath,%t
// RUN: %run %t/shared 2>&1 | FileCheck %s --check-prefix=ONE

#include <stdio.h>

#if defined(BUILD_HSA)
static int Calls;
int hsa_init(void) {
  fprintf(stderr, "real %d\n", ++Calls);
  return 7;
}
#elif defined(BUILD_DSO)
int hip_touch(void) { return 0; }
#elif defined(SHARED)
#  include <dlfcn.h>
int hip_touch(void);
int main(void) {
  hip_touch();
  int (*Init)(void) = (int (*)(void))dlsym(RTLD_DEFAULT, "hsa_init");
  if (!Init)
    return 1;
  fprintf(stderr, "status %d\n", Init());
  return 0;
}
#else
int hip_touch(void);
int hsa_init(void);
int main(void) {
  hip_touch();
  fprintf(stderr, "status %d\n", hsa_init());
  return 0;
}
#endif

// MIX: ERROR: DeviceAddressSanitizer: incompatible runtimes
// MIX-NOT: real

// ONE: real 1
// ONE-NOT: real 2
// ONE: status 7
