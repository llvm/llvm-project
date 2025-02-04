// Test that captures the current behavior of indicator-based ASan ODR checker
// when globals get unregistered.
//
// RUN: %clangxx_asan -g -O0 -DSHARED_LIB -DSIZE=1 %s -fPIC -shared -o %t-so-1.so
// RUN: %clangxx_asan -g -O0 -DSHARED_LIB -DSIZE=2 %s -fPIC -shared -o %t-so-2.so
// RUN: %clangxx_asan -g -O0 %s %libdl -Wl,--export-dynamic -o %t
// RUN: %env_asan_opts=report_globals=2:detect_odr_violation=1 %run %t 2>&1 | FileCheck %s

// FIXME: Checks do not match on Android.
// UNSUPPORTED: android

#include <cstdlib>
#include <dlfcn.h>
#include <stdio.h>
#include <string>

#ifdef SHARED_LIB
namespace foo {
char G[SIZE];
}
#else // SHARED_LIB
void *dlopen_or_die(std::string &path) {
  void *handle = dlopen(path.c_str(), RTLD_NOW);
  if (handle) {
    printf("Successfully called dlopen() on %s\n", path.c_str());
  } else {
    printf("Error in dlopen(): %s\n", dlerror());
    std::exit(1);
  }

  return handle;
}

void dlclose_or_die(void *handle, std::string &path) {
  if (!dlclose(handle)) {
    printf("Successfully called dlclose() on %s\n", path.c_str());
  } else {
    printf("Error in dlclose(): %s\n", dlerror());
    std::exit(1);
  }
}

namespace foo {
char G[1];
}

// main has its own version of foo::G
// CHECK: Added Global[[MAIN_G:[^\s]+]] size=1/32 name=foo::G {{.*}}
int main(int argc, char *argv[]) {
  std::string base_path = std::string(argv[0]);

  std::string path1 = base_path + "-so-1.so";
  // dlopen() brings another foo::G but it matches MAIN_G in size so it's not a
  // violation
  //
  //
  // CHECK: Added Global[[SO1_G:[^\s]+]] size=1/32 name=foo::G {{.*}}
  // CHECK-NOT: ERROR: AddressSanitizer: odr-violation
  void *handle1 = dlopen_or_die(path1);
  // CHECK: Removed Global[[SO1_G]] size=1/32 name=foo::G {{.*}}
  dlclose_or_die(handle1, path1);

  // At this point the indicator for foo::G is switched to UNREGISTERED for
  // **both** MAIN_G and SO1_G because the indicator value is shared.

  std::string path2 = base_path + "-so-2.so";
  // CHECK: Added Global[[SO2_G:[^\s]+]] size=2/32 name=foo::G {{.*}}
  //
  // This brings another foo::G but now different in size from MAIN_G. We
  // should've reported a violation, but we actually don't because of what's
  // described on line60
  //
  // CHECK-NOT: ERROR: AddressSanitizer: odr-violation
  void *handle2 = dlopen_or_die(path2);
  // CHECK: Removed Global[[MAIN_G]] size=1/32 name=foo::G {{.*}}
  // CHECK: Removed Global[[SO2_G]] size=2/32 name=foo::G {{.*}}
}

#endif // SHARED_LIB
