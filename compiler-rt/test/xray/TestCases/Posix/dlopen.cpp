// Check that we can patch and un-patch DSOs loaded with dlopen.
//

// RUN: split-file %s %t
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -shared -std=c++11 %t/testlib.cpp -o %t/testlib.so
// RUN: %clangxx_xray -g -fPIC -rdynamic -fxray-instrument -fxray-shared -std=c++11 %t/main.cpp -o %t/main.o
//
// RUN: XRAY_OPTIONS="patch_premain=true" %run %t/main.o %t/testlib.so 2>&1 | FileCheck %s

// REQUIRES: target={{(aarch64|x86_64)-.*}}

//--- main.cpp

#include "xray/xray_interface.h"

#include <cstdio>
#include <dlfcn.h>

void test_handler(int32_t fid, XRayEntryType type) {
  printf("called: %d, type=%d\n", fid, static_cast<int32_t>(type));
}

[[clang::xray_always_instrument]] void instrumented_in_executable() {
  printf("instrumented_in_executable called\n");
}

typedef void (*dso_func_type)();

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Shared library argument missing\n");
    // CHECK-NOT: Shared library argument missing
    return 1;
  }

  const char *dso_path = argv[1];

  void *dso_handle = dlopen(dso_path, RTLD_LAZY);
  if (!dso_handle) {
    printf("Failed to load shared library\n");
    char *error = dlerror();
    if (error) {
      fprintf(stderr, "%s\n", error);
      return 1;
    }
    return 1;
  }

  dso_func_type instrumented_in_dso =
      (dso_func_type)dlsym(dso_handle, "_Z19instrumented_in_dsov");
  if (!instrumented_in_dso) {
    printf("Failed to find symbol\n");
    char *error = dlerror();
    if (error) {
      fprintf(stderr, "%s\n", error);
      return 1;
    }
    return 1;
  }

  __xray_set_handler(test_handler);

  instrumented_in_executable();
  // CHECK: called: {{.*}}, type=0
  // CHECK-NEXT: instrumented_in_executable called
  // CHECK-NEXT: called: {{.*}}, type=1
  instrumented_in_dso();
  // CHECK-NEXT: called: {{.*}}, type=0
  // CHECK-NEXT: instrumented_in_dso called
  // CHECK-NEXT: called: {{.*}}, type=1

  auto status = __xray_unpatch();
  printf("unpatching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: unpatching status: 1

  instrumented_in_executable();
  // CHECK-NEXT: instrumented_in_executable called
  instrumented_in_dso();
  // CHECK-NEXT: instrumented_in_dso called

  status = __xray_patch();
  printf("patching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: patching status: 1

  instrumented_in_executable();
  // CHECK-NEXT: called: {{.*}}, type=0
  // CHECK-NEXT: instrumented_in_executable called
  // CHECK-NEXT: called: {{.*}}, type=1
  instrumented_in_dso();
  // CHECK-NEXT: called: {{.*}}, type=0
  // CHECK-NEXT: instrumented_in_dso called
  // CHECK-NEXT: called: {{.*}}, type=1

  dlclose(dso_handle);

  status = __xray_unpatch();
  printf("unpatching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: unpatching status: 1
}

//--- testlib.cpp

#include <cstdio>

[[clang::xray_always_instrument]] void instrumented_in_dso() {
  printf("instrumented_in_dso called\n");
}
