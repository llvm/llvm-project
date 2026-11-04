// Check that loading libraries with different modes (RTLD_LOCAL/RTLD_GLOBAL)
// and dependencies on other DSOs work correctly.
//

// RUN: split-file %s %t
//
// Build shared libs with dependencies b->c and e->f
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -shared -std=c++11 %t/testliba.cpp -o %t/testliba.so
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -shared -std=c++11 %t/testlibc.cpp -o %t/testlibc.so
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -shared -std=c++11 %t/testlibb.cpp %t/testlibc.so -o %t/testlibb.so
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -shared -std=c++11 %t/testlibd.cpp -o %t/testlibd.so
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -shared -std=c++11 %t/testlibf.cpp -o %t/testlibf.so
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -shared -std=c++11 %t/testlibe.cpp %t/testlibf.so -o %t/testlibe.so
//
// Executable links with a and b explicitly and loads d and e at runtime.
// RUN: %clangxx_xray -g -fPIC -rdynamic -fxray-instrument -fxray-shared -std=c++11 %t/main.cpp %t/testliba.so %t/testlibb.so -o %t/main.o
//
// RUN:  env XRAY_OPTIONS="patch_premain=true" %run %t/main.o %t/testlibd.so %t/testlibe.so 2>&1 | FileCheck %s

// REQUIRES: target={{(aarch64|x86_64)-.*}}

//--- main.cpp

#include "xray/xray_interface.h"

#include <cstdio>
#include <dlfcn.h>

[[clang::xray_never_instrument]] void test_handler(int32_t fid,
                                                   XRayEntryType type) {
  printf("called: %d, object=%d, fn=%d, type=%d\n", fid, (fid >> 24) & 0xFF,
         fid & 0x00FFFFFF, static_cast<int32_t>(type));
}

[[clang::xray_always_instrument]] void instrumented_in_executable() {
  printf("instrumented_in_executable called\n");
}

typedef void (*dso_func_type)();

[[clang::xray_never_instrument]] void *load_dso(const char *path, int mode) {
  void *dso_handle = dlopen(path, mode);
  if (!dso_handle) {
    printf("failed to load shared library\n");
    char *error = dlerror();
    if (error) {
      fprintf(stderr, "%s\n", error);
    }
    return nullptr;
  }
  return dso_handle;
}

[[clang::xray_never_instrument]] void find_and_call(void *dso_handle,
                                                    const char *fn) {
  dso_func_type dso_fn = (dso_func_type)dlsym(dso_handle, fn);
  if (!dso_fn) {
    printf("failed to find symbol\n");
    char *error = dlerror();
    if (error) {
      fprintf(stderr, "%s\n", error);
    }
    return;
  }
  dso_fn();
}

extern void a();
extern void b();

int main(int argc, char **argv) {

  if (argc < 3) {
    printf("Shared library arguments missing\n");
    // CHECK-NOT: Shared library arguments missing
    return 1;
  }

  const char *dso_path_d = argv[1];
  const char *dso_path_e = argv[2];

  __xray_set_handler(test_handler);

  instrumented_in_executable();
  // CHECK: called: {{[0-9]+}}, object=0, fn={{[0-9]+}}, type=0
  // CHECK-NEXT: instrumented_in_executable called
  // CHECK-NEXT: called: {{[0-9]+}}, object=0, fn={{[0-9]+}}, type=1

  a();
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ1:[0-9]+]], fn=1, type=0
  // CHECK-NEXT: a called
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ1]], fn=1, type=1

  // Make sure this object ID does not appear again
  // CHECK-NOT: called: {{[0-9]+}}, object=[[OBJ1]]

  b(); // b calls c
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ2:[0-9]+]], fn=1, type=0
  // CHECK-NEXT: b called
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ3:[0-9]+]], fn=1, type=0
  // CHECK-NEXT: c called
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ3]], fn=1, type=1
  // CHECK-NOT: called: {{[0-9]+}}, object=[[OBJ3]]
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ2]], fn=1, type=1
  // CHECK-NOT: called: {{[0-9]+}}, object=[[OBJ2]]

  // Now check explicit loading with RTLD_LOCAL

  void *dso_handle_d = load_dso(dso_path_d, RTLD_LAZY | RTLD_LOCAL);
  void *dso_handle_e = load_dso(dso_path_e, RTLD_LAZY | RTLD_LOCAL);
  // CHECK-NOT: failed to load shared library

  find_and_call(dso_handle_d, "_Z1dv");
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ4:[0-9]+]], fn=1, type=0
  // CHECK-NEXT: d called
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ4]], fn=1, type=1
  // CHECK-NOT: called: {{[0-9]+}}, object=[[OBJ4]]

  find_and_call(dso_handle_e, "_Z1ev");
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ5:[0-9]+]], fn=1, type=0
  // CHECK-NEXT: e called
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ6:[0-9]+]], fn=1, type=0
  // CHECK-NEXT: f called
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ6]], fn=1, type=1
  // CHECK-NOT: called: {{[0-9]+}}, object=[[OBJ6]]
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ5]], fn=1, type=1
  // CHECK-NOT: called: {{[0-9]+}}, object=[[OBJ5]]

  // Unload DSOs
  dlclose(dso_handle_d);
  dlclose(dso_handle_e);

  // Repeat test with RTLD_GLOBAL
  dso_handle_d = load_dso(dso_path_d, RTLD_LAZY | RTLD_GLOBAL);
  dso_handle_e = load_dso(dso_path_e, RTLD_LAZY | RTLD_GLOBAL);
  // CHECK-NOT: failed to load shared library

  find_and_call(dso_handle_d, "_Z1dv");
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ7:[0-9]+]], fn=1, type=0
  // CHECK-NEXT: d called
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ7]], fn=1, type=1
  // CHECK-NOT: called: {{[0-9]+}}, object=[[OBJ7]]

  find_and_call(dso_handle_e, "_Z1ev");
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ8:[0-9]+]], fn=1, type=0
  // CHECK-NEXT: e called
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ9:[0-9]+]], fn=1, type=0
  // CHECK-NEXT: f called
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ9]], fn=1, type=1
  // CHECK-NOT: called: {{[0-9]+}}, object=[[OBJ9]]
  // CHECK-NEXT: called: {{[0-9]+}}, object=[[OBJ8]], fn=1, type=1
  // CHECK-NOT: called: {{[0-9]+}}, object=[[OBJ8]]

  auto status = __xray_unpatch();
  printf("unpatching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: unpatching status: 1

  dlclose(dso_handle_d);
  dlclose(dso_handle_e);
}

//--- libgenmacro.inc
#include <cstdio>
// Helper macros to quickly generate libraries containing a single function.
#define GENERATE_LIB(NAME)                                                     \
  [[clang::xray_always_instrument]] void NAME() { printf(#NAME " called\n"); }

#define GENERATE_LIB_WITH_CALL(NAME, FN)                                       \
  extern void FN();                                                            \
  [[clang::xray_always_instrument]] void NAME() {                              \
    printf(#NAME " called\n");                                                 \
    FN();                                                                      \
  }

//--- testliba.cpp
#include "libgenmacro.inc"
GENERATE_LIB(a)

//--- testlibb.cpp
#include "libgenmacro.inc"
GENERATE_LIB_WITH_CALL(b, c)

//--- testlibc.cpp
#include "libgenmacro.inc"
GENERATE_LIB(c)

//--- testlibd.cpp
#include "libgenmacro.inc"
GENERATE_LIB(d)

//--- testlibe.cpp
#include "libgenmacro.inc"
GENERATE_LIB_WITH_CALL(e, f)

//--- testlibf.cpp
#include "libgenmacro.inc"
GENERATE_LIB(f)
