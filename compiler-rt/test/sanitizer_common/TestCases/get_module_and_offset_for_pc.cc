// RUN: mkdir -p %t-dir
// RUN: %clangxx -DSHARED %s -shared -o %t-dir/get_module_and_offset_for_pc.so -fPIC
// RUN: %clangxx -DSO_DIR=\"%t-dir\" -O0 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: i386-darwin
// XFAIL: android

// Tests __sanitizer_get_module_and_offset_for_pc.

#include <assert.h>
#include <dlfcn.h>
#include <sanitizer/common_interface_defs.h>
#include <stdio.h>

#ifdef SHARED
extern "C" {
int foo() { return 1; }
}
#else

void Test(void *pc, const char *name) {
  char module_name[1024];
  void *offset;
  int ok = __sanitizer_get_module_and_offset_for_pc(
      pc, module_name, sizeof(module_name), &offset);
  if (!ok) {
    printf("NOT FOUND %s: %p\n", name, pc);
  } else {
    printf("FOUND %s: %s %p\n", name, module_name, offset);
  }
}

void TestCallerPc() { Test(__builtin_return_address(0), "callerpc"); }

void TestDlsym() {
  void *handle = dlopen(SO_DIR "/get_module_and_offset_for_pc.so", RTLD_LAZY);
  assert(handle);
  void *foo = dlsym(handle, "foo");
  assert(foo);
  Test(foo, "foo");
  dlclose(handle);
}

// Call __sanitizer_get_module_and_offset_for_pc lots of times
// to make sure it is not too slow.
void TestLoop() {
  void *pc = __builtin_return_address(0);
  char module_name[1024];
  void *offset;
  for (int i = 0; i < 1000000; ++i) {
    __sanitizer_get_module_and_offset_for_pc(pc, module_name,
                                             sizeof(module_name), &offset);
  }
}

int main() {
  Test(0, "null");
  TestCallerPc();
  TestDlsym();
  TestLoop();
}
#endif
// CHECK: NOT FOUND null: {{.*}}
// CHECK-NEXT: FOUND callerpc: {{.*}}/get_module_and_offset_for_pc.cc.tmp {{.*}}
// CHECK-NEXT: FOUND foo: {{.*}}/get_module_and_offset_for_pc.so {{.*}}
