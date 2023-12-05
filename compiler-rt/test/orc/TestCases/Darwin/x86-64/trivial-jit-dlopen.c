// Test that __orc_rt_macho_jit_dlopen and __orc_rt_macho_jit_dlclose work as
// expected for a straightforward dlopen; dlclose sequence: first the
// constructors should be run, then destructors.
//
// RUN: %clang -c -o %t.inits.o %p/Inputs/standalone-ctor-and-cxa-atexit-dtor.S
// RUN: %clang -c -o %t.test.o %s
// RUN: %llvm_jitlink \
// RUN:   -alias _dlopen=___orc_rt_macho_jit_dlopen \
// RUN:   -alias _dlclose=___orc_rt_macho_jit_dlclose \
// RUN:   %t.test.o -jd inits %t.inits.o -lmain | FileCheck %s

// CHECK: entering main
// CHECK-NEXT: constructor
// CHECK-NEXT: destructor
// CHECK-NEXT: leaving main

int printf(const char * restrict format, ...);
void *dlopen(const char* path, int mode);
int dlclose(void *handle);

int main(int argc, char *argv[]) {
  printf("entering main\n");
  void *H = dlopen("inits", 0);
  if (!H) {
    printf("failed\n");
    return -1;
  }
  if (dlclose(H) == -1) {
    printf("failed\n");
    return -1;
  }
  printf("leaving main\n");
  return 0;
}
