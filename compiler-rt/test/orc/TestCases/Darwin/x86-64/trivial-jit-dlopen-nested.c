// Test that __orc_rt_macho_jit_dlopen and __orc_rt_macho_jit_dlclose work as
// expected for a dlopen; dlclose; dlopen; dlclose; sequence: the first dlclose
// should run destructors, and the second dlopen should re-run initializers.
//
// RUN: %clang -c -o %t.inits.o %p/Inputs/standalone-ctor-and-cxa-atexit-dtor.S
// RUN: %clang -c -o %t.test.o %s
// RUN: %llvm_jitlink \
// RUN:   -alias Platform:_dlopen=___orc_rt_macho_jit_dlopen \
// RUN:   -alias Platform:_dlclose=___orc_rt_macho_jit_dlclose \
// RUN:   %t.test.o -jd inits %t.inits.o -lmain | FileCheck %s

// CHECK: entering main
// CHECK-NEXT: first dlopen
// CHECK-NEXT: constructor
// CHECK-NEXT: second dlopen
// CHECK-NEXT: first dlclose
// CHECK-NEXT: second dlclose
// CHECK-NEXT: destructor
// CHECK-NEXT: leaving main

int printf(const char * restrict format, ...);
void *dlopen(const char* path, int mode);
int dlclose(void *handle);

int main(int argc, char *argv[]) {
  printf("entering main\n");
  printf("first dlopen\n");
  void *H = dlopen("inits", 0);
  if (!H) {
    printf("failed\n");
    return -1;
  }
  printf("second dlopen\n");
  void *I = dlopen("inits", 0);
  if (!I) {
    printf("failed\n");
    return -1;
  }
  if (I != H) {
    printf("handles do not match\n");
    return -1;
  }
  printf("first dlclose\n");
  if (dlclose(I) == -1) {
    printf("failed\n");
    return -1;
  }
  printf("second dlclose\n");
  if (dlclose(H) == -1) {
    printf("failed\n");
    return -1;
  }
  printf("leaving main\n");
  return 0;
}
