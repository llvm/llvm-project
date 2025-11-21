// REQUIRES: system-linux || system-darwin
// RUN: %clang_tysan %s -g -shared -fpic -o %t.so -DBUILD_SO
// RUN: %clang_tysan %s -g -o %t
// RUN: %run %t %t.so 2>&1 | FileCheck %s

// Compilers can't optimize using type aliasing across the bounds of dynamic librarys
// When passing memory between instrumented executables and dlls, you may want to alter TySan's
// shadow to prevent it from catching technically correct, yet harmless aliasing violations

#ifdef BUILD_SO
float useFloatArray(float *mem) {
  mem[0] = 2.f;
  mem[1] = 3.f;
  return mem[0] + mem[1];
}

int useIntArray(int *mem) {
  mem[0] = 2;
  mem[1] = 3;
  mem[2] = 5;
  return mem[0] + mem[1] + mem[2];
}
#else

#  include <assert.h>
#  include <dlfcn.h>
#  include <sanitizer/tysan_interface.h>
#  include <stdio.h>

typedef float (*lib_func1_t)(float *);
typedef int (*lib_func2_t)(int *);

void print_flush(const char *message) {
  printf("%s\n", message);
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  assert(argc >= 2);
  void *libHandle = dlopen(argv[1], RTLD_LAZY);
  assert(libHandle);

  lib_func1_t useFloatArray = (lib_func1_t)dlsym(libHandle, "useFloatArray");
  lib_func2_t useIntArray = (lib_func2_t)dlsym(libHandle, "useIntArray");

  char memory[sizeof(int) * 3];
  int iResult = 0;
  float fResult = 0.f;
  print_flush("Calling with omnipotent char memory");
  fResult = useFloatArray((float *)memory);
  print_flush("Shadow now has floats in");
  iResult = useIntArray((int *)memory);

  // CHECK: Calling with omnipotent char memory
  // CHECK-NEXT: Shadow now has floats in
  // CHECK-NEXT: ERROR: TypeSanitizer: type-aliasing-violation on address
  // CHECK-NEXT: WRITE of size 4 at 0x{{.*}} with type int accesses an existing object of type float

  __tysan_reset_shadow(memory, sizeof(memory));
  print_flush("Shadow has been reset");
  useIntArray((int *)memory);
  print_flush("Completed int array");

  // CHECK: Shadow has been reset
  // CHECK-NEXT: Completed int array

  // Set shadow type to float
  __tysan_copy_shadow_array(memory, &fResult, sizeof(float), 3);
  print_flush("Float array with float set shadow");
  useFloatArray((float *)memory);
  print_flush("Int array with float set shadow");
  useIntArray((int *)memory);

  // CHECK: Float array with float set shadow
  // CHECK-NEXT: Int array with float set shadow
  // CHECK-NEXT: ERROR: TypeSanitizer: type-aliasing-violation on address
  // CHECK-NEXT: WRITE of size 4 at 0x{{.*}} with type int accesses an existing object of type float

  // Set shadow type to int
  for (size_t i = 0; i < 3; i++) {
    __tysan_copy_shadow(&memory[sizeof(int) * i], &iResult, sizeof(int));
  }
  print_flush("Float array with int set shadow");
  useFloatArray((float *)memory);
  print_flush("Int array with int set shadow");
  useIntArray((int *)memory);
  print_flush("Completed int array");

  // CHECK: Float array with int set shadow
  // CHECK-NEXT: ERROR: TypeSanitizer: type-aliasing-violation on address
  // CHECK-NEXT: WRITE of size 4 at 0x{{.*}} with type float accesses an existing object of type int
  // CHECK: Int array with int set shadow
  // CHECK-NEXT: Completed int array

  dlclose(libHandle);
  return 0;
}

#endif
