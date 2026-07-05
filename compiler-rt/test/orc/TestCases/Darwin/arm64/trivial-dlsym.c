// Test that __orc_rt_macho_jit_dlsym works as expected.
//
// RUN: %clang -c -o %t.sym.o %p/Inputs/ret_self.S
// RUN: %clang -c -o %t.test.o %s
// RUN: %llvm_jitlink \
// RUN:   -alias Platform:_dlopen=___orc_rt_macho_jit_dlopen \
// RUN:   -alias Platform:_dlsym=___orc_rt_macho_jit_dlsym \
// RUN:   -alias Platform:_dlclose=___orc_rt_macho_jit_dlclose \
// RUN:   %t.test.o -lextra_sym -jd extra_sym %t.sym.o | FileCheck %s

// CHECK: entering main
// CHECK-NEXT: found "ret_self" at
// CHECK-NEXT: address of "ret_self" is consistent
// CHECK-NEXT: leaving main

int printf(const char *restrict format, ...);
void *dlopen(const char *path, int mode);
void *dlsym(void *handle, const char *symbol);
int dlclose(void *handle);

int main(int argc, char *argv[]) {
  printf("entering main\n");
  void *H = dlopen("extra_sym", 0);
  if (!H) {
    printf("failed\n");
    return -1;
  }

  void *(*ret_self)(void) = (void *(*)(void))dlsym(H, "ret_self");
  if (ret_self)
    printf("found \"ret_self\" at %p\n", ret_self);
  else
    printf("failed to find \"ret_self\" via dlsym\n");

  printf("address of \"ret_self\" is %s\n",
         ret_self() == ret_self ? "consistent" : "inconsistent");

  if (dlclose(H) == -1) {
    printf("failed\n");
    return -1;
  }
  printf("leaving main\n");
  return 0;
}
