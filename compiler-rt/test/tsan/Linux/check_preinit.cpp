// RUN: %clang_tsan -fno-sanitize=thread -shared -fPIC -O1 -DBUILD_SO=1 %s -o \
// RUN:  %t.so && \
// RUN:   %clang_tsan -O1 %s %t.so -o %t && %run %t 2>&1 | FileCheck %s
// RUN: llvm-objdump -t %t | FileCheck %s --check-prefix=CHECK-DUMP
// CHECK-DUMP:  {{[.]preinit_array.*__local_tsan_preinit}}

// SANITIZER_CAN_USE_PREINIT_ARRAY is undefined on android.
// UNSUPPORTED: android

// Test checks if __tsan_init is called from .preinit_array.
// Without initialization from .preinit_array, __tsan_init will be called from
// constructors of the binary which are called after constructors of shared
// library.

#include <sanitizer/tsan_interface.h>
#include <stdio.h>

#if BUILD_SO

// "volatile" is needed to avoid compiler optimize-out constructors.
volatile int counter = 0;
volatile int lib_constructor_call = 0;
volatile int tsan_init_call = 0;

__attribute__ ((constructor))
void LibConstructor() {
  lib_constructor_call = ++counter;
};

#else  // BUILD_SO

extern int counter;
extern int lib_constructor_call;
extern int tsan_init_call;

volatile int bin_constructor_call = 0;

__attribute__ ((constructor))
void BinConstructor() {
  bin_constructor_call = ++counter;
};

namespace __tsan {

void OnInitialize() {
  tsan_init_call = ++counter;
}

}

int main() {
  // CHECK: TSAN_INIT 1
  // CHECK: LIB_CONSTRUCTOR 2
  // CHECK: BIN_CONSTRUCTOR 3
  printf("TSAN_INIT %d\n", tsan_init_call);
  printf("LIB_CONSTRUCTOR %d\n", lib_constructor_call);
  printf("BIN_CONSTRUCTOR %d\n", bin_constructor_call);
  return 0;
}

#endif  // BUILD_SO
