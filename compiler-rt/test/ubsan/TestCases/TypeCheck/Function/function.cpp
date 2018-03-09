// RUN: %clangxx -fsanitize=function %s -O3 -g -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// Verify that we can disable symbolization if needed:
// RUN: %env_ubsan_opts=symbolize=0 %run %t 2>&1 | FileCheck %s --check-prefix=NOSYM
// XFAIL: win32,win64

#include <stdint.h>

void f() {}

void g(int x) {}

void make_valid_call() {
  // CHECK-NOT: runtime error: call to function g
  reinterpret_cast<void (*)(int)>(reinterpret_cast<uintptr_t>(g))(42);
}

void make_invalid_call() {
  // CHECK: function.cpp:[[@LINE+4]]:3: runtime error: call to function f() through pointer to incorrect function type 'void (*)(int)'
  // CHECK-NEXT: function.cpp:[[@LINE-11]]: note: f() defined here
  // NOSYM: function.cpp:[[@LINE+2]]:3: runtime error: call to function (unknown) through pointer to incorrect function type 'void (*)(int)'
  // NOSYM-NEXT: ({{.*}}+0x{{.*}}): note: (unknown) defined here
  reinterpret_cast<void (*)(int)>(reinterpret_cast<uintptr_t>(f))(42);
}

int main(void) {
  make_valid_call();
  make_invalid_call();
  // Check that no more errors will be printed.
  // CHECK-NOT: runtime error: call to function
  // NOSYM-NOT: runtime error: call to function
  make_invalid_call();
}
