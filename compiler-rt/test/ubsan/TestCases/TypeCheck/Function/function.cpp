// RUN: %clangxx -std=c++17 -fsanitize=function %s -O3 -g -o %t
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

void f1(int) {}
void f2(unsigned int) {}
void f3(int) noexcept {}
void f4(unsigned int) noexcept {}

void check_noexcept_calls() {
  void (*p1)(int);
  p1 = &f1;
  p1(0);
  p1 = reinterpret_cast<void (*)(int)>(&f2);
  // CHECK: function.cpp:[[@LINE+2]]:3: runtime error: call to function f2(unsigned int) through pointer to incorrect function type 'void (*)(int)'
  // NOSYM: function.cpp:[[@LINE+1]]:3: runtime error: call to function (unknown) through pointer to incorrect function type 'void (*)(int)'
  p1(0);
  p1 = &f3;
  p1(0);
  p1 = reinterpret_cast<void (*)(int)>(&f4);
  // CHECK: function.cpp:[[@LINE+2]]:3: runtime error: call to function f4(unsigned int) through pointer to incorrect function type 'void (*)(int)'
  // NOSYM: function.cpp:[[@LINE+1]]:3: runtime error: call to function (unknown) through pointer to incorrect function type 'void (*)(int)'
  p1(0);

  void (*p2)(int) noexcept;
  p2 = reinterpret_cast<void (*)(int) noexcept>(&f1);
  // TODO: Unclear whether calling a non-noexcept function through a pointer to
  // nexcept function should cause an error.
  // CHECK-NOT: function.cpp:[[@LINE+2]]:3: runtime error: call to function f1(int) through pointer to incorrect function type 'void (*)(int) noexcept'
  // NOSYM-NOT: function.cpp:[[@LINE+1]]:3: runtime error: call to function (unknown) through pointer to incorrect function type 'void (*)(int) noexcept'
  p2(0);
  p2 = reinterpret_cast<void (*)(int) noexcept>(&f2);
  // CHECK: function.cpp:[[@LINE+2]]:3: runtime error: call to function f2(unsigned int) through pointer to incorrect function type 'void (*)(int) noexcept'
  // NOSYM: function.cpp:[[@LINE+1]]:3: runtime error: call to function (unknown) through pointer to incorrect function type 'void (*)(int) noexcept'
  p2(0);
  p2 = &f3;
  p2(0);
  p2 = reinterpret_cast<void (*)(int) noexcept>(&f4);
  // CHECK: function.cpp:[[@LINE+2]]:3: runtime error: call to function f4(unsigned int) through pointer to incorrect function type 'void (*)(int) noexcept'
  // NOSYM: function.cpp:[[@LINE+1]]:3: runtime error: call to function (unknown) through pointer to incorrect function type 'void (*)(int) noexcept'
  p2(0);
}

int main(void) {
  make_valid_call();
  make_invalid_call();
  check_noexcept_calls();
  // Check that no more errors will be printed.
  // CHECK-NOT: runtime error: call to function
  // NOSYM-NOT: runtime error: call to function
  make_invalid_call();
}
