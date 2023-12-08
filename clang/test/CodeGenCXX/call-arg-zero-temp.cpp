// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-apple-darwin -S %s -o %t-64.s
// RUN: FileCheck -check-prefix CHECK-LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -triple i386-apple-darwin -S %s -o %t-32.s
// RUN: FileCheck -check-prefix CHECK-LP32 --input-file=%t-32.s %s


extern "C" int printf(...);

struct obj{ int a; float b; double d; };

void foo(obj o) {
  printf("%d  %f  %f\n", o.a, o.b, o.d);
}

int main() {
  obj o = obj();
  foo(obj());
}

// CHECK-LP64: callq    __Z3foo3obj

// CHECK-LP32: calll     __Z3foo3obj
