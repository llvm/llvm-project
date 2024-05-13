// Test clang diagnositic handler works in IR file compilation.

// REQUIRES: x86-registered-target

// RUN: llvm-profdata merge -o %t1.profdata %S/Inputs/thinlto_expect1.proftext
// RUN: %clang -target x86_64-linux-gnu -O2 -flto=thin -g -fprofile-use=%t1.profdata -c -o %t1.bo %s
// RUN: llvm-lto -thinlto -o %t %t1.bo
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -x ir %t1.bo -fthinlto-index=%t.thinlto.bc -emit-obj -Rpass-analysis=info 2>&1 | FileCheck %s -check-prefix=CHECK-REMARK
// RUN: llvm-profdata merge -o %t2.profdata %S/Inputs/thinlto_expect2.proftext
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -x ir %t1.bo -fthinlto-index=%t.thinlto.bc -fprofile-instrument-use-path=%t2.profdata -emit-obj 2>&1 | FileCheck %s -allow-empty -check-prefix=CHECK-NOWARNING

int sum;
__attribute__((noinline)) void bar(void) {
  sum = 1234;
}

__attribute__((noinline)) void foo(int m) {
  if (__builtin_expect(m > 9, 1))
    bar();
}
// CHECK-REMARK: remark: {{.*}}.c:
// CHECK-NOWARNING-NOT: warning: {{.*}}.c:{{[0-9]*}}:26: 50.00% (12 / 24)
