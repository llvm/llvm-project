// RUN: %clang_cc1 -triple avr -emit-llvm %s -o - | FileCheck %s

// This test case verifies https://github.com/llvm/llvm-project/issues/236067 .

extern void foo_0(void);
extern void foo_1(void);

void foo(int a) {
  void *p = &&label_0;
  if (a > 0)
    goto *p;
  foo_0();
  return;
label_0:
  foo_1();
}

// CHECK: store ptr addrspacecast (ptr addrspace(1) blockaddress(@foo, %label_0) to ptr), ptr %p, align 1
// CHECK: indirectbr ptr {{.*}} [label %label_0]
