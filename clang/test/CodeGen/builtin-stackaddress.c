// RUN: %clang -target x86_64 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=llvm
// RUN: %clang -target x86_64 -S %s -o - | FileCheck %s --check-prefix=x64

extern void f(int, int, int, long, long, long, long, long, long, long, long);

// llvm-LABEL: define {{[^@]+}} @a()
// llvm:       call   {{[^@]+}} @llvm.stackaddress.p0()
//
// x64-LABEL: a:
// x64:       movq  %rsp, %rax
void *a() {
  f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
  return __builtin_stack_address();
}
