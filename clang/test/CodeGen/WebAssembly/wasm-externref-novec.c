// RUN: %clang -O2 --target=wasm32 -S -emit-llvm -mreference-types -o - %s | FileCheck %s

// Issue: https://github.com/llvm/llvm-project/issues/69894

__externref_t foo(void);
// CHECK: declare ptr addrspace(10) @foo()

void bar(__externref_t);
// CHECK: declare void @bar(ptr addrspace(10))

void test(int flag, __externref_t ref1, __externref_t ref2) {
  if (flag) {
    ref1 = foo();
    ref2 = foo();
  }
  bar(ref1);
  bar(ref2);
}
