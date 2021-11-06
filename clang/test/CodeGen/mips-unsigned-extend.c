// RUN: %clang_cc1 -triple mips64-unknown-linux -O0 -target-abi n64 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=N64
// RUN: %clang_cc1 -triple mips64-unknown-linux -O0 -target-abi n32 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=N32
// RUN: %clang_cc1 -triple mips-unknown-linux -O0 -target-abi o32 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=O32

void foo(unsigned a) {
}

void foo1() {
  unsigned f = 0xffffffe0;
  foo(f);
}

// N64: call void @foo(i32 signext %{{[0-9]+}})
// N32: call void @foo(i32 signext %{{[0-9]+}})
// O32: call void @foo(i32 signext %{{[0-9]+}})
