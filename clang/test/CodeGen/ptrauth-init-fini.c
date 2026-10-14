// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-elf -target-feature +pauth -fptrauth-calls -fptrauth-init-fini    \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=COMMON,SIGNED %s

// RUN: %clang_cc1 -triple aarch64-elf -target-feature +pauth -fptrauth-calls -fptrauth-init-fini    \
// RUN:   -fptrauth-init-fini-address-discrimination -emit-llvm %s -o - | FileCheck --check-prefix=COMMON,ADDRDISC %s

// RUN: %clang_cc1 -triple aarch64-elf -target-feature +pauth -fptrauth-calls \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=COMMON,UNSIGNED %s

// RUN: %clang_cc1 -triple aarch64-elf -target-feature +pauth -fptrauth-calls -fptrauth-init-fini-address-discrimination \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=COMMON,UNSIGNED %s

// RUN: %clang_cc1 -triple aarch64-elf -target-feature +pauth                 -fptrauth-init-fini    \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=COMMON,UNSIGNED %s

// COMMON: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]
// COMMON: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @bar, ptr null }]

// COMMON: !llvm.module.flags = !{
// COMMON-SAME: !1
// COMMON-SAME: !2

// UNSIGNED: !1 = !{i32 1, !"ptrauth-init-fini", i32 0}
// UNSIGNED: !2 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 0}

// SIGNED:   !1 = !{i32 1, !"ptrauth-init-fini", i32 1}
// SIGNED:   !2 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 0}

// ADDRDISC: !1 = !{i32 1, !"ptrauth-init-fini", i32 1}
// ADDRDISC: !2 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 1}

volatile int x = 0;

__attribute__((constructor)) void foo(void) {
  x = 42;
}

__attribute__((destructor)) void bar(void) {
  x = 24;
}

int main() {
  return x;
}
