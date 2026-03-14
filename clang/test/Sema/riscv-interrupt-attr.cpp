// RUN: %clang_cc1 -x c++ -triple riscv32-unknown-elf -emit-llvm  -DCHECK_IR < %s | FileCheck %s
// RUN: %clang_cc1 -x c++ -triple riscv64-unknown-elf -emit-llvm  -DCHECK_IR < %s | FileCheck %s
// RUN: %clang_cc1 %s -triple riscv32-unknown-elf -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple riscv64-unknown-elf -verify -fsyntax-only

#if defined(CHECK_IR)
// CHECK-LABEL: @_Z11foo_defaultv() #0
// CHECK: ret void
[[gnu::interrupt]] void foo_default() {}
// CHECK: attributes #0
// CHECK: "interrupt"="machine"
#else
[[gnu::interrupt]] [[gnu::interrupt]] void foo1() {} // expected-warning {{repeated RISC-V 'interrupt' attribute}} \
                                                     // expected-note {{repeated RISC-V 'interrupt' attribute is here}}
[[gnu::interrupt]] void foo2() {}
#endif
