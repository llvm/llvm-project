// RUN: %clang_cc1 -Wall -Wno-unused-but-set-variable -Werror -triple riscv32 -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s
// RUN: %clang_cc1 -Wall -Wno-unused-but-set-variable -Werror -triple riscv64 -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

void test_eh_return_data_regno() {
  // CHECK: store volatile i32 10
  // CHECK: store volatile i32 11
  volatile int res;
  res = __builtin_eh_return_data_regno(0);
  res = __builtin_eh_return_data_regno(1);
}
