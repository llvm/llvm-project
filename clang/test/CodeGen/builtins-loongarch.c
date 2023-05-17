// RUN: %clang_cc1 -Wall -Wno-unused-but-set-variable -Werror -triple loongarch32 -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s
// RUN: %clang_cc1 -Wall -Wno-unused-but-set-variable -Werror -triple loongarch64 -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s

void test_eh_return_data_regno(void) {
  // CHECK: store volatile i32 4
  // CHECK: store volatile i32 5
  volatile int res;
  res = __builtin_eh_return_data_regno(0);
  res = __builtin_eh_return_data_regno(1);
}
