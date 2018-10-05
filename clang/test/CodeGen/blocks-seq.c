// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// CHECK: [[Vi:%.+]] = alloca %struct.__block_byref_i, align 8
// CHECK: call i32 (...) @rhs()
// CHECK: [[V7:%.+]] = getelementptr inbounds %struct.__block_byref_i, %struct.__block_byref_i* [[Vi]], i32 0, i32 1
// CHECK: load %struct.__block_byref_i*, %struct.__block_byref_i** [[V7]]
// CHECK: call i32 (...) @rhs()
// CHECK: [[V11:%.+]] = getelementptr inbounds %struct.__block_byref_i, %struct.__block_byref_i* [[Vi]], i32 0, i32 1
// CHECK: load %struct.__block_byref_i*, %struct.__block_byref_i** [[V11]]

int rhs();

void foo() {
  __block int i;
  ^{ (void)i; };
  i = rhs();
  i += rhs();
}
