// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// CHECK: [[Vi:%.+]] = alloca %struct.__block_byref_i, align 8
// CHECK: call i32 @rhs()
// CHECK: [[V7:%.+]] = getelementptr inbounds nuw %struct.__block_byref_i, ptr [[Vi]], i32 0, i32 1
// CHECK: load ptr, ptr [[V7]]
// CHECK: call i32 @rhs()
// CHECK: [[V11:%.+]] = getelementptr inbounds nuw %struct.__block_byref_i, ptr [[Vi]], i32 0, i32 1
// CHECK: load ptr, ptr [[V11]]

int rhs(void);

void foo(void) {
  __block int i;
  ^{ (void)i; };
  i = rhs();
  i += rhs();
}
