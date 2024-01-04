// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// LLVM: define void @zeroInit
// LLVM: [[RES:%.*]] = alloca [3 x i32], i64 1
// LLVM: store [3 x i32] zeroinitializer, ptr [[RES]]
void zeroInit() {
  int a[3] = {0, 0, 0};
}

