// RUN: %clang -O2  -fexperimental-new-pass-manager -fpseudo-probe-for-profiling -g -emit-llvm -S -o - %s | FileCheck %s

// Check the generation of pseudoprobe intrinsic call

void bar();
void go();

void foo(int x) {
  // CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 1, i32 0)
  if (x == 0)
    // CHECK: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 2, i32 0)
    bar();
  else
    // CHECK: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 3, i32 0)
    go();
  // CHECK: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 4, i32 0)
}
