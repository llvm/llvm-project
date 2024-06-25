// RUN: %clang_cc1 -O0 -fpseudo-probe-for-profiling -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -O2 -fpseudo-probe-for-profiling -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

// Check the generation of pseudoprobe intrinsic call

void bar(void);
void go(void);

void foo(int x) {
  // CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 1, i32 0, i64 -1)
  if (x == 0)
    // CHECK: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 2, i32 0, i64 -1)
    bar();  // probe id : 3
  else
    // CHECK: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 4, i32 0, i64 -1)
    go(); // probe id : 5
  // CHECK: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 6, i32 0, i64 -1)
}
