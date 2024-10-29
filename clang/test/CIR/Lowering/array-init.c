// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// LLVM-DAG: @__const.charInit3.arr
// LLVM-DAG: @__const.charInit2.arr
// LLVM-DAG: @charInit1.ar = internal global [4 x [4 x i8]] {{.*}}4 x i8] c"aa\00\00", [4 x i8] c"aa\00\00", [4 x i8] c"aa\00\00", [4 x i8] c"aa\00\00"], align 16
char charInit1() {
  static char ar[][4] = {"aa", "aa", "aa", "aa"};
  return ar[0][0];
}

// LLVM: define dso_local void @zeroInit
// LLVM: [[RES:%.*]] = alloca [3 x i32], i64 1
// LLVM: store [3 x i32] zeroinitializer, ptr [[RES]]
void zeroInit() {
  int a[3] = {0, 0, 0};
}

// LLVM: %[[PTR:.*]] = alloca [4 x [1 x i8]], i64 1, align 1
// FIXME: OG uses @llvm.memcpy.p0.p0.i64
// LLVM: void @llvm.memcpy.p0.p0.i32(ptr %[[PTR]], ptr @__const.charInit2.arr, i32 4, i1 false)
void charInit2() {
  char arr[4][1] = {"a", "b", "c", "d"};
}

// LLVM: %[[PTR:.*]] = alloca [4 x [2 x i8]], i64 1, align 1
// FIXME: OG uses @llvm.memcpy.p0.p0.i64
// LLVM: call void @llvm.memcpy.p0.p0.i32(ptr %[[PTR]], ptr @__const.charInit3.arr, i32 8, i1 false), !dbg !16
void charInit3() {
  char arr[4][2] = {"ab", "cd", "ef", "gh"};
}