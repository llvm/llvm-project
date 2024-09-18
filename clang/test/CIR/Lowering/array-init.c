// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// LLVM: charInit1.ar = internal global [4 x [4 x i8]] {{.*}}4 x i8] c"aa\00\00", [4 x i8] c"aa\00\00", [4 x i8] c"aa\00\00", [4 x i8] c"aa\00\00"], align 16
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

// LLVM: %1 = alloca [4 x [1 x i8]], i64 1, align 1
// LLVM: store [4 x [1 x i8]] {{.*}}1 x i8] c"a", [1 x i8] c"b", [1 x i8] c"c", [1 x i8] c"d"], ptr %1, align 1
void charInit2() {
  char arr[4][1] = {"a", "b", "c", "d"};
}

// LLVM: %1 = alloca [4 x [2 x i8]], i64 1, align 1
// LLVM: store [4 x [2 x i8]] {{.*}}2 x i8] c"ab", [2 x i8] c"cd", [2 x i8] c"ef", [2 x i8] c"gh"], ptr %1, align 1
void charInit3() {
  char arr[4][2] = {"ab", "cd", "ef", "gh"};
}