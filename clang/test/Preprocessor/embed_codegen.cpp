// RUN: %clang_cc1 %s -triple x86_64 --embed-dir=%S/Inputs -emit-llvm -o - | FileCheck %s

// CHECK: @__const._Z3fooi.ca = private unnamed_addr constant [3 x i32] [i32 0, i32 106, i32 107], align 4
// CHECK: @__const._Z3fooi.sc = private unnamed_addr constant %struct.S1 { i32 106, i32 107, i32 0 }, align 4
// CHECK: @__const._Z3fooi.t = private unnamed_addr constant [3 x %struct.T] [%struct.T { [2 x i32] [i32 48, i32 49], %struct.S1 { i32 50, i32 51, i32 52 } }, %struct.T { [2 x i32] [i32 53, i32 54], %struct.S1 { i32 55, i32 56, i32 57 } }, %struct.T { [2 x i32] [i32 10, i32 0], %struct.S1 zeroinitializer }], align 16
// CHECK: @__const._Z3fooi.W = private unnamed_addr constant %struct.Wrapper { i32 48, %struct.HasCharArray { [10 x i8] c"123456789\0A" } }, align 4
void foo(int a) {
// CHECK: %a.addr = alloca i32, align 4
// CHECK: store i32 %a, ptr %a.addr, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %ca, ptr align 4 @__const._Z3fooi.ca, i64 12, i1 false)
int ca[] = {
0
#embed <jk.txt> prefix(,)
};

// CHECK: %arrayinit.element = getelementptr inbounds i32, ptr %notca, i64 1
// CHECK: store i32 106, ptr %arrayinit.element, align 4
// CHECK: %arrayinit.element1 = getelementptr inbounds i32, ptr %notca, i64 2
// CHECK: store i32 107, ptr %arrayinit.element1, align 4
int notca[] = {
a
#embed <jk.txt> prefix(,)
};

struct S1 {
  int x, y, z;
};

// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %sc, ptr align 4 @__const._Z3fooi.sc, i64 12, i1 false)
S1 sc = {
#embed <jk.txt> suffix(,)
0
};

// CHECK: %x = getelementptr inbounds nuw %struct.S1, ptr %s, i32 0, i32 0
// CHECK: store i32 106, ptr %x, align 4
// CHECK: %y = getelementptr inbounds nuw %struct.S1, ptr %s, i32 0, i32 1
// CHECK: store i32 107, ptr %y, align 4
// CHECK: %z = getelementptr inbounds nuw %struct.S1, ptr %s, i32 0, i32 2
// CHECK: %1 = load i32, ptr %a.addr, align 4
S1 s = {
#embed <jk.txt> suffix(,)
a
};

// CHECK: store i32 107, ptr %b, align 4
int b = (
#embed<jk.txt>
    )
;


struct T {
  int arr[2];
  struct S1 s;
};

// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %t, ptr align 16 @__const._Z3fooi.t, i64 60, i1 false)
constexpr struct T t[] = {
#embed <numbers.txt>
};

// CHECK:  %arr = getelementptr inbounds nuw %struct.T, ptr %tnonc, i32 0, i32 0
// CHECK:  %2 = load i32, ptr %a.addr, align 4
// CHECK:  store i32 %2, ptr %arr, align 4
// CHECK:  %arrayinit.element2 = getelementptr inbounds i32, ptr %arr, i64 1
// CHECK:  store i32 300, ptr %arrayinit.element2, align 4
// CHECK:  %s3 = getelementptr inbounds nuw %struct.T, ptr %tnonc, i32 0, i32 1
// CHECK:  %x4 = getelementptr inbounds nuw %struct.S1, ptr %s3, i32 0, i32 0
// CHECK:  store i32 1, ptr %x4, align 4
// CHECK:  %y5 = getelementptr inbounds nuw %struct.S1, ptr %s3, i32 0, i32 1
// CHECK:  store i32 2, ptr %y5, align 4
// CHECK:  %z6 = getelementptr inbounds nuw %struct.S1, ptr %s3, i32 0, i32 2
// CHECK:  store i32 3, ptr %z6, align 4
// CHECK:  %arrayinit.element7 = getelementptr inbounds %struct.T, ptr %tnonc, i64 1
// CHECK:  call void @llvm.memset.p0.i64(ptr align 4 %arrayinit.element7, i8 0, i64 20, i1 false)
// CHECK:  %arr8 = getelementptr inbounds nuw %struct.T, ptr %arrayinit.element7, i32 0, i32 0
// CHECK:  store i32 106, ptr %arr8, align 4
// CHECK:  %arrayinit.element9 = getelementptr inbounds i32, ptr %arr8, i64 1
// CHECK:  store i32 107, ptr %arrayinit.element9, align 4
struct T tnonc[] = {
  a, 300, 1, 2, 3
#embed <jk.txt> prefix(,)
};


struct HasCharArray { unsigned char h[10]; };
struct Wrapper { int a; struct HasCharArray d;  };
constexpr struct Wrapper W = {
#embed "numbers.txt"
};

}
