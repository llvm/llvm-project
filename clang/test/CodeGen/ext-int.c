// RUN: %clang_cc1 -std=c23 -triple x86_64-gnu-linux -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK64
// RUN: %clang_cc1 -std=c23 -triple x86_64-windows-pc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK64
// RUN: %clang_cc1 -std=c23 -triple i386-gnu-linux -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN32
// RUN: %clang_cc1 -std=c23 -triple i386-windows-pc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN32

// CHECK64: %struct.S1 = type { i17, [4 x i8], [24 x i8] }
// CHECK64: %struct.S2 = type { [40 x i8], i32, [4 x i8] }

//GH62207
unsigned _BitInt(1) GlobSize1 = 0;
// CHECK: @GlobSize1 = {{.*}}global i1 false

// CHECK64: @__const.foo.A = private unnamed_addr constant { i17, [4 x i8], <{ i8, [23 x i8] }> } { i17 1, [4 x i8] undef, <{ i8, [23 x i8] }> <{ i8 -86, [23 x i8] zeroinitializer }> }, align 8
// CHECK64: @BigGlob = {{.*}}global <{ i8, i8, [38 x i8] }> <{ i8 -68, i8 2, [38 x i8] zeroinitializer }>, align 8
// CHECK64: @f.p = internal global <{ i8, i8, [22 x i8] }> <{ i8 16, i8 39, [22 x i8] zeroinitializer }>, align 8

void GenericTest(_BitInt(3) a, unsigned _BitInt(3) b, _BitInt(4) c) {
  // CHECK: define {{.*}}void @GenericTest
  int which = _Generic(a, _BitInt(3): 1, unsigned _BitInt(3) : 2, _BitInt(4) : 3);
  // CHECK: store i32 1
  int which2 = _Generic(b, _BitInt(3): 1, unsigned _BitInt(3) : 2, _BitInt(4) : 3);
  // CHECK: store i32 2
  int which3 = _Generic(c, _BitInt(3): 1, unsigned _BitInt(3) : 2, _BitInt(4) : 3);
  // CHECK: store i32 3
}

void VLATest(_BitInt(3) A, _BitInt(99) B, _BitInt(123) C) {
  // CHECK: define {{.*}}void @VLATest
  int AR1[A];
  // CHECK: %[[A:.+]] = zext i3 %{{.+}} to i[[INDXSIZE:[0-9]+]]
  // CHECK: %[[VLA1:.+]] = alloca i32, i[[INDXSIZE]] %[[A]]
  int AR2[B];
  // CHECK: %[[B:.+]] = trunc i99 %{{.+}} to i[[INDXSIZE]]
  // CHECK: %[[VLA2:.+]] = alloca i32, i[[INDXSIZE]] %[[B]]
  int AR3[C];
  // CHECK: %[[C:.+]] = trunc i123 %{{.+}} to i[[INDXSIZE]]
  // CHECK: %[[VLA3:.+]] = alloca i32, i[[INDXSIZE]] %[[C]]
}

struct S {
  _BitInt(17) A;
  _BitInt(128) B;
  _BitInt(17) C;
};

void OffsetOfTest(void) {
  // CHECK: define {{.*}}void @OffsetOfTest
  int A = __builtin_offsetof(struct S,A);
  // CHECK: store i32 0, ptr %{{.+}}
  int B = __builtin_offsetof(struct S,B);
  // CHECK64: store i32 8, ptr %{{.+}}
  // LIN32: store i32 4, ptr %{{.+}}
  // WINCHECK32: store i32 8, ptr %{{.+}}
  int C = __builtin_offsetof(struct S,C);
  // CHECK64: store i32 24, ptr %{{.+}}
  // LIN32: store i32 20, ptr %{{.+}}
  // WIN32: store i32 24, ptr %{{.+}}
}

void Size1ExtIntParam(unsigned _BitInt(1) A) {
  // CHECK: define {{.*}}void @Size1ExtIntParam(i1{{.*}}  %[[PARAM:.+]])
  // CHECK: %[[PARAM_ADDR:.+]] = alloca i1
  // CHECK: %[[B:.+]] = alloca [5 x i1]
  // CHECK: store i1 %[[PARAM]], ptr %[[PARAM_ADDR]]
  unsigned _BitInt(1) B[5];

  // CHECK: %[[PARAM_LOAD:.+]] = load i1, ptr %[[PARAM_ADDR]]
  // CHECK: %[[IDX:.+]] = getelementptr inbounds [5 x i1], ptr %[[B]]
  // CHECK: store i1 %[[PARAM_LOAD]], ptr %[[IDX]]
  B[2] = A;
}

#if __BITINT_MAXWIDTH__ > 128
struct S1 {
  _BitInt(17) A;
  _BitInt(129) B;
};

int foo(int a) {
  // CHECK64: %A1 = getelementptr inbounds %struct.S1, ptr %B, i32 0, i32 0
  // CHECK64: store i17 1, ptr %A1, align 8
  // CHECK64: %B2 = getelementptr inbounds %struct.S1, ptr %B, i32 0, i32 2
  // CHECK64: %0 = load i32, ptr %a.addr, align 4
  // CHECK64: %conv = sext i32 %0 to i129
  // CHECK64: store i129 %conv, ptr %B2, align 8
  // CHECK64: %B3 = getelementptr inbounds %struct.S1, ptr %A, i32 0, i32 2
  // CHECK64: %1 = load i129, ptr %B3, align 8
  // CHECK64: %conv4 = trunc i129 %1 to i32
  // CHECK64: %B5 = getelementptr inbounds %struct.S1, ptr %B, i32 0, i32 2
  // CHECK64: %2 = load i129, ptr %B5, align 8
  struct S1 A = {1, 170};
  struct S1 B = {1, a};
  return (int)A.B + (int)B.B;
}

struct S2 {
  _BitInt(257) A;
  int B;
};

_BitInt(257) bar() {
  // CHECK64: define {{.*}}void @bar(ptr {{.*}} sret([40 x i8]) align 8 %[[RET:.+]])
  // CHECK64: %A = alloca %struct.S2, align 8
  // CHECK64: %0 = getelementptr inbounds { <{ i8, [39 x i8] }>, i32, [4 x i8] }, ptr %A, i32 0, i32 0
  // CHECK64: %1 = getelementptr inbounds <{ i8, [39 x i8] }>, ptr %0, i32 0, i32 0
  // CHECK64: store i8 1, ptr %1, align 8
  // CHECK64: %2 = getelementptr inbounds { <{ i8, [39 x i8] }>, i32, [4 x i8] }, ptr %A, i32 0, i32 1
  // CHECK64: store i32 10000, ptr %2, align 8
  // CHECK64: %A1 = getelementptr inbounds %struct.S2, ptr %A, i32 0, i32 0
  // CHECK64: %3 = load i257, ptr %A1, align 8
  // CHECK64: store i257 %3, ptr %[[RET]], align 8
  struct S2 A = {1, 10000};
  return A.A;
}

void TakesVarargs(int i, ...) {
  // CHECK64: define{{.*}} void @TakesVarargs(i32
__builtin_va_list args;
__builtin_va_start(args, i);

_BitInt(160) A = __builtin_va_arg(args, _BitInt(160));
  // CHECK64: %[[ARG:.+]] = load i160
  // CHECK64: store i160 %[[ARG]], ptr %A, align 8
}

_BitInt(129) *f1(_BitInt(129) *p) {
  // CHECK64: getelementptr inbounds [24 x i8], {{.*}} i64 1
  return p + 1;
}

char *f2(char *p) {
  // CHECK64: getelementptr inbounds i8, {{.*}} i64 24
  return p + sizeof(_BitInt(129));
}

auto BigGlob = (_BitInt(257))700;
// CHECK64: define {{.*}}void @foobar(ptr {{.*}} sret([40 x i8]) align 8 %[[RET1:.+]])
_BitInt(257) foobar() {
  // CHECK64: %A = alloca [40 x i8], align 8
  // CHECK64: %0 = load i257, ptr @BigGlob, align 8
  // CHECK64: %add = add nsw i257 %0, 1
  // CHECK64: store i257 %add, ptr %A, align 8
  // CHECK64: %1 = load i257, ptr %A, align 8
  // CHECK64: store i257 %1, ptr %[[RET1]], align 8
  _BitInt(257) A = BigGlob + 1;
  return A;
}

void f() {
  static _BitInt(130) p = {10000};
}

#endif
