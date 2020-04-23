// RUN: %clang_cc1 -triple x86_64-gnu-linux -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -triple x86_64-windows-pc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK


void GenericTest(_ExtInt(3) a, unsigned _ExtInt(3) b, _ExtInt(4) c) {
  // CHECK: define {{.*}}void @GenericTest
  int which = _Generic(a, _ExtInt(3): 1, unsigned _ExtInt(3) : 2, _ExtInt(4) : 3);
  // CHECK: store i32 1
  int which2 = _Generic(b, _ExtInt(3): 1, unsigned _ExtInt(3) : 2, _ExtInt(4) : 3);
  // CHECK: store i32 2
  int which3 = _Generic(c, _ExtInt(3): 1, unsigned _ExtInt(3) : 2, _ExtInt(4) : 3);
  // CHECK: store i32 3
}

void VLATest(_ExtInt(3) A, _ExtInt(99) B, _ExtInt(123456) C) {
  // CHECK: define {{.*}}void @VLATest
  int AR1[A];
  // CHECK: %[[A:.+]] = zext i3 %{{.+}} to i64
  // CHECK: %[[VLA1:.+]] = alloca i32, i64 %[[A]]
  int AR2[B];
  // CHECK: %[[B:.+]] = trunc i99 %{{.+}} to i64
  // CHECK: %[[VLA2:.+]] = alloca i32, i64 %[[B]]
  int AR3[C];
  // CHECK: %[[C:.+]] = trunc i123456 %{{.+}} to i64
  // CHECK: %[[VLA3:.+]] = alloca i32, i64 %[[C]]
}

struct S {
  _ExtInt(17) A;
  _ExtInt(16777200) B;
  _ExtInt(17) C;
};

void OffsetOfTest() {
  // CHECK: define {{.*}}void @OffsetOfTest 
  int A = __builtin_offsetof(struct S,A);
  // CHECK: store i32 0, i32* %{{.+}}
  int B = __builtin_offsetof(struct S,B);
  // CHECK: store i32 8, i32* %{{.+}}
  int C = __builtin_offsetof(struct S,C);
  // CHECK: store i32 2097160, i32* %{{.+}}
}


