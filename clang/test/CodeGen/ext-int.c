// RUN: %clang_cc1 -triple x86_64-gnu-linux -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN64,CHECK64
// RUN: %clang_cc1 -triple x86_64-windows-pc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN64,CHECK64
// RUN: %clang_cc1 -triple i386-gnu-linux -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN32,CHECK32
// RUNX: %clang_cc1 -triple i386-windows-pc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN32,CHECK32


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
  // CHECK: %[[A:.+]] = zext i3 %{{.+}} to i[[INDXSIZE:[0-9]+]]
  // CHECK: %[[VLA1:.+]] = alloca i32, i[[INDXSIZE]] %[[A]]
  int AR2[B];
  // CHECK: %[[B:.+]] = trunc i99 %{{.+}} to i[[INDXSIZE]]
  // CHECK: %[[VLA2:.+]] = alloca i32, i[[INDXSIZE]] %[[B]]
  int AR3[C];
  // CHECK: %[[C:.+]] = trunc i123456 %{{.+}} to i[[INDXSIZE]]
  // CHECK: %[[VLA3:.+]] = alloca i32, i[[INDXSIZE]] %[[C]]
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
  // CHECK64: store i32 8, i32* %{{.+}}
  // CHECK32: store i32 4, i32* %{{.+}}
  int C = __builtin_offsetof(struct S,C);
  // CHECK64: store i32 2097160, i32* %{{.+}}
  // CHECK32: store i32 2097156, i32* %{{.+}}
}

// Make sure 128 and 64 bit versions are passed like integers, and that >128
// is passed indirectly.
void ParamPassing(_ExtInt(129) a, _ExtInt(128) b, _ExtInt(64) c) {}
// LIN64: define void @ParamPassing(i129* byval(i129) align 8 %{{.+}}, i64 %{{.+}}, i64 %{{.+}}, i64 %{{.+}})
// WIN64: define dso_local void @ParamPassing(i129* %{{.+}}, i128* %{{.+}}, i64 %{{.+}})
// LIN32: define void @ParamPassing(i129* %{{.+}}, i128* %{{.+}}, i64 %{{.+}})
// WIN32: define dso_local void @ParamPassing(i129* %{{.+}}, i128* %{{.+}}, i64 %{{.+}})
void ParamPassing2(_ExtInt(129) a, _ExtInt(127) b, _ExtInt(63) c) {}
// LIN64: define void @ParamPassing2(i129* byval(i129) align 8 %{{.+}}, i64 %{{.+}}, i64 %{{.+}}, i64 %{{.+}})
// WIN64: define dso_local void @ParamPassing2(i129* %{{.+}}, i127* %{{.+}}, i63 %{{.+}})
// LIN32: define void @ParamPassing2(i129* %{{.+}}, i127* %{{.+}}, i63 %{{.+}})
// WIN32: define dso_local void @ParamPassing2(i129* %{{.+}}, i127* %{{.+}}, i63 %{{.+}})
_ExtInt(63) ReturnPassing(){}
// LIN64: define i64 @ReturnPassing(
// WIN64: define dso_local i63 @ReturnPassing(
// LIN32: define i63 @ReturnPassing(
// WIN32: define dso_local i64 @ReturnPassing(
_ExtInt(64) ReturnPassing2(){}
// LIN64: define i64 @ReturnPassing2(
// WIN64: define dso_local i64 @ReturnPassing2(
// LIN32: define i64 @ReturnPassing2(
// WIN32: define dso_local i64 @ReturnPassing2(
_ExtInt(127) ReturnPassing3(){}
// LIN64: define { i64, i64 } @ReturnPassing3(
// WIN64: define dso_local void @ReturnPassing3(i127* noalias sret
// LIN32: define i127 @ReturnPassing3(
// WIN32: define dso_local i127 @ReturnPassing3(
_ExtInt(128) ReturnPassing4(){}
// LIN64: define { i64, i64 } @ReturnPassing4(
// WIN64: define dso_local void @ReturnPassing4(i128* noalias sret
// LIN32: define i128 @ReturnPassing4(
// WIN32: define dso_local i64 @ReturnPassing4(
_ExtInt(129) ReturnPassing5(){}
// LIN64: define void @ReturnPassing5(i129* noalias sret
// WIN64: define dso_local void @ReturnPassing5(i129* noalias sret
// LIN32: define i129 @ReturnPassing5(
// WIN32: define dso_local i129 @ReturnPassing5(
