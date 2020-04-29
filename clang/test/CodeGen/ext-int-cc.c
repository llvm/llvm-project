// RUN: %clang_cc1 -triple x86_64-gnu-linux -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=LIN64
// RUN: %clang_cc1 -triple x86_64-windows-pc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=WIN64
// RUN: %clang_cc1 -triple i386-gnu-linux -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=LIN32
// RUN: %clang_cc1 -triple i386-windows-pc -O3 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=WIN32

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
// WIN32: define dso_local i63 @ReturnPassing(
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
// WIN32: define dso_local i128 @ReturnPassing4(
_ExtInt(129) ReturnPassing5(){}
// LIN64: define void @ReturnPassing5(i129* noalias sret
// WIN64: define dso_local void @ReturnPassing5(i129* noalias sret
// LIN32: define i129 @ReturnPassing5(
// WIN32: define dso_local i129 @ReturnPassing5(
