// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -triple x86_64-windows-pc -fms-compatibility -emit-llvm -o - %s | FileCheck %s --check-prefix=WINDOWS

struct OutOfLineDefs {
  int foo(int);
  int foo(int, int);
  __attribute__((cpu_specific(atom))) int foo(int, int, int) { return 1; }
};

int __attribute__((cpu_specific(atom))) OutOfLineDefs::foo(int) {
  return 1;
}
int __attribute__((cpu_specific(ivybridge))) OutOfLineDefs::foo(int) {
  return 2;
}
int __attribute__((cpu_dispatch(ivybridge, atom))) OutOfLineDefs::foo(int) {
}

int __attribute__((cpu_specific(atom))) OutOfLineDefs::foo(int, int) {
  return 1;
}
int __attribute__((cpu_specific(ivybridge))) OutOfLineDefs::foo(int, int) {
  return 2;
}
int __attribute__((cpu_dispatch(ivybridge, atom)))
OutOfLineDefs::foo(int, int) {
}

int __attribute__((cpu_specific(ivybridge))) OutOfLineDefs::foo(int, int, int) {
  return 2;
}
int __attribute__((cpu_specific(sandybridge)))
OutOfLineDefs::foo(int, int, int) {
  return 3;
}
int __attribute__((cpu_dispatch(sandybridge, ivybridge, atom)))
OutOfLineDefs::foo(int, int, int) {
}

// IFunc definitions, Linux only.
// LINUX: @_ZN13OutOfLineDefs3fooEi = weak_odr alias i32 (ptr, i32), ptr @_ZN13OutOfLineDefs3fooEi.ifunc
// LINUX: @_ZN13OutOfLineDefs3fooEii = weak_odr alias i32 (ptr, i32, i32), ptr @_ZN13OutOfLineDefs3fooEii.ifunc
// LINUX: @_ZN13OutOfLineDefs3fooEiii = weak_odr alias i32 (ptr, i32, i32, i32), ptr @_ZN13OutOfLineDefs3fooEiii.ifunc

// LINUX: @_ZN13OutOfLineDefs3fooEi.ifunc = weak_odr ifunc i32 (ptr, i32), ptr @_ZN13OutOfLineDefs3fooEi.resolver
// LINUX: @_ZN13OutOfLineDefs3fooEii.ifunc = weak_odr ifunc i32 (ptr, i32, i32), ptr @_ZN13OutOfLineDefs3fooEii.resolver
// LINUX: @_ZN13OutOfLineDefs3fooEiii.ifunc = weak_odr ifunc i32 (ptr, i32, i32, i32), ptr @_ZN13OutOfLineDefs3fooEiii.resolver

// Arity 1 version:
// LINUX: define dso_local noundef i32 @_ZN13OutOfLineDefs3fooEi.O
// LINUX: define dso_local noundef i32 @_ZN13OutOfLineDefs3fooEi.S
// LINUX: define weak_odr ptr @_ZN13OutOfLineDefs3fooEi.resolver()
// LINUX: ret ptr @_ZN13OutOfLineDefs3fooEi.S
// LINUX: ret ptr @_ZN13OutOfLineDefs3fooEi.O
// LINUX: call void @llvm.trap

// WINDOWS: define dso_local noundef i32 @"?foo@OutOfLineDefs@@QEAAHH@Z.O"
// WINDOWS: define dso_local noundef i32 @"?foo@OutOfLineDefs@@QEAAHH@Z.S"
// WINDOWS: define weak_odr dso_local i32 @"?foo@OutOfLineDefs@@QEAAHH@Z"(ptr %0, i32 %1)
// WINDOWS: musttail call i32 @"?foo@OutOfLineDefs@@QEAAHH@Z.S"(ptr %0, i32 %1)
// WINDOWS: musttail call i32 @"?foo@OutOfLineDefs@@QEAAHH@Z.O"(ptr %0, i32 %1)
// WINDOWS: call void @llvm.trap

// Arity 2 version:
// LINUX: define dso_local noundef i32 @_ZN13OutOfLineDefs3fooEii.O
// LINUX: define dso_local noundef i32 @_ZN13OutOfLineDefs3fooEii.S
// LINUX: define weak_odr ptr @_ZN13OutOfLineDefs3fooEii.resolver()
// LINUX: ret ptr @_ZN13OutOfLineDefs3fooEii.S
// LINUX: ret ptr @_ZN13OutOfLineDefs3fooEii.O
// LINUX: call void @llvm.trap

// WINDOWS: define dso_local noundef i32 @"?foo@OutOfLineDefs@@QEAAHHH@Z.O"
// WINDOWS: define dso_local noundef i32 @"?foo@OutOfLineDefs@@QEAAHHH@Z.S"
// WINDOWS: define weak_odr dso_local i32 @"?foo@OutOfLineDefs@@QEAAHHH@Z"(ptr %0, i32 %1, i32 %2)
// WINDOWS: musttail call i32 @"?foo@OutOfLineDefs@@QEAAHHH@Z.S"(ptr %0, i32 %1, i32 %2)
// WINDOWS: musttail call i32 @"?foo@OutOfLineDefs@@QEAAHHH@Z.O"(ptr %0, i32 %1, i32 %2)
// WINDOWS: call void @llvm.trap

// Arity 3 version:
// LINUX: define dso_local noundef i32 @_ZN13OutOfLineDefs3fooEiii.S
// LINUX: define dso_local noundef i32 @_ZN13OutOfLineDefs3fooEiii.R
// LINUX: define weak_odr ptr @_ZN13OutOfLineDefs3fooEiii.resolver()
// LINUX: ret ptr @_ZN13OutOfLineDefs3fooEiii.S
// LINUX: ret ptr @_ZN13OutOfLineDefs3fooEiii.R
// LINUX: ret ptr @_ZN13OutOfLineDefs3fooEiii.O
// LINUX: call void @llvm.trap
// LINUX: define linkonce_odr noundef i32 @_ZN13OutOfLineDefs3fooEiii.O

// WINDOWS: define dso_local noundef i32 @"?foo@OutOfLineDefs@@QEAAHHHH@Z.S"
// WINDOWS: define dso_local noundef i32 @"?foo@OutOfLineDefs@@QEAAHHHH@Z.R"
// WINDOWS: define weak_odr dso_local i32 @"?foo@OutOfLineDefs@@QEAAHHHH@Z"(ptr %0, i32 %1, i32 %2, i32 %3)
// WINDOWS: musttail call i32 @"?foo@OutOfLineDefs@@QEAAHHHH@Z.S"(ptr %0, i32 %1, i32 %2, i32 %3)
// WINDOWS: musttail call i32 @"?foo@OutOfLineDefs@@QEAAHHHH@Z.R"(ptr %0, i32 %1, i32 %2, i32 %3)
// WINDOWS: musttail call i32 @"?foo@OutOfLineDefs@@QEAAHHHH@Z.O"(ptr %0, i32 %1, i32 %2, i32 %3)
// WINDOWS: call void @llvm.trap
// WINDOWS: define linkonce_odr dso_local noundef i32 @"?foo@OutOfLineDefs@@QEAAHHHH@Z.O"

