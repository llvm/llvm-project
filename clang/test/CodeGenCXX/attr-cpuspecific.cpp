// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -triple x86_64-apple-macos -emit-llvm -o - %s | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -triple x86_64-windows-pc -fms-compatibility -emit-llvm -o - %s | FileCheck %s --check-prefix=WINDOWS

struct S {
  __attribute__((cpu_specific(atom)))
  void Func(){}
  __attribute__((cpu_dispatch(ivybridge,atom)))
  void Func(){}
};

void foo() {
  S s;
  s.Func();
}

// LINUX: @_ZN1S4FuncEv = weak_odr alias void (ptr), ptr @_ZN1S4FuncEv.ifunc
// LINUX: @_ZN1S4FuncEv.ifunc = weak_odr ifunc void (ptr), ptr @_ZN1S4FuncEv.resolver
// LINUX: define weak_odr ptr @_ZN1S4FuncEv.resolver
// LINUX: ret ptr @_ZN1S4FuncEv.S
// LINUX: ret ptr @_ZN1S4FuncEv.O
// LINUX: declare void @_ZN1S4FuncEv.S
// LINUX: define linkonce_odr void @_ZN1S4FuncEv.O

// WINDOWS: define weak_odr dso_local void @"?Func@S@@QEAAXXZ"(ptr %0) comdat
// WINDOWS: musttail call void @"?Func@S@@QEAAXXZ.S"(ptr %0)
// WINDOWS: musttail call void @"?Func@S@@QEAAXXZ.O"(ptr %0)
// WINDOWS: declare dso_local void @"?Func@S@@QEAAXXZ.S"
// WINDOWS: define linkonce_odr dso_local void @"?Func@S@@QEAAXXZ.O"
