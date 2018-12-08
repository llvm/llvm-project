// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LINUX

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

// LINUX: define linkonce_odr void @_ZN1S4FuncEv.O
// LINUX: define void (%struct.S*)* @_ZN1S4FuncEv.resolver
// LINUX: ret void (%struct.S*)* @_ZN1S4FuncEv.S
// LINUX: ret void (%struct.S*)* @_ZN1S4FuncEv.O
