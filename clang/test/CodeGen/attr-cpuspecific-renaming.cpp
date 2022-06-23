// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-linux-gnu -emit-llvm -o - -debug-info-kind=limited -dwarf-version=4 -debugger-tuning=gdb %s | FileCheck %s --check-prefixes=CHECK,LIN
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-windows-pc -emit-llvm -o - -debug-info-kind=limited -dwarf-version=4 -debugger-tuning=gdb %s | FileCheck %s --check-prefixes=CHECK,WIN

// LIN: @[[S1_NAME:.+]].ifunc = weak_odr ifunc void (%struct.S1*), void (%struct.S1*)* ()* @[[S1_NAME]].resolver
// LIN: @[[S2_NAME:.+]].ifunc = weak_odr ifunc void (%struct.S2*), void (%struct.S2*)* ()* @[[S2_NAME]].resolver
// WIN: $"[[S1_NAME:.+]]" = comdat any
// WIN: $"[[S2_NAME:.+]]" = comdat any

struct S1 {
  void foo();
  void mv();
};

void S1::foo(){}

__attribute__((cpu_dispatch(ivybridge, generic)))
void S1::mv() {}
// LIN: define weak_odr void (%struct.S1*)* @[[S1_NAME]].resolver
// WIN: define weak_odr dso_local void @"[[S1_NAME]]"(%struct.S1*
__attribute__((cpu_specific(generic)))
void S1::mv() {}
// CHECK: define dso_local {{.*}}void @{{\"?}}[[S1_NAME]].S{{\"?}}
// CHECK: define dso_local {{.*}}void @{{\"?}}[[S1_NAME]].A{{\"?}}
__attribute__((cpu_specific(ivybridge)))
void S1::mv() {}

struct S2 {
  void foo();
  void mv();
};

void S2::foo(){}

__attribute__((cpu_specific(generic)))
void S2::mv() {}
// CHECK: define dso_local {{.*}}void @{{\"?}}[[S2_NAME]].A{{\"?}}
__attribute__((cpu_dispatch(ivybridge, generic)))
void S2::mv() {}
// LIN: define weak_odr void (%struct.S2*)* @[[S2_NAME]].resolver
// WIN: define weak_odr dso_local void @"[[S2_NAME]]"(%struct.S2*
__attribute__((cpu_specific(ivybridge)))
void S2::mv() {}
// CHECK: define dso_local {{.*}}void @{{\"?}}[[S2_NAME]].S{{\"?}}
