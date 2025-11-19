// Test that clang doesn't emit llvm.expect when the counter is 0

// RUN: llvm-profdata merge %S/Inputs/cxx-never-executed-branch.proftext -o %t.profdata
// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -O2 -o - -emit-llvm -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata -disable-llvm-passes | FileCheck %s

int rand();

// CHECK: define {{.*}}@_Z13is_in_profilev
// CHECK-NOT: call {{.*}}@llvm.expect

int is_in_profile() {
  int rando = rand();
  int x = 0;
  if (rando == 0) [[likely]]
    x = 2;
  else
    x = 3;
  return x;
}

// CHECK: define {{.*}}@_Z17is_not_in_profilev
// CHECK: call {{.*}}@llvm.expect

int is_not_in_profile() {
  int rando = rand();
  int x = 0;
  if (rando == 0) [[likely]]
    x = 2;
  else
    x = 3;
  return x;
}
