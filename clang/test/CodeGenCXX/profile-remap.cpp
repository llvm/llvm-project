// REQUIRES: x86-registered-target
//
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fprofile-sample-use=%S/Inputs/profile-remap.samples -fprofile-remapping-file=%S/Inputs/profile-remap.map -O2 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-SAMPLES
// RUN: llvm-profdata merge -output %t.profdata %S/Inputs/profile-remap.proftext
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fprofile-instrument-use=llvm -fprofile-instrument-use-path=%t.profdata -fprofile-remapping-file=%S/Inputs/profile-remap.map -O2 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-INSTR
// RUN: llvm-profdata merge -output %t.profdata %S/Inputs/profile-remap_entry.proftext
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fprofile-instrument-use=llvm -fprofile-instrument-use-path=%t.profdata -fprofile-remapping-file=%S/Inputs/profile-remap.map -O2 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-INSTR

namespace Foo {
  struct X {};
  bool cond();
  void bar();
  void baz();
  void function(X x) {
    if (cond())
      bar();
    else
      baz();
  }
}

// CHECK: define {{.*}} @_ZN3Foo8functionENS_1XE() {{.*}} !prof [[FUNC_ENTRY:![0-9]*]]
// CHECK: br i1 {{.*}} !prof [[BR_WEIGHTS:![0-9]*]]
//
// FIXME: Laplace's rule of succession is applied to sample profiles...
// CHECK-SAMPLES-DAG: [[FUNC_ENTRY]] = !{!"function_entry_count", i64 1}
// CHECK-SAMPLES-DAG: [[BR_WEIGHTS]] = !{!"branch_weights", i32 11, i32 91}
//
// ... but not to instruction profiles.
// CHECK-INSTR-DAG: [[FUNC_ENTRY]] = !{!"function_entry_count", i64 100}
// CHECK-INSTR-DAG: [[BR_WEIGHTS]] = !{!"branch_weights", i32 10, i32 90}
