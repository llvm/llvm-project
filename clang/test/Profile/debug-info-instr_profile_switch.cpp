// Tests that we don't attach misleading debug locations to llvm.instrprof.increment
// counters.

// RUN: %clang_cc1 -x c++ %s -debug-info-kind=standalone -triple %itanium_abi_triple -main-file-name debug-info-instr_profile_switch.cpp -std=c++11 -o - -emit-llvm -fprofile-instrument=clang | FileCheck %s

int main(int argc, const char *argv[]) {
  switch(argc) {
    case 0:
      return 0;
    case 1:
      return 1;
  }
}

// CHECK: define {{.*}} @main({{.*}}) #0 !dbg ![[MAIN_SCOPE:[0-9]+]]

// CHECK:        switch i32 {{.*}}, label {{.*}} [
// CHECK-NEXT:     i32 0, label %[[CASE1_LBL:[a-z0-9.]+]]
// CHECK-NEXT:     i32 1, label %[[CASE2_LBL:[a-z0-9.]+]]
// CHECK-NEXT:   ], !dbg ![[SWITCH_LOC:[0-9]+]]

// CHECK:       [[CASE1_LBL]]:
// CHECK-NEXT:     %{{.*}} = load i64, ptr getelementptr inbounds ({{.*}}, ptr @__profc_main, {{.*}}), align {{.*}}, !dbg ![[CTR_LOC:[0-9]+]]
// CHECK-NEXT:     %{{.*}} = add {{.*}}, !dbg ![[CTR_LOC]]
// CHECK-NEXT:     store i64 {{.*}}, ptr getelementptr inbounds ({{.*}}, ptr @__profc_main, {{.*}}), align {{.*}}, !dbg ![[CTR_LOC]]
// CHECK-NEXT:     store i32 0, {{.*}} !dbg ![[CASE1_LOC:[0-9]+]]
// CHECK-NEXT:     br label {{.*}}, !dbg ![[CASE1_LOC]]

// CHECK:       [[CASE2_LBL]]:
// CHECK-NEXT:     %{{.*}} = load i64, ptr getelementptr inbounds ({{.*}}, ptr @__profc_main, {{.*}}), align {{.*}}, !dbg ![[CTR_LOC]]
// CHECK-NEXT:     %{{.*}} = add {{.*}}, !dbg ![[CTR_LOC]]
// CHECK-NEXT:     store i64 {{.*}}, ptr getelementptr inbounds ({{.*}}, ptr @__profc_main, {{.*}}), align {{.*}}, !dbg ![[CTR_LOC]]
// CHECK-NEXT:     store i32 1, {{.*}} !dbg ![[CASE2_LOC:[0-9]+]]
// CHECK-NEXT:     br label {{.*}}, !dbg ![[CASE2_LOC]]

// CHECK: ![[SWITCH_LOC]] = !DILocation({{.*}}, scope: ![[MAIN_SCOPE]])
// CHECK: ![[CTR_LOC]] = !DILocation(line: 0, scope: ![[BLOCK_SCOPE:[0-9]+]])
// CHECK: ![[BLOCK_SCOPE]] = distinct !DILexicalBlock(scope: ![[MAIN_SCOPE]]
// CHECK: ![[CASE1_LOC]] = !DILocation(line: {{.*}}, column: {{.*}}, scope: ![[BLOCK_SCOPE]])
// CHECK: ![[CASE2_LOC]] = !DILocation(line: {{.*}}, column: {{.*}}, scope: ![[BLOCK_SCOPE]])
