// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-captured.c %s -o - -emit-llvm -fprofile-instrument=clang | FileCheck -allow-deprecated-dag-overlap  -check-prefix=PGOGEN -check-prefix=PGOALL %s

// RUN: llvm-profdata merge %S/Inputs/c-captured.proftext -o %t.profdata
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-captured.c %s -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata | FileCheck -allow-deprecated-dag-overlap  -check-prefix=PGOUSE -check-prefix=PGOALL %s

// PGOGEN: @[[DCC:__profc_debug_captured]] = private global [3 x i64] zeroinitializer
// PGOGEN: @[[CSC:__profc_c_captured.c___captured_stmt]] = private global [2 x i64] zeroinitializer
// PGOGEN: @[[C1C:__profc_c_captured.c___captured_stmt.1]] = private global [3 x i64] zeroinitializer

// PGOALL-LABEL: define{{.*}} void @debug_captured()
// PGOGEN: store {{.*}} @[[DCC]], i32 0, i32 0
void debug_captured(void) {
  int x = 10;

// Check both debug_captured counters, so we can do this all in one pass
// PGOGEN: store {{.*}} @[[DCC]], i32 0, i32 1
// PGOUSE: br {{.*}} !prof ![[DC1:[0-9]+]]
// PGOGEN: store {{.*}} @[[DCC]], i32 0, i32 2
// PGOUSE: br {{.*}} !prof ![[DC2:[0-9]+]]
// PGOALL: ret

// PGOALL-LABEL: define internal void @__captured_stmt(
// PGOGEN: store {{.*}} @[[CSC]], i32 0, i32 0
#pragma clang __debug captured
  {
    // PGOGEN: store {{.*}} @[[CSC]], i32 0, i32 1
    // PGOUSE: br {{.*}} !prof ![[CS1:[0-9]+]]
    if (x) {}
    // PGOALL: ret
  }

  if (x) {} // This is DC1. Checked above.

// PGOALL-LABEL: define internal void @__captured_stmt.1(
// PGOGEN: store {{.*}} @[[C1C]], i32 0, i32 0
#pragma clang __debug captured
  {
    // PGOGEN: store {{.*}} @[[C1C]], i32 0, i32 1
    // PGOUSE: br {{.*}} !prof ![[C11:[0-9]+]]
    for (int i = 0; i < x; ++i) {}
    // PGOGEN: store {{.*}} @[[C1C]], i32 0, i32 2
    // PGOUSE: br {{.*}} !prof ![[C12:[0-9]+]]
    if (x) {}
    // PGOALL: ret
  }

  if (x) {} // This is DC2. Checked above.
}

// PGOUSE-DAG: ![[DC1]] = !{!"branch_weights", i32 2, i32 1}
// PGOUSE-DAG: ![[DC2]] = !{!"branch_weights", i32 2, i32 1}
// PGOUSE-DAG: ![[CS1]] = !{!"branch_weights", i32 2, i32 1}
// PGOUSE-DAG: ![[C11]] = !{!"branch_weights", i32 11, i32 2}
// PGOUSE-DAG: ![[C12]] = !{!"branch_weights", i32 2, i32 1}

int main(int argc, const char *argv[]) {
  debug_captured();
  return 0;
}
