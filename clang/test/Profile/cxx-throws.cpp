// Test instrumentation of C++ exception handling constructs.

// FIXME: Don't seek bb labels, like "if.else"
// REQUIRES: asserts

// RUN: %clang_cc1 %s -o - -emit-llvm -fprofile-instrument=clang -fexceptions -fcxx-exceptions -triple %itanium_abi_triple | FileCheck -check-prefix=PGOGEN %s
// RUN: %clang_cc1 %s -o - -emit-llvm -fprofile-instrument=clang -fexceptions -fcxx-exceptions -triple %itanium_abi_triple | FileCheck -check-prefix=PGOGEN-EXC %s

// RUN: llvm-profdata merge %S/Inputs/cxx-throws.proftext -o %t.profdata
// RUN: %clang_cc1 %s -o - -emit-llvm -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata -fexceptions -fcxx-exceptions -triple %itanium_abi_triple | FileCheck -check-prefix=PGOUSE %s
// RUN: %clang_cc1 %s -o - -emit-llvm -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata -fexceptions -fcxx-exceptions -triple %itanium_abi_triple | FileCheck -check-prefix=PGOUSE-EXC %s

// PGOGEN: @[[THC:__profc__Z6throwsv]] = {{(private|internal)}} global [9 x i64] zeroinitializer
// PGOGEN-EXC: @[[THC:__profc__Z6throwsv]] = {{(private|internal)}} global [9 x i64] zeroinitializer
// PGOGEN: @[[UNC:__profc__Z11unreachablei]] = {{(private|internal)}} global [3 x i64] zeroinitializer

// PGOGEN-LABEL: @_Z6throwsv()
// PGOUSE-LABEL: @_Z6throwsv()
// PGOGEN: store {{.*}} @[[THC]]
void throws() {
  // PGOGEN: store {{.*}} @[[THC]], i32 0, i32 1
  // PGOUSE: br {{.*}} !prof ![[TH1:[0-9]+]]
  for (int i = 0; i < 100; ++i) {
    try {
      // PGOGEN: store {{.*}} @[[THC]], i32 0, i32 3
      // PGOUSE: br {{.*}} !prof ![[TH2:[0-9]+]]
      if (i % 3) {
        // PGOGEN: store {{.*}} @[[THC]], i32 0, i32 4
        // PGOUSE: br {{.*}} !prof ![[TH3:[0-9]+]]
        if (i < 50)
          throw 1;
      } else {
        // The catch block may be emitted after the throw above, we can skip it
        // by looking for an else block, but this will break if anyone puts an
        // else in the catch
        // PGOUSE: if.else{{.*}}:
        // PGOGEN: if.else{{.*}}:

        // PGOGEN: store {{.*}} @[[THC]], i32 0, i32 5
        // PGOUSE: br {{.*}} !prof ![[TH4:[0-9]+]]
        if (i >= 50)
          throw 0;
      }
    } catch (int e) {
      // PGOUSE-EXC: catch{{.*}}:
      // PGOGEN-EXC: catch{{.*}}:

      // PGOGEN-EXC: store {{.*}} @[[THC]], i32 0, i32 6
      // PGOGEN-EXC: store {{.*}} @[[THC]], i32 0, i32 7
      // PGOUSE-EXC: br {{.*}} !prof ![[TH5:[0-9]+]]
      if (e) {}
    }
    // PGOGEN: store {{.*}} @[[THC]], i32 0, i32 2

    // PGOGEN: store {{.*}} @[[THC]], i32 0, i32 8
    // PGOUSE: br {{.*}} !prof ![[TH6:[0-9]+]]
    if (i < 100) {}
  }

  // PGOUSE-NOT: br {{.*}} !prof ![0-9]+
  // PGOUSE: ret void
}

// PGOGEN-LABEL: @_Z11unreachablei(i32
// PGOUSE-LABEL: @_Z11unreachablei(i32
// PGOGEN: store {{.*}} @[[UNC]]
void unreachable(int i) {
  // PGOGEN: store {{.*}} @[[UNC]], i32 0, i32 1
  // PGOUSE: br {{.*}} !prof ![[UN1:[0-9]+]]
  if (i)
    throw i;

  // PGOGEN: store {{.*}} @[[UNC]], i32 0, i32 2
  // Since we never reach here, the weights should all be zero (and skipped)
  // PGOUSE-NOT: br {{.*}} !prof !{{.*}}
  if (i) {}
}

// PGOUSE-DAG: ![[TH1]] = !{!"branch_weights", i32 101, i32 2}
// PGOUSE-DAG: ![[TH2]] = !{!"branch_weights", i32 67, i32 35}
// PGOUSE-DAG: ![[TH3]] = !{!"branch_weights", i32 34, i32 34}
// PGOUSE-DAG: ![[TH4]] = !{!"branch_weights", i32 18, i32 18}
// PGOUSE-EXC: ![[TH5]] = !{!"branch_weights", i32 34, i32 18}
// PGOUSE-DAG: ![[TH6]] = !{!"branch_weights", i32 101, i32 1}
// PGOUSE-DAG: ![[UN1]] = !{!"branch_weights", i32 2, i32 1}

int main(int argc, const char *argv[]) {
  throws();
  try {
    unreachable(1);
  } catch (int) {}
  return 0;
}
