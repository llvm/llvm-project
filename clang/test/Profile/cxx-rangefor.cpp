// Tests for instrumentation of C++11 range-for

// RUN: %clang_cc1 -x c++ %s -triple %itanium_abi_triple -main-file-name cxx-rangefor.cpp -std=c++11 -o - -emit-llvm -fprofile-instrument=clang > %tgen
// RUN: FileCheck --input-file=%tgen -check-prefix=CHECK -check-prefix=PGOGEN %s

// RUN: llvm-profdata merge %S/Inputs/cxx-rangefor.proftext -o %t.profdata
// RUN: %clang_cc1 -x c++ %s -triple %itanium_abi_triple -main-file-name cxx-rangefor.cpp -std=c++11 -o - -emit-llvm -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata > %tuse
// RUN: FileCheck --input-file=%tuse -check-prefix=CHECK -check-prefix=PGOUSE %s

// PGOGEN: @[[RFC:__profc__Z9range_forv]] = {{(private|internal)}} global [5 x i64] zeroinitializer

// CHECK-LABEL: define {{.*}}void @_Z9range_forv()
// PGOGEN: store {{.*}} @[[RFC]]
void range_for() {
  int arr[] = {1, 2, 3, 4, 5};
  int sum = 0;
  // PGOGEN: store {{.*}} @[[RFC]], i32 0, i32 1
  // PGOUSE: br {{.*}} !prof ![[RF1:[0-9]+]]
  for (auto i : arr) {
    // PGOGEN: store {{.*}} @[[RFC]], i32 0, i32 2
    // PGOUSE: br {{.*}} !prof ![[RF2:[0-9]+]]
    if (i == 3)
      continue;
    sum += i;
    // PGOGEN: store {{.*}} @[[RFC]], i32 0, i32 3
    // PGOUSE: br {{.*}} !prof ![[RF3:[0-9]+]]
    if (sum >= 7)
      break;
  }

  // PGOGEN: store {{.*}} @[[RFC]], i32 0, i32 4
  // PGOUSE: br {{.*}} !prof ![[RF4:[0-9]+]]
  if (sum) {}
}

// PGOUSE-DAG: ![[RF1]] = !{!"branch_weights", i32 5, i32 1}
// PGOUSE-DAG: ![[RF2]] = !{!"branch_weights", i32 2, i32 4}
// PGOUSE-DAG: ![[RF3]] = !{!"branch_weights", i32 2, i32 3}
// PGOUSE-DAG: ![[RF4]] = !{!"branch_weights", i32 2, i32 1}

int main(int argc, const char *argv[]) {
  range_for();
  return 0;
}
