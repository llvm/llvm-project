// Tests for instrumentation of C++11 lambdas

// RUN: %clang_cc1 -x c++ %s -triple %itanium_abi_triple -main-file-name cxx-lambda.cpp -std=c++11 -o - -emit-llvm -fprofile-instrument=clang > %tgen
// RUN: FileCheck -allow-deprecated-dag-overlap  --input-file=%tgen -check-prefix=PGOGEN %s
// RUN: FileCheck -allow-deprecated-dag-overlap  --input-file=%tgen -check-prefix=LMBGEN %s

// RUN: llvm-profdata merge %S/Inputs/cxx-lambda.proftext -o %t.profdata
// RUN: %clang_cc1 -x c++ %s -triple %itanium_abi_triple -main-file-name cxx-lambda.cpp -std=c++11 -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata > %tuse
// RUN: FileCheck -allow-deprecated-dag-overlap  --input-file=%tuse -check-prefix=PGOUSE %s
// RUN: FileCheck -allow-deprecated-dag-overlap  --input-file=%tuse -check-prefix=LMBUSE %s

// PGOGEN: @[[LWC:__profc__Z7lambdasv]] = {{(private|internal)}} global [4 x i64] zeroinitializer
// PGOGEN: @[[MAC:__profc_main]] = {{(private|internal)}} global [1 x i64] zeroinitializer
// LMBGEN: @[[LFC:"__profc_cxx_lambda.cpp__ZZ7lambdasvENK3\$_0clEi"]] = {{(private|internal)}} global [4 x i64] zeroinitializer

// PGOGEN-LABEL: define {{.*}}void @_Z7lambdasv()
// PGOUSE-LABEL: define {{.*}}void @_Z7lambdasv()
// PGOGEN: store {{.*}} @[[LWC]], i64 0, i64 0
void lambdas() {
  int i = 1;

  // LMBGEN-LABEL: define internal{{( [0-9_a-z]*cc)?( zeroext)?}} i1 @"_ZZ7lambdasvENK3$_0clEi"(
  // LMBUSE-LABEL: define internal{{( [0-9_a-z]*cc)?( zeroext)?}} i1 @"_ZZ7lambdasvENK3$_0clEi"(
  // LMBGEN: store {{.*}} @[[LFC]], i64 0, i64 0
  auto f = [&i](int k) {
    // LMBGEN: store {{.*}} @[[LFC]], i64 0, i64 1
    // LMBUSE: br {{.*}} !prof ![[LF1:[0-9]+]]
    if (i > 0) {}
    // LMBGEN: store {{.*}} @[[LFC]], i64 0, i64 2
    // LMBUSE: br {{.*}} !prof ![[LF2:[0-9]+]]
    return k && i;
  };

  // PGOGEN: store {{.*}} @[[LWC]], i64 0, i64 1
  // PGOUSE: br {{.*}} !prof ![[LW1:[0-9]+]]
  if (i) {}

  // PGOGEN: store {{.*}} @[[LWC]], i64 0, i64 2
  // PGOUSE: br {{.*}} !prof ![[LW2:[0-9]+]]
  for (i = 0; i < 10; ++i)
    f(9 - i);

  // PGOGEN: store {{.*}} @[[LWC]], i64 0, i64 3
  // PGOUSE: br {{.*}} !prof ![[LW3:[0-9]+]]
  if (i) {}
}

// PGOUSE-DAG: ![[LW1]] = !{!"branch_weights", i32 2, i32 1}
// PGOUSE-DAG: ![[LW2]] = !{!"branch_weights", i32 11, i32 2}
// PGOUSE-DAG: ![[LW3]] = !{!"branch_weights", i32 2, i32 1}

// LMBUSE-DAG: ![[LF1]] = !{!"branch_weights", i32 10, i32 2}
// LMBUSE-DAG: ![[LF2]] = !{!"branch_weights", i32 10, i32 2}

int main(int argc, const char *argv[]) {
  lambdas();
  return 0;
}
