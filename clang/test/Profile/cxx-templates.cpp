// Tests for instrumentation of templated code. Each instantiation of a template
// should be instrumented separately.

// RUN: %clang_cc1 -x c++ %s -triple %itanium_abi_triple -main-file-name cxx-templates.cpp -std=c++11 -o - -emit-llvm -fprofile-instrument=clang > %tgen
// RUN: FileCheck --input-file=%tgen -check-prefix=T0GEN -check-prefix=ALL %s
// RUN: FileCheck --input-file=%tgen -check-prefix=T100GEN -check-prefix=ALL %s

// RUN: llvm-profdata merge %S/Inputs/cxx-templates.proftext -o %t.profdata
// RUN: %clang_cc1 -x c++ %s -triple %itanium_abi_triple -main-file-name cxx-templates.cpp -std=c++11 -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata > %tuse
// RUN: FileCheck --input-file=%tuse -check-prefix=T0USE -check-prefix=ALL %s
// RUN: FileCheck --input-file=%tuse -check-prefix=T100USE -check-prefix=ALL %s

// The linkage can be target dependent, so accept all linkage here,
// the linkage tests for different target are in llvm/test/Instrumentation/InstrProfiling/profiling.ll
// T0GEN: @[[T0C:__profc__Z4loopILj0EEvv]] = {{.*}} global [2 x i64] zeroinitializer
// T100GEN: @[[T100C:__profc__Z4loopILj100EEvv]] = {{.*}} global [2 x i64] zeroinitializer

// T0GEN-LABEL: define linkonce_odr {{.*}}void @_Z4loopILj0EEvv()
// T0USE-LABEL: define linkonce_odr {{.*}}void @_Z4loopILj0EEvv()
// T100GEN-LABEL: define linkonce_odr {{.*}}void @_Z4loopILj100EEvv()
// T100USE-LABEL: define linkonce_odr {{.*}}void @_Z4loopILj100EEvv()
template <unsigned N> void loop() {
  // ALL-NOT: ret
  // T0GEN: store {{.*}} @[[T0C]], i32 0, i32 0
  // T100GEN: store {{.*}} @[[T100C]], i32 0, i32 0

  // ALL-NOT: ret
  // T0GEN: store {{.*}} @[[T0C]], i32 0, i32 1
  // T0USE: br {{.*}} !prof ![[T01:[0-9]+]]
  // T100GEN: store {{.*}} @[[T100C]], i32 0, i32 1
  // T100USE: br {{.*}} !prof ![[T1001:[0-9]+]]
  for (unsigned I = 0; I < N; ++I) {}

  // ALL: ret
}

// T0USE-DAG: ![[T01]] = !{!"branch_weights", i32 1, i32 2}
// T100USE-DAG: ![[T1001]] = !{!"branch_weights", i32 101, i32 2}

int main(int argc, const char *argv[]) {
  loop<0>();
  loop<100>();
  return 0;
}
