// Tests for instrumentation of C++17 statement initializers

// RUN: %clang_cc1 -x c++ %s -triple %itanium_abi_triple -main-file-name cxx-stmt-initializers.cpp -std=c++1z -o - -emit-llvm -fprofile-instrument=clang > %tgen
// RUN: FileCheck --input-file=%tgen -check-prefix=CHECK -check-prefix=PGOGEN %s

// PGOGEN: @[[SIC:__profc__Z11switch_initv]] = {{(private|internal)}} global [3 x i64] zeroinitializer
// PGOGEN: @[[IIC:__profc__Z7if_initv]] = {{(private|internal)}} global [3 x i64] zeroinitializer

// Note: We expect counters for the function entry block, the condition in the
// switch initializer, and the switch successor block.
//
// CHECK-LABEL: define {{.*}}void @_Z11switch_initv()
// PGOGEN: store {{.*}} @[[SIC]], i32 0, i32 0
void switch_init() {
  switch (int i = true ? 0 : 1; i) {}
  // PGOGEN: store {{.*}} @[[SIC]], i32 0, i32 2
  // PGOGEN: store {{.*}} @[[SIC]], i32 0, i32 1
}

// Note: We expect counters for the function entry block, the condition in the
// if initializer, and the if successor block.
//
// CHECK-LABEL: define {{.*}}void @_Z7if_initv()
// PGOGEN: store {{.*}} @[[IIC]], i32 0, i32 0
void if_init() {
  if (int i = true ? 0 : 1; i) {}
  // PGOGEN: store {{.*}} @[[IIC]], i32 0, i32 2
  // PGOGEN: store {{.*}} @[[IIC]], i32 0, i32 1
}
