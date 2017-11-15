// RUN: %clang_cc1 -S -debug-info-kind=standalone -emit-llvm -o - %s -finstrument-functions -disable-llvm-passes | FileCheck %s

int test1(int x) {
// CHECK: @test1(i32 {{.*}}%x) #[[ATTR1:[0-9]+]]
// CHECK: ret
  return x;
}

int test2(int) __attribute__((no_instrument_function));
int test2(int x) {
// CHECK: @test2(i32 {{.*}}%x) #[[ATTR2:[0-9]+]]
// CHECK: ret
  return x;
}

// CHECK: attributes #[[ATTR1]] =
// CHECK-SAME: "instrument-function-entry"="__cyg_profile_func_enter"
// CHECK-SAME: "instrument-function-exit"="__cyg_profile_func_exit"

// CHECK: attributes #[[ATTR2]] =
// CHECK-NOT: "instrument-function-entry"
