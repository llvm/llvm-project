// RUN: %clang_cc1 -S -debug-info-kind=standalone -emit-llvm -o - %s -finstrument-functions | FileCheck %s

// CHECK: @test1
int test1(int x) {
// CHECK: call void @__cyg_profile_func_enter({{.*}}, !dbg
// CHECK: call void @__cyg_profile_func_exit({{.*}}, !dbg
// CHECK: ret
  return x;
}

// CHECK: @test2
int test2(int) __attribute__((no_instrument_function));
int test2(int x) {
// CHECK-NOT: __cyg_profile_func_enter
// CHECK-NOT: __cyg_profile_func_exit
// CHECK: ret
  return x;
}
