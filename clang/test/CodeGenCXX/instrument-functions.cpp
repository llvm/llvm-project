// RUN: %clang_cc1 -S -emit-llvm -o - %s -finstrument-functions | FileCheck %s

// CHECK: @_Z5test1i
int test1(int x) {
// CHECK: __cyg_profile_func_enter
// CHECK: __cyg_profile_func_exit
// CHECK: ret
  return x;
}

// CHECK: @_Z5test2i
int test2(int) __attribute__((no_instrument_function));
int test2(int x) {
// CHECK-NOT: __cyg_profile_func_enter
// CHECK-NOT: __cyg_profile_func_exit
// CHECK: ret
  return x;
}

// This test case previously crashed code generation.  It exists solely
// to test -finstrument-function does not crash codegen for this trivial
// case.
namespace rdar9445102 {
  class Rdar9445102 {
    public:
      Rdar9445102();
  };
}
static rdar9445102::Rdar9445102 s_rdar9445102Initializer;

