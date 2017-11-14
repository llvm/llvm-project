// RUN: %clang_cc1 -S -emit-llvm -triple %itanium_abi_triple -o - %s -finstrument-functions -disable-llvm-passes | FileCheck %s

int test1(int x) {
// CHECK: define i32 @_Z5test1i(i32 %x) #[[ATTR1:[0-9]+]]
// CHECK: ret
  return x;
}

int test2(int) __attribute__((no_instrument_function));
int test2(int x) {
// CHECK: define i32 @_Z5test2i(i32 %x) #[[ATTR2:[0-9]+]]
// CHECK: ret
  return x;
}

// CHECK: attributes #[[ATTR1]] =
// CHECK-SAME: "instrument-function-entry"="__cyg_profile_func_enter"
// CHECK-SAME: "instrument-function-exit"="__cyg_profile_func_exit"

// CHECK: attributes #[[ATTR2]] =
// CHECK-NOT: "instrument-function-entry"


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

