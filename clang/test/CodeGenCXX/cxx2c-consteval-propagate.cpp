// RUN: %clang_cc1 -emit-llvm %s -std=c++20 -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

namespace GH63742 {

void side_effect();
consteval int f(int x) {
    if (!__builtin_is_constant_evaluated()) side_effect();
    return x;
}
struct SS {
    int x = f(42);
    SS();
};
SS::SS(){}

}

// CHECK-LABEL: @_ZN7GH637422SSC2Ev
// CHECK-NOT:   call
// CHECK:       store i32 42, ptr {{.*}}
// CHECK:       ret void
