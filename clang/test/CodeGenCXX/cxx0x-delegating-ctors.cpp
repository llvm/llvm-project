// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -fexceptions -fcxx-exceptions -std=c++11 -o - %s | FileCheck %s

struct non_trivial {
  non_trivial();
  ~non_trivial() noexcept(false);
};
non_trivial::non_trivial() {}
non_trivial::~non_trivial() noexcept(false) {}

// We use a virtual base to ensure that the constructor
// delegation optimization (complete->base) can't be
// performed.
struct delegator {
  non_trivial n; 
  delegator();
  delegator(int);
  delegator(char);
  delegator(bool);
};

delegator::delegator() {
  throw 0;
}


delegator::delegator(bool)
{}

// CHECK: define {{.*}} @_ZN9delegatorC1Ec
// CHECK: {{.*}} @_ZN9delegatorC1Eb
// CHECK: void @__cxa_throw
// CHECK: void @_ZSt9terminatev
// CHECK: {{.*}} @_ZN9delegatorD1Ev
// CHECK: define {{.*}} @_ZN9delegatorC2Ec
// CHECK: {{.*}} @_ZN9delegatorC2Eb
// CHECK: void @__cxa_throw
// CHECK: void @_ZSt9terminatev
// CHECK: {{.*}} @_ZN9delegatorD2Ev
delegator::delegator(char)
  : delegator(true) {
  throw 0;
}

// CHECK: define {{.*}} @_ZN9delegatorC1Ei
// CHECK: {{.*}} @_ZN9delegatorC1Ev
// CHECK-NOT: void @_ZSt9terminatev
// CHECK: ret
// CHECK-NOT: void @_ZSt9terminatev
// CHECK: define {{.*}} @_ZN9delegatorC2Ei
// CHECK: {{.*}} @_ZN9delegatorC2Ev
// CHECK-NOT: void @_ZSt9terminatev
// CHECK: ret
// CHECK-NOT: void @_ZSt9terminatev
delegator::delegator(int)
  : delegator()
{}

namespace PR12890 {
  class X {
    int x;
    X() = default;
    X(int);
  };
  X::X(int) : X() {}
}
// CHECK: define {{.*}} @_ZN7PR128901XC1Ei(%"class.PR12890::X"* %this, i32)
// CHECK: call void @llvm.memset.p0i8.{{i32|i64}}(i8* {{.*}}, i8 0, {{i32|i64}} 4, i32 4, i1 false)
