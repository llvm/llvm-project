// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Regression test: C++14 variable templates with non-constexpr
// constructors must not crash CIR with "ctorRegion failed to
// verify constraint: region with at most 1 blocks".

template<int N> class FixedInt {
public:
  static const int value = N;
  operator int() const { return value; }
  FixedInt() {}
};

template<int N>
static const FixedInt<N> fix{};

int test() {
  return fix<1> + fix<2>;
}

// CIR: cir.global "private" internal dso_local @_ZL3fixILi1EE
// CIR: cir.global "private" internal dso_local @_ZL3fixILi2EE
// CIR: cir.func {{.*}} @_Z4testv

// LLVM: @_ZL3fixILi1EE = internal global
// LLVM: @_ZL3fixILi2EE = internal global
// LLVM: define {{.*}} @_Z4testv

// OGCG: @_ZL3fixILi1EE = internal global
// OGCG: @_ZL3fixILi2EE = internal global
// OGCG: define {{.*}} @_Z4testv
