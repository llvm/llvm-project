// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O2 -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O2 -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// CIR-LABEL: cir.func {{.*}} @_Z4testv() -> !cir.bool
// CIR:       %[[RETVAL:.+]] = cir.alloca !cir.bool
// CIR:       %[[CONST_TRUE:.+]] = cir.const #true
// CIR:       cir.store{{.*}} %[[CONST_TRUE]], %[[RETVAL]]
// CIR:       %[[LOADED_VAL:.+]] = cir.load{{.*}} %[[RETVAL]]
// CIR:       cir.return %[[LOADED_VAL]]

// LLVM-LABEL: define dso_local {{.*}}i1 @_Z4testv()
// LLVM:         ret i1 true

namespace B {
template <class _0p> class B {
public:
  typedef _0p A;
  B() { __has_trivial_destructor(A); }
};
template <class _0p, class _0e0uence = B<_0p>> class A { _0e0uence A; };
} // namespace B

class A { public: B::A<A> A; };

bool test() {
  return __has_trivial_destructor(A);
}

