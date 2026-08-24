// -fkeep-inline-functions has an effect without optimization, although it is
// primarily useful with optimization enabled.

// RUN: %clang_cc1 -O2 -fkeep-inline-functions -emit-llvm %s -o - -triple x86_64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -O0 -fkeep-inline-functions -emit-llvm %s -o - -triple x86_64-unknown-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -O2 -fkeep-inline-functions -emit-llvm %s -o - -triple powerpc64-ibm-aix-xcoff | FileCheck %s
// RUN: %clang_cc1 -O0 -fkeep-inline-functions -emit-llvm %s -o - -triple powerpc64-ibm-aix-xcoff | FileCheck %s

// -fkeep-inline-functions retains inline function definitions available in
// this translation unit. Definitions emitted with available_externally
// linkage are excluded.

// Retained:
//   f1  explicit inline and referenced
//   f2  static inline
//   f3  constexpr (implicit inline)
//   f4  in-class member definition (implicit inline)
//   f7  explicit inline and unreferenced
//   f8  explicitly instantiated inline template
//   TestCtorDtor implicitly inline constructor/destructor variants.

//   Also exercises the GlobalDecl construction path in MustBeEmitted() for
//   constructors/destructors. Without the special handling, debug builds hit
//   the GlobalDecl(FunctionDecl*) assertion.

// Not retained:
//   f5  non-inline
//   f6  GNU extern inline
//   f9  extern template specialization

inline int f1(int x) { return x + 1; }

static inline int f2(int x) { return x + 2; }

constexpr int f3(int x) { return x + 3; }

struct S {
  int f4() { return 4; }
};

int f5(int x) { return x + 5; }

__attribute__((gnu_inline)) extern inline int f6(int x) { return x + 6; }

inline int f7(int x) { return x + 7; }

template <typename T> inline T f8(T x) { return x + 8; }
template int f8<int>(int);

template <typename T>
struct A {
  static int f9() { return 9; }
};
extern template int A<int>::f9();

// Implicitly inline ctor and dtor intentionally unreferenced so
// MustBeEmitted() is the sole mechanism that forces their emission.
struct TestCtorDtor {
  TestCtorDtor() {}
  ~TestCtorDtor() {}
};

int use(S s) {
  return f1(0) + f5(0) + f6(0) + A<int>::f9();
}
// CHECK: @llvm{{(\.compiler)?}}.used = appending global [10 x ptr]

// CHECK-DAG: define {{.*}}@_Z2f1i
// CHECK-DAG: define internal {{.*}}@_ZL2f2i
// CHECK-DAG: define {{.*}}@_Z2f3i
// CHECK-DAG: define {{.*}}@_ZN1S2f4Ev
// CHECK-DAG: define {{.*}}@_Z2f7i
// CHECK-DAG: define {{.*}}@_Z2f8IiET_S0_
// CHECK-DAG: define {{.*}}@_ZN12TestCtorDtorC1Ev
// CHECK-DAG: define {{.*}}@_ZN12TestCtorDtorD1Ev
// CHECK-DAG: define {{.*}}@_ZN12TestCtorDtorC2Ev
// CHECK-DAG: define {{.*}}@_ZN12TestCtorDtorD2Ev

