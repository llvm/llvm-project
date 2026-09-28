// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s
// RUN: %clang_cc1 -O0 -mconstructor-aliases -emit-llvm %s -o - \
// RUN:   -triple powerpc64-ibm-aix-xcoff \
// RUN:   | FileCheck %s --check-prefixes=XCOFF
// RUN: %clang_cc1 -O2 -mconstructor-aliases -emit-llvm %s -o - \
// RUN:   -triple powerpc64-ibm-aix-xcoff \
// RUN:   | FileCheck %s --check-prefixes=XCOFF
// RUN: %clang_cc1 -O0 -mconstructor-aliases -emit-llvm %s -o - \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   | FileCheck %s --check-prefixes=ELF
// RUN: %clang_cc1 -O0 -emit-llvm %s -o - \
// RUN:   -triple powerpc64-ibm-aix-xcoff \
// RUN:   | FileCheck %s --check-prefixes=XCOFF-NOALIAS

// clang++ not respecting __attribute__((used)) on destructors
struct X0 {
  // CHECK-DAG: define linkonce_odr {{.*}} @_ZN2X0C1Ev
  __attribute__((used)) X0() {}
  // CHECK-DAG: define linkonce_odr {{.*}} @_ZN2X0D1Ev
  __attribute__((used)) ~X0() {}
};

// PR19743: not emitting __attribute__((used)) inline methods in nested classes.
struct X1 {
  struct Nested {
    // CHECK-DAG: define linkonce_odr {{.*}} @_ZN2X16Nested1fEv
    void __attribute__((used)) f() {}
  };
};

struct X2 {
  // We must delay emission of bar() until foo() has had its body parsed,
  // otherwise foo() would not be emitted.
  void __attribute__((used)) bar() { foo(); }
  void foo() { }

  // CHECK-DAG: define linkonce_odr {{.*}} @_ZN2X23barEv
  // CHECK-DAG: define linkonce_odr {{.*}} @_ZN2X23fooEv
};

// Test that __attribute__((used)) on a constructor/destructor retains the
// C1/D1 complete variants when -mconstructor-aliases is active.
//
// Without the fix, -mconstructor-aliases causes C1/D1 to be silently replaced
// in the IR (RAUW) before SetCommonAttributes can add them to llvm.used, so
// __attribute__((used)) does not work as expected.

struct Foo {
  __attribute__((used)) Foo() {}
  __attribute__((used)) ~Foo() {}
};

namespace {
struct Bar {
  __attribute__((used)) Bar() {}
  __attribute__((used)) ~Bar() {}
};
} // namespace


// On XCOFF, Foo's constructors/destructors have linkonce_odr linkage. C1/D1
// are emitted as full definitions since linkonce_odr does not guarantee that
// an alias and its target will be retained from the same translation unit.
// Bar's C1/D1 have internal linkage, so they are confined to the translation
// unit and can safely be aliases to C2/D2.
// XCOFF-DAG: define linkonce_odr {{.*}}@_ZN3FooC1Ev
// XCOFF-DAG: define linkonce_odr {{.*}}@_ZN3FooC2Ev
// XCOFF-DAG: define linkonce_odr {{.*}}@_ZN3FooD1Ev
// XCOFF-DAG: define linkonce_odr {{.*}}@_ZN3FooD2Ev
// XCOFF-DAG: @_ZN12_GLOBAL__N_13BarC1Ev = internal {{.*}}alias{{.*}}@_ZN12_GLOBAL__N_13BarC2Ev
// XCOFF-DAG: @_ZN12_GLOBAL__N_13BarD1Ev = internal {{.*}}alias{{.*}}@_ZN12_GLOBAL__N_13BarD2Ev
// XCOFF-DAG: define internal {{.*}}@_ZN12_GLOBAL__N_13BarC2Ev
// XCOFF-DAG: define internal {{.*}}@_ZN12_GLOBAL__N_13BarD2Ev

// XCOFF without -mconstructor-aliases: all variants are full definitions.
// XCOFF-NOALIAS-DAG: define linkonce_odr {{.*}}@_ZN3FooC1Ev
// XCOFF-NOALIAS-DAG: define linkonce_odr {{.*}}@_ZN3FooC2Ev
// XCOFF-NOALIAS-DAG: define linkonce_odr {{.*}}@_ZN3FooD1Ev
// XCOFF-NOALIAS-DAG: define linkonce_odr {{.*}}@_ZN3FooD2Ev
// XCOFF-NOALIAS-DAG: define internal {{.*}}@_ZN12_GLOBAL__N_13BarC1Ev
// XCOFF-NOALIAS-DAG: define internal {{.*}}@_ZN12_GLOBAL__N_13BarC2Ev
// XCOFF-NOALIAS-DAG: define internal {{.*}}@_ZN12_GLOBAL__N_13BarD1Ev
// XCOFF-NOALIAS-DAG: define internal {{.*}}@_ZN12_GLOBAL__N_13BarD2Ev

// On ELF, C1/D1 are aliases to C2/D2 for both Foo and Bar.
// ELF-DAG: @_ZN3FooC1Ev = {{.*}}alias{{.*}}@_ZN3FooC2Ev
// ELF-DAG: @_ZN3FooD1Ev = {{.*}}alias{{.*}}@_ZN3FooD2Ev
// ELF-DAG: define {{.*}}@_ZN3FooC2Ev
// ELF-DAG: define {{.*}}@_ZN3FooD2Ev
// ELF-DAG: @_ZN12_GLOBAL__N_13BarC1Ev = internal {{.*}}alias{{.*}}@_ZN12_GLOBAL__N_13BarC2Ev
// ELF-DAG: @_ZN12_GLOBAL__N_13BarD1Ev = internal {{.*}}alias{{.*}}@_ZN12_GLOBAL__N_13BarD2Ev
// ELF-DAG: define internal {{.*}}@_ZN12_GLOBAL__N_13BarC2Ev
// ELF-DAG: define internal {{.*}}@_ZN12_GLOBAL__N_13BarD2Ev
