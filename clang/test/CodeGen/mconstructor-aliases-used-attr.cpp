// Test that __attribute__((used)) on a constructor/destructor retains the
// C1/D1 complete variants when -mconstructor-aliases is active.
//
// Without the fix, -mconstructor-aliases causes C1/D1 to be silently replaced
// in the IR (RAUW) before SetCommonAttributes can add them to llvm.used, so
// __attribute__((used)) does not work as expected.

// RUN: %clang_cc1 -O0 -mconstructor-aliases -emit-llvm %s -o - \
// RUN:   -triple powerpc64-ibm-aix-xcoff \
// RUN:   | FileCheck %s --check-prefixes=XCOFF,USED
// RUN: %clang_cc1 -O2 -mconstructor-aliases -emit-llvm %s -o - \
// RUN:   -triple powerpc64-ibm-aix-xcoff \
// RUN:   | FileCheck %s --check-prefixes=XCOFF,USED
// RUN: %clang_cc1 -O0 -mconstructor-aliases -emit-llvm %s -o - \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   | FileCheck %s --check-prefixes=ELF,USED

// Without -mconstructor-aliases the complete variants are always emitted as
// separate definitions and the normal path handles __attribute__((used)).
// RUN: %clang_cc1 -O0 -emit-llvm %s -o - \
// RUN:   -triple powerpc64-ibm-aix-xcoff \
// RUN:   | FileCheck %s --check-prefixes=XCOFF,USED

struct Foo {
  __attribute__((used)) Foo() {}
  __attribute__((used)) ~Foo() {}
};
// All four variants must appear in llvm.used/llvm.compiler.used.
// USED: @llvm{{(\.compiler)?}}.used = appending global [4 x ptr]

// On XCOFF, C1/D1 are full definitions
// XCOFF-DAG: define {{.*}}@_ZN3FooC1Ev
// XCOFF-DAG: define {{.*}}@_ZN3FooC2Ev
// XCOFF-DAG: define {{.*}}@_ZN3FooD1Ev
// XCOFF-DAG: define {{.*}}@_ZN3FooD2Ev

// On ELF, C1/D1 are aliases to C2/D2
// ELF-DAG: @_ZN3FooC1Ev = {{.*}}alias{{.*}}@_ZN3FooC2Ev
// ELF-DAG: @_ZN3FooD1Ev = {{.*}}alias{{.*}}@_ZN3FooD2Ev
// ELF-DAG: define {{.*}}@_ZN3FooC2Ev
// ELF-DAG: define {{.*}}@_ZN3FooD2Ev
