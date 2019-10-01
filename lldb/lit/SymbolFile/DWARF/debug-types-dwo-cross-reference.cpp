// Test that we can jump from a type unit in one dwo file into a type unit in a
// different dwo file.

// REQUIRES: lld

// RUN: %clang %s -target x86_64-pc-linux -fno-standalone-debug -g \
// RUN:   -fdebug-types-section -gsplit-dwarf -c -o %t1.o -DONE
// RUN: %clang %s -target x86_64-pc-linux -fno-standalone-debug -g \
// RUN:   -fdebug-types-section -gsplit-dwarf -c -o %t2.o -DTWO
// RUN: llvm-dwarfdump %t1.dwo -debug-types | FileCheck --check-prefix=ONEUNIT %s
// RUN: llvm-dwarfdump %t2.dwo -debug-types | FileCheck --check-prefix=ONEUNIT %s
// RUN: ld.lld %t1.o %t2.o -o %t
// RUN: %lldb %t -o "target var a b **b.a" -b | FileCheck %s

// ONEUNIT-COUNT-1: DW_TAG_type_unit

// CHECK:      (const A) a = (a = 42)
// CHECK:      (const B) b = {
// CHECK-NEXT:   a = 0x{{.*}}
// CHECK-NEXT: }
// CHECK:      (const A) **b.a = (a = 42)

struct A;

extern const A *a_ptr;
#ifdef ONE
struct A {
  int a = 42;
};
constexpr A a{};
const A *a_ptr = &a;
#else
struct B {
  const A **a;
};
extern constexpr B b{&a_ptr};
#endif
