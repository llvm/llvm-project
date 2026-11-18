// RUN: %clang_cc1 -triple %itanium_abi_triple -O1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
//
// A polymorphic class stores a vtable pointer at offset 0. Its TBAA type
// descriptor must model that pointer as a member, so a vtable-pointer store
// reconciles with the object's own type instead of looking like an aliasing
// violation.

struct A {
  virtual ~A();
  char byte;
};

// Reading a member forces A's struct-path base-type node to be emitted.
char getByte(A *a) { return a->byte; }

// A's base-type node lists the vtable pointer at offset 0, then 'byte' at 8:
// CHECK: = !{!"_ZTS1A", [[VPTR:![0-9]+]], i64 0, {{![0-9]+}}, i64 8}
// CHECK: [[VPTR]] = !{!"vtable pointer", {{.*}}, i64 0}
