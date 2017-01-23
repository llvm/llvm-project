// RUN: %clang_cc1 -triple x86_64-unknown_unknown -emit-llvm -debug-info-kind=limited %s -O1 -o - | FileCheck %s

// Ensure class definitions are not emitted to debug info just because the
// vtable is emitted for optimization purposes (as available_externally). The
// class definition debug info should only go where the vtable is actually
// emitted into the object file.

// CHECK: @_ZTV3foo = available_externally
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "foo"
// CHECK-SAME: DIFlagFwdDecl

struct foo {
  virtual void f();
};

foo f;
