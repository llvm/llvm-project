// RUN: %clang_cc1 -triple x86_64-unknown_unknown -emit-llvm -debug-info-kind=limited %s -O1 -disable-llvm-passes -o - | FileCheck %s

// Ensure class definitions are not emitted to debug info just because the
// vtable is emitted for optimization purposes (as available_externally). The
// class definition debug info should only go where the vtable is actually
// emitted into the object file.

// CHECK: @_ZTV3foo = available_externally

// Verify that this doesn't involve querying for the vtable of types that aren't
// dynamic (that would cause an assertion in the case below)

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "bar<int>"
template <typename> struct bar {};
extern template struct bar<int>;
bar<int> *p1;
bar<int> a;

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "foo"
// CHECK-SAME: DIFlagFwdDecl

struct foo {
  virtual void f();
};

foo f;
