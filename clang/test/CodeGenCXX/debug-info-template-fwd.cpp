// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -debug-info-kind=limited -emit-llvm -o - | FileCheck %s
// This test is for a crash when emitting debug info for not-yet-completed
// types.
// Test that we don't actually emit a forward decl for the offending class:
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Derived<int>"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}
// rdar://problem/15931354
template <class A> class Derived;

template <class A> class Base {
  static Derived<A> *create();
};

template <class A> struct Derived : Base<A> {
};

Base<int> *f;

// During the instantiation of Derived<int>, Base<int> becomes required to be
// complete - since the declaration has already been emitted (due to 'f',
// above), we immediately try to build debug info for Base<int> which then
// requires the (incomplete definition) of Derived<int> which is problematic.
//
// (if 'f' is not present, the point at which Base<int> becomes required to be
// complete during the instantiation of Derived<int> is a no-op because
// Base<int> was never emitted so we ignore it and carry on until we
// wire up the base class of Derived<int> in the debug info later on)
Derived<int> d;
