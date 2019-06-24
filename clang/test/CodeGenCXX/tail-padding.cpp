// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

// PR36992
namespace Implicit {
  struct A { char c; A(const A&); };
  struct B { int n; char c[3]; ~B(); };
  struct C : B, virtual A {};
  static_assert(sizeof(C) == sizeof(void*) + 8);
  C f(C c) { return c; }

  // CHECK: define {{.*}} @_ZN8Implicit1CC1EOS0_
  // CHECK: call {{.*}} @_ZN8Implicit1AC2ERKS0_(
  // Note: this must memcpy 7 bytes, not 8, to avoid trampling over the virtual base class.
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i{{32|64}}(i8* {{.*}}, i8* {{.*}}, i{{32|64}} 7, i1 false)
  // CHECK: store i32 {{.*}} @_ZTVN8Implicit1CE
}

namespace InitWithinNVSize {
  // This is the same as the previous test, except that the A base lies
  // entirely within the nvsize of C. This makes it valid to copy at the
  // full width.
  struct A { char c; A(const A&); };
  struct B { int n; char c[3]; ~B(); };
  struct C : B, virtual A { char x; };
  static_assert(sizeof(C) > sizeof(void*) + 8);
  C f(C c) { return c; }

  // CHECK: define {{.*}} @_ZN16InitWithinNVSize1CC1EOS0_
  // CHECK: call {{.*}} @_ZN16InitWithinNVSize1AC2ERKS0_(
  // This copies over the 'C::x' member, but that's OK because we've not initialized it yet.
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i{{32|64}}(i8* {{.*}}, i8* {{.*}}, i{{32|64}} 8, i1 false)
  // CHECK: store i32 {{.*}} @_ZTVN16InitWithinNVSize1CE
  // CHECK: store i8
}

namespace NoUniqueAddr {
  struct A { char c; A(const A&); };
  struct B { int n; char c[3]; ~B(); };
  struct C : virtual A { B b; };
  struct D : virtual A { [[no_unique_address]] B b; };
  struct E : virtual A { [[no_unique_address]] B b; char x; };
  static_assert(sizeof(C) == sizeof(void*) + 8 + alignof(void*));
  static_assert(sizeof(D) == sizeof(void*) + 8);
  static_assert(sizeof(E) == sizeof(void*) + 8 + alignof(void*));

  // CHECK: define {{.*}} @_ZN12NoUniqueAddr1CC1EOS0_
  // CHECK: call {{.*}} @_ZN12NoUniqueAddr1AC2ERKS0_(
  // CHECK: store i32 {{.*}} @_ZTVN12NoUniqueAddr1CE
  // Copy the full size of B.
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i{{32|64}}(i8* {{.*}}, i8* {{.*}}, i{{32|64}} 8, i1 false)
  C f(C c) { return c; }

  // CHECK: define {{.*}} @_ZN12NoUniqueAddr1DC1EOS0_
  // CHECK: call {{.*}} @_ZN12NoUniqueAddr1AC2ERKS0_(
  // CHECK: store i32 {{.*}} @_ZTVN12NoUniqueAddr1DE
  // Copy just the data size of B, to avoid overwriting the A base class.
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i{{32|64}}(i8* {{.*}}, i8* {{.*}}, i{{32|64}} 7, i1 false)
  D f(D d) { return d; }

  // CHECK: define {{.*}} @_ZN12NoUniqueAddr1EC1EOS0_
  // CHECK: call {{.*}} @_ZN12NoUniqueAddr1AC2ERKS0_(
  // CHECK: store i32 {{.*}} @_ZTVN12NoUniqueAddr1EE
  // We can copy the full size of B here. (As it happens, we fold the copy of 'x' into
  // this memcpy, so we're copying 8 bytes either way.)
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i{{32|64}}(i8* {{.*}}, i8* {{.*}}, i{{32|64}} 8, i1 false)
  E f(E e) { return e; }

  struct F : virtual A {
    F(const F &o) : A(o), b(o.b) {}
    [[no_unique_address]] B b;
  };

  // CHECK: define {{.*}} @_ZN12NoUniqueAddr1FC1ERKS0_
  // CHECK: call {{.*}} @_ZN12NoUniqueAddr1AC2ERKS0_(
  // CHECK: store i32 {{.*}} @_ZTVN12NoUniqueAddr1FE
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i{{32|64}}(i8* {{.*}}, i8* {{.*}}, i{{32|64}} 7, i1 false)
  F f(F x) { return x; }
}
