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
