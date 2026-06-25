// The exact dynamic_cast optimization is enabled for a weak vtable only on
// targets with unique vtables; for a class with a key function (a unique,
// externally defined vtable) it is also enabled where the optimization applies
// at all. The ENABLED/DISABLED prefixes track the weak vtable (class B); the
// KEY-ENABLED/KEY-DISABLED prefixes track the key-function class (WithKey).
//
// Baseline, unique vtables:
// RUN: %clang_cc1 -I%S %s -triple x86_64-unknown-linux-gnu -O1 -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK,ENABLED,KEY-ENABLED
// Disabled without optimization:
// RUN: %clang_cc1 -I%S %s -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK,DISABLED,KEY-DISABLED
// Disabled for a weak vtable with non-default visibility, but kept for the
// key-function class (its vtable is external):
// RUN: %clang_cc1 -I%S %s -triple x86_64-unknown-linux-gnu -O1 -fvisibility=hidden -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK,DISABLED,KEY-ENABLED
// Disabled under -fapple-kext:
// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -O1 -fapple-kext -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK,DISABLED,KEY-DISABLED
// Disabled by -fno-assume-unique-vtables:
// RUN: %clang_cc1 -I%S %s -triple x86_64-unknown-linux-gnu -O1 -fno-assume-unique-vtables -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK,DISABLED,KEY-DISABLED
// Disabled for a weak vtable on a target that may duplicate vtables (Apple
// Mach-O), but kept for the key-function class:
// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -O1 -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK,DISABLED,KEY-ENABLED

struct A { virtual ~A(); };
struct B final : A { };

// CHECK-LABEL: @_Z5exactP1A
B *exact(A *a) {
  // DISABLED: call {{.*}} @__dynamic_cast
  // ENABLED-NOT: call {{.*}} @__dynamic_cast
  return dynamic_cast<B*>(a);
}

struct C {
  virtual ~C();
};

struct D final : private C {

};

// CHECK-LABEL: @_Z5exactP1C
D *exact(C *a) {
  // DISABLED: call {{.*}} @__dynamic_cast
  // ENABLED: entry:
  // ENABLED-NEXT: ret ptr null
  return dynamic_cast<D*>(a);
}

// WithKey has a key function (the out-of-line g()), so its vtable has external
// linkage and a unique address even on a target that may duplicate vtables.
struct WithKey final : A { virtual void g(); };

// CHECK-LABEL: @_Z12cast_withkeyP1A
WithKey *cast_withkey(A *a) {
  // KEY-DISABLED: call {{.*}} @__dynamic_cast
  // KEY-ENABLED-NOT: call {{.*}} @__dynamic_cast
  // KEY-ENABLED: icmp eq ptr {{.*}}, {{.*}}@_ZTV7WithKey
  return dynamic_cast<WithKey*>(a);
}
