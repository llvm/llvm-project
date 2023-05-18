// RUN: %clang_cc1 -triple x86_64-apple -emit-llvm -fobjc-arc -o - %s -std=c++11 | FileCheck %s --check-prefix=ITANIUM
// RUN: %clang_cc1 -triple x86_64-windows -emit-llvm -fobjc-arc -o - %s -std=c++11
//
// Test cases where we weren't properly adding extended parameter info, which
// caused assertions to fire. Hence, minimal CHECKs.

struct VirtualBase {
  VirtualBase(__attribute__((ns_consumed)) id x,
              void * __attribute__((pass_object_size(0))));
};
struct WithVirtualBase : virtual VirtualBase {
  WithVirtualBase(__attribute__((ns_consumed)) id x);
};

WithVirtualBase::WithVirtualBase(__attribute__((ns_consumed)) id x)
    : VirtualBase(x, (void *)0) {}


struct VirtualBase2 {
  VirtualBase2(__attribute__((ns_consumed)) id x, void *y);
};

// In this case, we don't actually end up passing the `id` param from
// WithVirtualBaseLast's ctor to WithVirtualBaseMid's. So, we shouldn't emit
// ext param info for `id` to `Mid`. Itanium-only check since MSABI seems to
// emit the construction code inline.
struct WithVirtualBaseMid : virtual VirtualBase2 {
  // Ensure we only pass in `this` and a vtable. Otherwise this test is useless.
  // ITANIUM: define {{.*}} void @_ZN18WithVirtualBaseMidCI212VirtualBase2EP11objc_objectPv({{[^,]*}}, {{[^,]*}})
  using VirtualBase2::VirtualBase2;
};
struct WithVirtualBaseLast : WithVirtualBaseMid {
  using WithVirtualBaseMid::WithVirtualBaseMid;
};

void callLast(__attribute__((ns_consumed)) id x) {
  WithVirtualBaseLast{x, (void*)0};
}
