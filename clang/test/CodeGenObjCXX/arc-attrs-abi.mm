// RUN: %clang_cc1 -triple x86_64-apple -emit-llvm -fobjc-arc -o - %s
// RUN: %clang_cc1 -triple x86_64-windows -emit-llvm -fobjc-arc -o - %s
//
// Test caess where we weren't properly adding parameter infos declarations,
// which caused assertions to fire. Hence, no CHECKs.

struct VirtualBase {
  VirtualBase(__attribute__((ns_consumed)) id x);
};
struct WithVirtualBase : virtual VirtualBase {
  WithVirtualBase(__attribute__((ns_consumed)) id x);
};

WithVirtualBase::WithVirtualBase(__attribute__((ns_consumed)) id x)
    : VirtualBase(x) {}
