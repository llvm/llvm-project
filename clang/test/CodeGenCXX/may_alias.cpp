// RUN: %clang_cc1 %s -triple %itanium_abi_triple -emit-llvm -O2 -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple %ms_abi_triple -emit-llvm -O2 -disable-llvm-passes -o - | FileCheck %s

enum class __attribute__((may_alias)) E {};

template<typename T> struct A {
  using B __attribute__((may_alias)) = enum {};
};

template<typename T> using Alias = typename A<T>::B;

// CHECK-LABEL: define {{.*}}foo
// CHECK: load i{{[0-9]*}}, {{.*}}, !tbaa ![[MAY_ALIAS:[^ ,]*]]
auto foo(E &r) { return r; }

// CHECK-LABEL: define {{.*}}goo
// CHECK: load i{{[0-9]*}}, {{.*}}, !tbaa ![[MAY_ALIAS]]
auto goo(A<int>::B &r) { return r; }

// CHECK-LABEL: define {{.*}}hoo
// CHECK: load i{{[0-9]*}}, {{.*}}, !tbaa ![[MAY_ALIAS]]
auto hoo(Alias<int> &r) { return r; }

// CHECK: ![[CHAR:.*]] = !{!"omnipotent char", !{{.*}}, i64 0}
// CHECK: ![[MAY_ALIAS]] = !{![[CHAR]], ![[CHAR]], i64 0}
