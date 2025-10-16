// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK

struct polymorphic_base {
    virtual void func() {}
    virtual ~polymorphic_base() {}
};

struct Empty {};
struct derived_virtual : virtual Empty {};
struct derived : polymorphic_base {};

// CHECK: %struct.Holder1 = type { %struct.polymorphic_base }
// CHECK: %struct.polymorphic_base = type { ptr }
// CHECK: %struct.Holder2 = type { %struct.derived_virtual }
// CHECK: %struct.derived_virtual = type { ptr }
// CHECK: %struct.Holder3 = type { %struct.derived }
// CHECK: %struct.derived = type { %struct.polymorphic_base }

struct Holder1 {
  polymorphic_base a{};
} g_holder1;

// CHECK: @{{.*}} = {{.*}}global %struct.Holder1 { %struct.polymorphic_base { ptr {{.*}} } }

struct Holder2 {
  derived_virtual a{};
} g_holder2;

// CHECK: @{{.*}} = {{.*}}global %struct.Holder2 zeroinitializer

struct Holder3 {
  derived a{};
} g_holder3;

// CHECK: @{{.*}} = {{.*}}global { { ptr } } { { ptr } { ptr {{.*}} } }
