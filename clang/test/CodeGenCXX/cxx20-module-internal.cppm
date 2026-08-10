// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

// internal-linkage symbol mangling is implementation defined.  Let's
// not mangle in the module attachment -- that unnecessarily bloats
// the symbols.

export module A;

// CHECK-DAG: void @_ZL6addonev(
static void addone() {}
// CHECK-DAG: @_ZL1x =
static int x = 5;

namespace {
// CHECK-DAG: void @_ZN12_GLOBAL__N_14frobEv(
void frob() {}
// CHECK-DAG: @_ZN12_GLOBAL__N_11yE =
int y = 2;
struct Bill {
  void F();
};
// CHECK-DAG: void @_ZN12_GLOBAL__N_14Bill1FEv(
void Bill::F() {}
} // namespace

// CHECK-DAG: void @_ZL4FrobPN12_GLOBAL__N_14BillE(
static void Frob(Bill *b) {
  if (b)
    b->F();
}

namespace N {
// CHECK-DAG: void @_ZN1NL5innerEv(
static void inner() {}
// CHECK-DAG: @_ZN1NL1zE
static int z = 3;
} // namespace N

// CHECK-DAG: void @_ZW1A6addsixv(
void addsix() {
  Frob(nullptr);
  frob();
  addone();
  void(x + y + N::z);
  N::inner();
}
