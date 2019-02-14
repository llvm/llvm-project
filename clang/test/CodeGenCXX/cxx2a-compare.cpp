// RUN: %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple %itanium_abi_triple | \
// RUN:    FileCheck %s \
// RUN:          '-DWE="class.std::__1::weak_equality"' \
// RUN:          '-DSO="class.std::__1::strong_ordering"' \
// RUN:          '-DSE="class.std::__1::strong_equality"' \
// RUN:          '-DPO="class.std::__1::partial_ordering"' \
// RUN:           -DEQ=0 -DLT=-1 -DGT=1 -DUNORD=-127 -DNE=1

#include "Inputs/std-compare.h"

// Ensure we don't emit definitions for the global variables
// since the builtins shouldn't ODR use them.
// CHECK-NOT: constant %[[SO]]
// CHECK-NOT: constant %[[SE]]
// CHECK-NOT: constant %[[WE]]
// CHECK-NOT: constant %[[PO]]

// CHECK-LABEL: @_Z11test_signedii
auto test_signed(int x, int y) {
  // CHECK: %[[DEST:retval|agg.result]]
  // CHECK: %cmp.lt = icmp slt i32 %0, %1
  // CHECK: %sel.lt = select i1 %cmp.lt, i8 [[LT]], i8 [[GT]]
  // CHECK: %cmp.eq = icmp eq i32 %0, %1
  // CHECK: %sel.eq = select i1 %cmp.eq, i8 [[EQ]], i8 %sel.lt
  // CHECK: %__value_ = getelementptr inbounds %[[SO]], %[[SO]]* %[[DEST]]
  // CHECK: store i8 %sel.eq, i8* %__value_, align 1
  // CHECK: ret
  return x <=> y;
}

// CHECK-LABEL: @_Z13test_unsignedjj
auto test_unsigned(unsigned x, unsigned y) {
  // CHECK: %[[DEST:retval|agg.result]]
  // CHECK: %cmp.lt = icmp ult i32 %0, %1
  // CHECK: %sel.lt = select i1 %cmp.lt, i8 [[LT]], i8 [[GT]]
  // CHECK: %cmp.eq = icmp eq i32 %0, %1
  // CHECK: %sel.eq = select i1 %cmp.eq, i8 [[EQ]], i8 %sel.lt
  // CHECK: %__value_ = getelementptr inbounds %[[SO]], %[[SO]]* %[[DEST]]
  // CHECK: store i8 %sel.eq, i8* %__value_
  // CHECK: ret
  return x <=> y;
}

// CHECK-LABEL: @_Z10float_testdd
auto float_test(double x, double y) {
  // CHECK: %[[DEST:retval|agg.result]]
  // CHECK: %cmp.eq = fcmp oeq double %0, %1
  // CHECK: %sel.eq = select i1 %cmp.eq, i8 [[EQ]], i8 [[UNORD]]
  // CHECK: %cmp.gt = fcmp ogt double %0, %1
  // CHECK: %sel.gt = select i1 %cmp.gt, i8 [[GT]], i8 %sel.eq
  // CHECK: %cmp.lt = fcmp olt double %0, %1
  // CHECK: %sel.lt = select i1 %cmp.lt, i8 [[LT]], i8 %sel.gt
  // CHECK: %__value_ = getelementptr inbounds %[[PO]], %[[PO]]* %[[DEST]]
  // CHECK: store i8 %sel.lt, i8* %__value_
  // CHECK: ret
  return x <=> y;
}

// CHECK-LABEL: @_Z8ptr_testPiS_
auto ptr_test(int *x, int *y) {
  // CHECK: %[[DEST:retval|agg.result]]
  // CHECK: %cmp.lt = icmp ult i32* %0, %1
  // CHECK: %sel.lt = select i1 %cmp.lt, i8 [[LT]], i8 [[GT]]
  // CHECK: %cmp.eq = icmp eq i32* %0, %1
  // CHECK: %sel.eq = select i1 %cmp.eq, i8 [[EQ]], i8 %sel.lt
  // CHECK: %__value_ = getelementptr inbounds %[[SO]], %[[SO]]* %[[DEST]]
  // CHECK: store i8 %sel.eq, i8* %__value_, align 1
  // CHECK: ret
  return x <=> y;
}

struct MemPtr {};
using MemPtrT = void (MemPtr::*)();
using MemDataT = int(MemPtr::*);

// CHECK-LABEL: @_Z12mem_ptr_testM6MemPtrFvvES1_
auto mem_ptr_test(MemPtrT x, MemPtrT y) {
  // CHECK: %[[DEST:retval|agg.result]]
  // CHECK: %cmp.ptr = icmp eq [[TY:i[0-9]+]] %lhs.memptr.ptr, %rhs.memptr.ptr
  // CHECK: %cmp.ptr.null = icmp eq [[TY]] %lhs.memptr.ptr, 0
  // CHECK: %cmp.adj = icmp eq [[TY]] %lhs.memptr.adj, %rhs.memptr.adj
  // CHECK: %[[OR:.*]] = or i1
  // CHECK-SAME: %cmp.adj
  // CHECK: %memptr.eq = and i1 %cmp.ptr, %[[OR]]
  // CHECK: %sel.eq = select i1 %memptr.eq, i8 [[EQ]], i8 [[NE]]
  // CHECK: %__value_ = getelementptr inbounds %[[SE]], %[[SE]]* %[[DEST]]
  // CHECK: store i8 %sel.eq, i8* %__value_, align 1
  // CHECK: ret
  return x <=> y;
}

// CHECK-LABEL: @_Z13mem_data_testM6MemPtriS0_
auto mem_data_test(MemDataT x, MemDataT y) {
  // CHECK: %[[DEST:retval|agg.result]]
  // CHECK: %[[CMP:.*]] = icmp eq i{{[0-9]+}} %0, %1
  // CHECK: %sel.eq = select i1 %[[CMP]], i8 [[EQ]], i8 [[NE]]
  // CHECK: %__value_ = getelementptr inbounds %[[SE]], %[[SE]]* %[[DEST]]
  // CHECK: store i8 %sel.eq, i8* %__value_, align 1
  return x <=> y;
}

// CHECK-LABEL: @_Z13test_constantv
auto test_constant() {
  // CHECK: %[[DEST:retval|agg.result]]
  // CHECK-NOT: icmp
  // CHECK: %__value_ = getelementptr inbounds %[[SO]], %[[SO]]* %[[DEST]]
  // CHECK-NEXT: store i8 -1, i8* %__value_
  // CHECK: ret
  const int x = 42;
  const int y = 101;
  return x <=> y;
}

// CHECK-LABEL: @_Z16test_nullptr_objPiDn
auto test_nullptr_obj(int* x, decltype(nullptr) y) {
  // CHECK: %[[DEST:retval|agg.result]]
  // CHECK: %cmp.eq = icmp eq i32* %0, null
  // CHECK: %sel.eq = select i1 %cmp.eq, i8 [[EQ]], i8 [[NE]]
  // CHECK: %__value_ = getelementptr inbounds %[[SE]], %[[SE]]* %[[DEST]]
  // CHECK: store i8 %sel.eq, i8* %__value_, align 1
  return x <=> y;
}

// CHECK-LABEL: @_Z18unscoped_enum_testijxy
void unscoped_enum_test(int i, unsigned u, long long l, unsigned long long ul) {
  enum EnumA : int { A };
  enum EnumB : unsigned { B };
  // CHECK: %[[I:.*]] = load {{.*}} %i.addr
  // CHECK: icmp slt i32 {{.*}} %[[I]]
  (void)(A <=> i);

  // CHECK: %[[U:.*]] = load {{.*}} %u.addr
  // CHECK: icmp ult i32 {{.*}} %[[U]]
  (void)(A <=> u);

  // CHECK: %[[L:.*]] = load {{.*}} %l.addr
  // CHECK: icmp slt i64 {{.*}} %[[L]]
  (void)(A <=> l);

  // CHECK: %[[U2:.*]] = load {{.*}} %u.addr
  // CHECK: icmp ult i32 {{.*}} %[[U2]]
  (void)(B <=> u);

  // CHECK: %[[UL:.*]] = load {{.*}} %ul.addr
  // CHECK: icmp ult i64 {{.*}} %[[UL]]
  (void)(B <=> ul);
}

namespace NullptrTest {
using nullptr_t = decltype(nullptr);

// CHECK-LABEL: @_ZN11NullptrTest4testEDnDn(
auto test(nullptr_t x, nullptr_t y) {
  // CHECK: %[[DEST:retval|agg.result]]
  // CHECK-NOT: select
  // CHECK: %__value_ = getelementptr inbounds %[[SE]], %[[SE]]* %[[DEST]]
  // CHECK-NEXT: store i8 [[EQ]], i8* %__value_
  // CHECK: ret
  return x <=> y;
}
} // namespace NullptrTest

namespace ComplexTest {

auto test_float(_Complex float x, _Complex float y) {
  // CHECK: %[[DEST:retval|agg.result]]
  // CHECK: %cmp.eq.r = fcmp oeq float %x.real, %y.real
  // CHECK: %cmp.eq.i = fcmp oeq float %x.imag, %y.imag
  // CHECK: %and.eq = and i1 %cmp.eq.r, %cmp.eq.i
  // CHECK: %sel.eq = select i1 %and.eq, i8 [[EQ]], i8 [[NE]]
  // CHECK: %__value_ = getelementptr inbounds %[[WE]], %[[WE]]* %[[DEST]]
  // CHECK: store i8 %sel.eq, i8* %__value_, align 1
  return x <=> y;
}

// CHECK-LABEL: @_ZN11ComplexTest8test_intECiS0_(
auto test_int(_Complex int x, _Complex int y) {
  // CHECK: %[[DEST:retval|agg.result]]
  // CHECK: %cmp.eq.r = icmp eq i32 %x.real, %y.real
  // CHECK: %cmp.eq.i = icmp eq i32 %x.imag, %y.imag
  // CHECK: %and.eq = and i1 %cmp.eq.r, %cmp.eq.i
  // CHECK: %sel.eq = select i1 %and.eq, i8 [[EQ]], i8 [[NE]]
  // CHECK: %__value_ = getelementptr inbounds %[[SE]], %[[SE]]* %[[DEST]]
  // CHECK: store i8 %sel.eq, i8* %__value_, align 1
  return x <=> y;
}

} // namespace ComplexTest
