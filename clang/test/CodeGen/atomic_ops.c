// XFAIL: hexagon,sparc
//        (due to not having native load atomic support)
// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple mips-linux-gnu -emit-llvm %s -o - | FileCheck %s

void foo(int x)
{
  _Atomic(int) i = 0;
  _Atomic(short) j = 0;
  // Check that multiply / divides on atomics produce a cmpxchg loop
  i *= 2;
  // CHECK: mul nsw i32
  // CHECK: {{(cmpxchg i32*|i1 @__atomic_compare_exchange\(i32 4,)}}
  i /= 2;
  // CHECK: sdiv i32
  // CHECK: {{(cmpxchg i32*|i1 @__atomic_compare_exchange\(i32 4, )}}
  j /= x;
  // CHECK: sdiv i32
  // CHECK: {{(cmpxchg i16*|i1 @__atomic_compare_exchange\(i32 2, )}}

}

extern _Atomic _Bool b;

_Bool bar() {
// CHECK-LABEL: @bar
// CHECK: %[[load:.*]] = load atomic i8, i8* @b seq_cst
// CHECK: %[[tobool:.*]] = trunc i8 %[[load]] to i1
// CHECK: ret i1 %[[tobool]]
  return b;
}

extern _Atomic(_Complex int) x;

void baz(int y) {
// CHECK-LABEL: @baz
// CHECK: {{store atomic|call void @__atomic_store}}
  x += y;
}

_Atomic(int) compound_add(_Atomic(int) in) {
// CHECK-LABEL: @compound_add
// CHECK: [[OLD:%.*]] = atomicrmw add i32* {{.*}}, i32 5 seq_cst
// CHECK: [[NEW:%.*]] = add i32 [[OLD]], 5
// CHECK: ret i32 [[NEW]]

  return (in += 5);
}

_Atomic(int) compound_sub(_Atomic(int) in) {
// CHECK-LABEL: @compound_sub
// CHECK: [[OLD:%.*]] = atomicrmw sub i32* {{.*}}, i32 5 seq_cst
// CHECK: [[NEW:%.*]] = sub i32 [[OLD]], 5
// CHECK: ret i32 [[NEW]]

  return (in -= 5);
}

_Atomic(int) compound_xor(_Atomic(int) in) {
// CHECK-LABEL: @compound_xor
// CHECK: [[OLD:%.*]] = atomicrmw xor i32* {{.*}}, i32 5 seq_cst
// CHECK: [[NEW:%.*]] = xor i32 [[OLD]], 5
// CHECK: ret i32 [[NEW]]

  return (in ^= 5);
}

_Atomic(int) compound_or(_Atomic(int) in) {
// CHECK-LABEL: @compound_or
// CHECK: [[OLD:%.*]] = atomicrmw or i32* {{.*}}, i32 5 seq_cst
// CHECK: [[NEW:%.*]] = or i32 [[OLD]], 5
// CHECK: ret i32 [[NEW]]

  return (in |= 5);
}

_Atomic(int) compound_and(_Atomic(int) in) {
// CHECK-LABEL: @compound_and
// CHECK: [[OLD:%.*]] = atomicrmw and i32* {{.*}}, i32 5 seq_cst
// CHECK: [[NEW:%.*]] = and i32 [[OLD]], 5
// CHECK: ret i32 [[NEW]]

  return (in &= 5);
}

_Atomic(int) compound_mul(_Atomic(int) in) {
// CHECK-LABEL: @compound_mul
// CHECK: cmpxchg i32* {{%.*}}, i32 {{%.*}}, i32 [[NEW:%.*]] seq_cst seq_cst
// CHECK: ret i32 [[NEW]]

  return (in *= 5);
}
