// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s \
// RUN:   -o - | FileCheck -check-prefixes=CHECK,NATIVE %s
// RUN: %clang_cc1 -triple riscv32 -target-feature -a -emit-llvm %s \
// RUN:   -o - | FileCheck -check-prefixes=CHECK,LIBCALL %s

void foo(int x)
{
  _Atomic(int) i = 0;
  _Atomic(short) j = 0;
  // Check that multiply / divides on atomics produce a cmpxchg loop
  i *= 2;
  // NATIVE: mul nsw i32
  // NATIVE: cmpxchg ptr {{.*}} seq_cst, align 4
  // LIBCALL: mul nsw i32
  // LIBCALL: i1 @__atomic_compare_exchange(i32 noundef 4,
  i /= 2;
  // NATIVE: sdiv i32
  // NATIVE: cmpxchg ptr {{.*}} seq_cst, align 4
  // LIBCALL: sdiv i32
  // LIBCALL: i1 @__atomic_compare_exchange(i32 noundef 4,
  j /= x;
  // NATIVE: sdiv i32
  // NATIVE: cmpxchg ptr {{.*}} seq_cst, align 2
  // LIBCALL: sdiv i32
  // LIBCALL: i1 @__atomic_compare_exchange(i32 noundef 2,

}

// LIBCALL: declare void @__atomic_load(i32, ptr, ptr, i32) [[LC_ATTRS:#[0-9]+]]
// LIBCALL: declare i1 @__atomic_compare_exchange(i32, ptr, ptr, ptr, i32, i32) [[LC_ATTRS:#[0-9]+]]

extern _Atomic _Bool b;

_Bool bar(void) {
// NATIVE-LABEL: @bar
// NATIVE: %[[load:.*]] = load atomic i8, ptr @b seq_cst, align 1
// NATIVE: %[[tobool:.*]] = trunc i8 %[[load]] to i1
// NATIVE: ret i1 %[[tobool]]
// LIBCALL-LABEL: @bar
// LIBCALL: call void @__atomic_load(i32 noundef 1, ptr noundef @b, ptr noundef %atomic-temp, i32 noundef 5)
// LIBCALL: %[[load:.*]] = load i8, ptr %atomic-temp
// LIBCALL: %[[tobool:.*]] = trunc i8 %[[load]] to i1
// LIBCALL: ret i1 %[[tobool]]

  return b;
}

extern _Atomic(_Complex int) x;

void baz(int y) {
// NATIVE-LABEL: @baz
// NATIVE: store atomic i64 {{.*}} seq_cst, align 8
// LIBCALL-LABEL: @baz
// LIBCALL: call void @__atomic_store

  x += y;
}

// LIBCALL: declare void @__atomic_store(i32, ptr, ptr, i32) [[LC_ATTRS:#[0-9]+]]

_Atomic(int) compound_add(_Atomic(int) in) {
// CHECK-LABEL: @compound_add
// CHECK: [[OLD:%.*]] = atomicrmw add ptr {{.*}}, i32 5 seq_cst, align 4
// CHECK: [[NEW:%.*]] = add i32 [[OLD]], 5
// CHECK: ret i32 [[NEW]]

  return (in += 5);
}

_Atomic(int) compound_sub(_Atomic(int) in) {
// CHECK-LABEL: @compound_sub
// CHECK: [[OLD:%.*]] = atomicrmw sub ptr {{.*}}, i32 5 seq_cst, align 4
// CHECK: [[NEW:%.*]] = sub i32 [[OLD]], 5
// CHECK: ret i32 [[NEW]]

  return (in -= 5);
}

_Atomic(int) compound_xor(_Atomic(int) in) {
// CHECK-LABEL: @compound_xor
// CHECK: [[OLD:%.*]] = atomicrmw xor ptr {{.*}}, i32 5 seq_cst, align 4
// CHECK: [[NEW:%.*]] = xor i32 [[OLD]], 5
// CHECK: ret i32 [[NEW]]

  return (in ^= 5);
}

_Atomic(int) compound_or(_Atomic(int) in) {
// CHECK-LABEL: @compound_or
// CHECK: [[OLD:%.*]] = atomicrmw or ptr {{.*}}, i32 5 seq_cst, align 4
// CHECK: [[NEW:%.*]] = or i32 [[OLD]], 5
// CHECK: ret i32 [[NEW]]

  return (in |= 5);
}

_Atomic(int) compound_and(_Atomic(int) in) {
// CHECK-LABEL: @compound_and
// CHECK: [[OLD:%.*]] = atomicrmw and ptr {{.*}}, i32 5 seq_cst, align 4
// CHECK: [[NEW:%.*]] = and i32 [[OLD]], 5
// CHECK: ret i32 [[NEW]]

  return (in &= 5);
}

_Atomic(int) compound_mul(_Atomic(int) in) {
// NATIVE-LABEL: @compound_mul
// NATIVE: cmpxchg ptr {{%.*}}, i32 {{%.*}}, i32 [[NEW:%.*]] seq_cst seq_cst, align 4
// NATIVE: ret i32 [[NEW]]
// LIBCALL-LABEL: @compound_mul
// LIBCALL: i1 @__atomic_compare_exchange(i32 noundef 4,

  return (in *= 5);
}

// LIBCALL: [[LC_ATTRS]] = { nounwind willreturn }
