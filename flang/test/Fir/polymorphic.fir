// RUN: tco %s | FileCheck %s

// Test code gen for unlimited polymorphic type descriptor.

func.func @_QMpolymorphic_testPtest_allocate_unlimited_polymorphic_non_derived() {
  %0 = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = "u", uniq_name = "_QMpolymorphic_testFtest_allocate_unlimited_polymorphic_non_derivedEu"}
  %1 = fir.zero_bits !fir.ptr<none>
  %2 = fir.embox %1 : (!fir.ptr<none>) -> !fir.class<!fir.ptr<none>>
  fir.store %2 to %0 : !fir.ref<!fir.class<!fir.ptr<none>>>
  return
}

// CHECK-LABEL: define void @_QMpolymorphic_testPtest_allocate_unlimited_polymorphic_non_derived() {
// CHECK:   %[[MEM:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }
// CHECK:   %[[DESC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1
// CHECK:   store { ptr, i64, i32, i8, i8, i8, i8 } { ptr null, i64 0, i32 20180515, i8 0, i8 -1, i8 1, i8 0 }, ptr %[[MEM]]
// CHECK:   %[[LOADED:.*]] = load { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[MEM]], align 8
// CHECK:   store { ptr, i64, i32, i8, i8, i8, i8 } %[[LOADED]], ptr %[[DESC]]
// CHECK:   ret void
// CHECK: }
