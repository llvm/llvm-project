// Test the option to avoid failing if derived type descriptors are not found.
// This is a debug option to allow manually writing derived type fir.embox without
// having to care with providing an ABI compliant derived type descriptor object.
// Missing derived type descriptor pointers are replaced by null pointers.
// RUN: tco --ignore-missing-type-desc -o - %s | FileCheck %s
// RUN: %flang_fc1 -emit-llvm  -mmlir --ignore-missing-type-desc -o - %s | FileCheck %s

!some_freestyle_type = !fir.type<some_not_mangled_type{j:i32}>

func.func private @bar(!fir.box<!some_freestyle_type>)

func.func @test_embox(%addr: !fir.ref<!some_freestyle_type>) {
  %0 = fir.embox %addr : (!fir.ref<!some_freestyle_type>) -> !fir.box<!some_freestyle_type>
  fir.call @bar(%0) : (!fir.box<!some_freestyle_type>) -> ()
  return
}
// CHECK-LABEL: define void @test_embox(
// CHECK-SAME: ptr captures(none) %[[ADDR:.*]])
// CHECK: insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
// CHECK-SAME: { ptr undef, i64 ptrtoint (ptr getelementptr (%some_not_mangled_type, ptr null, i32 1) to i64),
// CHECK-SAME: i32 20240719, i8 0, i8 42, i8 0, i8 1, ptr null, [1 x i64] zeroinitializer },
// CHECK-SAME: ptr %[[ADDR]], 0
