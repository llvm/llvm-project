// Test mapping between fir types and their runtime type descriptor objects.
// RUN: tco -o - %s | FileCheck %s

// Test mapping of derived type following the internal name ABI (i.e. that could come from an actual
// compiled Fortran program).

!sometype = !fir.type<_QFfooTsometype{num:i32,values:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>}>
fir.global internal @_QFfooE.dt.sometype constant : i8

fir.global internal @_QFfooEx : !fir.box<!fir.heap<!sometype>> {
  %0 = fir.zero_bits !fir.heap<!sometype>
  %1 = fir.embox %0 : (!fir.heap<!sometype>) -> !fir.box<!fir.heap<!sometype>>
  fir.has_value %1 : !fir.box<!fir.heap<!sometype>>
}
// CHECK: @_QFfooEx = internal global { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
// CHECK-SAME: { ptr null, i64 ptrtoint (ptr getelementptr (%_QFfooTsometype, ptr null, i32 1) to i64),
// CHECK-SAME: i32 20180515, i8 0, i8 42, i8 2, i8 1, ptr @_QFfooE.dt.sometype, [1 x i64] undef }

!some_pdt_type = !fir.type<_QFfooTsome_pdt_typeK42K43{num:i32,values:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>}>
fir.global internal @_QFfooE.dt.some_pdt_type.42.43 constant : i8

fir.global internal @_QFfooEx2 : !fir.box<!fir.heap<!some_pdt_type>> {
  %0 = fir.zero_bits !fir.heap<!some_pdt_type>
  %1 = fir.embox %0 : (!fir.heap<!some_pdt_type>) -> !fir.box<!fir.heap<!some_pdt_type>>
  fir.has_value %1 : !fir.box<!fir.heap<!some_pdt_type>>
}
// CHECK: @_QFfooEx2 = internal global { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
// CHECK-SAME: { ptr null, i64 ptrtoint (ptr getelementptr (%_QFfooTsome_pdt_typeK42K43, ptr null, i32 1) to i64),
// CHECK-SAME: i32 20180515, i8 0, i8 42, i8 2, i8 1, ptr @_QFfooE.dt.some_pdt_type.42.43, [1 x i64] undef }
