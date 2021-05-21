// RUN: tco -o - %s | FileCheck %s

!sometype = type !fir.type<_QFfooTsometype{num:i32,values:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>}>
fir.global internal @_QFfooE.dt.sometype : i8
fir.global internal @_QFfooEx : !fir.box<!fir.heap<!sometype>> {
  %0 = fir.zero_bits !fir.heap<!sometype>
  %1 = fir.embox %0 : (!fir.heap<!sometype>) -> !fir.box<!fir.heap<!sometype>>
  fir.has_value %1 : !fir.box<!fir.heap<!sometype>>
}

// CHECK: @_QFfooEx = internal global { %_QFfooTsometype*, i64, i32, i8, i8, i8, i8, i8*, i64, [1 x i64] }
// CHECK-SAME: { %_QFfooTsometype* null, i64 ptrtoint (%_QFfooTsometype* getelementptr (%_QFfooTsometype, %_QFfooTsometype* null, i32 1) to i64),
// CHECK-SAME: i32 20180515, i8 0, i8 34, i8 2, i8 1, i8* @_QFfooE.dt.sometype, i64 0, [1 x i64] undef }
