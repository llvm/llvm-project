// RUN: tco %s | FileCheck %s

// CHECK-LINE: define i64 @_QTtP.mem.size(i32 %0, i16 %1)
func.func @_QTtP.mem.size(%0 : i32, %1 : i16) -> index {
  %2 = call @_QTtP.f1.size(%0, %1) : (i32, i16) -> index
  %3 = call @_QTtP.f2.size(%0, %1) : (i32, i16) -> index
  %4 = arith.addi %2, %3 : index
  // CHECK: ret i64 8
  return %4 : index
}
// CHECK-LINE: define i64 @_QTtP.f1.size(i32 %0, i16 %1)
func.func @_QTtP.f1.size(%0 : i32, %1 : i16) -> index {
  %2 = arith.constant 4 : index
  // CHECK: ret i64 4
  return %2 : index
}
// CHECK-LINE: define i64 @_QTtP.f2.size(i32 %0, i16 %1)
func.func @_QTtP.f2.size(%0 : i32, %1 : i16) -> index {
  %2 = arith.constant 4 : index
  // CHECK: ret i64 4
  return %2 : index
}
// CHECK-LINE: define i32 @_QTtP.f1.offset(i32 %0, i16 %1)
func.func @_QTtP.f1.offset(%0 : i32, %1 : i16) -> i32 {
  %2 = arith.constant 0 : i32
  // CHECK: ret i32 0
  return %2 : i32
}
// CHECK-LINE: define i32 @_QTtP.f2.offset(i32 %0, i16 %1)
func.func @_QTtP.f2.offset(%0 : i32, %1 : i16) -> i32 {
  %2 = arith.constant 4 : i32
  // CHECK: ret i32 4
  return %2 : i32
}

// program p
//   type t(p1,p2)
//      integer, len :: p1
//      integer(kind=2), len :: p2
//      integer f1
//      real f2
//   end type t
//   type(t) var
//   var%f1 = 4
// end program p

// CHECK-LINE: define void @_QQmain(i32 %0, i16 %1)
func.func @_QQmain(%arg0 : i32, %arg1 : i16) {
  // CHECK: %[[size:.*]] = call i64 @_QTtP.mem.size(i32 %0, i16 %1)
  // CHECK: %[[alloc:.*]] = alloca i8, i64 %[[size]]
  %0 = fir.alloca !fir.type<_QTt(p1:i32,p2:i16){f1:i32,f2:f32}>(%arg0, %arg1 : i32, i16) {name = "_QEvar"}
  %1 = fir.field_index f1, !fir.type<_QTt(p1:i32,p2:i16){f1:i32,f2:f32}>(%arg0, %arg1 : i32, i16)
  %2 = fir.coordinate_of %0, %1 : (!fir.ref<!fir.type<_QTt(p1:i32,p2:i16){f1:i32,f2:f32}>>, !fir.field) -> !fir.ref<i32>
  %c4_i32 = arith.constant 4 : i32
  fir.store %c4_i32 to %2 : !fir.ref<i32>
  return
}

// CHECK-LINE: define i64 @_QTt1P.mem.size(i32 %0, i32 %1)
func.func @_QTt1P.mem.size(%0 : i32, %1 : i32) -> index {
  // CHECK: call i64 @_QTt1P.f1.size
  %2 = call @_QTt1P.f1.size(%0, %1) : (i32, i32) -> index
  // CHECK:  call i64 @_QTt1P.f2.size
  %3 = call @_QTt1P.f2.size(%0, %1) : (i32, i32) -> index
  %4 = arith.addi %2, %3 : index
  return %4 : index
}
// CHECK-LINE: define i64 @_QTt1P.f1.size(i32 %0, i32 %1)
func.func @_QTt1P.f1.size(%0 : i32, %1 : i32) -> index {
  %2 = fir.convert %0 : (i32) -> index
  return %2 : index
}
// CHECK-LINE: define i64 @_QTt1P.f2.size(i32 %0, i32 %1)
func.func @_QTt1P.f2.size(%0 : i32, %1 : i32) -> index {
  %2 = fir.convert %1 : (i32) -> index
  return %2 : index
}
// CHECK-LINE: define i32 @_QTt1P.f1.offset(i32 %0, i32 %1)
func.func @_QTt1P.f1.offset(%0 : i32, %1 : i32) -> i32 {
  %2 = arith.constant 0 : i32
  return %2 : i32
}
// CHECK-LINE: define i32 @_QTt1P.f2.offset(i32 %0, i32 %1)
func.func @_QTt1P.f2.offset(%0 : i32, %1 : i32) -> i32 {
  return %0 : i32
}
// subroutine foo(i,j)
//   type t(p1,p2)
//      integer, len :: p1
//      integer, len :: p2
//      character(LEN=p1) :: f1
//      character(LEN=p2) :: f2
//   end type t
//   type(t(i,j)) var
//   call bar(var%f2)
// end program p

func.func private @bar(!fir.ref<!fir.char<1,?>>)

// CHECK-LINE: define i8* @_QPfoo(i32 %0, i32 %1)
func.func @_QPfoo(%arg0 : i32, %arg1 : i32) {
  // CHECK: %[[size:.*]] = call i64 @_QTt1P.mem.size(i32 %0, i32 %1)
  // CHECK: %[[alloc:.*]] = alloca i8, i64 %[[size]]
  %0 = fir.alloca !fir.type<_QTt1(p1:i32,p2:i32){f1:!fir.char<1,?>,f2:!fir.char<1,?>}>(%arg0, %arg1 : i32, i32)
  %1 = fir.field_index f2, !fir.type<_QTt1>(%arg0, %arg1 : i32, i32)
  //%2 = fir.coordinate_of %0, %1 : (!fir.ref<!fir.type<_QTt1>>, !fir.field) -> !fir.ref<!fir.char<1,?>>
  %2 = fir.zero_bits !fir.ref<!fir.char<1,?>>
  fir.call @bar(%2) : (!fir.ref<!fir.char<1,?>>) -> ()
  return
}
