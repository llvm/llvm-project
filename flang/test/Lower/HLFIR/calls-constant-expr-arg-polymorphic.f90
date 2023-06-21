! RUN: bbc -emit-hlfir --polymorphic-type -o - %s | FileCheck %s

! Test when constant argument are copied in memory
! and passed to polymorphic arguments.
! The copy is done in case the dummy later appear in a
! copy-out that would create write to this memory location.
  type t1
    integer :: i
  end type
  type, extends(t1) :: t2
    integer :: j
  end type
  interface
  subroutine foo(x)
    import :: t1
    class(t1) :: x(:)
  end subroutine
  end interface

  call foo([t2(0,0)])
end
! CHECK-LABEL:   func.func @_QQmain() {
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_2:.*]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.1x_QFTt2.0"}
! CHECK:           %[[VAL_4:.*]] = hlfir.as_expr %[[VAL_3]]#0 : (!fir.ref<!fir.array<1x!fir.type<_QFTt2{i:i32,j:i32}>>>) -> !hlfir.expr<1x!fir.type<_QFTt2{i:i32,j:i32}>>
! CHECK:           %[[VAL_5:.*]]:3 = hlfir.associate %[[VAL_4]](%[[VAL_2]]) {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<1x!fir.type<_QFTt2{i:i32,j:i32}>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1x!fir.type<_QFTt2{i:i32,j:i32}>>>, !fir.ref<!fir.array<1x!fir.type<_QFTt2{i:i32,j:i32}>>>, i1)
! CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_5]]#0(%[[VAL_2]]) : (!fir.ref<!fir.array<1x!fir.type<_QFTt2{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<1x!fir.type<_QFTt2{i:i32,j:i32}>>>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.array<1x!fir.type<_QFTt2{i:i32,j:i32}>>>) -> !fir.class<!fir.array<?x!fir.type<_QFTt1{i:i32}>>>
! CHECK:           fir.call @_QPfoo(%[[VAL_7]]) {{.*}}: (!fir.class<!fir.array<?x!fir.type<_QFTt1{i:i32}>>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_5]]#1, %[[VAL_5]]#2 : !fir.ref<!fir.array<1x!fir.type<_QFTt2{i:i32,j:i32}>>>, i1
