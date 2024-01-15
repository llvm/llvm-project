! Test lowering of PASS procedure pointers components.
! RUN: bbc -emit-hlfir -polymorphic-type -o - %s | FileCheck %s

module m
  type t
    sequence
    integer :: i
    procedure(hello), pointer :: p
  end type
  type t2
    integer :: i
    procedure(goodbye), pointer :: p
  end type
  type t3
    sequence
    character(4) :: c
    procedure(char_func), pointer :: p
  end type

  interface
    subroutine takes_hello(p)
      import :: hello
      procedure(hello), pointer :: p
    end subroutine
  end interface
contains
subroutine hello(x)
  type(t) :: x
  print *, "hello"
end subroutine
subroutine goodbye(x)
  class(t2) :: x
  print *, "goodbye"
end subroutine
function char_func(x)
  type(t3) :: x
  character(4) :: char_func
  char_func = x%c
end function
end module

subroutine test1(x)
  use m, only : t
  type(t) :: x
  call x%p()
end subroutine
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> ()>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.boxproc<(!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> ()>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.boxproc<(!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> ()>) -> ((!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> ())
! CHECK:           fir.call %[[VAL_4]](%[[VAL_1]]#1) fastmath<contract> : (!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> ()

subroutine test2(x)
  use m, only : t2
  type(t2) :: x
  call x%p()
end subroutine
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>) -> !fir.box<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>) -> !fir.class<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.class<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>) -> !fir.ref<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>
! CHECK:           %[[VAL_5:.*]] = hlfir.designate %[[VAL_4]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>) -> !fir.ref<!fir.boxproc<(!fir.class<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>) -> ()>>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.boxproc<(!fir.class<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>) -> ()>>
! CHECK:           %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.boxproc<(!fir.class<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>) -> ()>) -> ((!fir.class<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>) -> ())
! CHECK:           fir.call %[[VAL_7]](%[[VAL_3]]) fastmath<contract> : (!fir.class<!fir.type<_QMmTt2{i:i32,p:!fir.boxproc<(!fir.class<!fir.type<_QMmTt2>>) -> ()>}>>) -> ()

subroutine test3(x)
  use m, only : t, takes_hello
  type(t) :: x
  call takes_hello(x%p)
end subroutine
! CHECK-LABEL:   func.func @_QPtest3(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> ()>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.boxproc<(!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> ()>>) -> !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:           fir.call @_QPtakes_hello(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.boxproc<() -> ()>>) -> ()

subroutine test4(x, y)
  use m, only : t
  type(t) :: x, y
  x%p => y%p
end subroutine
! CHECK-LABEL:   func.func @_QPtest4(
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]]  {{.*}}Ey
! CHECK:           %[[VAL_4:.*]] = hlfir.designate %[[VAL_2]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> ()>>
! CHECK:           %[[VAL_5:.*]] = hlfir.designate %[[VAL_3]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> ()>>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.boxproc<(!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> ()>>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_4]] : !fir.ref<!fir.boxproc<(!fir.ref<!fir.type<_QMmTt{i:i32,p:!fir.boxproc<(!fir.ref<!fir.type<_QMmTt>>) -> ()>}>>) -> ()>>

subroutine test5(x)
  use m, only : t3
  type(t3) :: x
  call takes_char(x%p())
end subroutine
! CHECK-LABEL:   func.func @_QPtest5(
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.char<1,4> {bindc_name = ".result"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:           %[[VAL_3:.*]] = hlfir.designate %[[VAL_2]]#1{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmTt3{c:!fir.char<1,4>,p:!fir.boxproc<(!fir.ref<!fir.char<1,4>>, index, !fir.ref<!fir.type<_QMmTt3>>) -> !fir.boxchar<1>>}>>) -> !fir.ref<!fir.boxproc<(!fir.ref<!fir.char<1,4>>, index, !fir.ref<!fir.type<_QMmTt3{c:!fir.char<1,4>,p:!fir.boxproc<(!fir.ref<!fir.char<1,4>>, index, !fir.ref<!fir.type<_QMmTt3>>) -> !fir.boxchar<1>>}>>) -> !fir.boxchar<1>>>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.boxproc<(!fir.ref<!fir.char<1,4>>, index, !fir.ref<!fir.type<_QMmTt3{c:!fir.char<1,4>,p:!fir.boxproc<(!fir.ref<!fir.char<1,4>>, index, !fir.ref<!fir.type<_QMmTt3>>) -> !fir.boxchar<1>>}>>) -> !fir.boxchar<1>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 4 : i64
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[VAL_7]] : index
! CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_6]], %[[VAL_7]] : index
! CHECK:           %[[VAL_10:.*]] = fir.call @llvm.stacksave.p0() fastmath<contract> : () -> !fir.ref<i8>
! CHECK:           %[[VAL_11:.*]] = fir.box_addr %[[VAL_4]] : (!fir.boxproc<(!fir.ref<!fir.char<1,4>>, index, !fir.ref<!fir.type<_QMmTt3{c:!fir.char<1,4>,p:!fir.boxproc<(!fir.ref<!fir.char<1,4>>, index, !fir.ref<!fir.type<_QMmTt3>>) -> !fir.boxchar<1>>}>>) -> !fir.boxchar<1>>) -> ((!fir.ref<!fir.char<1,4>>, index, !fir.ref<!fir.type<_QMmTt3{c:!fir.char<1,4>,p:!fir.boxproc<(!fir.ref<!fir.char<1,4>>, index, !fir.ref<!fir.type<_QMmTt3>>) -> !fir.boxchar<1>>}>>) -> !fir.boxchar<1>)
! CHECK:           %[[VAL_12:.*]] = fir.call %[[VAL_11]](%[[VAL_1]], %[[VAL_9]], %[[VAL_2]]#1) fastmath<contract> : (!fir.ref<!fir.char<1,4>>, index, !fir.ref<!fir.type<_QMmTt3{c:!fir.char<1,4>,p:!fir.boxproc<(!fir.ref<!fir.char<1,4>>, index, !fir.ref<!fir.type<_QMmTt3>>) -> !fir.boxchar<1>>}>>) -> !fir.boxchar<1>
