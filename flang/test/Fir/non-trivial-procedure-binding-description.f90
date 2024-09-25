! RUN: %flang_fc1 -emit-mlir %s -o - | FileCheck %s --check-prefix=BEFORE
! RUN: %flang_fc1 -emit-mlir %s -o - | fir-opt --abstract-result | FileCheck %s --check-prefix=AFTER
module a
  type f
  contains
! BEFORE: fir.address_of(@_QMaPfoo) : (!fir.ref<!fir.type<_QMaTf>>) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! AFTER: [[ADDRESS:%.*]] = fir.address_of(@_QMaPfoo) : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.type<_QMaTf>>) -> ()
! AFTER: fir.convert [[ADDRESS]] : ((!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.type<_QMaTf>>) -> ()) -> ((!fir.ref<!fir.type<_QMaTf>>) -> !fir.box<!fir.heap<!fir.char<1,?>>>)
! AFTER-NOT: fir.address_of(@_QMaPfoo) : (!fir.ref<!fir.type<_QMaTf>>) -> !fir.box<!fir.heap<!fir.char<1,?>>>
    procedure, nopass :: foo
  end type f
contains
  function foo(obj) result(bar)
    type(f) :: obj
    character(len=:), allocatable :: bar

    if (.TRUE.) then
      bar = "true"
    else
      bar = "false"
    endif
  end function foo
end module a

program main
  use a

  type(f) :: obj 
  print *, obj%foo(obj)
end program
