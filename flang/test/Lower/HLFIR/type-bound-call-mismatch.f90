! Test interface that lowering handles small interface mismatch with
! type bound procedures.
! RUN: bbc -emit-hlfir --polymorphic-type %s -o - -I nw | FileCheck %s

module dispatch_mismatch
type t
  integer :: i
end type
type, extends(t) :: t2
  contains
    procedure :: proc => foo
end type

interface
  subroutine foo(x)
    import :: t2
    class(t2) :: x
  end subroutine
end interface

end module

subroutine foo(x)
  use dispatch_mismatch, only : t
  ! mistmatch compared to the interface, but OK from an ABI
  ! point of view, and OKI because args compatible with t2 are
  ! compatible with t.
  class(t) :: x
end subroutine

subroutine test(x)
  use dispatch_mismatch, only : t2
  class(t2) :: x
  call x%proc()
end subroutine
!CHECK-LABEL:  func.func @_QPtest(
!CHECK:    %[[X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtestEx"}
!CHECK:    %[[CAST:.*]] = fir.convert %[[X]]#0 : (!fir.class<!fir.type<_QMdispatch_mismatchTt2{t:!fir.type<_QMdispatch_mismatchTt{i:i32}>}>>) -> !fir.class<!fir.type<_QMdispatch_mismatchTt{i:i32}>>
!CHECK:    fir.dispatch "proc"(%[[X]]#0 : !fir.class<!fir.type<_QMdispatch_mismatchTt2{t:!fir.type<_QMdispatch_mismatchTt{i:i32}>}>>) (%[[CAST]] : !fir.class<!fir.type<_QMdispatch_mismatchTt{i:i32}>>) {pass_arg_pos = 0 : i32}
