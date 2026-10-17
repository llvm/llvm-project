! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fopenmp \
! RUN:   -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck %s

! Resolved operations and implied-DO kinds must survive constant folding.
! Each pair has a 5/5 tie. These declarations attach to the same base symbol.

module different_operations
  type t
    integer :: value
  end type
  type(t) :: x
contains
  integer function first()
    first = 1
  end function
  integer function second()
    second = 2
  end function
  integer function base()
    base = 0
  end function
  integer function op1(a, b)
    type(t), intent(in) :: a, b
    op1 = 1
  end function
  integer function op2(a, b)
    type(t), intent(in) :: a, b
    op2 = 2
  end function
  subroutine attach_first()
    interface operator(+)
      module procedure op1
    end interface
    !$omp declare variant(base:first) &
    !$omp& match(user={condition(kind(x+x) == 4)}, &
    !$omp& implementation={vendor(score(4): llvm)})
  end subroutine
  subroutine attach_second()
    interface operator(+)
      module procedure op2
    end interface
    !$omp declare variant(base:second) &
    !$omp& match(user={condition(kind(x+x) == 4)}, device={kind(cpu)}, &
    !$omp& implementation={vendor(score(3): llvm)})
  end subroutine
  ! CHECK-LABEL: func.func @_QMdifferent_operationsPtest(
  ! CHECK-NOT: fir.call @_QMdifferent_operationsPsecond
  ! CHECK: fir.call @_QMdifferent_operationsPfirst
  ! CHECK-NOT: fir.call @_QMdifferent_operationsPsecond
  ! CHECK: return
  integer function test()
    test = base()
  end function
end module

module same_operation
  type t
    integer :: value
  end type
  type(t) :: x
contains
  integer function first()
    first = 1
  end function
  integer function second()
    second = 2
  end function
  integer function base()
    base = 0
  end function
  integer function op1(a, b)
    type(t), intent(in) :: a, b
    op1 = 1
  end function
  integer function op2(a, b)
    type(t), intent(in) :: a, b
    op2 = 2
  end function
  subroutine attach_first()
    interface operator(+)
      module procedure op1
    end interface
    !$omp declare variant(base:first) &
    !$omp& match(user={condition(kind(x+x) == 4)}, &
    !$omp& implementation={vendor(score(4): llvm)})
  end subroutine
  subroutine attach_second()
    interface operator(+)
      module procedure op1
    end interface
    !$omp declare variant(base:second) &
    !$omp& match(user={condition(kind(x+x) == 4)}, device={kind(cpu)}, &
    !$omp& implementation={vendor(score(3): llvm)})
  end subroutine
  ! CHECK-LABEL: func.func @_QMsame_operationPtest(
  ! CHECK-NOT: fir.call @_QMsame_operationPfirst
  ! CHECK: fir.call @_QMsame_operationPsecond
  ! CHECK-NOT: fir.call @_QMsame_operationPfirst
  ! CHECK: return
  integer function test()
    test = base()
  end function
end module

module implied_do_kinds
contains
  integer function first()
    first = 1
  end function
  integer function second()
    second = 2
  end function
  integer function base()
    base = 0
  end function
  subroutine attach_first()
    integer(4) :: i
    !$omp declare variant(base:first) &
    !$omp& match(user={condition(kind([(i, i=1,1)]) > 0)}, &
    !$omp& implementation={vendor(score(4): llvm)})
  end subroutine
  subroutine attach_second()
    integer(8) :: i
    !$omp declare variant(base:second) &
    !$omp& match(user={condition(kind([(i, i=1,1)]) > 0)}, device={kind(cpu)}, &
    !$omp& implementation={vendor(score(3): llvm)})
  end subroutine
  ! CHECK-LABEL: func.func @_QMimplied_do_kindsPtest(
  ! CHECK-NOT: fir.call @_QMimplied_do_kindsPsecond
  ! CHECK: fir.call @_QMimplied_do_kindsPfirst
  ! CHECK-NOT: fir.call @_QMimplied_do_kindsPsecond
  ! CHECK: return
  integer function test()
    test = base()
  end function
end module
