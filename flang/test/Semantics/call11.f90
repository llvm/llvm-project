! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Test 15.7 C1591 & others: contexts requiring pure subprograms

module m

  type :: t
   contains
    procedure, nopass :: tbp_pure => pure
    procedure, nopass :: tbp_impure => impure
  end type
  type, extends(t) :: t2
   contains
    !ERROR: An overridden pure type-bound procedure binding must also be pure
    procedure, nopass :: tbp_pure => impure ! 7.5.7.3
  end type

 contains

  pure integer function pure(n)
    integer, value :: n
    pure = n
  end function
  impure integer function impure(n)
    integer, value :: n
    impure = n
  end function

  subroutine test
    real :: a(pure(1)) ! ok
    !ERROR: Invalid specification expression: reference to impure function 'impure'
    real :: b(impure(1)) ! 10.1.11(4)
    forall (j=1:1)
      !ERROR: Impure procedure 'impure' may not be referenced in a FORALL
      a(j) = impure(j) ! C1037
    end forall
    forall (j=1:1)
      !ERROR: Impure procedure 'impure' may not be referenced in a FORALL
      a(j) = pure(impure(j)) ! C1037
    end forall
    !ERROR: DO CONCURRENT mask expression may not reference impure procedure 'impure'
    do concurrent (j=1:1, impure(j) /= 0) ! C1121
      !ERROR: Call to an impure procedure is not allowed in DO CONCURRENT
      a(j) = impure(j) ! C1139
    end do
  end subroutine

  subroutine test2
    type(t) :: x
    real :: a(x%tbp_pure(1)) ! ok
    !ERROR: Invalid specification expression: reference to impure function 'impure'
    real :: b(x%tbp_impure(1))
    forall (j=1:1)
      a(j) = x%tbp_pure(j) ! ok
    end forall
    forall (j=1:1)
      !ERROR: Impure procedure 'impure' may not be referenced in a FORALL
      a(j) = x%tbp_impure(j) ! C1037
    end forall
    do concurrent (j=1:1, x%tbp_pure(j) /= 0) ! ok
      a(j) = x%tbp_pure(j) ! ok
    end do
    !ERROR: DO CONCURRENT mask expression may not reference impure procedure 'impure'
    do concurrent (j=1:1, x%tbp_impure(j) /= 0) ! C1121
      !ERROR: Call to an impure procedure component is not allowed in DO CONCURRENT
      a(j) = x%tbp_impure(j) ! C1139
    end do
  end subroutine

  subroutine test3
    type :: t
      integer :: i
    end type
    type(t) :: a(10), b
    forall (i=1:10)
      a(i) = t(pure(i))  ! OK
    end forall
    forall (i=1:10)
      !ERROR: Impure procedure 'impure' may not be referenced in a FORALL
      a(i) = t(impure(i))  ! C1037
    end forall
  end subroutine

  subroutine test4(ch)
    type :: t
      real, allocatable :: x
    end type
    type(t) :: a(1), b(1)
    character(*), intent(in) :: ch
    allocate (b(1)%x)
    ! Intrinsic functions and a couple subroutines are pure; do not emit errors
    do concurrent (j=1:1)
      b(j)%x = cos(1.) + len(ch)
      call move_alloc(from=b(j)%x, to=a(j)%x)
    end do
  end subroutine

end module
