! RUN: %python %S/test_errors.py %s %flang_fc1
! C1141
! A reference to the procedure IEEE_SET_HALTING_MODE ! from the intrinsic
! module IEEE_EXCEPTIONS, shall not ! appear within a DO CONCURRENT construct.
!
! C1137
! An image control statement shall not appear within a DO CONCURRENT construct.
!
! C1136
! A RETURN statement shall not appear within a DO CONCURRENT construct.
!
! (11.1.7.5), paragraph 4
! In a DO CONCURRENT, can't have an i/o statement with an ADVANCE= specifier

subroutine do_concurrent_test1(i,n)
  implicit none
  integer :: i, n
  do 10 concurrent (i = 1:n)
!ERROR: An image control statement is not allowed in DO CONCURRENT
     SYNC ALL
!ERROR: An image control statement is not allowed in DO CONCURRENT
     SYNC IMAGES (*)
!ERROR: An image control statement is not allowed in DO CONCURRENT
     SYNC MEMORY
!ERROR: An image control statement is not allowed in DO CONCURRENT
     stop
!ERROR: An image control statement is not allowed in DO CONCURRENT
     if (.false.) stop
     error stop ! ok
!ERROR: RETURN is not allowed in DO CONCURRENT
     return
10 continue
end subroutine do_concurrent_test1

subroutine do_concurrent_test2(i,j,n,flag)
  use ieee_exceptions
  use iso_fortran_env, only: team_type
  implicit none
  integer :: i, n
  type(ieee_flag_type) :: flag
  logical :: flagValue, halting
  type(team_type) :: j
  type(ieee_status_type) :: status
  type(ieee_modes_type) :: modes
  do concurrent (i = 1:n)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    sync team (j)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    change team (j)
!ERROR: An image control statement is not allowed in DO CONCURRENT
      critical
      end critical
    end team
!ERROR: ADVANCE specifier is not allowed in DO CONCURRENT
    write(*,'(a35)',advance='no')
!ERROR: 'ieee_get_status' may not be called in DO CONCURRENT
    call ieee_get_status(status)
!ERROR: 'ieee_set_status' may not be called in DO CONCURRENT
    call ieee_set_status(status)
!ERROR: 'ieee_get_halting_mode' may not be called in DO CONCURRENT
    call ieee_get_halting_mode(flag, halting)
!ERROR: 'ieee_set_halting_mode' may not be called in DO CONCURRENT
    call ieee_set_halting_mode(flag, halting)
!ERROR: 'ieee_get_flag' may not be called in DO CONCURRENT
    call ieee_get_flag(flag, flagValue)
    call ieee_set_flag(flag, flagValue)
! TODO: IEEE_SET_MODES is SIMPLE per F2023 17.11.41 p2
! The impurity diagnostic is a known nonconformance, not an expectation
!ERROR: 'ieee_set_modes' may not be called in DO CONCURRENT
!ERROR: Impure procedure 'ieee_exceptions$__fortran_ieee_exceptions$ieee_set_modes_0' may not be referenced in DO CONCURRENT
    call ieee_set_modes(modes)
  end do
end subroutine do_concurrent_test2

! F2023 C1146: IEEE_SET_ROUNDING_MODE and IEEE_SET_UNDERFLOW_MODE
! are prohibited in DO CONCURRENT, even though they are SIMPLE
subroutine do_concurrent_ieee_arithmetic(n)
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer :: i, n
  type(ieee_round_type) :: rounding
  logical :: gradual
  do concurrent (i = 1:n)
!ERROR: 'ieee_set_rounding_mode' may not be called in DO CONCURRENT
    call ieee_set_rounding_mode(rounding)
!ERROR: 'ieee_set_rounding_mode' may not be called in DO CONCURRENT
    call ieee_set_rounding_mode(rounding, radix=2)
!ERROR: 'ieee_set_underflow_mode' may not be called in DO CONCURRENT
    call ieee_set_underflow_mode(gradual)
  end do
end subroutine do_concurrent_ieee_arithmetic

subroutine s1()
  use iso_fortran_env
  type(event_type), save :: x[*]
  do concurrent (i = 1:n)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    event post (x)
  end do
end subroutine s1

subroutine s2()
  use iso_fortran_env
  type(event_type), save :: x[*]
  do concurrent (i = 1:n)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    event wait (x)
  end do
end subroutine s2

subroutine s3()
  use iso_fortran_env
  type(team_type) :: t

  do concurrent (i = 1:n)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    form team(1, t)
  end do
end subroutine s3

subroutine s4()
  use iso_fortran_env
  type(lock_type), save :: l[*]

  do concurrent (i = 1:n)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    lock(l)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    unlock(l)
  end do
end subroutine s4

subroutine s5()
  do concurrent (i = 1:n)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    stop
  end do
end subroutine s5

subroutine s6()
  type :: type0
    integer, allocatable, dimension(:) :: type0_field
    integer, allocatable, dimension(:), codimension[:] :: coarray_type0_field
  end type

  type :: type1
    type(type0) :: type1_field
  end type

  type(type1), save :: pvar, qvar
  integer, allocatable, dimension(:) :: array1
  integer, allocatable, dimension(:) :: array2
  integer, allocatable, codimension[:] :: ca, cb
  integer, allocatable :: aa, ab

  ! All of the following are allowable outside a DO CONCURRENT
  allocate(array1(3), pvar%type1_field%type0_field(3), array2(9))
  allocate(pvar%type1_field%coarray_type0_field(3)[*])
  allocate(ca[*])
  allocate(ca[*], pvar%type1_field%coarray_type0_field(3)[*])

  do concurrent (i = 1:10)
    allocate(pvar%type1_field%type0_field(3))
  end do

  do concurrent (i = 1:10)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    allocate(ca[*])
  end do

  do concurrent (i = 1:10)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    deallocate(ca)
  end do

  do concurrent (i = 1:10)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    allocate(pvar%type1_field%coarray_type0_field(3)[*])
  end do

  do concurrent (i = 1:10)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    deallocate(pvar%type1_field%coarray_type0_field)
  end do

  do concurrent (i = 1:10)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    allocate(ca[*], pvar%type1_field%coarray_type0_field(3)[*])
  end do

  do concurrent (i = 1:10)
!ERROR: An image control statement is not allowed in DO CONCURRENT
    deallocate(ca, pvar%type1_field%coarray_type0_field)
  end do

! Call to MOVE_ALLOC of a coarray outside a DO CONCURRENT.  This is OK.
  call move_alloc(ca, cb)

! Call to MOVE_ALLOC with non-coarray arguments in a DO CONCURRENT.  This is OK.
  allocate(aa)
  do concurrent (i = 1:10)
    call move_alloc(aa, ab)
  end do

  do concurrent (i = 1:10)
!ERROR: An image control statement is not allowed in DO CONCURRENT
!ERROR: Impure procedure 'move_alloc' may not be referenced in DO CONCURRENT
    call move_alloc(ca, cb)
  end do

  do concurrent (i = 1:10)
!ERROR: An image control statement is not allowed in DO CONCURRENT
!ERROR: Impure procedure 'move_alloc' may not be referenced in DO CONCURRENT
    call move_alloc(pvar%type1_field%coarray_type0_field, qvar%type1_field%coarray_type0_field)
  end do
end subroutine s6

subroutine s7()
  interface
    pure integer function pf()
    end function pf
  end interface
  interface generic
    impure integer function ipf()
    end function ipf
  end interface

  type :: procTypeNotPure
    procedure(notPureFunc), pointer, nopass :: notPureProcComponent
  end type procTypeNotPure

  type :: procTypePure
    procedure(pf), pointer, nopass :: pureProcComponent
  end type procTypePure

  type(procTypeNotPure) :: procVarNotPure
  type(procTypePure) :: procVarPure
  integer :: ivar
  real :: rvar

  procVarPure%pureProcComponent => pureFunc

  do concurrent (i = 1:10)
    print *, "hello"
  end do

  do concurrent (i = 1:10)
    ivar = pureFunc()
  end do

  ! This should not generate errors
  do concurrent (i = 1:10)
    ivar = procVarPure%pureProcComponent()
  end do

  ! This should generate an error
  do concurrent (i = 1:10)
!ERROR: Impure procedure 'notpureproccomponent' may not be referenced in DO CONCURRENT
    ivar = procVarNotPure%notPureProcComponent()
  end do

  ! This should generate an error
  do concurrent (i = 1:10)
!ERROR: Impure procedure 'ipf' may not be referenced in DO CONCURRENT
    ivar = generic()
  end do

  ! This should generate an error
  do concurrent (i = 1:10)
!ERROR: Impure procedure 'irand' may not be referenced in DO CONCURRENT
    ivar = irand()
!ERROR: Impure procedure 'rand' may not be referenced in DO CONCURRENT
    rvar = rand()
  end do

  contains
    integer function notPureFunc()
      notPureFunc = 2
    end function notPureFunc

    pure integer function pureFunc()
      pureFunc = 3
    end function pureFunc

end subroutine s7

module m8
  type t
   contains
    procedure tbpAssign
    generic :: assignment(=) => tbpAssign
  end type
  interface assignment(=)
    module procedure nonTbpAssign
  end interface
 contains
  impure elemental subroutine tbpAssign(to, from)
    class(t), intent(out) :: to
    class(t), intent(in) :: from
    print *, 'impure due to I/O'
  end
  impure elemental subroutine nonTbpAssign(to, from)
    type(t), intent(out) :: to
    integer, intent(in) :: from
    print *, 'impure due to I/O'
  end
  subroutine test
    type(t) x, y
    do concurrent (j=1:1)
      !ERROR: The defined assignment subroutine 'tbpassign' is not pure
      x = y
      !ERROR: The defined assignment subroutine 'nontbpassign' is not pure
      x = 666
    end do
  end
end

! F2023 C1146 applies only to the intrinsic IEEE_ARITHMETIC module
module ieee_arithmetic
contains
  pure subroutine ieee_set_rounding_mode()
  end subroutine
end module
subroutine do_concurrent_non_intrinsic_ieee()
  use, non_intrinsic :: ieee_arithmetic
  implicit none
  integer :: i
  do concurrent (i = 1:10)
    call ieee_set_rounding_mode()
  end do
end subroutine

! F2023 C1146 permits IEEE_GET_ROUNDING_MODE and IEEE_GET_UNDERFLOW_MODE in DO CONCURRENT
subroutine do_concurrent_ieee_get(n)
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer :: i, n
  type(ieee_round_type) :: rounding(n)
  logical :: gradual(n)
  do concurrent (i = 1:n)
    call ieee_get_rounding_mode(rounding(i))
    call ieee_get_underflow_mode(gradual(i))
  end do
end subroutine
