! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52 -fopenmp-is-target-device
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s
!
! A procedure that is called from device code is implicitly DECLARE TARGET.
! Accesses to THREADPRIVATE variables in such procedures are diagnosed when
! compiling for the target device only.

module m
  integer :: mv, unused_mv
  !$omp threadprivate(mv, unused_mv)

  type :: t
    integer :: v
  contains
    procedure :: bound
  end type

  interface generic
    module procedure specific
  end interface

  interface assignment(=)
    module procedure assign
  end interface

contains
  subroutine bound(this)
    class(t), intent(inout) :: this
    !ERROR: A THREADPRIVATE variable 'mv' cannot be accessed in procedure 'bound', which is implicitly DECLARE TARGET
    this%v = mv
  end subroutine

  subroutine specific(i)
    integer, intent(inout) :: i
    !ERROR: A THREADPRIVATE variable 'mv' cannot be accessed in procedure 'specific', which is implicitly DECLARE TARGET
    i = mv
  end subroutine

  subroutine assign(a, i)
    type(t), intent(out) :: a
    integer, intent(in) :: i
    !ERROR: A THREADPRIVATE variable 'mv' cannot be accessed in procedure 'assign', which is implicitly DECLARE TARGET
    a%v = i + mv
  end subroutine

  integer function func(i)
    integer, intent(in) :: i
    !ERROR: A THREADPRIVATE variable 'mv' cannot be accessed in procedure 'func', which is implicitly DECLARE TARGET
    func = i + mv
  end function

  ! Only uses the module, which declares THREADPRIVATE variables.
  subroutine no_access(i)
    integer, intent(inout) :: i
    i = i + 1
  end subroutine

  ! Only called from the host.
  subroutine host_only(i)
    integer, intent(inout) :: i
    i = mv
  end subroutine

  ! Explicitly DECLARE TARGET for the host only.
  subroutine explicit_host(i)
    integer, intent(inout) :: i
    !$omp declare target enter(explicit_host) device_type(host)
    i = mv
  end subroutine
end module

! An intermediate procedure that does not access THREADPRIVATE variables.
subroutine middle(i)
  integer, intent(inout) :: i
  call leaf(i)
end subroutine

! Defined after its use.
subroutine leaf(i)
  integer, intent(inout) :: i
  integer, save :: sv
  integer :: x
  common /c/ x
  !$omp threadprivate(sv, /c/)
  !ERROR: A THREADPRIVATE variable 'sv' cannot be accessed in procedure 'leaf', which is implicitly DECLARE TARGET
  i = sv
  !ERROR: A THREADPRIVATE variable 'x' cannot be accessed in procedure 'leaf', which is implicitly DECLARE TARGET
  x = i
end subroutine

subroutine explicit_device(i)
  integer, intent(inout) :: i
  !$omp declare target
  call internal(i)
contains
  ! Called from a DECLARE TARGET procedure.
  subroutine internal(i)
    use m
    integer, intent(inout) :: i
    !ERROR: A THREADPRIVATE variable 'mv' cannot be accessed in procedure 'internal', which is implicitly DECLARE TARGET
    i = mv
  end subroutine
end subroutine

subroutine s(n)
  use m
  integer :: n
  type(t) :: obj
  interface
    subroutine middle(i)
      integer, intent(inout) :: i
    end subroutine
  end interface
  call host_only(n)
  !$omp target map(tofrom: n, obj)
  call middle(n)
  call generic(n)
  call obj%bound()
  obj = n
  n = func(n)
  call no_access(n)
  call explicit_host(n)
  !$omp end target
end subroutine
