! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
!
! OpenMP 5.2: 5.2 threadprivate directive restriction
! The effect of an access to a threadprivate variable in a target region is
! unspecified.

module m
  integer :: mv
  !$omp threadprivate(mv)
end module

subroutine target_regions(n)
  use m
  integer :: n, i, j
  integer, save :: sv
  real :: x, y(100)
  common /c/ x, y
  !$omp threadprivate(sv, /c/)

  !$omp target
  !ERROR: A THREADPRIVATE variable 'x' cannot be accessed in a TARGET region
  x = 1.0
  !$omp end target

  !$omp target teams
  !ERROR: A THREADPRIVATE variable 'sv' cannot be accessed in a TARGET region
  sv = 1
  !$omp end target teams

  !$omp target parallel do
  do i = 1, n
    !ERROR: A THREADPRIVATE variable 'y' cannot be accessed in a TARGET region
    y(i) = i
  end do

  !$omp target teams distribute parallel do
  do i = 1, n
    !ERROR: A THREADPRIVATE variable 'x' cannot be accessed in a TARGET region
    x = i
  end do

  !$omp target teams distribute parallel do simd
  do i = 1, n
    !ERROR: A THREADPRIVATE variable 'mv' cannot be accessed in a TARGET region
    mv = i
  end do

  !$omp target teams distribute
  do i = 1, n
    !$omp parallel do
    do j = 1, 10
      !ERROR: A THREADPRIVATE variable 'y' cannot be accessed in a TARGET region
      !ERROR: A THREADPRIVATE variable 'x' cannot be accessed in a TARGET region
      y(j) = x
    end do
  end do

  ! Clause expressions of the TARGET construct itself are evaluated on the host.
  !$omp target if(sv > 0)
  !$omp end target
end subroutine

subroutine declare_target_proc(n)
  integer :: n, i
  real :: x
  common /c/ x
  !$omp threadprivate(/c/)
  !$omp declare target
  !ERROR: A THREADPRIVATE variable 'x' cannot be accessed in a DECLARE TARGET procedure
  x = n
  ! Diagnosed once, for the innermost device context.
  !$omp target
  !ERROR: A THREADPRIVATE variable 'x' cannot be accessed in a TARGET region
  x = 1.0
  !$omp end target
contains
  subroutine internal()
    ! Internal procedures are only implicitly DECLARE TARGET, which is checked
    ! only when compiling for the device (threadprivate-target-implicit.f90).
    x = 2.0
  end subroutine
end subroutine

subroutine declare_target_host(n)
  integer :: n
  integer, save :: sv
  !$omp threadprivate(sv)
  !$omp declare target enter(declare_target_host) device_type(host)
  sv = n
end subroutine

subroutine declare_target_unused(n)
  use m
  integer :: n
  integer, save :: sv
  !$omp threadprivate(sv)
  !$omp declare target
  n = n + 1
end subroutine

subroutine indirect_accesses(n, m)
  integer :: n, m, i
  integer, save :: tp
  integer :: f, g
  real :: x
  common /c/ x
  !$omp threadprivate(tp, /c/)
  f(i) = i + tp
  g(i) = f(i) + 1
  !$omp target
  ! Through a statement function, also a nested one.
  !ERROR: A THREADPRIVATE variable 'tp' cannot be accessed in a TARGET region
  n = f(1)
  !ERROR: A THREADPRIVATE variable 'tp' cannot be accessed in a TARGET region
  m = g(1)
  ! In a clause that allows THREADPRIVATE variables.
  !ERROR: A THREADPRIVATE variable 'tp' cannot be accessed in a TARGET region
  !ERROR: A THREADPRIVATE variable '/c/' cannot be accessed in a TARGET region
  !$omp parallel copyin(tp, /c/)
  !$omp end parallel
  ! In a clause expression of a nested construct.
  !ERROR: A THREADPRIVATE variable 'tp' cannot be accessed in a TARGET region
  !$omp parallel if(tp > 0)
  !$omp end parallel
  !$omp end target
  ! Through ASSOCIATE names.
  associate (a => tp, b => x)
  !$omp target
  !ERROR: A THREADPRIVATE variable 'tp' cannot be accessed in a TARGET region
  !ERROR: A THREADPRIVATE variable 'x' cannot be accessed in a TARGET region
  n = a + b
  !$omp end target
  end associate
  ! As DO variable.
  !$omp target
  !ERROR: A THREADPRIVATE variable 'tp' cannot be accessed in a TARGET region
  do tp = 1, n
  end do
  !$omp end target
end subroutine

subroutine host_regions(n)
  integer :: n, i
  real :: x
  common /c/ x
  !$omp threadprivate(/c/)

  !$omp parallel do
  do i = 1, n
    x = i
  end do

  !$omp teams distribute parallel do
  do i = 1, n
    x = i
  end do
end subroutine