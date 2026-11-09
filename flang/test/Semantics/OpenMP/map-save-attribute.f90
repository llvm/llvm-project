!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52 -Werror -Wno-experimental-option
!RUN: %flang -fopenmp -fopenmp-version=52 -Wno-experimental-option -Wno-openmp-map-save-without-always -fsyntax-only %s 2>&1 | FileCheck --allow-empty --check-prefix=NOWARN %s
!NOWARN-NOT: warning:

! Verify that mapping a variable with the SAVE attribute (implicit or explicit)
! or that belongs to a COMMON block produces a warning when the ALWAYS map
! modifier is not specified -- but only when that variable is also part of a
! DECLARE TARGET directive, since that is what makes its device instance
! persist across regions.

subroutine explicit_save_warn
  integer, save :: s
  !$omp declare target(s)
!WARNING: Variable 's' has the SAVE attribute and appears in a MAP clause without the ALWAYS modifier; the map operation may be skipped when the variable is already present on the device [-Wopenmp-map-save-without-always]
  !$omp target map(tofrom: s)
  s = s + 1
  !$omp end target
end

subroutine explicit_save_with_always
  integer, save :: s
  !$omp declare target(s)
  !$omp target map(always, tofrom: s)
  s = s + 1
  !$omp end target
end

subroutine explicit_save_no_declare_target
  ! SAVE but not DECLARE TARGET -> no warning.
  integer, save :: s
  !$omp target map(tofrom: s)
  s = s + 1
  !$omp end target
end

subroutine data_init_implicit_save
  ! A variable initialized in a DATA statement has an implicit SAVE attribute.
  integer :: s
  data s /0/
  !$omp declare target(s)
!WARNING: Variable 's' has the SAVE attribute and appears in a MAP clause without the ALWAYS modifier; the map operation may be skipped when the variable is already present on the device [-Wopenmp-map-save-without-always]
  !$omp target map(tofrom: s)
  s = s + 1
  !$omp end target
end

subroutine initializer_implicit_save
  ! A variable with a default initializer has an implicit SAVE attribute.
  integer :: s = 0
  !$omp declare target(s)
!WARNING: Variable 's' has the SAVE attribute and appears in a MAP clause without the ALWAYS modifier; the map operation may be skipped when the variable is already present on the device [-Wopenmp-map-save-without-always]
  !$omp target map(tofrom: s)
  s = s + 1
  !$omp end target
end

subroutine non_save_no_warn(x)
  integer :: x
  ! No SAVE attribute -> no warning.
  !$omp target map(tofrom: x)
  x = x + 1
  !$omp end target
end

subroutine mixed_list
  integer, save :: s
  integer :: x
  !$omp declare target(s)
!WARNING: Variable 's' has the SAVE attribute and appears in a MAP clause without the ALWAYS modifier; the map operation may be skipped when the variable is already present on the device [-Wopenmp-map-save-without-always]
  !$omp target map(tofrom: s, x)
  s = s + x
  !$omp end target
end

module m
  ! Module variables have implicit SAVE.
  integer :: mv = 0
  integer, save :: mve = 0
  !$omp declare target(mv, mve)
contains
  subroutine use_module_var
!WARNING: Variable 'mv' has the SAVE attribute and appears in a MAP clause without the ALWAYS modifier; the map operation may be skipped when the variable is already present on the device [-Wopenmp-map-save-without-always]
!WARNING: Variable 'mve' has the SAVE attribute and appears in a MAP clause without the ALWAYS modifier; the map operation may be skipped when the variable is already present on the device [-Wopenmp-map-save-without-always]
    !$omp target map(tofrom: mv, mve)
    mv = mv + 1
    mve = mve + 1
    !$omp end target
  end
  subroutine use_module_var_always
    !$omp target map(always, tofrom: mv,mve)
    mv = mv + 1
    mve = mve + 1
    !$omp end target
  end
end module

module m_no_dt
  ! Module variables have implicit SAVE, but no DECLARE TARGET -> no warning.
  integer :: mv = 0
contains
  subroutine use_it
    !$omp target map(tofrom: mv)
    mv = mv + 1
    !$omp end target
  end
end module

subroutine explicit_save_allocatable_array
  integer, allocatable, save :: sa(:)
  !$omp declare target(sa)
  allocate(sa(10))
!WARNING: Variable 'sa' has the SAVE attribute and appears in a MAP clause without the ALWAYS modifier; the map operation may be skipped when the variable is already present on the device [-Wopenmp-map-save-without-always]
  !$omp target map(tofrom: sa)
  sa(1) = sa(1) + 1
  !$omp end target
end

subroutine explicit_save_allocatable_array_always
  integer, allocatable, save :: sa(:)
  !$omp declare target(sa)
  allocate(sa(10))
  !$omp target map(always, tofrom: sa)
  sa(1) = sa(1) + 1
  !$omp end target
end

module m_alloc
  ! Module-level allocatable arrays have implicit SAVE.
  integer, allocatable :: ma(:)
  !$omp declare target(ma)
contains
  subroutine use_module_allocatable
    allocate(ma(10))
!WARNING: Variable 'ma' has the SAVE attribute and appears in a MAP clause without the ALWAYS modifier; the map operation may be skipped when the variable is already present on the device [-Wopenmp-map-save-without-always]
    !$omp target map(tofrom: ma)
    ma(1) = ma(1) + 1
    !$omp end target
  end
  subroutine use_module_allocatable_always
    allocate(ma(10))
    !$omp target map(always, tofrom: ma)
    ma(1) = ma(1) + 1
    !$omp end target
  end
end module

subroutine common_block_member
  integer :: cv
  common /cb/ cv
  !$omp declare target(/cb/)
!WARNING: Variable 'cv' is a member of a COMMON block and appears in a MAP clause without the ALWAYS modifier; the map operation may be skipped when the variable is already present on the device [-Wopenmp-map-save-without-always]
  !$omp target map(tofrom: cv)
  cv = cv + 1
  !$omp end target
end

subroutine common_block_member_always
  integer :: cv
  common /cb/ cv
  !$omp declare target(/cb/)
  !$omp target map(always, tofrom: cv)
  cv = cv + 1
  !$omp end target
end

subroutine common_block_no_declare_target
  ! COMMON block but not DECLARE TARGET -> no warning.
  integer :: cv
  common /cb/ cv
  !$omp target map(tofrom: cv)
  cv = cv + 1
  !$omp end target
end

subroutine component_dedup
  type :: t
    integer :: a
    integer :: b
  end type
  type(t), save :: s
  !$omp declare target(s)
  ! The base variable is referenced twice (as `s` and as `s%a`), but only one
  ! warning about the base `s` should be emitted.
!WARNING: Variable 's' has the SAVE attribute and appears in a MAP clause without the ALWAYS modifier; the map operation may be skipped when the variable is already present on the device [-Wopenmp-map-save-without-always]
  !$omp target map(tofrom: s, s%a)
  s%a = s%a + 1
  !$omp end target
end
