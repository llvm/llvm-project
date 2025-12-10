!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52 -Werror

! The directives to check:
!   cancellation_point
!   declare_mapper
!   declare_reduction
!   declare_simd
!   declare_target
!   declare_variant
!   target_data
!   target_enter_data
!   target_exit_data
!   target_update

subroutine f00
  implicit none
  integer :: i

  !$omp parallel
  do i = 1, 10
!WARNING: Directive spelling 'CANCELLATION_POINT' is introduced in a later OpenMP version, try -fopenmp-version=60
    !$omp cancellation_point parallel
  enddo
  !$omp end parallel
end

subroutine f01
  type :: t
    integer :: x
  end type
!WARNING: Directive spelling 'DECLARE_MAPPER' is introduced in a later OpenMP version, try -fopenmp-version=60
  !$omp declare_mapper(t :: v) map(v%x)
end

subroutine f02
  type :: t
    integer :: x
  end type
!WARNING: Directive spelling 'DECLARE_REDUCTION' is introduced in a later OpenMP version, try -fopenmp-version=60
  !$omp declare_reduction(+ : t : omp_out%x = omp_out%x + omp_in%x)
end

subroutine f03
!WARNING: Directive spelling 'DECLARE_SIMD' is introduced in a later OpenMP version, try -fopenmp-version=60
  !$omp declare_simd
end

subroutine f04
!WARNING: Directive spelling 'DECLARE_TARGET' is introduced in a later OpenMP version, try -fopenmp-version=60
  !$omp declare_target
end

subroutine f05
  implicit none
  interface
    subroutine g05
    end
  end interface
!WARNING: Directive spelling 'DECLARE_VARIANT' is introduced in a later OpenMP version, try -fopenmp-version=60
  !$omp declare_variant(g05) match(user={condition(.true.)})
end

subroutine f06
  implicit none
  integer :: i
!WARNING: Directive spelling 'TARGET_DATA' is introduced in a later OpenMP version, try -fopenmp-version=60
  !$omp target_data map(tofrom: i)
  i = 0
!WARNING: Directive spelling 'TARGET_DATA' is introduced in a later OpenMP version, try -fopenmp-version=60
  !$omp end target_data
end

subroutine f07
  implicit none
  integer :: i
!WARNING: Directive spelling 'TARGET_ENTER_DATA' is introduced in a later OpenMP version, try -fopenmp-version=60
  !$omp target_enter_data map(to: i)
end

subroutine f08
  implicit none
  integer :: i
!WARNING: Directive spelling 'TARGET_EXIT_DATA' is introduced in a later OpenMP version, try -fopenmp-version=60
  !$omp target_exit_data map(from: i)
end

subroutine f09
  implicit none
  integer :: i
!WARNING: Directive spelling 'TARGET_UPDATE' is introduced in a later OpenMP version, try -fopenmp-version=60
  !$omp target_update to(i)
end
