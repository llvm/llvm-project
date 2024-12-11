! RUN: %flang -fsyntax-only -fopenmp %s 2>&1

! Check that OpenMP declarative directives can be used with objects that have
! an incomplete type.

subroutine test_decl
  ! OMPv5.2 5.2 threadprivate
  ! OMPv5.2 6.5 allocate
  implicit none
  save :: x1, y1
  !$omp threadprivate(x1)
  !$omp allocate(y1)
  integer :: x1, y1

  ! OMPv5.2 7.7 declare-simd
  external :: simd_func
  !$omp declare simd(simd_func)
  logical :: simd_func

  ! OMPv5.2 7.8.1 declare-target
  allocatable :: j
  !$omp declare target(j)
  save :: j
  real(kind=8) :: j(:)

  ! OMPv5.2 5.5.11 declare-reduction - crashes
  !external :: my_add_red
  !!$omp declare reduction(my_add_red : integer : my_add_red(omp_out, omp_in)) &
  !!$omp&  initializer(omp_priv=0)
  !integer :: my_add_red
end subroutine

subroutine test_decl2
  save x1, y1
  !$omp threadprivate(x1)
  !$omp allocate(y1)
  integer :: x1, y1

  ! implicit decl
  !$omp threadprivate(x2)
  !$omp allocate(y2)
  save x2, y2
end subroutine

module m1
  ! implicit decl
  !$omp threadprivate(x, y, z)
  integer :: y
  real :: z

contains
  subroutine sub
    !$omp parallel copyin(x, y, z)
    !$omp end parallel
  end subroutine
end module
