! RUN: %python %S/../test_modfile.py %s %flang_fc1 -fopenmp -fopenmp-version=52
! Check that PRIVATE declare reduction accessibility is preserved in module files.

!Expect: drm_private.mod
!module drm_private
!type::dt
!integer(4)::val=0_4
!endtype
!!$OMP DECLARE REDUCTION(+:dt: omp_out%val=omp_out%val+omp_in%val) INITIALIZER(&
!!$OMP&omp_priv=dt(0))
!interface operator(+)
!procedure::add_dt
!end interface
!private::operator(+)
!contains
!function add_dt(a,b)
!type(dt),intent(in)::a
!type(dt),intent(in)::b
!type(dt)::add_dt
!end
!end

module drm_private
  type :: dt
    integer :: val = 0
  end type
  private :: operator(+)
  interface operator(+)
    module procedure add_dt
  end interface
  !$omp declare reduction(+:dt:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=dt(0))
contains
  type(dt) function add_dt(a, b)
    type(dt), intent(in) :: a, b
    add_dt%val = a%val + b%val
  end function
end module drm_private
