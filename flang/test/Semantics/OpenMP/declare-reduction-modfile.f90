! RUN: %python %S/../test_modfile.py %s %flang_fc1 -fopenmp -fopenmp-version=52
! Check correct modfile generation for OpenMP DECLARE REDUCTION construct.

!Expect: drm.mod
!module drm
!type::t1
!integer(4)::val
!endtype
!!$OMP DECLARE REDUCTION (*:t1:omp_out = omp_out*omp_in) INITIALIZER(omp_priv=t&
!!$OMP&1(1))
!!$OMP METADIRECTIVE OTHERWISE(DECLARE REDUCTION(+:INTEGER))
!!$OMP DECLARE REDUCTION (.fluffy.:t1:omp_out = omp_out.fluffy.omp_in) INITIALI&
!!$OMP&ZER(omp_priv=t1(0))
!!$OMP DECLARE REDUCTION (.mul.:t1:omp_out = omp_out.mul.omp_in) INITIALIZER(om&
!!$OMP&p_priv=t1(1))
!interface operator(.mul.)
!procedure::mul
!end interface
!interface operator(.fluffy.)
!procedure::add
!end interface
!interface operator(*)
!procedure::mul
!end interface
!contains
!function mul(v1,v2)
!type(t1),intent(in)::v1
!type(t1),intent(in)::v2
!type(t1)::mul
!end
!function add(v1,v2)
!type(t1),intent(in)::v1
!type(t1),intent(in)::v2
!type(t1)::add
!end
!end

module drm
  type t1
    integer :: val
  end type t1
  interface operator(.mul.)
    procedure mul
  end interface
  interface operator(.fluffy.)
    procedure add
  end interface
  interface operator(*)
    module procedure mul
  end interface
!$omp declare reduction(*:t1:omp_out=omp_out*omp_in) initializer(omp_priv=t1(1))
!$omp declare reduction(.mul.:t1:omp_out=omp_out.mul.omp_in) initializer(omp_priv=t1(1))
!$omp declare reduction(.fluffy.:t1:omp_out=omp_out.fluffy.omp_in) initializer(omp_priv=t1(0))
!$omp metadirective otherwise(declare reduction(+: integer))
contains
  type(t1) function mul(v1, v2)
    type(t1), intent (in):: v1, v2
    mul%val = v1%val * v2%val
  end function
  type(t1) function add(v1, v2)
    type(t1), intent (in):: v1, v2
    add%val = v1%val + v2%val
  end function
end module drm

