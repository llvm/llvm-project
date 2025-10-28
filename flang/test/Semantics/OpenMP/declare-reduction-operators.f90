! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s

module vector_mod
  implicit none
  type :: Vector
    real :: x, y, z
  contains
    procedure :: add_vectors
    generic :: operator(+) => add_vectors
  end type Vector
contains
  ! Function implementing vector addition
  function add_vectors(a, b) result(res)
    class(Vector), intent(in) :: a, b
    type(Vector) :: res
    res%x = a%x + b%x
    res%y = a%y + b%y
    res%z = a%z + b%z
  end function add_vectors
end module vector_mod

!! Test user-defined operators. Two different varieties, using conventional and
!! unconventional names.
module m1
  interface operator(.mul.)
    procedure my_mul
  end interface
  interface operator(.fluffy.)
    procedure my_add
  end interface
  type t1
    integer :: val = 1
  end type
!$omp declare reduction(.mul.:t1:omp_out=omp_out.mul.omp_in)
!$omp declare reduction(.fluffy.:t1:omp_out=omp_out.fluffy.omp_in)
!CHECK: op.fluffy., PUBLIC: UserReductionDetails TYPE(t1)
!CHECK: op.mul., PUBLIC: UserReductionDetails TYPE(t1)   
contains
  function my_mul(x, y)
    type (t1), intent (in) :: x, y
    type (t1) :: my_mul
    my_mul%val = x%val * y%val
  end function
  function my_add(x, y)
    type (t1), intent (in) :: x, y
    type (t1) :: my_add
    my_add%val = x%val + y%val
  end function
end module m1

program test_vector
!CHECK-LABEL: MainProgram scope: TEST_VECTOR
  use vector_mod
!CHECK: add_vectors (Function): Use from add_vectors in vector_mod
  implicit none
  integer :: i
  type(Vector) :: v1(100), v2(100)

  !$OMP declare reduction(+:vector:omp_out=omp_out+omp_in) initializer(omp_priv=Vector(0,0,0))
!CHECK: op.+: UserReductionDetails TYPE(vector)
!CHECK: v1 size=1200 offset=4: ObjectEntity type: TYPE(vector) shape: 1_8:100_8
!CHECK: v2 size=1200 offset=1204: ObjectEntity type: TYPE(vector) shape: 1_8:100_8
!CHECK: vector: Use from vector in vector_mod

!CHECK: OtherConstruct scope:
!CHECK: omp_in size=12 offset=0: ObjectEntity type: TYPE(vector)
!CHECK: omp_out size=12 offset=12: ObjectEntity type: TYPE(vector)
!CHECK: OtherConstruct scope:
!CHECK: omp_orig size=12 offset=0: ObjectEntity type: TYPE(vector)
!CHECK: omp_priv size=12 offset=12: ObjectEntity type: TYPE(vector)

  v2 = Vector(0.0, 0.0, 0.0)
  v1 = Vector(1.0, 2.0, 3.0)
  !$OMP parallel do reduction(+:v2)
!CHECK: OtherConstruct scope
!CHECK: i (OmpPrivate, OmpPreDetermined): HostAssoc
!CHECK: v1 (OmpShared): HostAssoc
!CHECK: v2 (OmpReduction, OmpExplicit): HostAssoc

  do i = 1, 100
     v2(i) = v2(i) + v1(i)  ! Invokes add_vectors
  end do
  
  print *, 'v2 components:', v2%x, v2%y, v2%z
end program test_vector
