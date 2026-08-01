! A user-defined operator declare reduction that lists several types in a single
! declaration (Form A: `declare reduction(.op.:t1,t2:...)`) must lower to one
! omp.declare_reduction op per listed type, each with its own element type,
! combiner and initializer. Folding every type onto the first type's op is a
! silent miscompile (a real reduction lowered through an integer combiner). See
! https://github.com/llvm/llvm-project/issues/207255.
!
! t1 is integer-valued and t2 is real-valued, so a fold onto one op is glaring:
! the two ops must carry distinct element types (i32 vs f32) and each loop must
! bind its own type's op.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m
  type :: t1
    integer :: v = 0
  end type
  type :: t2
    real :: v = 0.0
  end type
  interface operator(.op.)
    module procedure a1, a2
  end interface
  !$omp declare reduction(.op.:t1,t2:omp_out%v=omp_out%v+omp_in%v)
contains
  type(t1) function a1(a,b)
    type(t1), intent(in) :: a,b
    a1%v=a%v+b%v
  end function
  type(t2) function a2(a,b)
    type(t2), intent(in) :: a,b
    a2%v=a%v+b%v
  end function
end module

program main
  use m
  type(t1) :: x
  type(t2) :: y
  integer :: i
  x = t1(0)
  !$omp parallel do reduction(.op.:x)
  do i=1,5
    x%v = x%v + i
  end do
  y = t2(0.0)
  !$omp parallel do reduction(.op.:y)
  do i=1,4
    y%v = y%v + real(i)
  end do
  print *, x%v, y%v
end program

! One distinct op per listed type, each with its own element type. The op name
! carries the owning scope (mangled "_QQ...") plus a per-type suffix, so the two
! types get two ops rather than colliding.
! CHECK-DAG: omp.declare_reduction @[[REDT1:_QQ[A-Za-z0-9_.]*op\.op\.[A-Za-z0-9_.]*t1]] : !fir.ref<!fir.type<{{[^>]*}}t1{v:i32}>>
! CHECK-DAG: omp.declare_reduction @[[REDT2:_QQ[A-Za-z0-9_.]*op\.op\.[A-Za-z0-9_.]*t2]] : !fir.ref<!fir.type<{{[^>]*}}t2{v:f32}>>

! The bare operator spelling must never be a declare_reduction name (it would
! collide across scopes/types).
! CHECK-NOT: omp.declare_reduction @op.op.

! Each parallel loop binds the op for its own variable's type.
! CHECK-DAG: reduction(byref @[[REDT1]] {{[^)]*}}t1{v:i32}
! CHECK-DAG: reduction(byref @[[REDT2]] {{[^)]*}}t2{v:f32}
