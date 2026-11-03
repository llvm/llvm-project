! Test lowering of an OpenMP REDUCTION clause that uses a locally-declared
! user-defined operator. See https://github.com/llvm/llvm-project/issues/204299:
! this used to ICE in ReductionProcessor because the defined-operator variant
! was assumed to be an intrinsic operator. This is the trivial (by-value)
! integer case from the issue reproducer.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine red_integer
  interface operator(.zzzz.)
    function zzzz_op(a, b)
      integer, intent(in) :: a, b
      integer :: zzzz_op
    end function zzzz_op
  end interface
  !$omp declare reduction(.zzzz.:integer:omp_out=omp_out+omp_in) &
  !$omp   initializer(omp_priv=0)
  integer :: x
  x = 0
  !$omp parallel reduction(.zzzz.:x)
  x = x .zzzz. 1
  !$omp end parallel
end subroutine red_integer

! The op name must be module-scoped (a mangled "_QQ..." generated name), NOT the
! bare operator spelling, so reductions with the same spelling in different
! modules do not collide. The directive and clause must reference the same name.
! CHECK: omp.declare_reduction @[[RED:_QQ[A-Za-z0-9_.]*op\.zzzz\.[A-Za-z0-9_.]*]] : i32
! CHECK-NOT: omp.declare_reduction @op.zzzz.
! CHECK: omp.parallel
! CHECK-SAME: reduction(@[[RED]]
