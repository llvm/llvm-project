! A named user-defined declare reduction that lists several types in a single
! declaration (declare reduction(myred: integer, real: ...)) must lower to one
! omp.declare_reduction op per listed type, each with its own element type and
! combiner, and each reduction clause must bind the op for its variable's type.
! Folding the whole declaration onto a single type-less op would bind a real
! reduction to an integer-typed op (an !fir.ref<f32> reduction on an i32 op, an
! IR type mismatch the verifier does not catch) and silently miscompile. See https://github.com/llvm/llvm-project/issues/207255.
!
! This test uses a strong oracle: it pins the full type-suffixed op names, checks
! each op's combiner arithmetic inside the combiner region (so a matching
! arith.addf/addi in the loop body cannot satisfy it), and binds each loop to the
! full type-specific name (the shared @_QQFmyred_ prefix must not let a substring
! match accept the wrong op).

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

program main
  integer :: i, isum
  real    :: rsum

  !$omp declare reduction(myred: integer, real : omp_out = omp_out + omp_in)

  isum = 0
  rsum = 0.0

  !$omp parallel do reduction(myred:isum)
  do i = 1, 3
     isum = isum + i
  end do

  !$omp parallel do reduction(myred:rsum)
  do i = 1, 3
     rsum = rsum + real(i)
  end do

  print *, isum, rsum
end program main

! There must be no single type-less op that both types fold onto.
! CHECK-NOT: omp.declare_reduction @_QQFmyred :

! One op per listed type, each with its own combiner arithmetic anchored inside
! the combiner region (the real op adds f32, the integer op adds i32).
! CHECK-LABEL: omp.declare_reduction @_QQFmyred_f32 : f32
! CHECK: combiner {
! CHECK: arith.addf
! CHECK-LABEL: omp.declare_reduction @_QQFmyred_i32 : i32
! CHECK: combiner {
! CHECK: arith.addi

! Each loop binds the op for its own variable's type, by full name.
! CHECK: omp.wsloop
! CHECK-SAME: reduction(@_QQFmyred_i32
! CHECK: omp.wsloop
! CHECK-SAME: reduction(@_QQFmyred_f32
