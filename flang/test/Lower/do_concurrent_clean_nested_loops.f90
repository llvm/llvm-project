! Test -fdo-concurrent-clean-nested-loops: a plain DO loop nested in a DO
! CONCURRENT body is lowered without the secondary-induction iter_arg (the DO
! variable is recomputed from the induction variable), while the Fortran
! post-loop value of the DO variable is still materialized after the loop. A
! plain DO loop that is not nested in a DO CONCURRENT body is unaffected.

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
! RUN: bbc -emit-hlfir -fdo-concurrent-clean-nested-loops -o - %s | FileCheck %s --check-prefixes=CHECK,CLEAN

subroutine nested(a, n)
  implicit none
  integer :: n, i, j
  integer :: a(n)
  do concurrent (i=1:n)
    do j = 1, 3
    end do
    a(i) = j
  end do
end subroutine

! CHECK-LABEL:   func.func @_QPnested
! CHECK:           %[[J_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFnestedEj"}
! CHECK:           fir.do_concurrent
! CHECK:             fir.do_concurrent.loop

! Default lowering: nested loop carries the DO variable as an iter_arg and the
! post-loop value is the loop result.
! DEFAULT:           %[[RES:.*]] = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32) {
! DEFAULT:           fir.store %[[RES]] to %[[J_DECL]]#0 : !fir.ref<i32>

! Clean lowering: nested loop has no iter_arg and the body recomputes the DO
! variable from the induction variable; the post-loop value is computed as
! lb + tripCount*step after the loop and stored to the DO variable.
! CLEAN:             fir.do_loop %{{.*}} = %[[LB:[^ ]+]] to %[[UB:[^ ]+]] step %[[ST:[^ ]+]] {
! CLEAN-NOT:           iter_args
! CLEAN:               %[[IV:.*]] = fir.convert %{{.*}} : (index) -> i32
! CLEAN:               fir.store %[[IV]] to %[[J_DECL]]#0 : !fir.ref<i32>
! CLEAN:             }
! CLEAN:             %[[LBI:.*]] = fir.convert %[[LB]] : (index) -> i32
! CLEAN:             %[[UBI:.*]] = fir.convert %[[UB]] : (index) -> i32
! CLEAN:             %[[STI:.*]] = fir.convert %[[ST]] : (index) -> i32
! CLEAN:             %[[C0:.*]] = arith.constant 0 : i32
! CLEAN:             %[[DIFF:.*]] = arith.subi %[[UBI]], %[[LBI]] overflow<nsw> : i32
! CLEAN:             %[[ADD:.*]] = arith.addi %[[DIFF]], %[[STI]] overflow<nsw> : i32
! CLEAN:             %[[TRIP:.*]] = arith.divsi %[[ADD]], %[[STI]] : i32
! CLEAN:             %[[CMP:.*]] = arith.cmpi slt, %[[TRIP]], %[[C0]] : i32
! CLEAN:             %[[SEL:.*]] = arith.select %[[CMP]], %[[C0]], %[[TRIP]] : i32
! CLEAN:             %[[MUL:.*]] = arith.muli %[[SEL]], %[[STI]] overflow<nsw> : i32
! CLEAN:             %[[LAST:.*]] = arith.addi %[[LBI]], %[[MUL]] overflow<nsw> : i32
! CLEAN:             fir.store %[[LAST]] to %[[J_DECL]]#0 : !fir.ref<i32>

! Non-unit lower bound and step: the post-loop value must still be lb +
! tripCount*step (here 2 + 4*2 = 10), not a hard-coded lb=1/step=1 form.
subroutine nested_stride(a, n)
  implicit none
  integer :: n, i, j
  integer :: a(n)
  do concurrent (i=1:n)
    do j = 2, 8, 2
    end do
    a(i) = j
  end do
end subroutine

! CHECK-LABEL:   func.func @_QPnested_stride
! CHECK:           %[[SJ_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFnested_strideEj"}
! DEFAULT:           %[[SRES:.*]] = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32) {
! DEFAULT:           fir.store %[[SRES]] to %[[SJ_DECL]]#0 : !fir.ref<i32>
! CLEAN:             fir.do_loop %{{.*}} = %[[SLB:[^ ]+]] to %[[SUB:[^ ]+]] step %[[SST:[^ ]+]] {
! CLEAN-NOT:           iter_args
! CLEAN:             }
! CLEAN:             %[[SLBI:.*]] = fir.convert %[[SLB]] : (index) -> i32
! CLEAN:             %[[SUBI:.*]] = fir.convert %[[SUB]] : (index) -> i32
! CLEAN:             %[[SSTI:.*]] = fir.convert %[[SST]] : (index) -> i32
! CLEAN:             %[[SC0:.*]] = arith.constant 0 : i32
! CLEAN:             %[[SDIFF:.*]] = arith.subi %[[SUBI]], %[[SLBI]] overflow<nsw> : i32
! CLEAN:             %[[SADD:.*]] = arith.addi %[[SDIFF]], %[[SSTI]] overflow<nsw> : i32
! CLEAN:             %[[STRIP:.*]] = arith.divsi %[[SADD]], %[[SSTI]] : i32
! CLEAN:             %[[SCMP:.*]] = arith.cmpi slt, %[[STRIP]], %[[SC0]] : i32
! CLEAN:             %[[SSEL:.*]] = arith.select %[[SCMP]], %[[SC0]], %[[STRIP]] : i32
! CLEAN:             %[[SMUL:.*]] = arith.muli %[[SSEL]], %[[SSTI]] overflow<nsw> : i32
! CLEAN:             %[[SLAST:.*]] = arith.addi %[[SLBI]], %[[SMUL]] overflow<nsw> : i32
! CLEAN:             fir.store %[[SLAST]] to %[[SJ_DECL]]#0 : !fir.ref<i32>

! Descending loop (step < 0): the same lb + tripCount*step formula must hold
! (here 8 + 4*(-2) = 0).
subroutine nested_stride_neg(a, n)
  implicit none
  integer :: n, i, j
  integer :: a(n)
  do concurrent (i=1:n)
    do j = 8, 2, -2
    end do
    a(i) = j
  end do
end subroutine

! CHECK-LABEL:   func.func @_QPnested_stride_neg
! CHECK:           %[[NJ_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFnested_stride_negEj"}
! DEFAULT:           %[[NRES:.*]] = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32) {
! DEFAULT:           fir.store %[[NRES]] to %[[NJ_DECL]]#0 : !fir.ref<i32>
! CLEAN:             fir.do_loop %{{.*}} = %[[NLB:[^ ]+]] to %[[NUB:[^ ]+]] step %[[NST:[^ ]+]] {
! CLEAN-NOT:           iter_args
! CLEAN:             }
! CLEAN:             %[[NLBI:.*]] = fir.convert %[[NLB]] : (index) -> i32
! CLEAN:             %[[NUBI:.*]] = fir.convert %[[NUB]] : (index) -> i32
! CLEAN:             %[[NSTI:.*]] = fir.convert %[[NST]] : (index) -> i32
! CLEAN:             %[[NC0:.*]] = arith.constant 0 : i32
! CLEAN:             %[[NDIFF:.*]] = arith.subi %[[NUBI]], %[[NLBI]] overflow<nsw> : i32
! CLEAN:             %[[NADD:.*]] = arith.addi %[[NDIFF]], %[[NSTI]] overflow<nsw> : i32
! CLEAN:             %[[NTRIP:.*]] = arith.divsi %[[NADD]], %[[NSTI]] : i32
! CLEAN:             %[[NCMP:.*]] = arith.cmpi slt, %[[NTRIP]], %[[NC0]] : i32
! CLEAN:             %[[NSEL:.*]] = arith.select %[[NCMP]], %[[NC0]], %[[NTRIP]] : i32
! CLEAN:             %[[NMUL:.*]] = arith.muli %[[NSEL]], %[[NSTI]] overflow<nsw> : i32
! CLEAN:             %[[NLAST:.*]] = arith.addi %[[NLBI]], %[[NMUL]] overflow<nsw> : i32
! CLEAN:             fir.store %[[NLAST]] to %[[NJ_DECL]]#0 : !fir.ref<i32>

! A DO CONCURRENT with multiple induction variables is lowered as a single
! fir.do_concurrent.loop carrying all the IVs (unaffected by the option); only
! the nested plain DO loop is cleaned.
subroutine multi_iv(a, n, m)
  implicit none
  integer :: n, m, i, j, k
  integer :: a(n, m)
  do concurrent (i=1:n, j=1:m)
    do k = 1, 3
      a(i, j) = a(i, j) + k
    end do
  end do
end subroutine

! CHECK-LABEL:   func.func @_QPmulti_iv
! CHECK:           %[[MK_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFmulti_ivEk"}
! CHECK:           fir.do_concurrent.loop (%{{[^)]*}}, %{{[^)]*}}) = (%{{[^)]*}}, %{{[^)]*}}) to (%{{[^)]*}}, %{{[^)]*}}) step (%{{[^)]*}}, %{{[^)]*}}) {
! DEFAULT:           %[[MRES:.*]] = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32) {
! DEFAULT:           fir.store %[[MRES]] to %[[MK_DECL]]#0 : !fir.ref<i32>
! CLEAN:             fir.do_loop %{{.*}} = %[[MLB:[^ ]+]] to %[[MUB:[^ ]+]] step %[[MST:[^ ]+]] {
! CLEAN-NOT:           iter_args
! CLEAN:             }
! CLEAN:             %[[MLBI:.*]] = fir.convert %[[MLB]] : (index) -> i32
! CLEAN:             %[[MUBI:.*]] = fir.convert %[[MUB]] : (index) -> i32
! CLEAN:             %[[MSTI:.*]] = fir.convert %[[MST]] : (index) -> i32
! CLEAN:             %[[MC0:.*]] = arith.constant 0 : i32
! CLEAN:             %[[MDIFF:.*]] = arith.subi %[[MUBI]], %[[MLBI]] overflow<nsw> : i32
! CLEAN:             %[[MADD:.*]] = arith.addi %[[MDIFF]], %[[MSTI]] overflow<nsw> : i32
! CLEAN:             %[[MTRIP:.*]] = arith.divsi %[[MADD]], %[[MSTI]] : i32
! CLEAN:             %[[MCMP:.*]] = arith.cmpi slt, %[[MTRIP]], %[[MC0]] : i32
! CLEAN:             %[[MSEL:.*]] = arith.select %[[MCMP]], %[[MC0]], %[[MTRIP]] : i32
! CLEAN:             %[[MMUL:.*]] = arith.muli %[[MSEL]], %[[MSTI]] overflow<nsw> : i32
! CLEAN:             %[[MLAST:.*]] = arith.addi %[[MLBI]], %[[MMUL]] overflow<nsw> : i32
! CLEAN:             fir.store %[[MLAST]] to %[[MK_DECL]]#0 : !fir.ref<i32>

! A plain DO loop not nested in a DO CONCURRENT body keeps its iter_arg even
! when the option is enabled.
subroutine not_nested(x)
  implicit none
  integer :: x, j
  do j = 1, 3
  end do
  x = j
end subroutine

! CHECK-LABEL:   func.func @_QPnot_nested
! CLEAN:           fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32) {
