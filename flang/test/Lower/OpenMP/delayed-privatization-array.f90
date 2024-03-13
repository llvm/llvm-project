! Test delayed privatization for arrays.

! RUN: split-file %s %t

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %t/one_dim_array.f90 2>&1 | FileCheck %s --check-prefix=ONE_DIM
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - \
! RUN:   %t/one_dim_array.f90 2>&1 | FileCheck %s --check-prefix=ONE_DIM

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %t/two_dim_array.f90 2>&1 | FileCheck %s --check-prefix=TWO_DIM
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %t/two_dim_array.f90 2>&1 |\
! RUN:   FileCheck %s --check-prefix=TWO_DIM

!--- one_dim_array.f90
subroutine delayed_privatization_private(var1, l1, u1)
  implicit none
  integer(8):: l1, u1
  integer, dimension(l1:u1) :: var1

!$omp parallel firstprivate(var1)
  var1(l1 + 1) = 10
!$omp end parallel
end subroutine

! ONE_DIM-LABEL: omp.private {type = firstprivate}
! ONE_DIM-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.box<!fir.array<\?xi32>>]] alloc {

! ONE_DIM-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! ONE_DIM-NEXT:   %[[C0:.*]] = arith.constant 0 : index
! ONE_DIM-NEXT:   %[[DIMS:.*]]:3 = fir.box_dims %[[PRIV_ARG]], %[[C0]] : ([[TYPE]], index) -> (index, index, index)
! ONE_DIM-NEXT:   %[[PRIV_ALLOCA:.*]] = fir.alloca !fir.array<{{\?}}xi32>, %[[DIMS]]#1 {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_privateEvar1"}
! ONE_DIM-NEXT:   %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[DIMS]]#0, %[[DIMS]]#1 : (index, index) -> !fir.shapeshift<1>
! ONE_DIM-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOCA]](%[[SHAPE_SHIFT]]) {uniq_name = "_QFdelayed_privatization_privateEvar1"}
! ONE_DIM-NEXT:  omp.yield(%[[PRIV_DECL]]#0 : [[TYPE]])

! ONE_DIM-NEXT: } copy {
! ONE_DIM-NEXT: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):
! ONE_DIM:        hlfir.assign %[[PRIV_ORIG_ARG]] to %[[PRIV_PRIV_ARG]] temporary_lhs
! ONE_DIM-NEXT:   omp.yield(%[[PRIV_PRIV_ARG]] : [[TYPE]])
! ONE_DIM-NEXT: }

!--- two_dim_array.f90
subroutine delayed_privatization_private(var1, l1, u1, l2, u2)
  implicit none
  integer(8):: l1, u1, l2, u2
  integer, dimension(l1:u1, l2:u2) :: var1

!$omp parallel firstprivate(var1)
  var1(l1 + 1, u2) = 10
!$omp end parallel
end subroutine

! TWO_DIM-LABEL: omp.private {type = firstprivate}
! TWO_DIM-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.box<!fir.array<\?x\?xi32>>]] alloc {

! TWO_DIM-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):
! TWO_DIM-NEXT:   %[[C0:.*]] = arith.constant 0 : index
! TWO_DIM-NEXT:   %[[DIMS0:.*]]:3 = fir.box_dims %[[PRIV_ARG]], %[[C0]] : ([[TYPE]], index) -> (index, index, index)

! TWO_DIM-NEXT:   %[[C1:.*]] = arith.constant 1 : index
! TWO_DIM-NEXT:   %[[DIMS1:.*]]:3 = fir.box_dims %[[PRIV_ARG]], %[[C1]] : ([[TYPE]], index) -> (index, index, index)

! TWO_DIM-NEXT:   %[[PRIV_ALLOCA:.*]] = fir.alloca !fir.array<{{\?}}x{{\?}}xi32>, %[[DIMS0]]#1, %[[DIMS1]]#1 {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_privateEvar1"}
! TWO_DIM-NEXT:   %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[DIMS0]]#0, %[[DIMS0]]#1, %[[DIMS1]]#0, %[[DIMS1]]#1 : (index, index, index, index) -> !fir.shapeshift<2>

! TWO_DIM-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOCA]](%[[SHAPE_SHIFT]]) {uniq_name = "_QFdelayed_privatization_privateEvar1"}
! TWO_DIM-NEXT:  omp.yield(%[[PRIV_DECL]]#0 : [[TYPE]])

! TWO_DIM-NEXT: } copy {
! TWO_DIM-NEXT: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):
! TWO_DIM:        hlfir.assign %[[PRIV_ORIG_ARG]] to %[[PRIV_PRIV_ARG]] temporary_lhs
! TWO_DIM-NEXT:   omp.yield(%[[PRIV_PRIV_ARG]] : [[TYPE]])
! TWO_DIM-NEXT: }
