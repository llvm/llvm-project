! Test delayed privatization for arrays.

! RUN: split-file %s %t

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %t/one_dim_array.f90 2>&1 | FileCheck %s --check-prefix=ONE_DIM
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - \
! RUN:   %t/one_dim_array.f90 2>&1 | FileCheck %s --check-prefix=ONE_DIM

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %t/two_dim_array.f90 2>&1 | FileCheck %s --check-prefix=TWO_DIM
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - \
! RUN:   %t/two_dim_array.f90 2>&1 | FileCheck %s --check-prefix=TWO_DIM

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %t/one_dim_array_default_lb.f90 2>&1 | FileCheck %s --check-prefix=ONE_DIM_DEFAULT_LB
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - \
! RUN:   %t/one_dim_array_default_lb.f90 2>&1 | FileCheck %s --check-prefix=ONE_DIM_DEFAULT_LB

!--- one_dim_array.f90
subroutine delayed_privatization_private_1d(var1, l1, u1)
  implicit none
  integer(8):: l1, u1
  integer, dimension(l1:u1) :: var1

!$omp parallel firstprivate(var1)
  var1(l1 + 1) = 10
!$omp end parallel
end subroutine

! ONE_DIM-LABEL: omp.private {type = firstprivate}
! ONE_DIM-SAME: @[[PRIVATIZER_SYM:.*]] : [[BOX_TYPE:!fir.box<!fir.array<\?xi32>>]] init {

! ONE_DIM-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE:!fir.ref<!fir.box<!fir.array<\?xi32>>>]], %[[PRIV_BOX_ALLOC:.*]]: [[TYPE]]):

! ONE_DIM-NEXT:   %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]]
! ONE_DIM-NEXT:   %[[C0:.*]] = arith.constant 0 : index
! ONE_DIM-NEXT:   %[[DIMS:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C0]]
! ONE_DIM-NEXT:   %[[SHAPE:.*]] = fir.shape %[[DIMS]]#1
! ONE_DIM-NEXT:   %[[ARRAY_ALLOC:.*]] = fir.allocmem !fir.array<?xi32>, %[[DIMS]]#1
! ONE_DIM-NEXT:   %[[TRUE:.*]] = arith.constant true
! ONE_DIM-NEXT:   %[[DECL:.*]]:2 = hlfir.declare %[[ARRAY_ALLOC]](%[[SHAPE]])
! ONE_DIM-NEXT:   %[[C0_0:.*]] = arith.constant 0
! ONE_DIM-NEXT:   %[[DIMS2:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C0_0]]
! ONE_DIM-NEXT:   %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[DIMS2]]#0, %[[DIMS2]]#1
! ONE_DIM-NEXT:   %[[REBOX:.*]] = fir.rebox %[[DECL]]#0(%[[SHAPE_SHIFT]])
! ONE_DIM-NEXT:   fir.store %[[REBOX]] to %[[PRIV_BOX_ALLOC]]
! ONE_DIM-NEXT:   omp.yield(%[[PRIV_BOX_ALLOC]] : [[TYPE]])

! ONE_DIM-NEXT: } copy {
! ONE_DIM-NEXT: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):
! ONE_DIM-NEXT:  %[[PRIV_ORIG_ARG_VAL:.*]] = fir.load %[[PRIV_ORIG_ARG:.*]] : [[TYPE]]
! ONE_DIM-NEXT:  hlfir.assign %[[PRIV_ORIG_ARG_VAL]] to %[[PRIV_PRIV_ARG]]
! ONE_DIM-NEXT:   omp.yield(%[[PRIV_PRIV_ARG]] : [[TYPE]])
! ONE_DIM-NEXT: }

!--- two_dim_array.f90
subroutine delayed_privatization_private_2d(var1, l1, u1, l2, u2)
  implicit none
  integer(8):: l1, u1, l2, u2
  integer, dimension(l1:u1, l2:u2) :: var1

!$omp parallel firstprivate(var1)
  var1(l1 + 1, u2) = 10
!$omp end parallel
end subroutine

! TWO_DIM-LABEL: omp.private {type = firstprivate}
! TWO_DIM-SAME: @[[PRIVATIZER_SYM:.*]] : [[BOX_TYPE:!fir.box<!fir.array<\?x\?xi32>>]] init {

! TWO_DIM-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE:!fir.ref<!fir.box<!fir.array<\?x\?xi32>>>]], %[[PRIV_BOX_ALLOC:.*]]: [[TYPE]]):
! TWO_DIM-NEXT:   %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]]
! TWO_DIM-NEXT:   %[[C0:.*]] = arith.constant 0 : index
! TWO_DIM-NEXT:   %[[DIMS_0:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C0]]
! TWO_DIM-NEXT:   %[[C1:.*]] = arith.constant 1 : index
! TWO_DIM-NEXT:   %[[DIMS_1:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C1]]
! TWO_DIM-NEXT:   %[[SHAPE:.*]] = fir.shape %[[DIMS_0]]#1, %[[DIMS_1]]#1
! TWO_DIM-NEXT:   %[[ARRAY_ALLOC:.*]] = fir.allocmem !fir.array<?x?xi32>, %[[DIMS_0]]#1, %[[DIMS_1]]#1
! TWO_DIM-NEXT:   %[[TRUE:.*]] = arith.constant true
! TWO_DIM-NEXT:   %[[DECL:.*]]:2 = hlfir.declare %[[ARRAY_ALLOC]](%[[SHAPE]])
! TWO_DIM-NEXT:   %[[C0_0:.*]] = arith.constant 0
! TWO_DIM-NEXT:   %[[DIMS2_0:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C0_0]]
! TWO_DIM-NEXT:   %[[C1_0:.*]] = arith.constant 1
! TWO_DIM-NEXT:   %[[DIMS2_1:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C1_0]]
! TWO_DIM-NEXT:   %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[DIMS2_0]]#0, %[[DIMS2_0]]#1, %[[DIMS2_1]]#0, %[[DIMS2_1]]#1
! TWO_DIM-NEXT:   %[[REBOX:.*]] = fir.rebox %[[DECL]]#0(%[[SHAPE_SHIFT]])
! TWO_DIM-NEXT:   fir.store %[[REBOX]] to %[[PRIV_BOX_ALLOC]]
! TWO_DIM-NEXT:   omp.yield(%[[PRIV_BOX_ALLOC]] : [[TYPE]])

! TWO_DIM-NEXT: } copy {
! TWO_DIM-NEXT: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):
! TWO_DIM-NEXT:  %[[PRIV_ORIG_ARG_VAL:.*]] = fir.load %[[PRIV_ORIG_ARG:.*]] : [[TYPE]]
! TWO_DIM-NEXT:  hlfir.assign %[[PRIV_ORIG_ARG_VAL]] to %[[PRIV_PRIV_ARG]]
! TWO_DIM-NEXT:   omp.yield(%[[PRIV_PRIV_ARG]] : [[TYPE]])
! TWO_DIM-NEXT: }

!--- one_dim_array_default_lb.f90
program main
  implicit none
  integer, dimension(10) :: var1

!$omp parallel private(var1)
  var1(1) = 10
!$omp end parallel
end program

! ONE_DIM_DEFAULT_LB-LABEL: omp.private {type = private}
! ONE_DIM_DEFAULT_LB-SAME: @[[PRIVATIZER_SYM:.*]] : [[BOX_TYPE:!fir.box<!fir.array<10xi32>>]] init {

! ONE_DIM_DEFAULT_LB-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE:!fir.ref<!fir.box<!fir.array<10xi32>>>]], %[[PRIV_BOX_ALLOC:.*]]: [[TYPE]]):
! ONE_DIM_DEFAULT_LB-NEXT:   %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]]
! ONE_DIM_DEFAULT_LB-NEXT:   %[[C10:.*]] = arith.constant 10 : index
! ONE_DIM_DEFAULT_LB-NEXT:   %[[SHAPE:.*]] = fir.shape %[[C10]]
! ONE_DIM_DEFAULT_LB-NEXT:   %[[ARRAY_ALLOC:.*]] = fir.allocmem !fir.array<10xi32>
! ONE_DIM_DEFAULT_LB-NEXT:   %[[TRUE:.*]] = arith.constant true
! ONE_DIM_DEFAULT_LB-NEXT:   %[[DECL:.*]]:2 = hlfir.declare %[[ARRAY_ALLOC]](%[[SHAPE]])
! ONE_DIM_DEFAULT_LB-NEXT:   %[[C0_0:.*]] = arith.constant 0
! ONE_DIM_DEFAULT_LB-NEXT:   %[[DIMS2:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C0_0]]
! ONE_DIM_DEFAULT_LB-NEXT:   %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[DIMS2]]#0, %[[DIMS2]]#1
! ONE_DIM_DEFAULT_LB-NEXT:   %[[EMBOX:.*]] = fir.embox %[[DECL]]#0(%[[SHAPE_SHIFT]])
! ONE_DIM_DEFAULT_LB-NEXT:   fir.store %[[EMBOX]] to %[[PRIV_BOX_ALLOC]]
! ONE_DIM_DEFAULT_LB-NEXT:   omp.yield(%[[PRIV_BOX_ALLOC]] : [[TYPE]])
