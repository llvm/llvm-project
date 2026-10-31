! Tests mapping of a basic `do concurrent` loop to
! `!$omp target teams distribute parallel do`.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

program do_concurrent_shape
    implicit none
    integer :: a(10, 20)
    integer :: i, j

    do concurrent (i=1:10, j=1:20)
        a(i, j) = i * j
    end do
end program do_concurrent_shape

! Constant extents are rematerialized on the device rather than mapped.
! CHECK-LABEL: func.func @_QQmain
! CHECK-NOT:    name("_QFEa.extent.dim{{[0-9]}}")
! CHECK:        omp.target
! CHECK-DAG:    %[[DIM0_EXT:.*]] = arith.constant 10 : index
! CHECK-DAG:    %[[DIM1_EXT:.*]] = arith.constant 20 : index
! CHECK:        %[[SHAPE:.*]] = fir.shape %[[DIM0_EXT]], %[[DIM1_EXT]]
! CHECK:        %{{.*}}:2 = hlfir.declare %{{.*}}(%[[SHAPE]]) uniq_name("_QFEa")

subroutine do_concurrent_shape_shift
    implicit none
    integer :: a(2:10)
    integer :: i

    do concurrent (i=1:10)
        a(i) = i
    end do
end subroutine do_concurrent_shape_shift

! CHECK-LABEL: func.func @_QPdo_concurrent_shape_shift
! CHECK-NOT:    name("_QF{{.*}}Ea.{{(start_idx|extent)}}.dim0")
! CHECK:        omp.target
! CHECK-DAG:    %[[DIM0_STRT:.*]] = arith.constant 2 : index
! CHECK-DAG:    %[[DIM0_EXT:.*]] = arith.constant 9 : index
! CHECK:        %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[DIM0_STRT]], %[[DIM0_EXT]]
! CHECK:        %{{.*}}:2 = hlfir.declare %{{.*}}(%[[SHAPE_SHIFT]]) uniq_name("_QF{{.*}}Ea")
