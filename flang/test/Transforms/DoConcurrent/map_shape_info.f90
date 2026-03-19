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

! CHECK: fir.store %{{c10.*}} to %[[DIM0_EXT:.*]] : !fir.ref<index>
! CHECK: fir.store %{{c20.*}} to %[[DIM1_EXT:.*]] : !fir.ref<index>

! CHECK: omp.map.info
! CHECK: omp.map.info
! CHECK: omp.map.info

! CHECK: omp.map.info
! CHECK: omp.map.info
! CHECK: omp.map.info

! CHECK: omp.map.info
! CHECK: omp.map.info
! CHECK: omp.map.info

! CHECK: %[[DIM0_EXT_MAP:.*]] = omp.map.info
! CHECK-SAME:   var_ptr(%[[DIM0_EXT]] : !fir.ref<index>, index)
! CHECK-SAME:   map_clauses(implicit)
! CHECK-SAME:   capture(ByCopy) -> !fir.ref<index> {name = "_QFEa.extent.dim0"}

! CHECK: %[[DIM1_EXT_MAP:.*]] = omp.map.info
! CHECK-SAME:   var_ptr(%[[DIM1_EXT]] : !fir.ref<index>, index)
! CHECK-SAME:   map_clauses(implicit)
! CHECK-SAME:   capture(ByCopy) -> !fir.ref<index> {name = "_QFEa.extent.dim1"}

! CHECK: omp.target host_eval({{.*}}) map_entries(
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %[[DIM0_EXT_MAP]] -> %[[DIM0_EXT_ARG:[^,]+]],
! CHECK-SAME:   %[[DIM1_EXT_MAP]] -> %[[DIM1_EXT_ARG:[^,]+]] : {{.*}})

! CHECK-DAG:    %[[DIM0_EXT_DEV:.*]] = fir.load %[[DIM0_EXT_ARG]]
! CHECK-DAG:    %[[DIM1_EXT_DEV:.*]] = fir.load %[[DIM1_EXT_ARG]]

! CHECK:        %[[SHAPE:.*]] = fir.shape %[[DIM0_EXT_DEV]], %[[DIM1_EXT_DEV]]
! CHECK:        %{{.*}}:2 = hlfir.declare %{{.*}}(%[[SHAPE]]) {uniq_name = "_QFEa"}

subroutine do_concurrent_shape_shift
    implicit none
    integer :: a(2:10)
    integer :: i

    do concurrent (i=1:10)
        a(i) = i
    end do
end subroutine do_concurrent_shape_shift

! CHECK: fir.store %{{c2.*}} to %[[DIM0_STRT:.*]] : !fir.ref<index>
! CHECK: fir.store %{{c9.*}} to %[[DIM0_EXT:.*]] : !fir.ref<index>

! CHECK: omp.map.info
! CHECK: omp.map.info
! CHECK: omp.map.info

! CHECK: omp.map.info
! CHECK: omp.map.info

! CHECK: %[[DIM0_STRT_MAP:.*]] = omp.map.info
! CHECK-SAME:   var_ptr(%[[DIM0_STRT]] : !fir.ref<index>, index)
! CHECK-SAME:   map_clauses(implicit)
! CHECK-SAME:   capture(ByCopy) -> !fir.ref<index> {name = "_QF{{.*}}Ea.start_idx.dim0"}

! CHECK: %[[DIM0_EXT_MAP:.*]] = omp.map.info
! CHECK-SAME:   var_ptr(%[[DIM0_EXT]] : !fir.ref<index>, index)
! CHECK-SAME:   map_clauses(implicit)
! CHECK-SAME:   capture(ByCopy) -> !fir.ref<index> {name = "_QF{{.*}}Ea.extent.dim0"}

! CHECK: omp.target host_eval({{.*}}) map_entries(
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! CHECK-SAME:   %[[DIM0_STRT_MAP]] -> %[[DIM0_STRT_ARG:[^,]+]],
! CHECK-SAME:   %[[DIM0_EXT_MAP]] -> %[[DIM0_EXT_ARG:[^,]+]] : {{.*}})

! CHECK-DAG:    %[[DIM0_STRT_DEV:.*]] = fir.load %[[DIM0_STRT_ARG]]
! CHECK-DAG:    %[[DIM0_EXT_DEV:.*]] = fir.load %[[DIM0_EXT_ARG]]

! CHECK:        %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[DIM0_STRT_DEV]], %[[DIM0_EXT_DEV]]
! CHECK:        %{{.*}}:2 = hlfir.declare %{{.*}}(%[[SHAPE_SHIFT]]) {uniq_name = "_QF{{.*}}Ea"}

