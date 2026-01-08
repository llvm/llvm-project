! Tests that we can map "unnamed" and non-reference/non-box values to device; for
! example, values that result from `fix.box_dims` ops.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

subroutine test_non_refernece
  integer i
  real, allocatable :: arr(:)

  associate(a => arr)
    do concurrent (i = 1:10)
      block
        real z(size(a,1))
      end block
    end do
  end associate
end subroutine test_non_refernece

! CHECK: omp.map.info var_ptr(%{{.*}} : !fir.ref<index>, index)
! CHECK: omp.map.info var_ptr(%{{.*}} : !fir.ref<index>, index)
! CHECK: omp.map.info var_ptr(%{{.*}} : !fir.ref<index>, index)

! CHECK:      %[[DIM_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<index>, index)
! CHECK-SAME:                     map_clauses(implicit)
! CHECK-SAME:                     capture(ByCopy) -> !fir.ref<index> {name = ""}


! CHECK:      omp.target host_eval({{.*}} : index, index, index)
! CHECK-SAME:   map_entries({{.*}}, %[[DIM_MAP]] -> %{{.*}} :
! CHECK-SAME:               !fir.ref<i32>, !fir.ref<index>)

