!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -mmlir -mlir-print-op-generic -o - | FileCheck %s
!RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=51 %s -mlir-print-op-generic -o - | FileCheck %s

! Check that we don't generate member information for the descriptor of `a`
! on entry to the target region.

integer function s(a)
  integer :: a(:)
  integer :: t
  !$omp target data map(to:a) use_device_addr(a)
  !$omp target map(from:t) has_device_addr(a)
  t = size(a, 1)
  !$omp end target
  !$omp end target data
  s = t
end

! Check that the map.info for `a` only takes a single parameter.

!CHECK-DAG: %[[MAP_A:[0-9]+]] = "omp.map.info"(%[[STORAGE_A:[0-9#]+]]) <{map_capture_type = #omp<variable_capture_kind(ByRef)>, map_type = 517 : ui64, name = "a", operandSegmentSizes = array<i32: 1, 0, 0, 0>, partial_map = false, var_type = !fir.box<!fir.array<?xi32>>}> : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.ref<!fir.array<?xi32>>
!CHECK-DAG: %[[MAP_T:[0-9]+]] = "omp.map.info"(%[[STORAGE_T:[0-9#]+]]) <{map_capture_type = #omp<variable_capture_kind(ByRef)>, map_type = 2 : ui64, name = "t", operandSegmentSizes = array<i32: 1, 0, 0, 0>, partial_map = false, var_type = i32}> : (!fir.ref<i32>) -> !fir.ref<i32>

!CHECK: "omp.target"(%[[MAP_A]], %[[MAP_T]])
