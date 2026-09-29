! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Non-descriptor optional arguments need empty mapping bounds when absent,
! while retaining their null base address for PRESENT inside the target.

subroutine scalar(x, found)
  real, optional :: x
  logical :: found
  !$omp target map(alloc:x) map(from:found)
    found = present(x)
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPscalar(
! CHECK: %[[X:[^ ,:]+]]:2 = hlfir.declare {{.*}}fortran_attrs = #fir.var_attrs<optional>
! CHECK: %[[PRESENT:[^ ,:]+]] = fir.is_present %[[X]]#1
! CHECK: %[[ZERO:[^ ,:]+]] = arith.constant 0 : index
! CHECK: %[[ONE:[^ ,:]+]] = arith.constant 1 : index
! CHECK: %[[NEGONE:[^ ,:]+]] = arith.constant -1 : index
! CHECK: %[[UB:[^ ,:]+]] = arith.select %[[PRESENT]], %[[ZERO]], %[[NEGONE]] : index
! CHECK: %[[ABSENT_EXTENT:[^ ,:]+]] = arith.constant 0 : index
! CHECK: %[[EXTENT:[^ ,:]+]] = arith.select %[[PRESENT]], %[[ONE]], %[[ABSENT_EXTENT]] : index
! CHECK: %[[BOUNDS:[^ ,:]+]] = omp.map.bounds lower_bound(%[[ZERO]] : index) upper_bound(%[[UB]] : index) extent(%[[EXTENT]] : index)
! CHECK: %[[MAP:[^ ,:]+]] = omp.map.info var_ptr(%[[X]]#1 : !fir.ref<f32>, f32) map_clauses(storage) capture(ByRef) bounds(%[[BOUNDS]])
! CHECK: omp.target {{.*}}map_entries(%[[MAP]] -> %[[ARG:[^ ,:]+]],
! CHECK: %[[DEVICE_X:[^ ,:]+]]:2 = hlfir.declare %[[ARG]] {fortran_attrs = #fir.var_attrs<optional>
! CHECK: fir.is_present %[[DEVICE_X]]#0

subroutine array(n, x, found)
  integer :: n
  real, optional :: x(n, 3)
  logical :: found
  !$omp target map(from:found)
    found = present(x)
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QParray(
! CHECK: %[[X:[^ ,:]+]]:2 = hlfir.declare {{.*}}fortran_attrs = #fir.var_attrs<optional>
! CHECK: omp.map.bounds lower_bound(%[[LB0:[^ ,:]+]] : index) upper_bound(%[[UB0:[^ ,:]+]] : index) extent(%[[EXT0:[^ ,:]+]] : index)
! CHECK: omp.map.bounds lower_bound(%[[LB1:[^ ,:]+]] : index) upper_bound(%[[UB1:[^ ,:]+]] : index) extent(%[[EXT1:[^ ,:]+]] : index)
! CHECK: %[[PRESENT:[^ ,:]+]] = fir.is_present %[[X]]#1
! CHECK: %[[ZERO0:[^ ,:]+]] = arith.constant 0 : index
! CHECK: %[[NEW_LB0:[^ ,:]+]] = arith.select %[[PRESENT]], %[[LB0]], %[[ZERO0]] : index
! CHECK: %[[NEGONE0:[^ ,:]+]] = arith.constant -1 : index
! CHECK: %[[NEW_UB0:[^ ,:]+]] = arith.select %[[PRESENT]], %[[UB0]], %[[NEGONE0]] : index
! CHECK: %[[ZERO_EXT0:[^ ,:]+]] = arith.constant 0 : index
! CHECK: %[[NEW_EXT0:[^ ,:]+]] = arith.select %[[PRESENT]], %[[EXT0]], %[[ZERO_EXT0]] : index
! CHECK: %[[BOUNDS0:[^ ,:]+]] = omp.map.bounds lower_bound(%[[NEW_LB0]] : index) upper_bound(%[[NEW_UB0]] : index) extent(%[[NEW_EXT0]] : index)
! CHECK: %[[ZERO1:[^ ,:]+]] = arith.constant 0 : index
! CHECK: %[[NEW_LB1:[^ ,:]+]] = arith.select %[[PRESENT]], %[[LB1]], %[[ZERO1]] : index
! CHECK: %[[NEGONE1:[^ ,:]+]] = arith.constant -1 : index
! CHECK: %[[NEW_UB1:[^ ,:]+]] = arith.select %[[PRESENT]], %[[UB1]], %[[NEGONE1]] : index
! CHECK: %[[ZERO_EXT1:[^ ,:]+]] = arith.constant 0 : index
! CHECK: %[[NEW_EXT1:[^ ,:]+]] = arith.select %[[PRESENT]], %[[EXT1]], %[[ZERO_EXT1]] : index
! CHECK: %[[BOUNDS1:[^ ,:]+]] = omp.map.bounds lower_bound(%[[NEW_LB1]] : index) upper_bound(%[[NEW_UB1]] : index) extent(%[[NEW_EXT1]] : index)
! CHECK: omp.map.info var_ptr(%[[X]]#1 : !fir.ref<!fir.array<?x3xf32>>, f32) map_clauses(implicit, tofrom) capture(ByRef) bounds(%[[BOUNDS0]], %[[BOUNDS1]])

! A section must also have zero offsets when absent, so that its null base
! address is not incremented when computing the start of the mapped data.
subroutine section(n, x, found)
  integer :: n
  real, optional :: x(n)
  logical :: found
  !$omp target map(tofrom:x(2:n)) map(from:found)
    found = present(x)
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPsection(
! CHECK: %[[X:[^ ,:]+]]:2 = hlfir.declare {{.*}}fortran_attrs = #fir.var_attrs<optional>
! CHECK: omp.map.bounds lower_bound(%[[LB:[^ ,:]+]] : index) upper_bound(%[[UB:[^ ,:]+]] : index) extent(%[[EXT:[^ ,:]+]] : index)
! CHECK: %[[PRESENT:[^ ,:]+]] = fir.is_present %[[X]]#1
! CHECK: %[[ZERO:[^ ,:]+]] = arith.constant 0 : index
! CHECK: %[[NEW_LB:[^ ,:]+]] = arith.select %[[PRESENT]], %[[LB]], %[[ZERO]] : index
! CHECK: %[[NEGONE:[^ ,:]+]] = arith.constant -1 : index
! CHECK: %[[NEW_UB:[^ ,:]+]] = arith.select %[[PRESENT]], %[[UB]], %[[NEGONE]] : index
! CHECK: %[[ZERO_EXT:[^ ,:]+]] = arith.constant 0 : index
! CHECK: %[[NEW_EXT:[^ ,:]+]] = arith.select %[[PRESENT]], %[[EXT]], %[[ZERO_EXT]] : index
! CHECK: %[[BOUNDS:[^ ,:]+]] = omp.map.bounds lower_bound(%[[NEW_LB]] : index) upper_bound(%[[NEW_UB]] : index) extent(%[[NEW_EXT]] : index)
! CHECK: omp.map.info var_ptr(%[[X]]#1 : !fir.ref<!fir.array<?xf32>>, f32) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]])

! Nonoptional scalar mappings do not need presence checks or bounds.
subroutine nonoptional(x)
  real :: x
  !$omp target map(tofrom:x)
    x = x + 1
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPnonoptional(
! CHECK-NOT: fir.is_present
! CHECK-NOT: omp.map.bounds
! CHECK: omp.target
