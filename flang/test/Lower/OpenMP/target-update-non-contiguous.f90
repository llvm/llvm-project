! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! A scalar subscript has extent one, but subsequent physical strides must
! include the full source extent of that dimension. Both motion directions
! need these strides to describe non-contiguous array sections.

! CHECK-LABEL: func.func @_QPupdate_to(
! CHECK: %[[ARRAY:.*]]:2 = hlfir.declare %arg0(
! CHECK: %[[D0:.*]]:3 = fir.box_dims %[[ARRAY]]#0,
! CHECK: %[[B0:.*]] = omp.map.bounds {{.*}} stride(%[[D0]]#2 : index) {{.*}} stride_in_bytes(true)
! CHECK: %[[D1:.*]]:3 = fir.box_dims %[[ARRAY]]#0,
! CHECK: %[[B1:.*]] = omp.map.bounds {{.*}} stride(%[[D1]]#2 : index) {{.*}} stride_in_bytes(true)
! CHECK: %[[D2:.*]]:3 = fir.box_dims %[[ARRAY]]#0,
! CHECK: %[[B2:.*]] = omp.map.bounds {{.*}} stride(%[[D2]]#2 : index) {{.*}} stride_in_bytes(true)
! CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(to) {{.*}} bounds(%[[B0]], %[[B1]], %[[B2]])
! CHECK: omp.target_update map_entries(%[[MAP]]
subroutine update_to(a, n, m, k)
  integer :: n, m, k
  real :: a(n, m, k)
  !$omp target update to(a(2, 3:, :))
end subroutine

! CHECK-LABEL: func.func @_QPupdate_from(
! CHECK: %[[ARRAY:.*]]:2 = hlfir.declare %arg0(
! CHECK: %[[D0:.*]]:3 = fir.box_dims %[[ARRAY]]#0,
! CHECK: %[[B0:.*]] = omp.map.bounds {{.*}} stride(%[[D0]]#2 : index) {{.*}} stride_in_bytes(true)
! CHECK: %[[D1:.*]]:3 = fir.box_dims %[[ARRAY]]#0,
! CHECK: %[[B1:.*]] = omp.map.bounds {{.*}} stride(%[[D1]]#2 : index) {{.*}} stride_in_bytes(true)
! CHECK: %[[D2:.*]]:3 = fir.box_dims %[[ARRAY]]#0,
! CHECK: %[[B2:.*]] = omp.map.bounds {{.*}} stride(%[[D2]]#2 : index) {{.*}} stride_in_bytes(true)
! CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(from) {{.*}} bounds(%[[B0]], %[[B1]], %[[B2]])
! CHECK: omp.target_update map_entries(%[[MAP]]
subroutine update_from(a, n, m, k)
  integer :: n, m, k
  real :: a(n, m, k)
  !$omp target update from(a(2, 3:, :))
end subroutine

! CHECK-LABEL: func.func @_QPupdate_explicit(
! CHECK: %[[ONE:.*]] = arith.constant 1 : index
! CHECK: %[[FOUR:.*]] = arith.constant 4 : index
! CHECK: %[[B0:.*]] = omp.map.bounds {{.*}} extent(%[[ONE]] : index) stride(%[[ONE]] : index)
! CHECK: %[[TWENTY:.*]] = arith.constant 20 : index
! CHECK: %[[B1:.*]] = omp.map.bounds {{.*}} stride(%[[FOUR]] : index)
! CHECK: %[[B2:.*]] = omp.map.bounds {{.*}} stride(%[[TWENTY]] : index)
! CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(to) {{.*}} bounds(%[[B0]], %[[B1]], %[[B2]])
! CHECK: omp.target_update map_entries(%[[MAP]]
! CHECK: %[[FROM_FOUR:.*]] = arith.constant 4 : index
! CHECK: %[[FROM_B0:.*]] = omp.map.bounds {{.*}} extent(%{{.*}} : index) stride(%{{.*}} : index)
! CHECK: %[[FROM_TWENTY:.*]] = arith.constant 20 : index
! CHECK: %[[FROM_B1:.*]] = omp.map.bounds {{.*}} stride(%[[FROM_FOUR]] : index)
! CHECK: %[[FROM_B2:.*]] = omp.map.bounds {{.*}} stride(%[[FROM_TWENTY]] : index)
! CHECK: %[[FROM_MAP:.*]] = omp.map.info {{.*}} map_clauses(from) {{.*}} bounds(%[[FROM_B0]], %[[FROM_B1]], %[[FROM_B2]])
! CHECK: omp.target_update map_entries(%[[FROM_MAP]]
subroutine update_explicit(a)
  real :: a(4, 5, 6)
  !$omp target update to(a(2, 3:, :))
  !$omp target update from(a(2, 3:, :))
end subroutine
