! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=51 %s -o /dev/null

! CHECK-LABEL: func.func @_QPstandalone_common
! CHECK: %[[STANDALONE_X_MAP:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "x"\}|name\("x"\))}}
! CHECK: %[[STANDALONE_Y_MAP:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "y"\}|name\("y"\))}}
! CHECK: omp.target_data use_device_addr(%[[STANDALONE_X_MAP]] -> %[[STANDALONE_X_ARG:.*]], %[[STANDALONE_Y_MAP]] -> %[[STANDALONE_Y_ARG:.*]] : {{.*}}) {
! CHECK: %[[STANDALONE_X:.*]]:2 = hlfir.declare %[[STANDALONE_X_ARG]]
! CHECK: %[[STANDALONE_Y:.*]]:2 = hlfir.declare %[[STANDALONE_Y_ARG]]
! CHECK: %[[STANDALONE_LOAD:.*]] = fir.load %[[STANDALONE_Y]]#0
! CHECK: %[[STANDALONE_SUM:.*]] = arith.addi %[[STANDALONE_LOAD]]
! CHECK: hlfir.assign %[[STANDALONE_SUM]] to %[[STANDALONE_X]]#0
subroutine standalone_common
  integer :: x, y
  common /standalone/ x, y
  !$omp target data use_device_addr(/standalone/)
    x = y + 1
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPall_cptr_common
! CHECK-NOT: use_device_ptr
! CHECK: %[[CPTR_P_MAP:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "p"\}|name\("p"\))}}
! CHECK: %[[CPTR_Q_MAP:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "q"\}|name\("q"\))}}
! CHECK: omp.target_data use_device_addr(%[[CPTR_P_MAP]] -> %[[CPTR_P_ARG:.*]], %[[CPTR_Q_MAP]] -> %[[CPTR_Q_ARG:.*]] : {{.*}}) {
! CHECK-NOT: use_device_ptr
! CHECK: %[[CPTR_P:.*]]:2 = hlfir.declare %[[CPTR_P_ARG]]
! CHECK: %[[CPTR_Q:.*]]:2 = hlfir.declare %[[CPTR_Q_ARG]]
! CHECK: hlfir.assign %[[CPTR_Q]]#0 to %[[CPTR_P]]#0
subroutine all_cptr_common
  use iso_c_binding, only : c_ptr
  type(c_ptr) :: p, q
  common /all_cptr/ p, q
  !$omp target data use_device_addr(/all_cptr/)
    p = q
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPmixed_common
! CHECK-NOT: use_device_ptr
! CHECK: %[[MIXED_P_MAP:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "p"\}|name\("p"\))}}
! CHECK: %[[MIXED_X_MAP:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "x"\}|name\("x"\))}}
! CHECK: omp.target_data use_device_addr(%[[MIXED_P_MAP]] -> %[[MIXED_P_ARG:.*]], %[[MIXED_X_MAP]] -> %[[MIXED_X_ARG:.*]] : {{.*}}) {
! CHECK-NOT: use_device_ptr
! CHECK: %[[MIXED_P:.*]]:2 = hlfir.declare %[[MIXED_P_ARG]]
! CHECK: %[[MIXED_X:.*]]:2 = hlfir.declare %[[MIXED_X_ARG]]
! CHECK: %[[MIXED_LOAD:.*]] = fir.load %[[MIXED_X]]#0
! CHECK: hlfir.assign %{{.*}} to %[[MIXED_X]]#0
subroutine mixed_common
  use iso_c_binding, only : c_ptr
  type(c_ptr) :: p
  integer :: x
  common /mixed/ p, x
  !$omp target data use_device_addr(/mixed/)
    x = x + 1
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPmapped_common
! CHECK: %[[MAPPED_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} {{(\{name = "mapped"\}|name\("mapped"\))}}
! CHECK: %[[MAPPED_UDA:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "mapped"\}|name\("mapped"\))}}
! CHECK: omp.target_data map_entries(%[[MAPPED_MAP]] : {{.*}}) use_device_addr(%[[MAPPED_UDA]] -> %[[MAPPED_ARG:.*]] : {{.*}}) {
! CHECK: %[[MAPPED_X_COORD:.*]] = fir.coordinate_of %[[MAPPED_ARG]], %{{.*}}
! CHECK: %[[MAPPED_X_REF:.*]] = fir.convert %[[MAPPED_X_COORD]]
! CHECK: %[[MAPPED_X:.*]]:2 = hlfir.declare %[[MAPPED_X_REF]] storage(%[[MAPPED_ARG]][0])
! CHECK: %[[MAPPED_Y_COORD:.*]] = fir.coordinate_of %[[MAPPED_ARG]], %{{.*}}
! CHECK: %[[MAPPED_Y_REF:.*]] = fir.convert %[[MAPPED_Y_COORD]]
! CHECK: %[[MAPPED_Y:.*]]:2 = hlfir.declare %[[MAPPED_Y_REF]] storage(%[[MAPPED_ARG]][4])
! CHECK: %[[MAPPED_LOAD:.*]] = fir.load %[[MAPPED_Y]]#0
! CHECK: %[[MAPPED_SUM:.*]] = arith.addi %[[MAPPED_LOAD]]
! CHECK: hlfir.assign %[[MAPPED_SUM]] to %[[MAPPED_X]]#0
subroutine mapped_common
  integer :: x, y
  common /mapped/ x, y
  !$omp target data map(tofrom: /mapped/) use_device_addr(/mapped/)
    x = y + 1
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPexplicit_map_common
! CHECK: %[[EXPLICIT_X_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} {{(\{name = "x"\}|name\("x"\))}}
! CHECK: %[[EXPLICIT_Y_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} {{(\{name = "y"\}|name\("y"\))}}
! CHECK: %[[EXPLICIT_X_UDA:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "x"\}|name\("x"\))}}
! CHECK: %[[EXPLICIT_Y_UDA:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "y"\}|name\("y"\))}}
! CHECK: omp.target_data map_entries(%[[EXPLICIT_X_MAP]], %[[EXPLICIT_Y_MAP]] : {{.*}}) use_device_addr(%[[EXPLICIT_X_UDA]] -> %[[EXPLICIT_X_ARG:.*]], %[[EXPLICIT_Y_UDA]] -> %[[EXPLICIT_Y_ARG:.*]] : {{.*}}) {
! CHECK: %[[EXPLICIT_X:.*]]:2 = hlfir.declare %[[EXPLICIT_X_ARG]]
! CHECK: %[[EXPLICIT_Y:.*]]:2 = hlfir.declare %[[EXPLICIT_Y_ARG]]
! CHECK: %[[EXPLICIT_LOAD:.*]] = fir.load %[[EXPLICIT_Y]]#0
! CHECK: hlfir.assign %{{.*}} to %[[EXPLICIT_X]]#0
subroutine explicit_map_common
  integer :: x, y
  common /explicit_map/ x, y
  !$omp target data map(tofrom: x, y) use_device_addr(/explicit_map/)
    x = y + 1
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPouter_map_common
! CHECK: %[[OUTER_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} {{(\{name = "outer_map"\}|name\("outer_map"\))}}
! CHECK: omp.target_data map_entries(%[[OUTER_MAP]] : {{.*}}) {
! CHECK: %[[OUTER_X_UDA:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "x"\}|name\("x"\))}}
! CHECK: %[[OUTER_Y_UDA:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "y"\}|name\("y"\))}}
! CHECK: omp.target_data use_device_addr(%[[OUTER_X_UDA]] -> %[[OUTER_X_ARG:.*]], %[[OUTER_Y_UDA]] -> %[[OUTER_Y_ARG:.*]] : {{.*}}) {
! CHECK: %[[OUTER_X:.*]]:2 = hlfir.declare %[[OUTER_X_ARG]]
! CHECK: %[[OUTER_Y:.*]]:2 = hlfir.declare %[[OUTER_Y_ARG]]
! CHECK: %[[OUTER_LOAD:.*]] = fir.load %[[OUTER_Y]]#0
! CHECK: hlfir.assign %{{.*}} to %[[OUTER_X]]#0
subroutine outer_map_common
  integer :: x, y
  common /outer_map/ x, y
  !$omp target data map(tofrom: /outer_map/)
    !$omp target data use_device_addr(/outer_map/)
      x = y + 1
    !$omp end target data
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPreordered_common
! CHECK: %[[SECOND_R_MAP:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "r"\}|name\("r"\))}}
! CHECK: %[[SECOND_S_MAP:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "s"\}|name\("s"\))}}
! CHECK: %[[PTR_CHILD:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = ""\}|name\(""\))}}
! CHECK: %[[PTR_MAP:.*]] = omp.map.info {{.*}} members(%[[PTR_CHILD]] {{.*}}) {{.*}}{{(\{name = "ptr"\}|name\("ptr"\))}}
! CHECK: %[[PTR_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} {{(\{name = "ptr"\}|name\("ptr"\))}}
! CHECK: %[[FIRST_A_MAP:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "a"\}|name\("a"\))}}
! CHECK: %[[FIRST_B_MAP:.*]] = omp.map.info {{.*}} map_clauses(return_param) {{.*}} {{(\{name = "b"\}|name\("b"\))}}
! CHECK: omp.target_data map_entries(%[[PTR_ATTACH]] : {{.*}}) use_device_addr(%[[SECOND_R_MAP]] -> %[[SECOND_R_ARG:.*]], %[[SECOND_S_MAP]] -> %[[SECOND_S_ARG:.*]], %[[PTR_MAP]] -> %[[PTR_ARG:.*]], %[[FIRST_A_MAP]] -> %[[FIRST_A_ARG:.*]], %[[FIRST_B_MAP]] -> %[[FIRST_B_ARG:.*]], %[[PTR_CHILD]] -> %[[PTR_CHILD_ARG:.*]] : {{.*}}) {
! CHECK: %[[SECOND_R:.*]]:2 = hlfir.declare %[[SECOND_R_ARG]]
! CHECK: %[[SECOND_S:.*]]:2 = hlfir.declare %[[SECOND_S_ARG]]
! CHECK: %[[PTR:.*]]:2 = hlfir.declare %[[PTR_ARG]]
! CHECK: %[[FIRST_A:.*]]:2 = hlfir.declare %[[FIRST_A_ARG]]
! CHECK: %[[FIRST_B:.*]]:2 = hlfir.declare %[[FIRST_B_ARG]]
! CHECK: %[[SECOND_LOAD:.*]] = fir.load %[[SECOND_S]]#0
! CHECK: %[[FIRST_LOAD:.*]] = fir.load %[[FIRST_A]]#0
! CHECK: hlfir.assign %{{.*}} to %[[SECOND_R]]#0
! CHECK: %[[FIRST_B_LOAD:.*]] = fir.load %[[FIRST_B]]#0
! CHECK: %[[PTR_LOAD:.*]] = fir.load %[[PTR]]#0
subroutine reordered_common(ptr)
  integer, pointer :: ptr
  integer :: a, b
  real :: r, s
  common /first/ a, b
  common /second/ r, s
  !$omp target data use_device_addr(/second/, ptr, /first/)
    r = s + real(a)
    ptr = b
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPnested_common
! CHECK: %[[HOST_X:.*]]:2 = hlfir.declare {{.*}} storage({{.*}}[0])
! CHECK: %[[HOST_Y:.*]]:2 = hlfir.declare {{.*}} storage({{.*}}[4])
! CHECK: omp.target_data use_device_addr({{.*}} -> %[[OUTER_X_ARG:.*]], {{.*}} -> %[[OUTER_Y_ARG:.*]] : {{.*}}) {
! CHECK: %[[NESTED_OUTER_X:.*]]:2 = hlfir.declare %[[OUTER_X_ARG]]
! CHECK: %[[NESTED_OUTER_Y:.*]]:2 = hlfir.declare %[[OUTER_Y_ARG]]
! CHECK: %[[OUTER_INITIAL_LOAD:.*]] = fir.load %[[NESTED_OUTER_Y]]#0
! CHECK: hlfir.assign %{{.*}} to %[[NESTED_OUTER_X]]#0
! CHECK: omp.target_data use_device_addr({{.*}} -> %[[INNER_X_ARG:.*]], {{.*}} -> %[[INNER_Y_ARG:.*]] : {{.*}}) {
! CHECK: %[[NESTED_INNER_X:.*]]:2 = hlfir.declare %[[INNER_X_ARG]]
! CHECK: %[[NESTED_INNER_Y:.*]]:2 = hlfir.declare %[[INNER_Y_ARG]]
! CHECK: %[[INNER_LOAD:.*]] = fir.load %[[NESTED_INNER_X]]#0
! CHECK: hlfir.assign %{{.*}} to %[[NESTED_INNER_Y]]#0
! CHECK: %[[OUTER_RESTORED_LOAD:.*]] = fir.load %[[NESTED_OUTER_Y]]#0
! CHECK: hlfir.assign %{{.*}} to %[[NESTED_OUTER_X]]#0
! CHECK: %[[HOST_RESTORED_LOAD:.*]] = fir.load %[[HOST_X]]#0
! CHECK: hlfir.assign %{{.*}} to %[[HOST_Y]]#0
subroutine nested_common
  integer :: x, y
  common /nested/ x, y
  !$omp target data use_device_addr(/nested/)
    x = y + 1
    !$omp target data use_device_addr(/nested/)
      y = x + 2
    !$omp end target data
    x = y + 3
  !$omp end target data
  y = x + 4
end subroutine

! CHECK-LABEL: func.func @_QPunstructured_common
! CHECK: omp.target_data use_device_addr({{.*}} -> %[[UNSTRUCTURED_X_ARG:.*]], {{.*}} -> %[[UNSTRUCTURED_Y_ARG:.*]] : {{.*}}) {
! CHECK: %[[UNSTRUCTURED_X:.*]]:2 = hlfir.declare %[[UNSTRUCTURED_X_ARG]]
! CHECK: %[[UNSTRUCTURED_Y:.*]]:2 = hlfir.declare %[[UNSTRUCTURED_Y_ARG]]
! CHECK: %[[UNSTRUCTURED_Y_LOAD:.*]] = fir.load %[[UNSTRUCTURED_Y]]#0
! CHECK: hlfir.assign %{{.*}} to %[[UNSTRUCTURED_X]]#0
! CHECK: cf.br ^[[DEST:.*]]
! CHECK: ^[[DEST]]:
! CHECK: %[[UNSTRUCTURED_X_LOAD:.*]] = fir.load %[[UNSTRUCTURED_X]]#0
! CHECK: hlfir.assign %{{.*}} to %[[UNSTRUCTURED_Y]]#0
subroutine unstructured_common
  integer :: x, y
  common /unstructured/ x, y
  !$omp target data use_device_addr(/unstructured/)
    x = y + 1
    goto 10
10  y = x + 2
  !$omp end target data
end subroutine
