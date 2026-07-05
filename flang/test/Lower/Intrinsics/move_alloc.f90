  ! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPto_from_only
subroutine to_from_only
  integer, allocatable :: from(:), to(:)
  ! CHECK: %[[a1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK: %[[from:.*]]:2 = hlfir.declare %[[a1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFto_from_onlyEfrom"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  ! CHECK: %[[b1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK: %[[to:.*]]:2 = hlfir.declare %[[b1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFto_from_onlyEto"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  allocate(from(20))
  ! CHECK: %[[errMsg:.*]] = fir.absent !fir.box<none>
  ! CHECK: %[[false:.*]] = arith.constant false
  ! CHECK: %[[to_conv:.*]] = fir.convert %[[to]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[from_conv:.*]] = fir.convert %[[from]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  call move_alloc(from, to)
  ! CHECK: fir.call @_FortranAMoveAlloc(%[[to_conv]], %[[from_conv]], %{{.*}}, %[[false]], %[[errMsg]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
end subroutine to_from_only

! CHECK-LABEL: func @_QPto_from_stat
subroutine to_from_stat
  integer, allocatable :: from(:), to(:)
  integer :: stat
  ! CHECK-DAG: %[[a1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[from:.*]]:2 = hlfir.declare %[[a1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFto_from_statEfrom"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  ! CHECK-DAG: %[[b1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[to:.*]]:2 = hlfir.declare %[[b1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFto_from_statEto"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  allocate(from(20))
  ! CHECK: %[[errMsg:.*]] = fir.absent !fir.box<none>
  ! CHECK: %[[true:.*]] = arith.constant true
  ! CHECK: %[[to_conv:.*]] = fir.convert %[[to]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[from_conv:.*]] = fir.convert %[[from]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  call move_alloc(from, to, stat)
  ! CHECK: %[[stat:.*]] = fir.call @_FortranAMoveAlloc(%[[to_conv]], %[[from_conv]], %{{.*}}, %[[true]], %[[errMsg]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
end subroutine to_from_stat

! CHECK-LABEL: func @_QPto_from_stat_errmsg
subroutine to_from_stat_errmsg
  integer, allocatable :: from(:), to(:)
  integer :: stat
  character :: errMsg*64
  ! CHECK-DAG: %[[errMsg1:.*]] = fir.alloca !fir.char<1,64>
  ! CHECK-DAG: %[[errMsg_decl:.*]]:2 = hlfir.declare %[[errMsg1]] typeparams {{.*}} {uniq_name = "_QFto_from_stat_errmsgEerrmsg"} : (!fir.ref<!fir.char<1,64>>, index) -> (!fir.ref<!fir.char<1,64>>, !fir.ref<!fir.char<1,64>>)
  ! CHECK-DAG: %[[a1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[from:.*]]:2 = hlfir.declare %[[a1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFto_from_stat_errmsgEfrom"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  ! CHECK-DAG: %[[b1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[to:.*]]:2 = hlfir.declare %[[b1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFto_from_stat_errmsgEto"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  allocate(from(20))
  ! CHECK: %[[errMsg2:.*]] = fir.embox %[[errMsg_decl]]#0 : (!fir.ref<!fir.char<1,64>>) -> !fir.box<!fir.char<1,64>>
  ! CHECK: %[[true:.*]] = arith.constant true
  ! CHECK: %[[to_conv:.*]] = fir.convert %[[to]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[from_conv:.*]] = fir.convert %[[from]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[errMsg3:.*]] = fir.convert %[[errMsg2]] : (!fir.box<!fir.char<1,64>>) -> !fir.box<none>
  call move_alloc(from, to, stat, errMsg)
  ! CHECK: %[[stat:.*]] = fir.call @_FortranAMoveAlloc(%[[to_conv]], %[[from_conv]], %{{.*}}, %[[true]], %[[errMsg3]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
end subroutine to_from_stat_errmsg
