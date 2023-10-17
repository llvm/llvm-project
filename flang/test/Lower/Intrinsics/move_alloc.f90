  ! RUN: bbc --use-desc-for-alloc=false -emit-fir %s -o - | FileCheck  %s
  ! RUN: flang-new -fc1 -mllvm --use-desc-for-alloc=false -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: to_from_only
subroutine to_from_only
  ! CHECK: %[[a1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK: %[[b1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  integer, allocatable :: from(:), to(:)
  allocate(from(20))
  ! CHECK: %[[errMsg:.*]] = fir.absent !fir.box<none>
  ! CHECK: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[a2:.*]] = fir.convert %[[a1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[b2:.*]] = fir.convert %[[b1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  call move_alloc(from, to)
  ! CHECK: fir.call @_FortranAMoveAlloc(%[[b2]], %[[a2]], %{{.*}}, %[[false]], %[[errMsg]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
  ! CHECK-DAG:  %[[a3:.*]] = fir.load %[[a1:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK-DAG:  %[[a4:.*]] = fir.box_addr %[[a3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK-DAG:  %[[b3:.*]] = fir.load %[[b1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK-DAG:  %[[b4:.*]] = fir.box_addr %[[b3:.*]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
end subroutine to_from_only

! CHECK-LABEL: to_from_stat
subroutine to_from_stat
  ! CHECK-DAG: %[[a1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[b1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  integer, allocatable :: from(:), to(:)
  ! CHECK-DAG: %[[stat1:.*]] = fir.alloca i32
  integer :: stat
  allocate(from(20))
  ! CHECK: %[[errMsg:.*]] = fir.absent !fir.box<none>
  ! CHECK: %[[true:.*]] = arith.constant true
  ! CHECK-DAG: %[[a2:.*]] = fir.convert %[[a1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[b2:.*]] = fir.convert %[[b1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  call move_alloc(from, to, stat)
  ! CHECK: %[[stat:.*]] = fir.call @_FortranAMoveAlloc(%[[b2]], %[[a2]], %{{.*}}, %[[true]], %[[errMsg]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
  ! CHECK-DAG:  %[[a3:.*]] = fir.load %[[a1:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK-DAG:  %[[a4:.*]] = fir.box_addr %[[a3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK-DAG:  %[[b3:.*]] = fir.load %[[b1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK-DAG:  %[[b4:.*]] = fir.box_addr %[[b3:.*]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
end subroutine to_from_stat

! CHECK-LABEL: to_from_stat_errmsg
subroutine to_from_stat_errmsg
  ! CHECK-DAG: %[[errMsg1:.*]] = fir.alloca !fir.char<1,64>
  ! CHECK-DAG: %[[a1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[b1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  integer, allocatable :: from(:), to(:)
  ! CHECK-DAG: %[[stat1:.*]] = fir.alloca i32
  integer :: stat
  character :: errMsg*64
  allocate(from(20))
  ! CHECK: %[[errMsg2:.*]] = fir.embox %[[errMsg1]] : (!fir.ref<!fir.char<1,64>>) -> !fir.box<!fir.char<1,64>>
  ! CHECK: %[[true:.*]] = arith.constant true
  ! CHECK-DAG: %[[errMsg3:.*]] = fir.convert %[[errMsg2]] : (!fir.box<!fir.char<1,64>>) -> !fir.box<none>
  ! CHECK-DAG: %[[a2:.*]] = fir.convert %[[a1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[b2:.*]] = fir.convert %[[b1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  call move_alloc(from, to, stat, errMsg)
  ! CHECK: %[[stat:.*]] = fir.call @_FortranAMoveAlloc(%[[b2]], %[[a2]], %{{.*}}, %[[true]], %[[errMsg3]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
  ! CHECK-DAG:  %[[a3:.*]] = fir.load %[[a1:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK-DAG:  %[[a4:.*]] = fir.box_addr %[[a3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK-DAG:  %[[b3:.*]] = fir.load %[[b1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK-DAG:  %[[b4:.*]] = fir.box_addr %[[b3:.*]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
end subroutine to_from_stat_errmsg
