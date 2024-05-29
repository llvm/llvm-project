! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

module spread_mod

type :: p1
  integer :: a
end type

type, extends(p1) :: p2
  integer :: b
end type

contains

! CHECK-LABEL: func @_QMspread_modPspread_test(
! CHECK-SAME: %[[arg0:[^:]+]]: !fir.ref<i32>{{.*}}, %[[arg1:[^:]+]]: !fir.ref<i32>{{.*}}, %[[arg2:[^:]+]]: !fir.ref<i32>{{.*}}, %[[arg3:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine spread_test(s,d,n,r)
    integer :: s,d,n
    integer :: r(:)
  ! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG:  %[[a1:.*]] = fir.load %[[arg1]] : !fir.ref<i32>
  ! CHECK-DAG:  %[[a2:.*]] = fir.load %[[arg2]] : !fir.ref<i32>
  ! CHECK-DAG:  %[[a3:.*]] = fir.embox %[[arg0]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG:  %[[a9:.*]] = fir.convert %[[a3]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK-DAG:  %[[a10:.*]] = fir.convert %[[a2]] : (i32) -> i64
    r = spread(s,d,n)
  ! CHECK:  %{{.*}} = fir.call @_FortranASpread(%[[a8]], %[[a9]], %[[a1]], %[[a10]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i64, !fir.ref<i8>, i32) -> none
  ! CHECK-DAG:  %[[a13:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK-DAG:  %[[a15:.*]] = fir.box_addr %[[a13]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK:  fir.freemem %[[a15]]
end subroutine
  
! CHECK-LABEL: func @_QMspread_modPspread_test2(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[arg1:[^:]+]]: !fir.ref<i32>{{.*}}, %[[arg2:[^:]+]]: !fir.ref<i32>{{.*}}, %[[arg3:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}) {
subroutine spread_test2(s,d,n,r)
    integer :: s(:),d,n
    integer :: r(:,:)
  ! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK-DAG:  %[[a1:.*]] = fir.load %[[arg1]] : !fir.ref<i32>
  ! CHECK-DAG:  %[[a2:.*]] = fir.load %[[arg2]] : !fir.ref<i32>
  ! CHECK-DAG:  %[[a7:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG:  %[[a9:.*]] = fir.convert %[[a2]] : (i32) -> i64
    r = spread(s,d,n)
  ! CHECK:  %{{.*}} = fir.call @_FortranASpread(%[[a7]], %[[a8]], %[[a1]], %[[a9]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i64, !fir.ref<i8>, i32) -> none
  ! CHECK-DAG:  %[[a12:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK-DAG:  %[[a15:.*]] = fir.box_addr %[[a12]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
  ! CHECK:  fir.freemem %[[a15:.*]]
end subroutine

! CHECK-LABEL: func.func @_QMspread_modPspread_test_polymorphic_source(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>> {fir.bindc_name = "p"}) {
subroutine spread_test_polymorphic_source(p)
  class(*), pointer :: p(:,:)
  class(*), allocatable :: r(:,:,:)
  r = spread(p(:,::2), dim=1, ncopies=2)
! CHECK: %[[res:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x?xnone>>>
! CHECK: %[[load_p:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>>
! CHECK: %[[source_box:.*]] = fir.rebox %[[load_p]](%{{.*}}) [%{{.*}}] : (!fir.class<!fir.ptr<!fir.array<?x?xnone>>>, !fir.shift<2>, !fir.slice<2>) -> !fir.class<!fir.array<?x?xnone>>
! CHECK: %[[embox:.*]] = fir.embox %{{.*}}(%{{.*}}) source_box %[[source_box]] : (!fir.heap<!fir.array<?x?x?xnone>>, !fir.shape<3>, !fir.class<!fir.array<?x?xnone>>) -> !fir.class<!fir.heap<!fir.array<?x?x?xnone>>>
! CHECK: fir.store %[[embox]] to %[[res]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x?xnone>>>>
! CHECK: %[[res_box_none:.*]] = fir.convert %[[res]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[source_box_none:.*]] = fir.convert %[[source_box]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranASpread(%[[res_box_none]], %[[source_box_none]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i64, !fir.ref<i8>, i32) -> none

end subroutine

end module
