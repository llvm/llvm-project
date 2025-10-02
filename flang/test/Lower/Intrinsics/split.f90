! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPsplit_test1(
! CHECK-SAME: %[[s1:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[s2:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[p:[^:]+]]: !fir.ref<i32>{{.*}})
subroutine split_test1(s1, s2, p)
character(*) :: s1, s2
integer :: p
! CHECK: %[[c1:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[c2:.*]]:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[pos:.*]] = fir.load %arg2 : !fir.ref<i32>
! CHECK: %false = arith.constant false
! CHECK: %[[c1base:.*]] = fir.convert %[[c1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: %[[c1len:.*]] = fir.convert %[[c1]]#1 : (index) -> i64
! CHECK: %[[c2base:.*]] = fir.convert %[[c2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: %[[c2len:.*]] = fir.convert %[[c2]]#1 : (index) -> i64
! CHECK: %[[pos1:.*]] = fir.convert %[[pos]] : (i32) -> i64
! CHECK: %[[pos2:.*]] = fir.call @_FortranASplit1(%[[c1base]], %[[c1len]], %[[c2base]], %[[c2len]], %[[pos1]], %false, {{.*}}) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i64, i1, !fir.ref<i8>, i32) -> i64
! CHECK: %[[pos3:.*]] = fir.convert %[[pos2]] : (i64) -> i32
! CHECK: fir.store %[[pos3]] to %[[p]] : !fir.ref<i32>
! CHECK: return
call split(s1, s2, p)
end subroutine split_test1

! CHECK-LABEL: func @_QPsplit_test2(
! CHECK-SAME: %[[s1:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[s2:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[p:[^:]+]]: !fir.ref<i32>{{.*}})
subroutine split_test2(s1, s2, p)
character(*) :: s1, s2
integer :: p
! CHECK: %[[c1:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[c2:.*]]:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %true = arith.constant true
! CHECK: %[[pos:.*]] = fir.load %arg2 : !fir.ref<i32>
! CHECK: %[[c1base:.*]] = fir.convert %[[c1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: %[[c1len:.*]] = fir.convert %[[c1]]#1 : (index) -> i64
! CHECK: %[[c2base:.*]] = fir.convert %[[c2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: %[[c2len:.*]] = fir.convert %[[c2]]#1 : (index) -> i64
! CHECK: %[[pos1:.*]] = fir.convert %[[pos]] : (i32) -> i64
! CHECK: %[[pos2:.*]] = fir.call @_FortranASplit1(%[[c1base]], %[[c1len]], %[[c2base]], %[[c2len]], %[[pos1]], %true, {{.*}}) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i64, i1, !fir.ref<i8>, i32) -> i64
! CHECK: %[[pos3:.*]] = fir.convert %[[pos2]] : (i64) -> i32
! CHECK: fir.store %[[pos3]] to %[[p]] : !fir.ref<i32>
! CHECK: return
call split(s1, s2, p, .true.)
end subroutine split_test2
