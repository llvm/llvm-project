! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPscan_test(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function scan_test(s1, s2)
character(*) :: s1, s2
! CHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK-DAG: %[[c:.*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-DAG: %[[cBox:.*]] = fir.embox %[[c]]#0 typeparams %[[c]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-DAG: %[[cBoxNone:.*]] = fir.convert %[[cBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-DAG: %[[c2:.*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-DAG: %[[cBox2:.*]] = fir.embox %[[c2]]#0 typeparams %[[c2]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-DAG: %[[cBoxNone2:.*]] = fir.convert %[[cBox2]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-DAG: %[[backOptBox:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG: %[[backBox:.*]] = fir.convert %[[backOptBox]] : (!fir.box<i1>) -> !fir.box<none>
! CHECK-DAG: %[[kindConstant:.*]] = arith.constant 4 : i32
! CHECK-DAG: %[[resBox:.*]] = fir.convert %[[tmpBox:.*]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: fir.call @{{.*}}Scan(%[[resBox]], %[[cBoxNone]], %[[cBoxNone2]], %[[backBox]], %[[kindConstant]], {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()
scan_test = scan(s1, s2, kind=4)
! CHECK-DAG: %[[tmpAddr:.*]] = fir.box_addr
! CHECK: fir.freemem %[[tmpAddr]]
end function scan_test

! CHECK-LABEL: func @_QPscan_test2(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}},
! CHECK-SAME: %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function scan_test2(s1, s2)
character(*) :: s1, s2
! CHECK: %[[st:[^:]*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[sst:[^:]*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[a1:.*]] = fir.convert %[[st]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: %[[a2:.*]] = fir.convert %[[st]]#1 : (index) -> i64
! CHECK: %[[a3:.*]] = fir.convert %[[sst]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: %[[a4:.*]] = fir.convert %[[sst]]#1 : (index) -> i64
! CHECK: = fir.call @_FortranAScan1(%[[a1]], %[[a2]], %[[a3]], %[[a4]], %{{.*}}) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
scan_test2 = scan(s1, s2, .true.)
end function scan_test2

! CHECK-LABEL: func @_QPtest_optional(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.boxchar<1>
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine test_optional(string, set, back)
character (*) :: string(:), set
logical, optional :: back(:)
print *, scan(string, set, back)
! CHECK:  %[[VAL_11:.*]] = fir.is_present %[[VAL_2]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i1
! CHECK:  %[[VAL_12:.*]] = fir.zero_bits !fir.ref<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_15:.*]] = fir.embox %[[VAL_12]](%[[VAL_14]]) : (!fir.ref<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[VAL_16:.*]] = arith.select %[[VAL_11]], %[[VAL_2]], %[[VAL_15]] : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[VAL_17:.*]] = fir.array_load %[[VAL_16]] {fir.optional} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.array<?x!fir.logical<4>>
! CHECK:  fir.do_loop %[[VAL_25:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<?xi32>) {
! CHECK:  %[[VAL_31:.*]] = fir.if %[[VAL_11]] -> (!fir.logical<4>) {
  ! CHECK:  %[[VAL_32:.*]] = fir.array_fetch %[[VAL_17]], %[[VAL_25]] : (!fir.array<?x!fir.logical<4>>, index) -> !fir.logical<4>
  ! CHECK:  fir.result %[[VAL_32]] : !fir.logical<4>
! CHECK:  } else {
  ! CHECK:  %[[VAL_33:.*]] = arith.constant false
  ! CHECK:  %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i1) -> !fir.logical<4>
  ! CHECK:  fir.result %[[VAL_34]] : !fir.logical<4>
! CHECK:  }
! CHECK:  %[[VAL_39:.*]] = fir.convert %[[VAL_31]] : (!fir.logical<4>) -> i1
! CHECK:  fir.call @_FortranAScan1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_39]]) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK:  }
! CHECK:  fir.array_merge_store
end subroutine

! CHECK-LABEL: func @_QPtest_optional_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.boxchar<1>
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<!fir.logical<4>>
subroutine test_optional_scalar(string, set, back)
character (*) :: string(:), set
logical, optional :: back
print *, scan(string, set, back)
! CHECK:  %[[VAL_11:.*]] = fir.is_present %[[VAL_2]] : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:  %[[VAL_12:.*]] = fir.if %[[VAL_11]] -> (!fir.logical<4>) {
! CHECK:  %[[VAL_13:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.logical<4>>
! CHECK:  fir.result %[[VAL_13]] : !fir.logical<4>
! CHECK:  } else {
! CHECK:  %[[VAL_14:.*]] = arith.constant false
! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i1) -> !fir.logical<4>
! CHECK:  fir.result %[[VAL_15]] : !fir.logical<4>
! CHECK:  }
! CHECK:  fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<?xi32>) {
! CHECK:  %[[VAL_39:.*]] = fir.convert %[[VAL_12]] : (!fir.logical<4>) -> i1
! CHECK:  fir.call @_FortranAScan1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_39]]) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK:  }
! CHECK:  fir.array_merge_store
end subroutine
