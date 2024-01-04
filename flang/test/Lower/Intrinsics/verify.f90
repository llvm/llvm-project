! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPverify_test(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) -> i32 {
integer function verify_test(s1, s2)
! CHECK: %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK: %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "verify_test", uniq_name = "_QFverify_testEverify_test"}
! CHECK: %[[VAL_6:.*]] = arith.constant 4 : i32
! CHECK: %[[VAL_7:.*]] = fir.absent !fir.box<i1>
! CHECK: %[[VAL_8:.*]] = fir.embox %[[VAL_3]]#0 typeparams %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK: %[[VAL_9:.*]] = fir.embox %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK: %[[VAL_10:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK: %[[VAL_11:.*]] = fir.embox %[[VAL_10]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK: fir.store %[[VAL_11]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK: %[[VAL_12:.*]] = fir.address_of(@_QQclX{{[0-9a-z]+}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
! CHECK: %[[VAL_13:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK: %[[VAL_14:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[VAL_15:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK: %[[VAL_16:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK: %[[VAL_17:.*]] = fir.convert %[[VAL_7]] : (!fir.box<i1>) -> !fir.box<none>
! CHECK: %[[VAL_18:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
! CHECK: %[[VAL_19:.*]] = fir.call @_FortranAVerify(%[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_6]], %[[VAL_18]], %[[VAL_13]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
! CHECK: %[[VAL_20:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK: %[[VAL_21:.*]] = fir.box_addr %[[VAL_20]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK: %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.heap<i32>
! CHECK: fir.freemem %[[VAL_21]]
! CHECK: fir.store %[[VAL_22]] to %[[VAL_5]] : !fir.ref<i32>
! CHECK: %[[VAL_23:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK: return %[[VAL_23]] : i32
  character(*) :: s1, s2
  verify_test = verify(s1, s2, kind=4)
end function verify_test

! CHECK-LABEL: func @_QPverify_test2(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) -> i32 {
integer function verify_test2(s1, s2)
! CHECK: %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "verify_test2", uniq_name = "_QFverify_test2Everify_test2"}
! CHECK: %[[VAL_5:.*]] = arith.constant true
! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
! CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_3]]#1 : (index) -> i64
! CHECK: %[[VAL_10:.*]] = fir.call @_FortranAVerify1(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_5]]) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> i32
! CHECK: fir.store %[[VAL_11]] to %[[VAL_4]] : !fir.ref<i32>
! CHECK: %[[VAL_12:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK: return %[[VAL_12]] : i32
  character(*) :: s1, s2
  verify_test2 = verify(s1, s2, .true.)
end function verify_test2

! CHECK-LABEL: func @_QPtest_optional(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.boxchar<1>
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine test_optional(string, set, back)
  character (*) :: string(:), set
  logical, optional :: back(:)
  print *, verify(string, set, back)
! CHECK:  %[[VAL_11:.*]] = fir.is_present %[[VAL_2]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i1
! CHECK:  %[[VAL_12:.*]] = fir.zero_bits !fir.ref<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_15:.*]] = fir.embox %[[VAL_12]](%[[VAL_14]]) : (!fir.ref<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[VAL_16:.*]] = arith.select %[[VAL_11]], %[[VAL_2]], %[[VAL_15]] : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[VAL_17:.*]] = fir.array_load %[[VAL_16]] {fir.optional} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.array<?x!fir.logical<4>>
! CHECK:  %[[VAL_24:.*]] = fir.do_loop %[[VAL_25:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered iter_args(%[[VAL_26:.*]] = %{{.*}}) -> (!fir.array<?xi32>) {
  ! CHECK:  %[[VAL_31:.*]] = fir.if %[[VAL_11]] -> (!fir.logical<4>) {
    ! CHECK:  %[[VAL_32:.*]] = fir.array_fetch %[[VAL_17]], %[[VAL_25]] : (!fir.array<?x!fir.logical<4>>, index) -> !fir.logical<4>
    ! CHECK:  fir.result %[[VAL_32]] : !fir.logical<4>
  ! CHECK:  } else {
    ! CHECK:  %[[VAL_33:.*]] = arith.constant false
    ! CHECK:  %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i1) -> !fir.logical<4>
    ! CHECK:  fir.result %[[VAL_34]] : !fir.logical<4>
  ! CHECK:  }
  ! CHECK:  %[[VAL_39:.*]] = fir.convert %[[VAL_31]] : (!fir.logical<4>) -> i1
  ! CHECK:  fir.call @_FortranAVerify1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_39]]) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK:  }
! CHECK:  fir.array_merge_store
end subroutine

! CHECK: func private @{{.*}}Verify(
! CHECK: func private @{{.*}}Verify1(
