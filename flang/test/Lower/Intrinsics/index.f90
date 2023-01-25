! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPindex_test(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function index_test(s1, s2)
  character(*) :: s1, s2
  ! CHECK: %[[st:[^:]*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sst:[^:]*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[a1:.*]] = fir.convert %[[st]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a2:.*]] = fir.convert %[[st]]#1 : (index) -> i64
  ! CHECK: %[[a3:.*]] = fir.convert %[[sst]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a4:.*]] = fir.convert %[[sst]]#1 : (index) -> i64
  ! CHECK: = fir.call @_FortranAIndex1(%[[a1]], %[[a2]], %[[a3]], %[[a4]], %{{.*}}) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
  index_test = index(s1, s2)
end function index_test

! CHECK-LABEL: func @_QPindex_test2(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function index_test2(s1, s2)
  character(*) :: s1, s2
  ! CHECK: %[[mut:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
  ! CHECK: %[[st:[^:]*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sst:[^:]*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sb:.*]] = fir.embox %[[st]]#0 typeparams %[[st]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[ssb:.*]] = fir.embox %[[sst]]#0 typeparams %[[sst]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[back:.*]] = fir.embox %{{.*}} : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  ! CHECK: %[[hb:.*]] = fir.embox %{{.*}} : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  ! CHECK: %[[a0:.*]] = fir.convert %[[mut]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[a1:.*]] = fir.convert %[[sb]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: %[[a2:.*]] = fir.convert %[[ssb]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: %[[a3:.*]] = fir.convert %[[back]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
  ! CHECK: %[[a5:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:  fir.call @_FortranAIndex(%[[a0]], %[[a1]], %[[a2]], %[[a3]], %{{.*}}, %[[a5]], %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  index_test2 = index(s1, s2, .true., 4)
  ! CHECK: %[[ld1:.*]] = fir.load %[[mut]] : !fir.ref<!fir.box<!fir.heap<i32>>>
  ! CHECK: %[[ad1:.*]] = fir.box_addr %[[ld1]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  ! CHECK: %[[ld2:.*]] = fir.load %[[ad1]] : !fir.heap<i32>
  ! CHECK: fir.freemem %[[ad1]]
end function index_test2

! CHECK-LABEL: func @_QPindex_test3
integer function index_test3(s, i)
  character(*) :: s
  integer :: i
  ! CHECK: %[[tmpChar:.*]] = fir.alloca !fir.char<1>
  ! CHECK: fir.store %{{.*}} to %[[tmpChar]] : !fir.ref<!fir.char<1>>
  ! CHECK: %[[tmpCast:.*]] = fir.convert %[[tmpChar]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAIndex1(%{{.*}}, %{{.*}}, %[[tmpCast]], %{{.*}}, %{{.*}})
  index_test3 = index(s, char(i))
end function

! CHECK-LABEL: func @_QPtest_optional(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.boxchar<1>
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine test_optional(string, substring, back)
  character (*) :: string(:), substring
  logical, optional :: back(:)
  print *, index(string, substring, back)
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
  ! CHECK:  fir.call @_FortranAIndex1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_39]]) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK:  }
! CHECK:  fir.array_merge_store
end subroutine
