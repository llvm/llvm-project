! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPscan_test(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function scan_test(s1, s2)
character(*) :: s1, s2
! CHECK: fir.alloca !fir.box<!fir.heap<i32>>
! CHECK: arith.constant 4 : i32
! CHECK: fir.absent !fir.box<i1>
! CHECK: fir.call @_FortranAScan({{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()
! CHECK: fir.box_addr %{{.*}} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK: fir.freemem %{{.*}}
scan_test = scan(s1, s2, kind=4)
end function scan_test

! CHECK-LABEL: func @_QPscan_test2(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}},
! CHECK-SAME: %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function scan_test2(s1, s2)
character(*) :: s1, s2
! CHECK: arith.constant true
! CHECK: fir.call @_FortranAScan1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
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
! CHECK:  fir.is_present %{{.*}} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i1
! CHECK:  hlfir.elemental %{{.*}} unordered
! CHECK:  fir.if %{{.*}} -> (!fir.logical<4>)
! CHECK:  fir.call @_FortranAScan1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK:  hlfir.end_associate
! CHECK:  hlfir.destroy
end subroutine

! CHECK-LABEL: func @_QPtest_optional_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.boxchar<1>
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<!fir.logical<4>>
subroutine test_optional_scalar(string, set, back)
character (*) :: string(:), set
logical, optional :: back
print *, scan(string, set, back)
! CHECK:  fir.is_present %{{.*}} : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:  hlfir.elemental %{{.*}} unordered
! CHECK:  fir.if %{{.*}} -> (!fir.logical<4>)
! CHECK:  fir.call @_FortranAScan1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK:  hlfir.end_associate
! CHECK:  hlfir.destroy
end subroutine
