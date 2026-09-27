! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPindex_test(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function index_test(s1, s2)
  character(*) :: s1, s2
  ! CHECK: %[[st:[^:]*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFindex_testEs1"}
  ! CHECK: %[[sst:[^:]*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFindex_testEs2"}
  ! CHECK: %[[res:.*]] = hlfir.index %[[sst]]#0 in %[[st]]#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> i32
  ! CHECK: hlfir.assign %[[res]] to {{.*}} : i32, !fir.ref<i32>
  index_test = index(s1, s2)
end function index_test

! CHECK-LABEL: func @_QPindex_test2(
! CHECK-SAME: %[[s:[^:]+]]: !fir.boxchar<1>{{.*}}, %[[ss:[^:]+]]: !fir.boxchar<1>{{.*}}) -> i32
integer function index_test2(s1, s2)
  character(*) :: s1, s2
  ! CHECK: %[[st:[^:]*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFindex_test2Es1"}
  ! CHECK: %[[sst:[^:]*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFindex_test2Es2"}
  ! CHECK: %true = arith.constant true
  ! CHECK: %[[res:.*]] = hlfir.index %[[sst]]#0 in %[[st]]#0 back %true : (!fir.boxchar<1>, !fir.boxchar<1>, i1) -> i32
  ! CHECK: hlfir.assign %[[res]] to {{.*}} : i32, !fir.ref<i32>
  index_test2 = index(s1, s2, .true., 4)
end function index_test2

! CHECK-LABEL: func @_QPindex_test3
integer function index_test3(s, i)
  character(*) :: s
  integer :: i
  ! CHECK: %[[st:[^:]*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFindex_test3Es"}
  ! CHECK: hlfir.index {{.*}} in %[[st]]#0 : (!hlfir.expr<!fir.char<1>>, !fir.boxchar<1>) -> i32
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
! CHECK-DAG:  %[[BACKDECL:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}}uniq_name = "_QFtest_optionalEback"
! CHECK-DAG:  %[[STRDECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFtest_optionalEstring"}
! CHECK-DAG:  %[[SUBDECL:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_optionalEsubstring"}
! CHECK:  %[[ISPRES:.*]] = fir.is_present %[[BACKDECL]]#0
! CHECK:  hlfir.elemental {{.*}} {
! CHECK:    %[[ELEM:.*]] = hlfir.designate %[[STRDECL]]#0
! CHECK:    fir.if %[[ISPRES]] -> (!fir.logical<4>) {
! CHECK:      hlfir.designate %[[BACKDECL]]#0
! CHECK:      fir.load
! CHECK:    } else {
! CHECK:      arith.constant false
! CHECK:      fir.convert {{.*}} : (i1) -> !fir.logical<4>
! CHECK:    }
! CHECK:    hlfir.index %[[SUBDECL]]#0 in %[[ELEM]] back {{.*}} : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.logical<4>) -> i32
! CHECK:  }
end subroutine
