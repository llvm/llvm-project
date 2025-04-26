! RUN: bbc %s -o - -emit-hlfir | FileCheck %s --check-prefixes=ALL,REALLOCLHS
! RUN: bbc %s -o - -emit-hlfir -frealloc-lhs | FileCheck %s --check-prefixes=ALL,REALLOCLHS
! RUN: bbc %s -o - -emit-hlfir -frealloc-lhs=false | FileCheck %s --check-prefixes=ALL,NOREALLOCLHS
! RUN: %flang_fc1 %s -o - -emit-hlfir | FileCheck %s --check-prefixes=ALL,REALLOCLHS
! RUN: %flang_fc1 %s -o - -emit-hlfir -frealloc-lhs | FileCheck %s --check-prefixes=ALL,REALLOCLHS
! RUN: %flang_fc1 %s -o - -emit-hlfir -fno-realloc-lhs | FileCheck %s --check-prefixes=ALL,NOREALLOCLHS

subroutine test1(a, b)
  integer, allocatable :: a(:), b(:)
  a = b + 1
end

! ALL-LABEL:   func.func @_QPtest1(
! ALL:           %[[VAL_3:.*]]:2 = hlfir.declare{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest1Ea"}
! REALLOCLHS:    hlfir.assign %{{.*}} to %[[VAL_3]]#0 realloc : !hlfir.expr<?xi32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

! NOREALLOCLHS:  %[[VAL_20:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! NOREALLOCLHS:  hlfir.assign %{{.*}} to %[[VAL_20]] : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>

subroutine test2(a, b)
  character(len=*), allocatable :: a(:)
  character(len=*) :: b(:)
  a = b
end subroutine test2

! ALL-LABEL:   func.func @_QPtest2(
! ALL:           %[[VAL_3:.*]]:2 = hlfir.declare{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest2Ea"}
! REALLOCLHS:    hlfir.assign %{{.*}} to %[[VAL_3]]#0 realloc keep_lhs_len : !fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>

! NOREALLOCLHS:  %[[VAL_7:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! NOREALLOCLHS:  hlfir.assign %{{.*}} to %[[VAL_7]] : !fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>

