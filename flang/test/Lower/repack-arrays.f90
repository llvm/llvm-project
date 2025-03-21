! RUN: bbc -emit-hlfir -frepack-arrays -fstack-arrays -frepack-arrays-continuity-whole %s -o - -I nowhere | FileCheck --check-prefixes=ALL,STACK,WHOLE %s
! RUN: bbc -emit-hlfir -frepack-arrays -fstack-arrays=false -frepack-arrays-continuity-whole %s -o - -I nowhere | FileCheck --check-prefixes=ALL,HEAP,WHOLE %s
! RUN: bbc -emit-hlfir -frepack-arrays -fstack-arrays -frepack-arrays-continuity-whole=false %s -o - -I nowhere | FileCheck --check-prefixes=ALL,STACK,INNER %s
! RUN: bbc -emit-hlfir -frepack-arrays -fstack-arrays=false -frepack-arrays-continuity-whole=false %s -o - -I nowhere | FileCheck --check-prefixes=ALL,HEAP,INNER %s

! ALL-LABEL:   func.func @_QPtest1(
! ALL-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
subroutine test1(x)
  real :: x(:)
! ALL:           %[[VAL_2:.*]] = fir.pack_array %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! ALL-NOT:       no_copy
! ALL-SAME       : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
! ALL:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %{{.*}} {uniq_name = "_QFtest1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! ALL:           fir.unpack_array %[[VAL_2]] to %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-NOT:       no_copy
! ALL-SAME:      : !fir.box<!fir.array<?xf32>>
end subroutine test1

! ALL-LABEL:   func.func @_QPtest2(
! ALL-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! ALL-SAME:                        %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
subroutine test2(n, x)
  integer :: n
  character(n) :: x(:,:)
! ALL:           %[[VAL_8:.*]] = fir.pack_array %[[VAL_1]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! INNER-SAME:    innermost
! ALL-NOT:       no_copy
! ALL-SAME:      typeparams %[[VAL_7:.*]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
! ALL:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_7]] dummy_scope %{{.*}} {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>)
! ALL:           fir.unpack_array %[[VAL_8]] to %[[VAL_1]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-NOT:       no_copy
! ALL-SAME:      : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
end subroutine test2

! ALL-LABEL:   func.func @_QPtest3(
! ALL-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.type<_QFtest3Tt>>> {fir.bindc_name = "x"}) {
subroutine test3(x)
  type t
  end type t
  type(t) :: x(:,:)
! ALL:           %[[VAL_2:.*]] = fir.pack_array %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! INNER-SAME:    innermost
! ALL-NOT:       no_copy
! ALL-SAME:      : (!fir.box<!fir.array<?x?x!fir.type<_QFtest3Tt>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QFtest3Tt>>>
! ALL:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %{{.*}} {uniq_name = "_QFtest3Ex"} : (!fir.box<!fir.array<?x?x!fir.type<_QFtest3Tt>>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.type<_QFtest3Tt>>>, !fir.box<!fir.array<?x?x!fir.type<_QFtest3Tt>>>)
! ALL:           fir.unpack_array %[[VAL_2]] to %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-NOT:       no_copy
! ALL-SAME:      : !fir.box<!fir.array<?x?x!fir.type<_QFtest3Tt>>>
end subroutine test3

! ALL-LABEL:   func.func @_QPtest4(
! ALL-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
subroutine test4(x)
  real, intent(inout) :: x(:)
! ALL:           %[[VAL_2:.*]] = fir.pack_array %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! ALL-NOT:       no_copy
! ALL-SAME       : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
! ALL:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFtest4Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! ALL:           fir.unpack_array %[[VAL_2]] to %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-NOT:       no_copy
! ALL-SAME       : !fir.box<!fir.array<?xf32>>
end subroutine test4

! ALL-LABEL:   func.func @_QPtest5(
! ALL-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
subroutine test5(x)
  real, intent(in) :: x(:)
! ALL:           %[[VAL_2:.*]] = fir.pack_array %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! ALL-NOT:       no_copy
! ALL-SAME:      (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
! ALL:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest5Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! ALL:           fir.unpack_array %[[VAL_2]] to %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-SAME       no_copy : !fir.box<!fir.array<?xf32>>
end subroutine test5

! ALL-LABEL:   func.func @_QPtest6(
! ALL-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
subroutine test6(x)
  real, intent(out) :: x(:)
! ALL:           %[[VAL_2:.*]] = fir.pack_array %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! ALL-SAME       no_copy : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
! ALL:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFtest6Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! ALL:           fir.unpack_array %[[VAL_2]] to %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-NOT:       no_copy
! ALL-SAME       : !fir.box<!fir.array<?xf32>>
end subroutine test6

! ALL-LABEL:   func.func @_QPtest7(
! ALL-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.class<!fir.array<?x!fir.type<_QFtest7Tt>>> {fir.bindc_name = "x"}) {
subroutine test7(x)
  type t
  end type t
  class(t) :: x(:)
! ALL:           %[[VAL_2:.*]] = fir.pack_array %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! ALL-NOT:       no_copy
! ALL-SAME       : (!fir.class<!fir.array<?x!fir.type<_QFtest7Tt>>>) -> !fir.class<!fir.array<?x!fir.type<_QFtest7Tt>>>
! ALL:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %{{.*}} {uniq_name = "_QFtest7Ex"} : (!fir.class<!fir.array<?x!fir.type<_QFtest7Tt>>>, !fir.dscope) -> (!fir.class<!fir.array<?x!fir.type<_QFtest7Tt>>>, !fir.class<!fir.array<?x!fir.type<_QFtest7Tt>>>)
! ALL:           fir.unpack_array %[[VAL_2]] to %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-NOT:       no_copy
! ALL-SAME       : !fir.class<!fir.array<?x!fir.type<_QFtest7Tt>>>
end subroutine test7

! ALL-LABEL:   func.func @_QPtest8(
! ALL-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
subroutine test8(x)
  real :: x(:)
! ALL:           %[[VAL_2:.*]] = fir.pack_array %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! ALL-NOT:       no_copy
! ALL-SAME       : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
! ALL:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %{{.*}} {uniq_name = "_QFtest8Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  call inner(x(1))
! ALL:           fir.call @_QFtest8Pinner
! ALL:           fir.unpack_array %[[VAL_2]] to %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-NOT:       no_copy
! ALL-SAME       : !fir.box<!fir.array<?xf32>>
contains
! ALL-LABEL:   func.func private @_QFtest8Pinner(
  subroutine inner(y)
! ALL-NOT: fir.pack_array
! ALL-NOT: fir.unpack_array
    real :: y
    y = 1.0
  end subroutine inner
end subroutine test8

! ALL-LABEL:   func.func @_QPtest9(
! ALL-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) -> f32 {
real function test9(x)
  real :: x(:)
! ALL:           %[[VAL_6:.*]] = fir.pack_array %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! ALL-NOT:       no_copy
! ALL-SAME       : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
! ALL:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] dummy_scope %{{.*}} {uniq_name = "_QFtest9Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  real :: y(10)
  test9 = x(1)
! ALL:           fir.unpack_array %[[VAL_6]] to %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-NOT:       no_copy
! ALL-SAME       : !fir.box<!fir.array<?xf32>>
! ALL-NEXT:      return
  return

! ALL-LABEL:   func.func @_QPtest9_alt(
  entry test9_alt(y)
! ALL-NOT: fir.pack_array
! ALL-NOT: fir.unpack_array
  rest9_ = y(1)
end function test9

! ALL-LABEL:   func.func @_QPtest10(
! ALL-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x", fir.optional}) {
subroutine test10(x)
  real, optional :: x(:,:)
! ALL:           %[[VAL_2:.*]] = fir.pack_array %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! INNER-SAME:    innermost
! ALL-NOT:       no_copy
! ALL-SAME:      : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
! ALL:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest10Ex"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! ALL:           fir.unpack_array %[[VAL_2]] to %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-NOT:       no_copy
! ALL-SAME:      : !fir.box<!fir.array<?x?xf32>>
end subroutine test10

! ALL-LABEL:   func.func @_QPtest11(
! ALL-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x!fir.char<1,10>>> {fir.bindc_name = "x"}) {
subroutine test11(x)
  character(10) :: x(:)
! ALL:           %[[VAL_3:.*]] = fir.pack_array %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! INNER-SAME:    whole
! ALL-NOT:       no_copy
! ALL-SAME:      : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.box<!fir.array<?x!fir.char<1,10>>>
! ALL:           fir.unpack_array %[[VAL_3]] to %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-NOT:       no_copy
! ALL-SAME:      : !fir.box<!fir.array<?x!fir.char<1,10>>>
end subroutine test11

! ALL-LABEL:   func.func @_QPtest12(
! ALL-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
subroutine test12(x)
  character(*) :: x(:)
! ALL:           %[[VAL_2:.*]] = fir.pack_array %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! WHOLE-SAME:    whole
! INNER-SAME:    whole
! ALL-NOT:       no_copy
! ALL-SAME:      : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! ALL:           fir.unpack_array %[[VAL_2]] to %[[VAL_0]]
! STACK-SAME:    stack
! HEAP-SAME:     heap
! ALL-NOT:       no_copy
! ALL-SAME:      : !fir.box<!fir.array<?x!fir.char<1,?>>>
end subroutine test12
