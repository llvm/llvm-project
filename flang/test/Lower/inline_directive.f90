! RUN: %flang_fc1 -emit-fir -o - %s | FileCheck %s

subroutine test_inline()
  integer :: x, y
!CHECK:  %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_inlineEx"}
!CHECK:  %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "_QFtest_inlineEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
!CHECK:  %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFtest_inlineEy"}
!CHECK:  %[[VAL_3:.*]] = fir.declare %[[VAL_2]] {uniq_name = "_QFtest_inlineEy"} : (!fir.ref<i32>) -> !fir.ref<i32>

  !dir$ forceinline
  y = g(x)
  !CHECK:  %[[VAL_4:.*]] = fir.call @_QFtest_inlinePg(%[[VAL_1]]) fastmath<contract> {inline_attr = #fir.inline_attrs<always_inline>} : (!fir.ref<i32>) -> i32
  !CHECK:  fir.store %[[VAL_4]] to %[[VAL_3]] : !fir.ref<i32>
  
  !dir$ forceinline
  call f(x, y)
  !CHECK:  fir.call @_QFtest_inlinePf(%[[VAL_1]], %[[VAL_3]]) fastmath<contract> {inline_attr = #fir.inline_attrs<always_inline>} : (!fir.ref<i32>, !fir.ref<i32>) -> ()

  !dir$ noinline
  y = g(x) + 7 * (8 + g(y))
  !CHECK: %[[VAL_8:.*]] = fir.call @_QFtest_inlinePg(%[[VAL_1]]) fastmath<contract> {inline_attr = #fir.inline_attrs<no_inline>} : (!fir.ref<i32>) -> i32
  !CHECK: %[[VAL_9:.*]] = fir.call @_QFtest_inlinePg(%[[VAL_3]]) fastmath<contract> {inline_attr = #fir.inline_attrs<no_inline>} : (!fir.ref<i32>) -> i32
  !CHECK: %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[C8:.*]] : i32
  !CHECK: %[[VAL_11:.*]] = fir.no_reassoc %[[VAL_10]] : i32
  !CHECK: %[[VAL_12:.*]] = arith.muli %[[VAL_11]], %[[C7:.*]] : i32
  !CHECK: %[[VAL_13:.*]] = arith.addi %[[VAL_8]], %[[VAL_12]] : i32
  !CHECK: fir.store %[[VAL_13]] to %[[VAL_3]] : !fir.ref<i32>

  !dir$ noinline
  call f(x, y)
  !CHECK:  fir.call @_QFtest_inlinePf(%[[VAL_1]], %[[VAL_3]]) fastmath<contract> {inline_attr = #fir.inline_attrs<no_inline>} : (!fir.ref<i32>, !fir.ref<i32>) -> ()

  !dir$ inline
  call f(x, y)
  !CHECK:  fir.call @_QFtest_inlinePf(%[[VAL_1]], %[[VAL_3]]) fastmath<contract> {inline_attr = #fir.inline_attrs<inline_hint>} : (!fir.ref<i32>, !fir.ref<i32>) -> ()

  !dir$ forceinline
  do i = 1, 100
    !CHECK: fir.do_loop %[[ARG_0:.*]] = %[[FROM:.*]] to %[[TO:.*]] step %[[C1:.*]]  iter_args(%[[ARG_1:.*]] = {{.*}}) -> (i32) {
    !CHECK:  fir.call @_QFtest_inlinePf(%[[VAL_1]], %[[VAL_3]]) fastmath<contract> {inline_attr = #fir.inline_attrs<always_inline>} : (!fir.ref<i32>, !fir.ref<i32>) -> ()
    call f(x, y)
  enddo

  !dir$ inline
  do i = 1, 100
    !CHECK: fir.do_loop %[[ARG_0:.*]] = %[[FROM:.*]] to %[[TO:.*]] step %[[C1:.*]]  iter_args(%[[ARG_1:.*]] = {{.*}}) -> (i32) {
    !CHECK:  fir.call @_QFtest_inlinePf(%[[VAL_1]], %[[VAL_3]]) fastmath<contract> {inline_attr = #fir.inline_attrs<inline_hint>} : (!fir.ref<i32>, !fir.ref<i32>) -> ()
    call f(x, y)
  enddo
!CHECK:  return
  contains
    subroutine f(x, y)
      integer, intent(in) :: x
      integer, intent(out) :: y
      y = x*2
    end subroutine f
    integer function g(x)
      integer :: x
      g = x*2
    end function g
end subroutine test_inline
