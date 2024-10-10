! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! Simple tests for structured concurrent loops with loop-control.

pure function bar(n, m)
   implicit none
   integer, intent(in) :: n, m
   integer :: bar
   bar = n + m
end function

subroutine sub1(n)
   implicit none
   integer :: n, m, i, j
   integer, dimension(n) :: a
!CHECK: %[[LB1:.*]] = arith.constant 1 : i32
!CHECK: %[[LB1_CVT:.*]] = fir.convert %[[LB1]] : (i32) -> index
!CHECK: %[[UB1:.*]] = fir.load %5#0 : !fir.ref<i32>
!CHECK: %[[UB1_CVT:.*]] = fir.convert %[[UB1]] : (i32) -> index
!CHECK: %[[LB2:.*]] = arith.constant 1 : i32
!CHECK: %[[LB2_CVT:.*]] = fir.convert %[[LB2]] : (i32) -> index
!CHECK: %[[UB2:.*]] = fir.call @_QPbar(%{{.*}}, %{{.*}}) proc_attrs<pure> fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32>) -> i32
!CHECK: %[[UB2_CVT:.*]] = fir.convert %[[UB2]] : (i32) -> index
!CHECK: fir.do_loop %{{.*}} = %[[LB1_CVT]] to %[[UB1_CVT]] step %{{.*}} unordered
!CHECK: fir.do_loop %{{.*}} = %[[LB2_CVT]] to %[[UB2_CVT]] step %{{.*}} unordered
   do concurrent(i=1:n, j=1:bar(n*m, n/m))
      a(i) = n
   end do
end subroutine

subroutine sub2(n)
   implicit none
   integer :: n, m, i, j
   integer, dimension(n) :: a
!CHECK: %[[LB1:.*]] = arith.constant 1 : i32
!CHECK: %[[LB1_CVT:.*]] = fir.convert %[[LB1]] : (i32) -> index
!CHECK: %[[UB1:.*]] = fir.load %5#0 : !fir.ref<i32>
!CHECK: %[[UB1_CVT:.*]] = fir.convert %[[UB1]] : (i32) -> index
!CHECK: %[[LB2:.*]] = arith.constant 1 : i32
!CHECK: %[[LB2_CVT:.*]] = fir.convert %[[LB2]] : (i32) -> index
!CHECK: %[[UB2:.*]] = fir.call @_QPbar(%{{.*}}, %{{.*}}) proc_attrs<pure> fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32>) -> i32
!CHECK: %[[UB2_CVT:.*]] = fir.convert %[[UB2]] : (i32) -> index
!CHECK: fir.do_loop %{{.*}} = %[[LB1_CVT]] to %[[UB1_CVT]] step %{{.*}} unordered
!CHECK: fir.do_loop %{{.*}} = %[[LB2_CVT]] to %[[UB2_CVT]] step %{{.*}} unordered
   do concurrent(i=1:n)
      do concurrent(j=1:bar(n*m, n/m))
         a(i) = n
      end do
   end do
end subroutine
