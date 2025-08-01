! Tests that if `do concurrent` is not perfectly nested in its parent loop, that
! we skip converting the not-perfectly nested `do concurrent` loop.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s

program main
   integer, parameter :: n = 10
   integer, parameter :: m = 20
   integer, parameter :: l = 30
   integer x;
   integer :: a(n, m, l)

   do concurrent(i=1:n)
     x = 10
     do concurrent(j=1:m, k=1:l)
       a(i,j,k) = i * j + k
     end do
   end do
end

! CHECK: omp.parallel {

! CHECK: omp.wsloop {
! CHECK: omp.loop_nest ({{[^[:space:]]+}}) {{.*}} {
! CHECK:   fir.do_concurrent {

! CHECK:     %[[ORIG_J_ALLOC:.*]] = fir.alloca i32 {bindc_name = "j"}
! CHECK:     %[[ORIG_J_DECL:.*]]:2 = hlfir.declare %[[ORIG_J_ALLOC]]

! CHECK:     %[[ORIG_K_ALLOC:.*]] = fir.alloca i32 {bindc_name = "k"}
! CHECK:     %[[ORIG_K_DECL:.*]]:2 = hlfir.declare %[[ORIG_K_ALLOC]]

! CHECK:     fir.do_concurrent.loop (%[[J_IV:.*]], %[[K_IV:.*]]) = {{.*}} {
! CHECK:       %[[J_IV_CONV:.*]] = fir.convert %[[J_IV]] : (index) -> i32
! CHECK:       fir.store %[[J_IV_CONV]] to %[[ORIG_J_DECL]]#0

! CHECK:       %[[K_IV_CONV:.*]] = fir.convert %[[K_IV]] : (index) -> i32
! CHECK:       fir.store %[[K_IV_CONV]] to %[[ORIG_K_DECL]]#0
! CHECK:     }
! CHECK:   }
! CHECK: omp.yield
! CHECK: }
! CHECK: }
! CHECK: omp.terminator
! CHECK: }
