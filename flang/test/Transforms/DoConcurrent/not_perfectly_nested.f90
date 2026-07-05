! Tests that if `do concurrent` is not perfectly nested in its parent loop, that
! we skip converting the not-perfectly nested `do concurrent` loop.


! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s --check-prefixes=COMMON

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s --check-prefixes=DEVICE,COMMON

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



! DEVICE: omp.target {{.*}}map_entries(
! DEVICE-SAME:   %{{[[:alnum:]]+}} -> %{{[^,]+}},
! DEVICE-SAME:   %{{[[:alnum:]]+}} -> %{{[^,]+}},
! DEVICE-SAME:   %{{[[:alnum:]]+}} -> %{{[^,]+}},
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[I_ARG:[^,]+]],
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[X_ARG:[^,]+]],
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[A_ARG:[^,]+]],
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %{{[^:]+}} :
! DEVICE-SAME:   {{.*}}) {

! DEVICE: omp.teams

! COMMON: omp.parallel {

! DEVICE: omp.distribute

! COMMON: omp.wsloop {
! COMMON: omp.loop_nest ({{[^[:space:]]+}}) {{.*}} {
! COMMON:   fir.do_concurrent {

! COMMON:     %[[ORIG_J_ALLOC:.*]] = fir.alloca i32 {bindc_name = "j"}
! COMMON:     %[[ORIG_J_DECL:.*]]:2 = hlfir.declare %[[ORIG_J_ALLOC]]

! COMMON:     %[[ORIG_K_ALLOC:.*]] = fir.alloca i32 {bindc_name = "k"}
! COMMON:     %[[ORIG_K_DECL:.*]]:2 = hlfir.declare %[[ORIG_K_ALLOC]]

! COMMON:     fir.do_concurrent.loop (%[[J_IV:.*]], %[[K_IV:.*]]) = {{.*}} {
! COMMON:       %[[J_IV_CONV:.*]] = fir.convert %[[J_IV]] : (index) -> i32
! COMMON:       fir.store %[[J_IV_CONV]] to %[[ORIG_J_DECL]]#0

! COMMON:       %[[K_IV_CONV:.*]] = fir.convert %[[K_IV]] : (index) -> i32
! COMMON:       fir.store %[[K_IV_CONV]] to %[[ORIG_K_DECL]]#0
! COMMON:     }
! COMMON:   }
! COMMON: omp.yield
! COMMON: }
! COMMON: }
! COMMON: omp.terminator
! COMMON: }
