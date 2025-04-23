! Tests that if `do concurrent` is not perfectly nested in its parent loop, that
! we skip converting the not-perfectly nested `do concurrent` loop.


! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s --check-prefixes=HOST,COMMON

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

! HOST: %[[ORIG_K_ALLOC:.*]] = fir.alloca i32 {bindc_name = "k"}
! HOST: %[[ORIG_K_DECL:.*]]:2 = hlfir.declare %[[ORIG_K_ALLOC]]

! HOST: %[[ORIG_J_ALLOC:.*]] = fir.alloca i32 {bindc_name = "j"}
! HOST: %[[ORIG_J_DECL:.*]]:2 = hlfir.declare %[[ORIG_J_ALLOC]]

! DEVICE: omp.target {{.*}}map_entries(
! DEVICE-SAME:   %{{[[:alnum:]]+}} -> %{{[^,]+}},
! DEVICE-SAME:   %{{[[:alnum:]]+}} -> %{{[^,]+}},
! DEVICE-SAME:   %{{[[:alnum:]]+}} -> %{{[^,]+}},
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[I_ARG:[^,]+]],
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[X_ARG:[^,]+]],
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[J_ARG:[^,]+]],
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[K_ARG:[^,]+]],
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[A_ARG:[^,]+]],
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %{{[^,]+}},
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %{{[^:]+}} :
! DEVICE-SAME:   {{.*}}) {

! DEVICE: %[[TARGET_J_DECL:.*]]:2 = hlfir.declare %[[J_ARG]] {uniq_name = "_QFEj"}
! DEVICE: %[[TARGET_K_DECL:.*]]:2 = hlfir.declare %[[K_ARG]] {uniq_name = "_QFEk"}

! DEVICE: omp.teams

! COMMON: omp.parallel {

! DEVICE: omp.distribute

! COMMON: omp.wsloop {
! COMMON: omp.loop_nest ({{[^[:space:]]+}}) {{.*}} {
! COMMON:   fir.do_loop %[[J_IV:.*]] = {{.*}} {
! COMMON:     %[[J_IV_CONV:.*]] = fir.convert %[[J_IV]] : (index) -> i32
! HOST:       fir.store %[[J_IV_CONV]] to %[[ORIG_J_DECL]]#0
! DEVICE:     fir.store %[[J_IV_CONV]] to %[[TARGET_J_DECL]]#0

! COMMON:     fir.do_loop %[[K_IV:.*]] = {{.*}} {
! COMMON:       %[[K_IV_CONV:.*]] = fir.convert %[[K_IV]] : (index) -> i32
! HOST:         fir.store %[[K_IV_CONV]] to %[[ORIG_K_DECL]]#0
! DEVICE:       fir.store %[[K_IV_CONV]] to %[[TARGET_K_DECL]]#0
! COMMON:     }
! COMMON:   }
! COMMON: omp.yield
! COMMON: }
! COMMON: }
! COMMON: omp.terminator
! COMMON: }
