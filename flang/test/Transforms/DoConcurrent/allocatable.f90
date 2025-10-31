! Verifies that proper `omp.map.bounds` ops are emitted when an allocatable is
! implicitly mapped by a `do concurrent` loop.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s
program main
   implicit none

   integer,parameter :: n = 1000000
   real, allocatable, dimension(:) :: y
   integer :: i

   allocate(y(1:n))

   do concurrent(i=1:n)
       y(i) = 42
   end do

   deallocate(y)
end program main

! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEy"}
! CHECK: %[[Y_VAL:.*]] = fir.load %[[Y_DECL]]#0
! CHECK: %[[Y_DIM0:.*]]:3 = fir.box_dims %[[Y_VAL]], %{{c0_.*}}
! CHECK: %[[Y_LB:.*]] = arith.constant 0 : index
! CHECK: %[[Y_UB:.*]] = arith.subi %[[Y_DIM0]]#1, %{{c1_.*}} : index
! CHECK: %[[Y_BOUNDS:.*]] = omp.map.bounds lower_bound(%[[Y_LB]] : index) upper_bound(%[[Y_UB]] : index) extent(%[[Y_DIM0]]#1 : index)
! CHECK: %[[MEM_MAP:.*]] = omp.map.info {{.*}} bounds(%[[Y_BOUNDS]])
! CHECK: omp.map.info var_ptr(%[[Y_DECL]]#1 : {{.*}}) {{.*}} members(%[[MEM_MAP]] : {{.*}})
