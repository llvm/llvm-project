! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - 2>&1 | FileCheck %s
! RUN: bbc -fopenmp -emit-hlfir -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - 2>&1 | FileCheck %s
! RUN: bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
!
! Test that the calls to omp_lib's omp_get_num_threads and omp_set_num_threads
! get lowered even though their implementation is not in the omp_lib module
! (and this matters because this is an intrinsic module - and calls to
! intrinsics are specially resolved).

program main
  use omp_lib
  integer(omp_integer_kind) :: num_threads
  integer(omp_integer_kind), parameter :: requested_num_threads = 4
  call omp_set_num_threads(requested_num_threads)
  num_threads = omp_get_num_threads()
  print *, num_threads
end program

!CHECK-NOT: not yet implemented: intrinsic: omp_set_num_threads
!CHECK-NOT: not yet implemented: intrinsic: omp_get_num_threads
!CHECK: fir.call @omp_set_num_threads
!CHECK: fir.call @omp_get_num_threads
