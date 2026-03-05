!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! Regression test for https://github.com/llvm/llvm-project/issues/143330
! Verify that forall constructs with sliced arrays inside workshare are
! correctly lowered without causing race conditions or crashes.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix HLFIR
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefix FIR
!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s --check-prefix LLVM

subroutine workshare_forall_sliced(a1)
  integer :: a1(2,1,3)
  !$omp parallel
  !$omp workshare
  forall (n1=1:1)
     forall (n2=1:3)
        a1(:,n1,n2) = a1(:,n1,n2)+1
     end forall
  end forall
  !$omp end workshare
  !$omp end parallel
end subroutine

! HLFIR-LABEL: func.func @_QPworkshare_forall_sliced
! HLFIR:       omp.parallel {
! HLFIR:         omp.workshare {
! HLFIR:           hlfir.forall
! HLFIR:             hlfir.forall
! HLFIR:               hlfir.region_assign
! HLFIR:           omp.terminator
! HLFIR:         }
! HLFIR:         omp.terminator
! HLFIR:       }

! After workshare lowering, the forall should be in omp.single (since it
! contains operations that are not safe to parallelize across threads).
! The key fix is that this compiles without crashing and the runtime
! executes correctly.

! FIR-LABEL: func.func @_QPworkshare_forall_sliced
! FIR:       omp.parallel {
! FIR:         omp.single
! FIR:           fir.do_loop
! FIR:             fir.do_loop
! FIR:         omp.barrier
! FIR:         omp.terminator
! FIR:       }

! Verify LLVM IR is generated successfully (the original issue caused crashes)
! LLVM-LABEL: define {{.*}}workshare_forall_sliced
! LLVM:       call {{.*}}@__kmpc_fork_call
