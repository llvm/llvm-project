!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix HLFIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix FIR
!RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix LLVM

! Test that parallel workshare with firstprivate(P) where P is a pointer
! correctly places stores through the pointer target in omp.single rather
! than parallelizing them. The pointer descriptor is thread-local (firstprivate),
! but the target data is shared memory.

subroutine test_workshare_firstprivate_pointer(P)
  integer, pointer, intent(in) :: P(:)
  integer :: i
  !$omp parallel workshare firstprivate(P)
  forall (i = 1:SIZE(P)) P(i) = i
  !$omp end parallel workshare
end subroutine

! HLFIR:     omp.parallel {
! HLFIR:       omp.workshare {
! The firstprivate copy: alloca, zero-init, declare, then copy from original
! HLFIR:         fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! HLFIR:         fir.store
! HLFIR:         hlfir.declare
! HLFIR:         fir.load
! HLFIR:         fir.store
! HLFIR:         hlfir.forall
! HLFIR:         omp.terminator
! HLFIR:       }
! HLFIR:       omp.terminator
! HLFIR:     }

! After workshare lowering, the forall body (which stores through the pointer
! target) must be inside omp.single, not parallelized.
! FIR:     omp.parallel {
! FIR:       %[[DESC:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! The firstprivate init + copy and the forall loop must be in omp.single
! FIR:       omp.single copyprivate(%[[DESC]]
! FIR:         fir.store
! FIR:         fir.declare
! FIR:         fir.load
! FIR:         fir.store
! The forall loop accesses pointer target (shared memory) - must stay in single
! FIR:         fir.do_loop
! FIR:           fir.array_coor
! FIR:           fir.store
! FIR:         omp.terminator
! FIR:       }
! FIR:       omp.barrier
! FIR:       omp.terminator

! At LLVM IR level, verify the OpenMP fork call exists and the loop body
! is inside the outlined function.
! LLVM:     call void {{.*}}__kmpc_fork_call
! LLVM:     define internal void @test_workshare_firstprivate_pointer_..omp_par
! The single construct must be present in the outlined function
! LLVM:       call i32 @__kmpc_single
