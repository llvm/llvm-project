!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

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

! At LLVM IR level, verify the OpenMP fork call exists and the loop body
! is inside the outlined function.
! LLVM:       call void {{.*}}__kmpc_fork_call({{.*}}@test_workshare_firstprivate_pointer_..omp_par{{.*}})
! LLVM: {{.*}}test_workshare_firstprivate_pointer_..omp_par{{.*}}
! LLVM-LABEL: omp.par.region{{[0-9]+}}:
! LLVM:       call i32 @__kmpc_single
! LLVM:       icmp ne i32
! LLVM-LABEL: omp_region.end:
! LLVM:       call void @__kmpc_copyprivate
! LLVM:       call void {{.*}}__kmpc_barrier
! LLVM-LABEL: omp.single.region:
! LLVM:       call void @llvm.memcpy{{.*}}
! LLVM:       getelementptr {{.*}} i32 0, i32 7
! LLVM:       load i64{{.*}}
! LLVM-LABEL: omp_region.finalize:
! LLVM:       call void @__kmpc_end_single
! LLVM:       store i32 %{{.*}}, ptr %{{.*}}
! LLVM:       getelementptr {{.*}}i8
! LLVM:       ret void
