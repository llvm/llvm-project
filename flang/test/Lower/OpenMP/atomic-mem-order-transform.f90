!RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=50 %s -o - 2>&1 | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK51

! Test that incompatible memory orderings on atomic operations are
! transformed to valid orderings during lowering.

! read + release -> relaxed
!CHECK: omp.atomic.read %{{.*}} = %{{.*}} memory_order(relaxed)
!CHECK51: omp.atomic.read %{{.*}} = %{{.*}} memory_order(relaxed)
subroutine test_read_release(x, v)
  integer :: x, v
  !$omp atomic read release
  v = x
end

! read + acq_rel -> acquire (5.0 only; untouched in 5.1)
!CHECK: omp.atomic.read %{{.*}} = %{{.*}} memory_order(acquire)
!CHECK51: omp.atomic.read %{{.*}} = %{{.*}} memory_order(acq_rel)
subroutine test_read_acq_rel(x, v)
  integer :: x, v
  !$omp atomic read acq_rel
  v = x
end

! write + acquire -> relaxed
!CHECK: omp.atomic.write %{{.*}} = %{{.*}} memory_order(relaxed)
!CHECK51: omp.atomic.write %{{.*}} = %{{.*}} memory_order(relaxed)
subroutine test_write_acquire(x, v)
  integer :: x, v
  !$omp atomic write acquire
  x = v
end

! write + acq_rel -> release (5.0 only; untouched in 5.1)
!CHECK: omp.atomic.write %{{.*}} = %{{.*}} memory_order(release)
!CHECK51: omp.atomic.write %{{.*}} = %{{.*}} memory_order(acq_rel)
subroutine test_write_acq_rel(x, v)
  integer :: x, v
  !$omp atomic write acq_rel
  x = v
end

! update + acquire -> relaxed (5.0 only; untouched in 5.1)
!CHECK: omp.atomic.update memory_order(relaxed)
!CHECK51: omp.atomic.update memory_order(acquire)
subroutine test_update_acquire(x)
  integer :: x
  !$omp atomic update acquire
  x = x + 1
end

! update + acq_rel -> release (5.0 only; untouched in 5.1)
!CHECK: omp.atomic.update memory_order(release)
!CHECK51: omp.atomic.update memory_order(acq_rel)
subroutine test_update_acq_rel(x)
  integer :: x
  !$omp atomic update acq_rel
  x = x + 1
end
