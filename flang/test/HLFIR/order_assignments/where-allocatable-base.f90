! Test scheduling of WHERE assignments on allocatable array sections.
! The LHS and RHS regions load the allocatable descriptor independently, so the
! section bases are distinct SSA values even though they are the same variable.
! The scheduler must still recognize identical sections and avoid RHS temps.

! RUN: bbc -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments)" \
! RUN:   --debug-only=flang-ordered-assignment \
! RUN:   -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s
! REQUIRES: asserts

subroutine alloc_base(center, kinetic_constant, mpi_is, mpi_ie, omp_js, omp_je)
  real(8), allocatable :: center(:,:,:,:,:), kinetic_constant(:,:,:,:)
  integer :: mpi_is, mpi_ie, omp_js, omp_je

  where (kinetic_constant(mpi_is:mpi_ie, omp_js:omp_je, :, 1) .ne. 0.d0)
    center(mpi_is:mpi_ie, omp_js:omp_je, :, 1, 1) = &
      center(mpi_is:mpi_ie, omp_js:omp_je, :, 1, 1) + 1.d0
  end where
end subroutine

! CHECK-LABEL: ------------ scheduling where in _QPalloc_base ------------
! CHECK: conflict (aligned):
! CHECK-NEXT: run 1 evaluate: where/region_assign1
! CHECK-NOT: save
! CHECK-NOT: run 2
