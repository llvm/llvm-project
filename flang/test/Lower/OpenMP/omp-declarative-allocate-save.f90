! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 %s -o - 2>&1 | FileCheck %s \
! RUN:   --implicit-check-not='TODO : OpenMP declarative ALLOCATE on SAVE variables or COMMON blocks' \
! RUN:   --implicit-check-not=omp.allocate_dir \
! RUN:   --implicit-check-not=omp.allocate_free

subroutine save_allocate
  implicit none
  integer, save :: counter = 100

  !$omp allocate(counter) allocator(1)
  counter = counter + 1
end subroutine save_allocate

subroutine implicit_save_allocate
  implicit none
  integer :: implicit_counter = 100

  !$omp allocate(implicit_counter) allocator(1)
  implicit_counter = implicit_counter + 1
end subroutine implicit_save_allocate

subroutine common_allocate
  implicit none
  real :: cb_a, cb_b
  common /myblock/ cb_a, cb_b

  !$omp allocate(/myblock/) allocator(1)
  cb_a = 1.0
end subroutine common_allocate

subroutine save_allocate_align
  implicit none
  integer, save :: aligned_counter = 100

  !$omp allocate(aligned_counter) align(64) allocator(1)
  aligned_counter = aligned_counter + 1
end subroutine save_allocate_align

subroutine common_allocate_align
  implicit none
  real :: ca, cb
  common /alignedblock/ ca, cb

  !$omp allocate(/alignedblock/) align(128) allocator(1)
  ca = 1.0
end subroutine common_allocate_align

! CHECK: fir.global common @alignedblock_{{.*}}alignment = 128
! CHECK: fir.global common @myblock_

! CHECK-LABEL: func.func @_QPsave_allocate
! CHECK: fir.address_of(@_QFsave_allocateEcounter)

! CHECK-LABEL: func.func @_QPimplicit_save_allocate
! CHECK: fir.address_of(@_QFimplicit_save_allocateEimplicit_counter)

! CHECK-LABEL: func.func @_QPcommon_allocate
! CHECK: fir.address_of(@myblock_)

! CHECK-LABEL: func.func @_QPsave_allocate_align
! CHECK: fir.address_of(@_QFsave_allocate_alignEaligned_counter)

! CHECK-LABEL: func.func @_QPcommon_allocate_align
! CHECK: fir.address_of(@alignedblock_)

! CHECK: fir.global {{.*}}@_QFsave_allocateEcounter
! CHECK: fir.global {{.*}}@_QFimplicit_save_allocateEimplicit_counter
! CHECK: fir.global {{.*}}@_QFsave_allocate_alignEaligned_counter{{.*}}alignment = 64
