! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! Verify that the inlined allocation path checks the allocation status before
! calling the actual allocation function

! CHECK-LABEL: func.func @_QPallocate_twice()
subroutine allocate_twice()
  integer, allocatable :: array(:)

  ! CHECK: %[[IS_ALLOCATED_1:.*]] = arith.cmpi ne
  ! CHECK: fir.if %[[IS_ALLOCATED_1]] {
  ! CHECK:   fir.call @_FortranAReportFatalUserError
  ! CHECK: }
  ! CHECK: fir.allocmem
  allocate(array(5))

  ! CHECK: %[[IS_ALLOCATED_2:.*]] = arith.cmpi ne
  ! CHECK: fir.if %[[IS_ALLOCATED_2]] {
  ! CHECK:   fir.call @_FortranAReportFatalUserError
  ! CHECK: }
  ! CHECK: fir.allocmem
  allocate(array(8))
end subroutine
