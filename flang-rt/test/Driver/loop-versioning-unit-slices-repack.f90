! Test FIR loop versioning by checking the generated fast path and verifying
! that its execution produces the expected results.
! Verify slice versioning through a frontend-generated fir.pack_array.
! REQUIRES: llvm-flang
! UNSUPPORTED: offload-cuda
! RUN: %flang -S -O3 -fversion-loops-for-stride \
! RUN:   -frepack-arrays -frepack-arrays-contiguity=whole \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-versioning \
! RUN:   %s -o /dev/null 2>&1 | \
! RUN:   FileCheck %s --check-prefix=VERSIONED --enable-var-scope
! RUN: %flang %isysroot -L"%libdir" -O3 -fversion-loops-for-stride \
! RUN:   -frepack-arrays -frepack-arrays-contiguity=whole %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

module loop_versioning_unit_slices_repack_m
  implicit none
contains
  subroutine fill_repacked(values, indices, record)
    real, intent(inout) :: values(:, :)
    integer, intent(in) :: indices(2)
    character(*), intent(in) :: record

    read(record, *) values(2:3, indices)
  end subroutine
end module

program loop_versioning_unit_slices_repack
  use loop_versioning_unit_slices_repack_m, only: fill_repacked
  implicit none
  real :: storage(6, 2), expected(6, 2)
  integer :: indices(2)

  storage = -1.0
  expected = -1.0
  indices = [1, 2]

  ! The actual argument is noncontiguous in its first dimension. Whole-array
  ! repacking gives the callee a contiguous temporary, and the epilogue must
  ! copy the values written through the byte fast path back to storage.
  call fill_repacked(storage(1:5:2, :), indices, '1 2 3 4')
  expected(3:5:2, 1) = [1.0, 2.0]
  expected(3:5:2, 2) = [3.0, 4.0]
  if (any(storage /= expected)) error stop 1

  ! CHECK: PASS
  print '(A)', 'PASS'
end program

! The compile-time half proves that the executed repacking configuration uses
! the packed descriptor in the guarded byte-address fast path.
! VERSIONED-LABEL: func.func @_QMloop_versioning_unit_slices_repack_mPfill_repacked(
! VERSIONED: %[[PACKED:.*]] = fir.pack_array %[[ORIGINAL:.*]] heap whole
! VERSIONED: %[[DECLARED:.*]] = fir.declare %[[PACKED]]
! VERSIONED: %[[PRED:.*]] = arith.cmpi eq, %{{.*}}#2, %{{.*}} : index
! VERSIONED-NEXT: fir.if %[[PRED]]
! VERSIONED: %[[BYTE_BOX:.*]] = fir.convert %[[DECLARED]]
! VERSIONED-SAME: -> !fir.box<!fir.array<?xi8>>
! VERSIONED: %[[BASE:.*]] = fir.box_addr %[[BYTE_BOX]]
! VERSIONED: %[[ADDRESS:.*]] = fir.coordinate_of %[[BASE]],
! VERSIONED-NEXT: %[[FAST:.*]] = fir.convert %[[ADDRESS]]
! VERSIONED-NEXT: %{{.*}} = fir.call @_FortranAioInputReal32({{.*}}, %[[FAST]])
! VERSIONED: fir.unpack_array %[[PACKED]] to %[[ORIGINAL]] heap
