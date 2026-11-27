! Tests that PowerPC vector locals (fir::VectorType) are excluded from
! -finit-local= initialization because fir::VectorType does not implement
! DataLayoutTypeInterface at the HLFIR level.
!
! History of pre-fix behavior:
!   - Direct vector local: silently fell back to zero initialization
!     regardless of the requested mode (i.e. hex mode produced zero, not
!     the requested byte pattern).
!   - Derived-type local with a vector component: compilation crashed in
!     record-size calculation when it encountered the unsupported
!     fir::VectorType.  The crash occurred before any store was emitted.
!   At the current head, if a vector type somehow reached genByteSplatInit,
!   it would hit llvm_unreachable rather than silently produce zero.
!
! REQUIRES: target=powerpc{{.*}}
!
! RUN: %flang_fc1 -emit-hlfir -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX %s
! RUN: %flang_fc1 -emit-hlfir -finit-local=zero %s -o - | FileCheck --check-prefix=ZERO %s

! ---------------------------------------------------------------------------
! Positive control: ordinary record (no vector component) IS initialized.
! This ensures the vector-exclusion checks below are not trivially satisfied
! by a compiler that never initializes any record.
! ---------------------------------------------------------------------------
subroutine test_plain_record(res)
  type :: tp
    integer(4) :: a
    integer(4) :: b
  end type
  type(tp) :: x
  integer :: res
  res = x%a
end subroutine

! HEX-LABEL: func.func @_QPtest_plain_record(
! HEX:        hlfir.declare {{.*}}_QFtest_plain_recordEx
! HEX:        fir.do_loop
! HEX:        return

! ZERO-LABEL: func.func @_QPtest_plain_record(
! ZERO:        hlfir.declare {{.*}}_QFtest_plain_recordEx
! ZERO:        fir.do_loop
! ZERO:        return

! ---------------------------------------------------------------------------
! Direct vector(real(4)) local -- excluded; no initialization store or loop.
! ---------------------------------------------------------------------------
subroutine test_vec_real4(res)
  vector(real(4)) :: res
  vector(real(4)) :: x
  res = x
end subroutine

! HEX-LABEL: func.func @_QPtest_vec_real4(
! HEX:        hlfir.declare {{.*}}_QFtest_vec_real4Ex
! HEX-NOT:    fir.do_loop
! HEX-NOT:    fir.store
! HEX:        return

! ZERO-LABEL: func.func @_QPtest_vec_real4(
! ZERO:        hlfir.declare {{.*}}_QFtest_vec_real4Ex
! ZERO-NOT:    fir.do_loop
! ZERO-NOT:    fir.store
! ZERO:        return

! ---------------------------------------------------------------------------
! Derived type with a vector(real(4)) component -- excluded recursively.
! The containsVectorComponent helper walks component scopes so that record-
! size calculation is never attempted for this type.
! ---------------------------------------------------------------------------
subroutine test_derived_with_vec(res)
  type :: tv
    integer(4)      :: i
    vector(real(4)) :: v
  end type
  type(tv) :: x
  integer :: res
  res = x%i
end subroutine

! HEX-LABEL: func.func @_QPtest_derived_with_vec(
! HEX:        hlfir.declare {{.*}}_QFtest_derived_with_vecEx
! HEX-NOT:    fir.do_loop
! HEX-NOT:    fir.store
! HEX:        return

! ZERO-LABEL: func.func @_QPtest_derived_with_vec(
! ZERO:        hlfir.declare {{.*}}_QFtest_derived_with_vecEx
! ZERO-NOT:    fir.do_loop
! ZERO-NOT:    fir.store
! ZERO:        return
