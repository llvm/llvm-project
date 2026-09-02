! Tests that PowerPC vector locals (fir::VectorType) are excluded from
! -finit-local= initialization because fir::VectorType does not implement
! DataLayoutTypeInterface at the HLFIR level.
! Before the fix, a direct vector local silently fell back to zero
! initialization regardless of the requested mode (not a DataLayout crash).
! A derived-type local with a vector component would crash record-size
! calculation; that case is covered by test_derived_with_vec below.
! The local variable 'x' must not receive any initialization store.
!
! REQUIRES: target=powerpc{{.*}}
!
! RUN: %flang_fc1 -emit-hlfir -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX %s
! RUN: %flang_fc1 -emit-hlfir -finit-local=zero %s -o - | FileCheck --check-prefix=ZERO %s

! vector(real(4)) -- excluded; must not produce any fir.store for 'x'.
subroutine test_vec_real4(res)
  vector(real(4)) :: res
  vector(real(4)) :: x
  res = x
end subroutine

! HEX-LABEL: func.func @_QPtest_vec_real4(
! HEX:        hlfir.declare {{.*}}_QFtest_vec_real4Ex
! HEX-NOT:    fir.store {{.*}} to %{{.*}}#0

! ZERO-LABEL: func.func @_QPtest_vec_real4(
! ZERO:        hlfir.declare {{.*}}_QFtest_vec_real4Ex
! ZERO-NOT:    fir.store {{.*}} to %{{.*}}#0

! Derived type with a vector(real(4)) component -- the eligibility check must
! walk components recursively and exclude this record, just as it excludes a
! direct vector local.  Before the fix, compilation would crash in record-size
! calculation when it encountered the unsupported fir::VectorType component.
subroutine test_derived_with_vec(res)
  type :: tv
    integer(4)    :: i
    vector(real(4)) :: v
  end type
  type(tv) :: x
  res = x%i
end subroutine

! HEX-LABEL: func.func @_QPtest_derived_with_vec(
! HEX:        hlfir.declare {{.*}}_QFtest_derived_with_vecEx
! HEX-NOT:    fir.store {{.*}} to %{{.*}}#0

! ZERO-LABEL: func.func @_QPtest_derived_with_vec(
! ZERO:        hlfir.declare {{.*}}_QFtest_derived_with_vecEx
! ZERO-NOT:    fir.store {{.*}} to %{{.*}}#0
