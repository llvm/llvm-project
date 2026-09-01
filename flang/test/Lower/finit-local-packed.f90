! AIX BIND(C) regression test for -finit-local= with packed derived types.
!
! On AIX, ConvertType creates a packed layout for a BIND(C) record whose
! REAL(8) component is not first.  The initialization loop must use the
! correct packed size.
!
! For {integer(4), complex(8)}: i32 store=4B, complex<f64> store=16B.
! Packed size = 4+16 = 20B.  Loop upper bound must be 19 (trip count 20).
!
! Note: on AIX f64 has 4-byte ABI alignment, so the non-packed size of
! {i32, complex<f64>} also happens to be 20B (no inter-field gap needed).
! The platform-independent test that distinguishes packed from non-packed
! (using x86-64 alignment where they differ) lives in
! flang/test/Fir/box-elesize-canonicalize.fir.
!
! REQUIRES: system-aix
!
! RUN: bbc -emit-hlfir -finit-local=0xAA -o - %s 2>/dev/null | FileCheck --check-prefix=HEX  %s
! RUN: bbc -emit-hlfir -finit-local=zero  -o - %s 2>/dev/null | FileCheck --check-prefix=ZERO %s

subroutine test_bindc_aix(res)
  type, bind(c) :: tp
    integer(4) :: i
    complex(8) :: z
  end type
  type(tp) :: x
  res = x%i
end subroutine

! HEX-LABEL:  func.func @_QPtest_bindc_aix
! HEX:        arith.constant 19 : index
! HEX:        fir.do_loop
! HEX:        fir.store %{{.*}} : !fir.ref<i8>

! ZERO-LABEL: func.func @_QPtest_bindc_aix
! ZERO:       arith.constant 19 : index
! ZERO:       fir.do_loop
! ZERO:       fir.store %{{.*}} : !fir.ref<i8>
