! Tests that -finit-local= preserves the requested bit pattern for LOGICAL
! variables in the final LLVM IR. fir.convert from integer to !fir.logical
! normalizes any nonzero value to .TRUE. (i.e. 1); the fix stores via a
! bitcasted integer address instead so the bit pattern is preserved.
!
! RUN: %flang_fc1 -emit-llvm -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX %s
! RUN: %flang_fc1 -emit-llvm -finit-local=nan  %s -o - | FileCheck --check-prefix=NAN %s
! RUN: %flang_fc1 -emit-llvm -finit-local=zero %s -o - | FileCheck --check-prefix=ZERO %s

! ---------------------------------------------------------------------------
! LOGICAL(1) -- 1-byte storage; 0xAA byte-splat = -86 (i8), NOT i8 1
! ---------------------------------------------------------------------------
subroutine test_logical1(res)
  logical(1) :: res
  logical(1) :: x
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_logical1{{.*}}(
! HEX:  store i8 -86,
! HEX-NOT: store i8 1,

! NAN-LABEL: define {{.*}}@{{.*}}test_logical1{{.*}}(
! NAN:  store i8 -86,
! NAN-NOT: store i8 1,

! ZERO-LABEL: define {{.*}}@{{.*}}test_logical1{{.*}}(
! ZERO: store i8 0,

! ---------------------------------------------------------------------------
! LOGICAL(4) -- 4-byte storage; 0xAA byte-splat = -1431655766 (i32), NOT i32 1
! ---------------------------------------------------------------------------
subroutine test_logical4(res)
  logical(4) :: res
  logical(4) :: x
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_logical4{{.*}}(
! HEX:  store i32 -1431655766,
! HEX-NOT: store i32 1,

! NAN-LABEL: define {{.*}}@{{.*}}test_logical4{{.*}}(
! NAN:  store i32 -1431655766,
! NAN-NOT: store i32 1,

! ZERO-LABEL: define {{.*}}@{{.*}}test_logical4{{.*}}(
! ZERO: store i32 0,
