! LLVM IR-level regression test for volatile byte-fill stores.
! Verifies that volatile locals produce "store volatile" in LLVM IR for
! every byte written by the -finit-local= byte-fill paths: derived-type
! record fill, real(10) allocation-gap fill, integer array element fill,
! and both the compile-time-length and runtime-length CHARACTER byte loops.
!
! At -O2 a non-volatile fill of a volatile variable can be eliminated
! entirely.  These checks confirm the stores carry the volatile flag.
!
! REQUIRES: x86-registered-target
!
! RUN: %flang_fc1 -emit-llvm -triple x86_64-unknown-linux-gnu \
! RUN:     -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX %s
! RUN: %flang_fc1 -emit-llvm -triple x86_64-unknown-linux-gnu \
! RUN:     -finit-local=zero %s -o - | FileCheck --check-prefix=ZERO %s

! ---------------------------------------------------------------------------
! Volatile derived-type local -- record byte-fill loop.
! Every store into the record's bytes must be "store volatile i8".
! ---------------------------------------------------------------------------
subroutine test_volatile_derived(oa)
  type :: t
    integer(4) :: a
    integer(1) :: b
  end type
  type(t), volatile :: v
  integer :: oa
  v%a = 1
  oa = v%a
end subroutine

! HEX-LABEL: define {{.*}}@{{.*}}test_volatile_derived{{.*}}(
! HEX:        store volatile i8

! ZERO-LABEL: define {{.*}}@{{.*}}test_volatile_derived{{.*}}(
! ZERO:        store volatile i8

! ---------------------------------------------------------------------------
! Volatile real(10) local -- allocation-gap byte-fill loop.
! real(10) = x86_fp80: 10-byte store size, 16-byte allocation.
! ---------------------------------------------------------------------------
subroutine test_volatile_real10(res)
  real(10), volatile :: x
  real(10) :: res
  res = x
end subroutine

! HEX-LABEL: define {{.*}}@{{.*}}test_volatile_real10{{.*}}(
! HEX:        store volatile i8

! ZERO-LABEL: define {{.*}}@{{.*}}test_volatile_real10{{.*}}(
! ZERO:        store volatile i8

! ---------------------------------------------------------------------------
! Volatile integer array local -- rank-1 array-view byte-fill loop.
! ---------------------------------------------------------------------------
subroutine test_volatile_array(res)
  integer(4), volatile :: x(4)
  integer :: res
  res = x(1)
end subroutine

! HEX-LABEL: define {{.*}}@{{.*}}test_volatile_array{{.*}}(
! HEX:        store volatile i32

! ZERO-LABEL: define {{.*}}@{{.*}}test_volatile_array{{.*}}(
! ZERO:        store volatile i32

! ---------------------------------------------------------------------------
! Volatile fixed-length CHARACTER local -- compile-time-length byte-fill loop.
! hex: loop over nLen bytes -> store volatile i8.
! zero: fir.zero_bits store -> store volatile i8 0.
! ---------------------------------------------------------------------------
subroutine test_volatile_char_fixed(res)
  character(10), volatile :: x
  character(10) :: res
  res = x
end subroutine

! HEX-LABEL: define {{.*}}@{{.*}}test_volatile_char_fixed{{.*}}(
! HEX:        store volatile i8

! ZERO-LABEL: define {{.*}}@{{.*}}test_volatile_char_fixed{{.*}}(
! ZERO:        store volatile i8

! ---------------------------------------------------------------------------
! Volatile runtime-length CHARACTER local -- dynamic-length byte-fill loop.
! hex: loop over rtLen bytes -> store volatile i8.
! zero: loop over rtLen bytes -> store volatile i8 0.
! ---------------------------------------------------------------------------
subroutine test_volatile_char_runtime(res, n)
  integer, intent(in) :: n
  character(n), volatile :: x
  character(n) :: res
  res = x
end subroutine

! HEX-LABEL: define {{.*}}@{{.*}}test_volatile_char_runtime{{.*}}(
! HEX:        store volatile i8

! ZERO-LABEL: define {{.*}}@{{.*}}test_volatile_char_runtime{{.*}}(
! ZERO:        store volatile i8
