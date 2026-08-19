! Tests that -finit-local= fills the full allocation for REAL(10) and
! COMPLEX(10) on x86. x86_fp80 has a 10-byte store size but a 16-byte
! allocation size; a typed store would leave 6 bytes uninitialized.
! The fix detects the gap via DataLayout and emits a byte-fill loop over
! the full allocation instead.
!
! REQUIRES: x86-registered-target
!
! RUN: %flang_fc1 -emit-llvm -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX %s
! RUN: %flang_fc1 -emit-llvm -finit-local=zero %s -o - | FileCheck --check-prefix=ZERO %s

! ---------------------------------------------------------------------------
! REAL(10) -- x86_fp80: store=10 bytes, alloc=16 bytes on x86-64.
! ---------------------------------------------------------------------------
subroutine test_real10(res)
  real(10) :: res
  real(10) :: x
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_real10{{.*}}(
! HEX:         phi i64 [ {{.*}}, {{.*}} ], [ 16, %{{.*}} ]
! HEX:         getelementptr i8, ptr {{.*}}, i64
! HEX:         store i8 -86,

! ZERO-LABEL: define {{.*}}@{{.*}}test_real10{{.*}}(
! ZERO:        phi i64 [ {{.*}}, {{.*}} ], [ 16, %{{.*}} ]
! ZERO:        getelementptr i8, ptr {{.*}}, i64
! ZERO:        store i8 0,

! ---------------------------------------------------------------------------
! COMPLEX(10) -- two x86_fp80 parts: store=20 bytes, alloc=32 bytes on x86-64.
! ---------------------------------------------------------------------------
subroutine test_complex10(res)
  complex(10) :: res
  complex(10) :: x
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_complex10{{.*}}(
! HEX:         phi i64 [ {{.*}}, {{.*}} ], [ 32, %{{.*}} ]
! HEX:         getelementptr i8, ptr {{.*}}, i64
! HEX:         store i8 -86,

! ZERO-LABEL: define {{.*}}@{{.*}}test_complex10{{.*}}(
! ZERO:        phi i64 [ {{.*}}, {{.*}} ], [ 32, %{{.*}} ]
! ZERO:        getelementptr i8, ptr {{.*}}, i64
! ZERO:        store i8 0,
