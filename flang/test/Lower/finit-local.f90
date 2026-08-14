! Tests for -finit-local= local variable initialization.
!
! Covers every Fortran type listed in the RFC type-mapping table:
!   INTEGER(k) k=1,2,4,8
!   REAL(k)    k=4,8  (k=16 in finit-local-f128.f90, requires flang-supports-f128-math)
!   COMPLEX(k) k=4,8  (k=16 in finit-local-f128.f90, requires flang-supports-f128-math)
!   LOGICAL(k) k=1,4
!   CHARACTER(n)
!   Derived type (struct with plain-int and real components)
!   Arrays of integer and real
!
! Modes exercised: zero, nan, snan, 0xAA (hex), and off (no flag).
!
! RUN: bbc -emit-hlfir -finit-local=zero  -o - %s | FileCheck --check-prefix=ZERO  %s
! RUN: bbc -emit-hlfir -finit-local=nan   -o - %s | FileCheck --check-prefix=NAN   %s
! RUN: bbc -emit-hlfir -finit-local=snan  -o - %s | FileCheck --check-prefix=SNAN  %s
! RUN: bbc -emit-hlfir -finit-local=0xAA  -o - %s | FileCheck --check-prefix=HEX   %s
! RUN: bbc -emit-hlfir                    -o - %s | FileCheck --check-prefix=OFF   %s
! RUN: bbc -emit-hlfir -finit-local-zero  -o - %s | FileCheck --check-prefix=ZERO  %s

! ---------------------------------------------------------------------------
! INTEGER(1) -- 1-byte: pattern 0xAA = -86 (signed) = 170 (unsigned)
! ---------------------------------------------------------------------------
subroutine test_int1(res)
  integer(1) :: res
  integer(1) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_int1
! ZERO: fir.alloca i8
! ZERO: fir.zero_bits i8
! ZERO: fir.store {{.*}} : !fir.ref<i8>

! NAN-LABEL:  func.func @_QPtest_int1
! NAN:  arith.constant -86 : i8
! NAN:  fir.store {{.*}} : !fir.ref<i8>

! HEX-LABEL:  func.func @_QPtest_int1
! HEX:  arith.constant -86 : i8
! HEX:  fir.store {{.*}} : !fir.ref<i8>

! OFF-LABEL: func.func @_QPtest_int1
! OFF-NOT: fir.store {{.*}} : !fir.ref<i8>

! ---------------------------------------------------------------------------
! INTEGER(2) -- 2-byte: pattern 0xAAAA = -21846 (signed)
! ---------------------------------------------------------------------------
subroutine test_int2(res)
  integer(2) :: res
  integer(2) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_int2
! ZERO: fir.zero_bits i16

! NAN-LABEL:  func.func @_QPtest_int2
! NAN:  arith.constant -21846 : i16
! NAN:  fir.store {{.*}} : !fir.ref<i16>

! HEX-LABEL:  func.func @_QPtest_int2
! HEX:  arith.constant -21846 : i16
! HEX:  fir.store {{.*}} : !fir.ref<i16>

! ---------------------------------------------------------------------------
! INTEGER(4) -- 4-byte: pattern 0xAAAAAAAA = -1431655766 (signed)
! ---------------------------------------------------------------------------
subroutine test_int4(res)
  integer(4) :: res
  integer(4) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_int4
! ZERO: fir.zero_bits i32

! NAN-LABEL:  func.func @_QPtest_int4
! NAN:  arith.constant -1431655766 : i32
! NAN:  fir.store {{.*}} : !fir.ref<i32>

! SNAN-LABEL: func.func @_QPtest_int4
! SNAN: arith.constant -1431655766 : i32
! SNAN: fir.store {{.*}} : !fir.ref<i32>

! HEX-LABEL:  func.func @_QPtest_int4
! HEX:  arith.constant -1431655766 : i32
! HEX:  fir.store {{.*}} : !fir.ref<i32>

! OFF-LABEL: func.func @_QPtest_int4
! OFF-NOT: fir.zero_bits

! ---------------------------------------------------------------------------
! INTEGER(8) -- 8-byte: pattern 0xAAAAAAAAAAAAAAAA = -6148914691236517206 (signed)
! ---------------------------------------------------------------------------
subroutine test_int8(res)
  integer(8) :: res
  integer(8) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_int8
! ZERO: fir.zero_bits i64

! NAN-LABEL:  func.func @_QPtest_int8
! NAN:  arith.constant -6148914691236517206 : i64
! NAN:  fir.store {{.*}} : !fir.ref<i64>

! HEX-LABEL:  func.func @_QPtest_int8
! HEX:  arith.constant -6148914691236517206 : i64
! HEX:  fir.store {{.*}} : !fir.ref<i64>

! ---------------------------------------------------------------------------
! REAL(4) -- zero fills with fir.zero_bits; nan/snan with FP constant; hex bitcast
! ---------------------------------------------------------------------------
subroutine test_real4(res)
  real(4) :: res
  real(4) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_real4
! ZERO: fir.zero_bits f32
! ZERO: fir.store {{.*}} : !fir.ref<f32>

! NAN-LABEL:  func.func @_QPtest_real4
! NAN:  arith.constant {{.*}} : f32
! NAN:  fir.store {{.*}} : !fir.ref<f32>

! SNAN-LABEL: func.func @_QPtest_real4
! SNAN: arith.constant {{.*}} : f32
! SNAN: fir.store {{.*}} : !fir.ref<f32>

! HEX-LABEL:  func.func @_QPtest_real4
! HEX:  arith.constant -1431655766 : i32
! HEX:  arith.bitcast {{.*}} : i32 to f32
! HEX:  fir.store {{.*}} : !fir.ref<f32>

! OFF-LABEL: func.func @_QPtest_real4
! OFF-NOT: fir.zero_bits

! ---------------------------------------------------------------------------
! REAL(8) -- 8-byte FP
! ---------------------------------------------------------------------------
subroutine test_real8(res)
  real(8) :: res
  real(8) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_real8
! ZERO: fir.zero_bits f64
! ZERO: fir.store {{.*}} : !fir.ref<f64>

! NAN-LABEL:  func.func @_QPtest_real8
! NAN:  arith.constant {{.*}} : f64
! NAN:  fir.store {{.*}} : !fir.ref<f64>

! SNAN-LABEL: func.func @_QPtest_real8
! SNAN: arith.constant {{.*}} : f64
! SNAN: fir.store {{.*}} : !fir.ref<f64>

! HEX-LABEL:  func.func @_QPtest_real8
! HEX:  arith.constant -6148914691236517206 : i64
! HEX:  arith.bitcast {{.*}} : i64 to f64
! HEX:  fir.store {{.*}} : !fir.ref<f64>

! ---------------------------------------------------------------------------
! COMPLEX(4) -- two f32 parts; stored as complex<f32>
! nan/snan: both parts get NaN; hex: both parts get bitcast pattern
! ---------------------------------------------------------------------------
subroutine test_complex4(res)
  complex(4) :: res
  complex(4) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_complex4
! ZERO: fir.zero_bits complex<f32>
! ZERO: fir.store {{.*}} : !fir.ref<complex<f32>>

! NAN-LABEL:  func.func @_QPtest_complex4
! NAN:  arith.constant {{.*}} : f32
! NAN:  complex.create {{.*}} : complex<f32>
! NAN:  fir.store {{.*}} : !fir.ref<complex<f32>>

! SNAN-LABEL: func.func @_QPtest_complex4
! SNAN: arith.constant {{.*}} : f32
! SNAN: complex.create {{.*}} : complex<f32>
! SNAN: fir.store {{.*}} : !fir.ref<complex<f32>>

! HEX-LABEL:  func.func @_QPtest_complex4
! HEX:  arith.constant -1431655766 : i32
! HEX:  arith.bitcast {{.*}} : i32 to f32
! HEX:  complex.create {{.*}} : complex<f32>
! HEX:  fir.store {{.*}} : !fir.ref<complex<f32>>

! ---------------------------------------------------------------------------
! COMPLEX(8) -- two f64 parts; stored as complex<f64>
! ---------------------------------------------------------------------------
subroutine test_complex8(res)
  complex(8) :: res
  complex(8) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_complex8
! ZERO: fir.zero_bits complex<f64>
! ZERO: fir.store {{.*}} : !fir.ref<complex<f64>>

! NAN-LABEL:  func.func @_QPtest_complex8
! NAN:  arith.constant {{.*}} : f64
! NAN:  complex.create {{.*}} : complex<f64>
! NAN:  fir.store {{.*}} : !fir.ref<complex<f64>>

! SNAN-LABEL: func.func @_QPtest_complex8
! SNAN: arith.constant {{.*}} : f64
! SNAN: complex.create {{.*}} : complex<f64>
! SNAN: fir.store {{.*}} : !fir.ref<complex<f64>>

! HEX-LABEL:  func.func @_QPtest_complex8
! HEX:  arith.constant -6148914691236517206 : i64
! HEX:  arith.bitcast {{.*}} : i64 to f64
! HEX:  complex.create {{.*}} : complex<f64>
! HEX:  fir.store {{.*}} : !fir.ref<complex<f64>>

! ---------------------------------------------------------------------------
! LOGICAL(1) -- stored as i8; pattern 0xAA = -86
! ---------------------------------------------------------------------------
subroutine test_logical1(res)
  logical(1) :: res
  logical(1) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_logical1
! ZERO: fir.zero_bits !fir.logical<1>

! NAN-LABEL:  func.func @_QPtest_logical1
! NAN:  arith.constant -86 : i8
! NAN:  fir.convert {{.*}} : (i8) -> !fir.logical<1>
! NAN:  fir.store {{.*}} : !fir.ref<!fir.logical<1>>

! HEX-LABEL:  func.func @_QPtest_logical1
! HEX:  arith.constant {{.*}} : i8
! HEX:  fir.convert {{.*}} : (i8) -> !fir.logical<1>
! HEX:  fir.store {{.*}} : !fir.ref<!fir.logical<1>>

! ---------------------------------------------------------------------------
! LOGICAL(4) -- stored as i32; pattern 0xAAAAAAAA = -1431655766
! ---------------------------------------------------------------------------
subroutine test_logical4(res)
  logical(4) :: res
  logical(4) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_logical4
! ZERO: fir.zero_bits !fir.logical<4>

! NAN-LABEL:  func.func @_QPtest_logical4
! NAN:  arith.constant -1431655766 : i32
! NAN:  fir.convert {{.*}} : (i32) -> !fir.logical<4>
! NAN:  fir.store {{.*}} : !fir.ref<!fir.logical<4>>

! HEX-LABEL:  func.func @_QPtest_logical4
! HEX:  arith.constant {{.*}} : i32
! HEX:  fir.convert {{.*}} : (i32) -> !fir.logical<4>
! HEX:  fir.store {{.*}} : !fir.ref<!fir.logical<4>>

! ---------------------------------------------------------------------------
! CHARACTER(10) -- fir::CharacterType is not mlir::FloatType/IntegerType/ComplexType
! nan/snan/hex: fall back to fir.zero_bits (known limitation, TODO)
! ---------------------------------------------------------------------------
subroutine test_char10(res)
  character(10) :: res
  character(10) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_char10
! ZERO: fir.zero_bits !fir.char<1,10>
! ZERO: fir.store {{.*}} : !fir.ref<!fir.char<1,10>>

! NAN-LABEL:  func.func @_QPtest_char10
! NAN:  fir.zero_bits !fir.char<1,10>
! NAN:  fir.store {{.*}} : !fir.ref<!fir.char<1,10>>

! SNAN-LABEL: func.func @_QPtest_char10
! SNAN: fir.zero_bits !fir.char<1,10>
! SNAN: fir.store {{.*}} : !fir.ref<!fir.char<1,10>>

! HEX-LABEL:  func.func @_QPtest_char10
! HEX:  fir.zero_bits !fir.char<1,10>
! HEX:  fir.store {{.*}} : !fir.ref<!fir.char<1,10>>

! ---------------------------------------------------------------------------
! Derived type -- struct with an INTEGER(4) and a REAL(4) field
! nan/hex: field-by-field walk (integer: 0xAA; real: NaN or bitcast)
! ---------------------------------------------------------------------------
subroutine test_derived(res)
  type :: mytype
    integer(4) :: i
    real(4) :: r
  end type
  type(mytype) :: res
  type(mytype) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_derived
! ZERO: fir.zero_bits !fir.type<{{.*}}>
! ZERO: fir.store {{.*}} : !fir.ref<!fir.type<{{.*}}>>

! NAN-LABEL:  func.func @_QPtest_derived
! NAN:  fir.coordinate_of {{.*}} -> !fir.ref<i32>
! NAN:  arith.constant {{.*}} : i32
! NAN:  fir.store {{.*}} : !fir.ref<i32>
! NAN:  fir.coordinate_of {{.*}} -> !fir.ref<f32>
! NAN:  arith.constant {{.*}} : f32
! NAN:  fir.store {{.*}} : !fir.ref<f32>

! HEX-LABEL:  func.func @_QPtest_derived
! HEX:  fir.coordinate_of {{.*}} -> !fir.ref<i32>
! HEX:  arith.constant {{.*}} : i32
! HEX:  fir.store {{.*}} : !fir.ref<i32>
! HEX:  fir.coordinate_of {{.*}} -> !fir.ref<f32>
! HEX:  arith.bitcast {{.*}} : i32 to f32
! HEX:  fir.store {{.*}} : !fir.ref<f32>


! ---------------------------------------------------------------------------
! Array INTEGER(4)(4) -- 1-D; filled via insert_on_range
! ---------------------------------------------------------------------------
subroutine test_int_array(res)
  integer(4) :: res(4)
  integer(4) :: x(4)
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_int_array
! ZERO: fir.insert_on_range {{.*}} from (0) to (3)
! ZERO: fir.store {{.*}} : !fir.ref<!fir.array<4xi32>>

! NAN-LABEL:  func.func @_QPtest_int_array
! NAN:  fir.insert_on_range {{.*}} from (0) to (3)
! NAN:  fir.store {{.*}} : !fir.ref<!fir.array<4xi32>>

! HEX-LABEL:  func.func @_QPtest_int_array
! HEX:  fir.insert_on_range {{.*}} from (0) to (3)
! HEX:  fir.store {{.*}} : !fir.ref<!fir.array<4xi32>>

! OFF-LABEL: func.func @_QPtest_int_array
! OFF-NOT: fir.insert_on_range

! ---------------------------------------------------------------------------
! Array REAL(4)(4) -- 1-D; nan/snan: NaN element; hex: bitcast element
! ---------------------------------------------------------------------------
subroutine test_real_array(res)
  real(4) :: res(4)
  real(4) :: x(4)
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_real_array
! ZERO: fir.insert_on_range {{.*}} from (0) to (3)
! ZERO: fir.store {{.*}} : !fir.ref<!fir.array<4xf32>>

! NAN-LABEL:  func.func @_QPtest_real_array
! NAN:  fir.insert_on_range {{.*}} from (0) to (3)
! NAN:  fir.store {{.*}} : !fir.ref<!fir.array<4xf32>>

! SNAN-LABEL: func.func @_QPtest_real_array
! SNAN: fir.insert_on_range {{.*}} from (0) to (3)
! SNAN: fir.store {{.*}} : !fir.ref<!fir.array<4xf32>>

! HEX-LABEL:  func.func @_QPtest_real_array
! HEX:  fir.insert_on_range {{.*}} from (0) to (3)
! HEX:  fir.store {{.*}} : !fir.ref<!fir.array<4xf32>>

! ---------------------------------------------------------------------------
! Array INTEGER(4)(3,4) -- 2-D; insert_on_range with two-dimension bounds
! ---------------------------------------------------------------------------
subroutine test_int_array_2d(res)
  integer(4) :: res(3,4)
  integer(4) :: x(3,4)
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_int_array_2d
! ZERO: fir.insert_on_range {{.*}} from (0, 0) to (2, 3)
! ZERO: fir.store {{.*}} : !fir.ref<!fir.array<3x4xi32>>

! HEX-LABEL: func.func @_QPtest_int_array_2d
! HEX:  fir.insert_on_range {{.*}} from (0, 0) to (2, 3)
! HEX:  fir.store {{.*}} : !fir.ref<!fir.array<3x4xi32>>

! ---------------------------------------------------------------------------
! Exclusion: explicit init (= 42) -- must NOT be touched
! ---------------------------------------------------------------------------
subroutine test_explicit_init(res)
  integer(4) :: res
  integer(4) :: x = 42
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_explicit_init
! ZERO-NOT: fir.zero_bits

! NAN-LABEL:  func.func @_QPtest_explicit_init
! NAN-NOT:  arith.constant -1431655766 : i32

! HEX-LABEL:  func.func @_QPtest_explicit_init
! HEX-NOT:  arith.bitcast

! ---------------------------------------------------------------------------
! Exclusion: DATA statement init -- must NOT be touched
! ---------------------------------------------------------------------------
subroutine test_data_init(res)
  integer(4) :: res
  integer(4) :: x
  data x /99/
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_data_init
! ZERO-NOT: fir.zero_bits i32

! NAN-LABEL:  func.func @_QPtest_data_init
! NAN-NOT:  arith.constant -1431655766 : i32

! HEX-LABEL:  func.func @_QPtest_data_init
! HEX-NOT:  arith.bitcast

! ---------------------------------------------------------------------------
! Exclusion: derived-type default component init -- must NOT be touched
! ---------------------------------------------------------------------------
subroutine test_default_comp_init(res)
  type :: inittype
    integer(4) :: i = 7
    real(4)    :: r = 3.14
  end type
  type(inittype) :: res
  type(inittype) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_default_comp_init
! ZERO-NOT: fir.zero_bits

! NAN-LABEL:  func.func @_QPtest_default_comp_init
! NAN-NOT:  arith.constant -1431655766 : i32

! HEX-LABEL:  func.func @_QPtest_default_comp_init
! HEX-NOT:  arith.bitcast

! ---------------------------------------------------------------------------
! Exclusion: SAVE -- must NOT be touched
! ---------------------------------------------------------------------------
subroutine test_save(res)
  integer(4) :: res
  integer(4), save :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_save
! ZERO-NOT: fir.zero_bits i32

! HEX-LABEL: func.func @_QPtest_save
! HEX-NOT: arith.constant -1431655766 : i32

! ---------------------------------------------------------------------------
! Exclusion: dummy argument -- must NOT be touched
! ---------------------------------------------------------------------------
subroutine test_dummy(x, res)
  integer(4), intent(in) :: x
  integer(4) :: res
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_dummy
! ZERO-NOT: fir.zero_bits i32

! HEX-LABEL: func.func @_QPtest_dummy
! HEX-NOT: arith.constant -1431655766 : i32

! ---------------------------------------------------------------------------
! Exclusion: ALLOCATABLE -- must NOT be touched
! ---------------------------------------------------------------------------
subroutine test_allocatable(res)
  integer(4), allocatable :: x
  integer(4) :: res
  if (allocated(x)) res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_allocatable
! ZERO-NOT: fir.zero_bits i32

! HEX-LABEL: func.func @_QPtest_allocatable
! HEX-NOT: arith.constant -1431655766 : i32

! ---------------------------------------------------------------------------
! Exclusion: EQUIVALENCE -- must NOT be touched
! ---------------------------------------------------------------------------
subroutine test_equivalence(res)
  integer(4) :: res
  integer(4) :: x, y
  equivalence (x, y)
  res = x + y
end subroutine
! ZERO-LABEL: func.func @_QPtest_equivalence
! ZERO-NOT: fir.zero_bits
! ZERO: return

! NAN-LABEL:  func.func @_QPtest_equivalence
! NAN-NOT:  arith.constant -1431655766 : i32
! NAN: return

! HEX-LABEL:  func.func @_QPtest_equivalence
! HEX-NOT:  arith.bitcast
! HEX: return
