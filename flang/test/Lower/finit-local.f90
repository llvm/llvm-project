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
! Modes exercised: zero, 0xAA (hex), and off (no flag).
!
! RUN: bbc -emit-hlfir -finit-local=zero  -o - %s | FileCheck --check-prefix=ZERO  %s
! RUN: bbc -emit-hlfir -finit-local=0xAA  -o - %s | FileCheck --check-prefix=HEX   %s
! RUN: bbc -emit-hlfir                    -o - %s | FileCheck --check-prefix=OFF   %s
! RUN: bbc -emit-hlfir -finit-local-zero  -o - %s | FileCheck --check-prefix=ZERO  %s
! --- Empty value should be rejected by bbc ---
! RUN: not bbc -emit-hlfir -finit-local=   -o - %s 2>&1 | FileCheck --check-prefix=EMPTY %s
! --- Last option wins before validation: -finit-local-zero after a bad value selects zero ---
! RUN: bbc -emit-hlfir -finit-local=   -finit-local-zero -o - %s | FileCheck --check-prefix=ZERO %s
! RUN: bbc -emit-hlfir -finit-local=bogus -finit-local-zero -o - %s | FileCheck --check-prefix=ZERO %s

! EMPTY: bbc: invalid -finit-local= value: (empty)

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


! HEX-LABEL:  func.func @_QPtest_int8
! HEX:  arith.constant -6148914691236517206 : i64
! HEX:  fir.store {{.*}} : !fir.ref<i64>

! ---------------------------------------------------------------------------
! REAL(4) -- zero: fir.zero_bits; hex: bitcast from integer splat
! ---------------------------------------------------------------------------
subroutine test_real4(res)
  real(4) :: res
  real(4) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_real4
! ZERO: fir.zero_bits f32
! ZERO: fir.store {{.*}} : !fir.ref<f32>



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



! HEX-LABEL:  func.func @_QPtest_real8
! HEX:  arith.constant -6148914691236517206 : i64
! HEX:  arith.bitcast {{.*}} : i64 to f64
! HEX:  fir.store {{.*}} : !fir.ref<f64>

! ---------------------------------------------------------------------------
! COMPLEX(4) -- two f32 parts; stored as complex<f32>
! hex: both parts get bitcast pattern
! ---------------------------------------------------------------------------
subroutine test_complex4(res)
  complex(4) :: res
  complex(4) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_complex4
! ZERO: fir.zero_bits complex<f32>
! ZERO: fir.store {{.*}} : !fir.ref<complex<f32>>



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


! HEX-LABEL:  func.func @_QPtest_logical1
! HEX:  arith.constant {{.*}} : i8
! HEX:  fir.convert {{.*}} : (!fir.ref<!fir.logical<1>>) -> !fir.ref<i8>
! HEX:  fir.store {{.*}} : !fir.ref<i8>

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


! HEX-LABEL:  func.func @_QPtest_logical4
! HEX:  arith.constant {{.*}} : i32
! HEX:  fir.convert {{.*}} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<i32>
! HEX:  fir.store {{.*}} : !fir.ref<i32>

! ---------------------------------------------------------------------------
! CHARACTER(10) -- fixed-length scalar.
! zero: fir.zero_bits over the whole character type.
! hex: byte-loop over 10 singleton code-units.
! ---------------------------------------------------------------------------
subroutine test_char10(res)
  character(10) :: res
  character(10) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_char10
! ZERO: fir.zero_bits !fir.char<1,10>
! ZERO: fir.store {{.*}} : !fir.ref<!fir.char<1,10>>



! HEX-LABEL:  func.func @_QPtest_char10
! HEX:        fir.do_loop
! HEX:          fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! HEX:          arith.constant {{.*}} : i8
! HEX:          fir.store {{.*}} : !fir.ref<i8>

! ---------------------------------------------------------------------------
! CHARACTER(0) -- zero-length: no store should be emitted (guard for
! zero-byte allocation; writing through it would be out of bounds).
! ---------------------------------------------------------------------------
subroutine test_char0(res)
  character(0) :: res
  character(0) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_char0
! ZERO-NOT: fir.store {{.*}} : !fir.ref<!fir.char<1,0>>

! HEX-LABEL:  func.func @_QPtest_char0
! HEX-NOT:  fir.store {{.*}} : !fir.ref<!fir.char<1,0>>

! ---------------------------------------------------------------------------
! CHARACTER(n) -- runtime-length: emit a fir.do_loop over [0, n-1] so
! every byte is initialised. The loop body uses fir.coordinate_of on a
! rank-1 unknown-extent array view of the allocation.
! When n == 0 the trip count is 0 and the body is never entered.
! ---------------------------------------------------------------------------
subroutine test_charN(res, n)
  integer, intent(in) :: n
  character(n) :: res
  character(n) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_charn
! ZERO:       fir.do_loop
! ZERO:         fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! ZERO:         fir.zero_bits !fir.char<1>
! ZERO:         fir.store {{.*}} : !fir.ref<!fir.char<1>>



! HEX-LABEL:  func.func @_QPtest_charn
! HEX:        fir.do_loop
! HEX:          fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! HEX:          arith.constant {{.*}} : i8
! HEX:          fir.store {{.*}} : !fir.ref<i8>

! ---------------------------------------------------------------------------
! Derived type -- struct with an INTEGER(4) and a REAL(4) field
! hex: byte-loop over the whole struct (covers typed fields and padding)
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
! ZERO: fir.do_loop
! ZERO:   fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! ZERO:   arith.constant 0 : i8
! ZERO:   fir.store {{.*}} : !fir.ref<i8>


! HEX-LABEL:  func.func @_QPtest_derived
! HEX:       fir.do_loop
! HEX:         fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! HEX:         arith.constant {{.*}} : i8
! HEX:         fir.store {{.*}} : !fir.ref<i8>


! ---------------------------------------------------------------------------
! Array INTEGER(4)(4) -- 1-D; all modes use do_loop + rank-1 view
! ---------------------------------------------------------------------------
subroutine test_int_array(res)
  integer(4) :: res(4)
  integer(4) :: x(4)
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_int_array
! ZERO: fir.do_loop
! ZERO: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! ZERO: fir.store {{.*}} : !fir.ref<i32>


! HEX-LABEL:  func.func @_QPtest_int_array
! HEX:  fir.do_loop
! HEX:  fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! HEX:  fir.store {{.*}} : !fir.ref<i32>

! OFF-LABEL: func.func @_QPtest_int_array
! OFF-NOT: fir.do_loop
! OFF-NOT: fir.insert_on_range

! ---------------------------------------------------------------------------
! Array REAL(4)(4) -- 1-D; hex: bitcast element
! ---------------------------------------------------------------------------
subroutine test_real_array(res)
  real(4) :: res(4)
  real(4) :: x(4)
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_real_array
! ZERO: fir.do_loop
! ZERO: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! ZERO: fir.store {{.*}} : !fir.ref<f32>



! HEX-LABEL:  func.func @_QPtest_real_array
! HEX:  fir.do_loop
! HEX:  fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! HEX:  fir.store {{.*}} : !fir.ref<f32>

! ---------------------------------------------------------------------------
! Array INTEGER(4)(3,4) -- 2-D; all modes use do_loop + rank-1 view
! ---------------------------------------------------------------------------
subroutine test_int_array_2d(res)
  integer(4) :: res(3,4)
  integer(4) :: x(3,4)
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_int_array_2d
! ZERO: fir.do_loop
! ZERO: fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! ZERO: fir.store {{.*}} : !fir.ref<i32>

! HEX-LABEL: func.func @_QPtest_int_array_2d
! HEX:  fir.do_loop
! HEX:  fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! HEX:  fir.store {{.*}} : !fir.ref<i32>

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


! HEX-LABEL:  func.func @_QPtest_equivalence
! HEX-NOT:  arith.bitcast
! HEX: return

! ---------------------------------------------------------------------------
! CHARACTER(kind=2, len=3) -- fixed-length, higher kind.
! hex: byte-loop over 3*2=6 bytes using a kind=1 singleton view.
! zero: fir.zero_bits over the whole type.
! ---------------------------------------------------------------------------
subroutine test_char2_fixed(res)
  character(kind=2, len=3) :: res
  character(kind=2, len=3) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_char2_fixed
! ZERO: fir.zero_bits !fir.char<2,3>
! ZERO: fir.store {{.*}} : !fir.ref<!fir.char<2,3>>

! HEX-LABEL:  func.func @_QPtest_char2_fixed
! HEX:        fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
! HEX:          fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! HEX:          arith.constant {{.*}} : i8
! HEX:          fir.store {{.*}} : !fir.ref<i8>

! ---------------------------------------------------------------------------
! CHARACTER(kind=4, len=2) -- fixed-length, wider kind.
! hex: byte-loop over 2*4=8 bytes using a kind=1 singleton view.
! ---------------------------------------------------------------------------
subroutine test_char4_fixed(res)
  character(kind=4, len=2) :: res
  character(kind=4, len=2) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_char4_fixed
! ZERO: fir.zero_bits !fir.char<4,2>
! ZERO: fir.store {{.*}} : !fir.ref<!fir.char<4,2>>

! HEX-LABEL:  func.func @_QPtest_char4_fixed
! HEX:        fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
! HEX:          fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! HEX:          arith.constant {{.*}} : i8
! HEX:          fir.store {{.*}} : !fir.ref<i8>

! ---------------------------------------------------------------------------
! CHARACTER(kind=2, len=n) -- runtime-length, higher kind.
! hex/zero: byte-loop over n*2 bytes; skipped when n==0.
! ---------------------------------------------------------------------------
subroutine test_char2_runtime(res, n)
  integer, intent(in) :: n
  character(kind=2, len=n) :: res
  character(kind=2, len=n) :: x
  res = x
end subroutine
! ZERO-LABEL: func.func @_QPtest_char2_runtime
! ZERO:       fir.do_loop
! ZERO:         fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! ZERO:         fir.zero_bits !fir.char<1>
! ZERO:         fir.store {{.*}} : !fir.ref<!fir.char<1>>

! HEX-LABEL:  func.func @_QPtest_char2_runtime
! HEX:        fir.do_loop
! HEX:          fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! HEX:          arith.constant {{.*}} : i8
! HEX:          fir.store {{.*}} : !fir.ref<i8>
