!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix=HLFIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix=FIR
!RUN: %if x86-registered-target %{ %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix=LLVM %}
!RUN: %if aarch64-registered-target %{ %flang_fc1 -triple aarch64-unknown-linux-gnu -emit-llvm -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix=LLVM %}

! OpenMP atomic compare-capture tracked through every stage of the compiler
! (HLFIR, FIR/omp dialect, LLVM IR) for every operand type: integer, real,
! logical and complex under ==, plus integer and real under min/max (<, >).
! Equality lowers to a cmpxchg; min/max lowers to an atomicrmw.

! ===========================================================================
! integer ==  (postfix): cmpxchg + select(new value) + store into v
! HLFIR-LABEL: func.func @_QPcc_integer(
! HLFIR:         %[[X:.*]]:2 = hlfir.declare %{{.*}}Ex"
! HLFIR:         omp.atomic.capture memory_order(relaxed) {
! HLFIR:           omp.atomic.compare %[[X]]#0 : !fir.ref<i32> {
! HLFIR:             arith.cmpi eq
! HLFIR:           }
! HLFIR:           omp.atomic.read %{{.*}} = %[[X]]#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_integer(
! FIR:         omp.atomic.capture memory_order(relaxed) {
! FIR:           omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! FIR:             arith.cmpi eq
! FIR:           }
! FIR:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! FIR:         }
! LLVM-LABEL: define void @cc_integer_(
! LLVM:         %[[RES:.*]] = cmpxchg ptr %{{.*}}, i32 %{{.*}}, i32 %[[D:.*]] monotonic monotonic
! LLVM:         %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
! LLVM:         %[[OK:.*]] = extractvalue { i32, i1 } %[[RES]], 1
! LLVM:         %[[NEW:.*]] = select i1 %[[OK]], i32 %[[D]], i32 %[[OLD]]
! LLVM:         store i32 %[[NEW]], ptr
subroutine cc_integer(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) x = d
  v = x
  !$omp end atomic
end

! ===========================================================================
! real ==  (postfix): equality on floats uses the HandleFPNegZero cmpxchg path.
! HLFIR-LABEL: func.func @_QPcc_real(
! HLFIR:         omp.atomic.compare %{{.*}}#0 : !fir.ref<f32> {
! HLFIR:           arith.cmpf oeq
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_real(
! FIR:         omp.atomic.compare %{{.*}} : !fir.ref<f32> {
! FIR:           arith.cmpf oeq
! FIR:         }
! LLVM-LABEL: define void @cc_real_(
! LLVM:         cmpxchg ptr
! LLVM:         store float %{{.*}}, ptr
subroutine cc_real(x, e, d, v)
  real :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) x = d
  v = x
  !$omp end atomic
end

! ===========================================================================
! logical ==  (postfix): the logical atom is converted to i32, so the omp op
! operates on !fir.ref<i32> and lowers to an i32 cmpxchg.
! HLFIR-LABEL: func.func @_QPcc_logical(
! HLFIR:         fir.convert %{{.*}} : (!fir.logical<4>) -> i32
! HLFIR:         omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! HLFIR:           arith.cmpi eq
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_logical(
! FIR:         fir.convert %{{.*}} : (!fir.logical<4>) -> i32
! FIR:         omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! LLVM-LABEL: define void @cc_logical_(
! LLVM:         cmpxchg ptr %{{.*}}, i32
! LLVM:         store i32 %{{.*}}, ptr
subroutine cc_logical(x, e, d, v)
  logical :: x, e, d, v
  !$omp atomic compare capture
  if (x .eqv. e) x = d
  v = x
  !$omp end atomic
end

! ===========================================================================
! complex ==  (postfix): fir.cmpc region; lowers to an i64 (bitcast) cmpxchg
! and the captured new value is a { float, float } struct.
! HLFIR-LABEL: func.func @_QPcc_complex(
! HLFIR:         omp.atomic.compare %{{.*}}#0 : !fir.ref<complex<f32>> {
! HLFIR:           fir.cmpc "oeq"
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_complex(
! FIR:         omp.atomic.compare %{{.*}} : !fir.ref<complex<f32>> {
! FIR:           fir.cmpc "oeq"
! FIR:         }
! LLVM-LABEL: define void @cc_complex_(
! LLVM:         %[[REEQ:.*]] = fcmp oeq float %[[REX:.*]], %[[REE:.*]]
! LLVM:         %[[IMEQ:.*]] = fcmp oeq float %[[IMX:.*]], %[[IME:.*]]
! LLVM:         cmpxchg ptr %{{.*}}, i64
! LLVM:         store { float, float } %{{.*}}, ptr
subroutine cc_complex(x, e, d, v)
  complex :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) x = d
  v = x
  !$omp end atomic
end

! ===========================================================================
! integer min (x > e, postfix): atomicrmw min, v gets the new value smin(old,e).
! HLFIR-LABEL: func.func @_QPcc_integer_min(
! HLFIR:         omp.atomic.compare %{{.*}}#0 : !fir.ref<i32> {
! HLFIR:           arith.cmpi sgt
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_integer_min(
! FIR:         omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! FIR:           arith.cmpi sgt
! FIR:         }
! LLVM-LABEL: define void @cc_integer_min_(
! LLVM:         %[[OLD:.*]] = atomicrmw min ptr %{{.*}}, i32 %[[E:.*]] monotonic
! LLVM:         %[[NEW:.*]] = call i32 @llvm.smin.i32(i32 %[[OLD]], i32 %[[E]])
! LLVM:         store i32 %[[NEW]], ptr
subroutine cc_integer_min(x, e, v)
  integer :: x, e, v
  !$omp atomic compare capture
  if (x > e) x = e
  v = x
  !$omp end atomic
end

! ===========================================================================
! integer max (x < e, postfix): atomicrmw max, v gets the new value smax(old,e).
! HLFIR-LABEL: func.func @_QPcc_integer_max(
! HLFIR:         omp.atomic.compare %{{.*}}#0 : !fir.ref<i32> {
! HLFIR:           arith.cmpi slt
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_integer_max(
! FIR:         omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! FIR:           arith.cmpi slt
! FIR:         }
! LLVM-LABEL: define void @cc_integer_max_(
! LLVM:         %[[OLD:.*]] = atomicrmw max ptr %{{.*}}, i32 %[[E:.*]] monotonic
! LLVM:         %[[NEW:.*]] = call i32 @llvm.smax.i32(i32 %[[OLD]], i32 %[[E]])
! LLVM:         store i32 %[[NEW]], ptr
subroutine cc_integer_max(x, e, v)
  integer :: x, e, v
  !$omp atomic compare capture
  if (x < e) x = e
  v = x
  !$omp end atomic
end

! ===========================================================================
! real min (x > e, postfix): atomicrmw fmin, v gets minnum(old, e).
! HLFIR-LABEL: func.func @_QPcc_real_min(
! HLFIR:         omp.atomic.compare %{{.*}}#0 : !fir.ref<f32> {
! HLFIR:           arith.cmpf ogt
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_real_min(
! FIR:         omp.atomic.compare %{{.*}} : !fir.ref<f32> {
! FIR:           arith.cmpf ogt
! FIR:         }
! LLVM-LABEL: define void @cc_real_min_(
! LLVM:         %[[OLD:.*]] = atomicrmw fmin ptr %{{.*}}, float %[[E:.*]] monotonic
! LLVM:         %[[NEW:.*]] = call float @llvm.minnum.f32(float %[[OLD]], float %[[E]])
! LLVM:         store float %[[NEW]], ptr
subroutine cc_real_min(x, e, v)
  real :: x, e, v
  !$omp atomic compare capture
  if (x > e) x = e
  v = x
  !$omp end atomic
end

! ===========================================================================
! real max (x < e, postfix): atomicrmw fmax, v gets maxnum(old, e).
! HLFIR-LABEL: func.func @_QPcc_real_max(
! HLFIR:         omp.atomic.compare %{{.*}}#0 : !fir.ref<f32> {
! HLFIR:           arith.cmpf olt
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_real_max(
! FIR:         omp.atomic.compare %{{.*}} : !fir.ref<f32> {
! FIR:           arith.cmpf olt
! FIR:         }
! LLVM-LABEL: define void @cc_real_max_(
! LLVM:         %[[OLD:.*]] = atomicrmw fmax ptr %{{.*}}, float %[[E:.*]] monotonic
! LLVM:         %[[NEW:.*]] = call float @llvm.maxnum.f32(float %[[OLD]], float %[[E]])
! LLVM:         store float %[[NEW]], ptr
subroutine cc_real_max(x, e, v)
  real :: x, e, v
  !$omp atomic compare capture
  if (x < e) x = e
  v = x
  !$omp end atomic
end

! The remaining cases exercise the different *capture-statement* forms allowed
! for atomic compare-capture (all with an integer == atom).

! ===========================================================================
! Prefix form  {v = x; cond-update-stmt}:  the read precedes the update, so v
! captures the OLD value of x - the read is emitted first and v is a direct
! store of the extracted old value (no select).
! HLFIR-LABEL: func.func @_QPcc_prefix(
! HLFIR:         omp.atomic.capture memory_order(relaxed) {
! HLFIR:           omp.atomic.read %{{.*}} = %[[X:.*]]#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! HLFIR:           omp.atomic.compare %[[X]]#0 : !fir.ref<i32> {
! HLFIR:             arith.cmpi eq
! HLFIR:           }
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_prefix(
! FIR:         omp.atomic.capture memory_order(relaxed) {
! FIR:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! FIR:           omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! FIR:         }
! LLVM-LABEL: define void @cc_prefix_(
! LLVM:         %[[RES:.*]] = cmpxchg ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}} monotonic monotonic
! LLVM:         %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
! LLVM:         store i32 %[[OLD]], ptr
subroutine cc_prefix(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
  v = x
  if (x == e) x = d
  !$omp end atomic
end

! ===========================================================================
! Fail-only form  {if(x == e){x = d;} else {v = x;}}:  v is captured only when
! the compare fails; the capture carries the fail_only attribute and lowers to
! a conditional store of the old value.
! HLFIR-LABEL: func.func @_QPcc_failonly(
! HLFIR:         omp.atomic.capture memory_order(relaxed) {
! HLFIR:           omp.atomic.compare %{{.*}}#0 : !fir.ref<i32> {
! HLFIR:             arith.cmpi eq
! HLFIR:           }
! HLFIR:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! HLFIR:         } {fail_only}
! FIR-LABEL: func.func @_QPcc_failonly(
! FIR:         omp.atomic.capture memory_order(relaxed) {
! FIR:         } {fail_only}
! LLVM-LABEL: define void @cc_failonly_(
! LLVM:         %[[RES:.*]] = cmpxchg ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}} monotonic monotonic
! LLVM:         %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
! LLVM:         %[[OK:.*]] = extractvalue { i32, i1 } %[[RES]], 1
! LLVM:         br i1 %[[OK]], label %[[EXIT:.*]], label %[[CONT:.*]]
! LLVM:         [[CONT]]:
! LLVM:         store i32 %[[OLD]], ptr
subroutine cc_failonly(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) then
    x = d
  else
    v = x
  end if
  !$omp end atomic
end

! ===========================================================================
! Reversed comparison operand order  {if(e == x){x = d;} ; v = x}:  the front
! end normalises 'e == x' to the same compare-capture as 'x == e', so this
! accepts the alternate spelling and lowers identically (cmpxchg + select).
! HLFIR-LABEL: func.func @_QPcc_eq_reversed(
! HLFIR:         omp.atomic.compare %{{.*}}#0 : !fir.ref<i32> {
! HLFIR:           arith.cmpi eq
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_eq_reversed(
! LLVM-LABEL: define void @cc_eq_reversed_(
! LLVM:         %[[RES:.*]] = cmpxchg ptr %{{.*}}, i32 %{{.*}}, i32 %[[D:.*]] monotonic monotonic
! LLVM:         %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
! LLVM:         %[[OK:.*]] = extractvalue { i32, i1 } %[[RES]], 1
! LLVM:         %[[NEW:.*]] = select i1 %[[OK]], i32 %[[D]], i32 %[[OLD]]
! LLVM:         store i32 %[[NEW]], ptr
subroutine cc_eq_reversed(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
  if (e == x) x = d
  v = x
  !$omp end atomic
end

! ===========================================================================
! The prefix and fail-only forms for the remaining operand types (real,
! logical, complex), to confirm each type honours every capture form.

! real, prefix: v gets the old value of x (float).
! HLFIR-LABEL: func.func @_QPcc_real_prefix(
! HLFIR:         omp.atomic.capture memory_order(relaxed) {
! HLFIR:           omp.atomic.read %{{.*}} = %[[X:.*]]#0 : !fir.ref<f32>, !fir.ref<f32>, f32
! HLFIR:           omp.atomic.compare %[[X]]#0 : !fir.ref<f32> {
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_real_prefix(
! FIR:           omp.atomic.read %{{.*}} : !fir.ref<f32>, !fir.ref<f32>, f32
! FIR:           omp.atomic.compare %{{.*}} : !fir.ref<f32> {
! LLVM-LABEL: define void @cc_real_prefix_(
! LLVM:         cmpxchg ptr
! LLVM:         store float %{{.*}}, ptr
subroutine cc_real_prefix(x, e, d, v)
  real :: x, e, d, v
  !$omp atomic compare capture
  v = x
  if (x == e) x = d
  !$omp end atomic
end

! real, fail-only: conditional store of the old value (float).
! HLFIR-LABEL: func.func @_QPcc_real_failonly(
! HLFIR:         omp.atomic.capture memory_order(relaxed) {
! HLFIR:           omp.atomic.compare %{{.*}}#0 : !fir.ref<f32> {
! HLFIR:         } {fail_only}
! FIR-LABEL: func.func @_QPcc_real_failonly(
! FIR:         } {fail_only}
! LLVM-LABEL: define void @cc_real_failonly_(
! LLVM:         cmpxchg ptr
! LLVM:         br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
! LLVM:         store float %{{.*}}, ptr
subroutine cc_real_failonly(x, e, d, v)
  real :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) then
    x = d
  else
    v = x
  end if
  !$omp end atomic
end

! logical, prefix: v gets the old value (via the i32 atom).
! HLFIR-LABEL: func.func @_QPcc_logical_prefix(
! HLFIR:         omp.atomic.capture memory_order(relaxed) {
! HLFIR:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! HLFIR:           omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! FIR-LABEL: func.func @_QPcc_logical_prefix(
! FIR:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! FIR:           omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! LLVM-LABEL: define void @cc_logical_prefix_(
! LLVM:         %[[RES:.*]] = cmpxchg ptr %{{.*}}, i32
! LLVM:         %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
subroutine cc_logical_prefix(x, e, d, v)
  logical :: x, e, d, v
  !$omp atomic compare capture
  v = x
  if (x .eqv. e) x = d
  !$omp end atomic
end

! logical, fail-only: conditional store (via the i32 atom).
! HLFIR-LABEL: func.func @_QPcc_logical_failonly(
! HLFIR:         } {fail_only}
! FIR-LABEL: func.func @_QPcc_logical_failonly(
! FIR:         } {fail_only}
! LLVM-LABEL: define void @cc_logical_failonly_(
! LLVM:         %[[RES:.*]] = cmpxchg ptr %{{.*}}, i32
! LLVM:         %[[OK:.*]] = extractvalue { i32, i1 } %[[RES]], 1
! LLVM:         br i1 %[[OK]], label %{{.*}}, label %{{.*}}
subroutine cc_logical_failonly(x, e, d, v)
  logical :: x, e, d, v
  !$omp atomic compare capture
  if (x .eqv. e) then
    x = d
  else
    v = x
  end if
  !$omp end atomic
end

! complex, prefix: v gets the old complex value (struct), no select.
! HLFIR-LABEL: func.func @_QPcc_complex_prefix(
! HLFIR:         omp.atomic.capture memory_order(relaxed) {
! HLFIR:           omp.atomic.read %{{.*}} = %[[X:.*]]#0 : !fir.ref<complex<f32>>, !fir.ref<complex<f32>>, complex<f32>
! HLFIR:           omp.atomic.compare %[[X]]#0 : !fir.ref<complex<f32>> {
! FIR-LABEL: func.func @_QPcc_complex_prefix(
! FIR:           omp.atomic.read %{{.*}} : !fir.ref<complex<f32>>, !fir.ref<complex<f32>>, complex<f32>
! LLVM-LABEL: define void @cc_complex_prefix_(
! LLVM:         %[[REEQ:.*]] = fcmp oeq float %[[REX:.*]], %[[REE:.*]]
! LLVM:         %[[IMEQ:.*]] = fcmp oeq float %[[IMX:.*]], %[[IME:.*]]
! LLVM:         %[[RES:.*]] = cmpxchg ptr %{{.*}}, i64
! LLVM:         store { float, float } %{{.*}}, ptr
subroutine cc_complex_prefix(x, e, d, v)
  complex :: x, e, d, v
  !$omp atomic compare capture
  v = x
  if (x == e) x = d
  !$omp end atomic
end

! complex, fail-only: conditional store of the old complex value.
! HLFIR-LABEL: func.func @_QPcc_complex_failonly(
! HLFIR:         omp.atomic.compare %{{.*}}#0 : !fir.ref<complex<f32>> {
! HLFIR:         } {fail_only}
! FIR-LABEL: func.func @_QPcc_complex_failonly(
! FIR:         } {fail_only}
! LLVM-LABEL: define void @cc_complex_failonly_(
! LLVM:         %[[REEQ:.*]] = fcmp oeq float %[[REX:.*]], %[[REE:.*]]
! LLVM:         %[[IMEQ:.*]] = fcmp oeq float %[[IMX:.*]], %[[IME:.*]]
! LLVM:         cmpxchg ptr %{{.*}}, i64
! LLVM:         %[[OK:.*]] = phi i1
! LLVM:         %[[FAILED:.*]] = xor i1 %[[OK]], true
! LLVM:         br i1 %[[FAILED]], label %{{.*}}, label %{{.*}}
! LLVM:         store { float, float } %{{.*}}, ptr
subroutine cc_complex_failonly(x, e, d, v)
  complex :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) then
    x = d
  else
    v = x
  end if
  !$omp end atomic
end

! ===========================================================================
! weak clause (postfix): cmpxchg gets the 'weak' modifier.
! HLFIR-LABEL: func.func @_QPcc_weak(
! HLFIR:         omp.atomic.capture memory_order(relaxed) {
! HLFIR:           omp.atomic.compare %{{.*}}#0 : !fir.ref<i32> {
! HLFIR:             arith.cmpi eq
! HLFIR:           } {weak}
! HLFIR:           omp.atomic.read %{{.*}} = %{{.*}}#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! HLFIR:         }
! FIR-LABEL: func.func @_QPcc_weak(
! FIR:         omp.atomic.capture memory_order(relaxed) {
! FIR:           omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! FIR:             arith.cmpi eq
! FIR:           } {weak}
! FIR:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! FIR:         }
! LLVM-LABEL: define void @cc_weak_(
! LLVM:         %[[RES:.*]] = cmpxchg weak ptr %{{.*}}, i32 %{{.*}}, i32 %[[D:.*]] monotonic monotonic
! LLVM:         %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
! LLVM:         %[[OK:.*]] = extractvalue { i32, i1 } %[[RES]], 1
! LLVM:         %[[NEW:.*]] = select i1 %[[OK]], i32 %[[D]], i32 %[[OLD]]
! LLVM:         store i32 %[[NEW]], ptr
subroutine cc_weak(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture weak
  if (x == e) x = d
  v = x
  !$omp end atomic
end
