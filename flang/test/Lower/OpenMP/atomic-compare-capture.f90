! This test checks lowering of atomic compare capture constructs to HLFIR.
! RUN: bbc %openmp_flags -fopenmp-version=51 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 %s -o - | FileCheck %s

! ---------------------------------------------------------------------------
! integer == postfix: { if (x==e) x=d; v=x }
! CHECK-LABEL: func.func @_QPcc_int_postfix(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[E:.*]]: !fir.ref<i32> {fir.bindc_name = "e"},
! CHECK-SAME:    %[[D:.*]]: !fir.ref<i32> {fir.bindc_name = "d"},
! CHECK-SAME:    %[[V:.*]]: !fir.ref<i32> {fir.bindc_name = "v"})
! CHECK:         %[[D_DECL:.*]]:2 = hlfir.declare %[[D]] {{.*}}
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[V_DECL:.*]]:2 = hlfir.declare %[[V]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:         %[[DVAL:.*]] = fir.load %[[D_DECL]]#0 : !fir.ref<i32>
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK:           ^bb0(%[[XVAL:.*]]: i32):
! CHECK:             %[[CMP:.*]] = arith.cmpi eq, %[[XVAL]], %[[EVAL]] : i32
! CHECK:             %[[SEL:.*]] = arith.select %[[CMP]], %[[DVAL]], %[[XVAL]] : i32
! CHECK:             omp.yield(%[[SEL]] : i32)
! CHECK:           }
! CHECK:           omp.atomic.read %[[V_DECL]]#0 = %[[X_DECL]]#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:         }
subroutine cc_int_postfix(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) x = d
  v = x
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! integer == prefix: { v=x; if (x==e) x=d }
! CHECK-LABEL: func.func @_QPcc_int_prefix(
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %arg1 {{.*}}Ee"
! CHECK:         %[[E_VAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.read %{{.*}}#0 = %[[X:.*]]#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:           omp.atomic.compare %[[X]]#0 : !fir.ref<i32> {
! CHECK:           ^bb0(%[[XVAL:.*]]: i32):
! CHECK:             arith.cmpi eq, %[[XVAL]], %[[E_VAL]] : i32
! CHECK:             omp.yield
! CHECK:           }
! CHECK:         }
subroutine cc_int_prefix(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! integer == fail-only: { if (x==e) x=d; else v=x }
! CHECK-LABEL: func.func @_QPcc_int_failonly(
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %arg1 {{.*}}Ee"
! CHECK:         %[[E_VAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<i32> {
! CHECK:           ^bb0(%[[XVAL:.*]]: i32):
! CHECK:             arith.cmpi eq, %[[XVAL]], %[[E_VAL]] : i32
! CHECK:             omp.yield
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}}#0 = %{{.*}}#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:         } {fail_only}
subroutine cc_int_failonly(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) then
    x = d
  else
    v = x
  end if
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! integer == fail-only with `fail` clause: v is captured on the failure path,
! and the fail clause sets that path's ordering independently of the success
! ordering (seq_cst success, acquire failure).
! CHECK-LABEL: func.func @_QPcc_int_failonly_fail(
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %arg1 {{.*}}Ee"
! CHECK:         %[[E_VAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:         omp.atomic.capture memory_order(seq_cst) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<i32> {
! CHECK:           ^bb0(%[[XVAL:.*]]: i32):
! CHECK:             arith.cmpi eq, %[[XVAL]], %[[E_VAL]] : i32
! CHECK:             omp.yield
! CHECK:           } {fail_memory_order = #omp<memoryorderkind acquire>}
! CHECK:           omp.atomic.read %{{.*}}#0 = %{{.*}}#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:         } {fail_only}
subroutine cc_int_failonly_fail(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture seq_cst fail(acquire)
  if (x == e) then
    x = d
  else
    v = x
  end if
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! integer == reversed (e==x): same lowering as x==e
! CHECK-LABEL: func.func @_QPcc_int_eq_reversed(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<i32> {
! CHECK:             arith.cmpi eq
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:         }
subroutine cc_int_eq_reversed(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
  if (e == x) x = d
  v = x
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! real == postfix
! CHECK-LABEL: func.func @_QPcc_real_postfix(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<f32> {
! CHECK:             arith.cmpf oeq
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<f32>, !fir.ref<f32>, f32
! CHECK:         }
subroutine cc_real_postfix(x, e, d, v)
  real :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) x = d
  v = x
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! real == prefix
! CHECK-LABEL: func.func @_QPcc_real_prefix(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.read %{{.*}} = %[[X:.*]]#0 : !fir.ref<f32>, !fir.ref<f32>, f32
! CHECK:           omp.atomic.compare %[[X]]#0 : !fir.ref<f32> {
! CHECK:             arith.cmpf oeq
! CHECK:           }
! CHECK:         }
subroutine cc_real_prefix(x, e, d, v)
  real :: x, e, d, v
  !$omp atomic compare capture
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! real == fail-only
! CHECK-LABEL: func.func @_QPcc_real_failonly(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<f32> {
! CHECK:             arith.cmpf oeq
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<f32>, !fir.ref<f32>, f32
! CHECK:         } {fail_only}
subroutine cc_real_failonly(x, e, d, v)
  real :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) then
    x = d
  else
    v = x
  end if
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! logical .eqv. postfix
! CHECK-LABEL: func.func @_QPcc_logical_postfix(
! CHECK:         fir.convert %{{.*}} : (!fir.logical<4>) -> i32
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! CHECK:             arith.cmpi eq
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:         }
subroutine cc_logical_postfix(x, e, d, v)
  logical :: x, e, d, v
  !$omp atomic compare capture
  if (x .eqv. e) x = d
  v = x
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! logical .eqv. prefix
! CHECK-LABEL: func.func @_QPcc_logical_prefix(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:           omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! CHECK:             arith.cmpi eq
! CHECK:           }
! CHECK:         }
subroutine cc_logical_prefix(x, e, d, v)
  logical :: x, e, d, v
  !$omp atomic compare capture
  v = x
  if (x .eqv. e) x = d
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! logical .eqv. fail-only
! CHECK-LABEL: func.func @_QPcc_logical_failonly(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}} : !fir.ref<i32> {
! CHECK:             arith.cmpi eq
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:         } {fail_only}
subroutine cc_logical_failonly(x, e, d, v)
  logical :: x, e, d, v
  !$omp atomic compare capture
  if (x .eqv. e) then
    x = d
  else
    v = x
  end if
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! complex == postfix
! CHECK-LABEL: func.func @_QPcc_complex_postfix(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<complex<f32>> {
! CHECK:             fir.cmpc "oeq"
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<complex<f32>>, !fir.ref<complex<f32>>, complex<f32>
! CHECK:         }
subroutine cc_complex_postfix(x, e, d, v)
  complex :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) x = d
  v = x
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! complex == prefix
! CHECK-LABEL: func.func @_QPcc_complex_prefix(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.read %{{.*}} = %[[X:.*]]#0 : !fir.ref<complex<f32>>, !fir.ref<complex<f32>>, complex<f32>
! CHECK:           omp.atomic.compare %[[X]]#0 : !fir.ref<complex<f32>> {
! CHECK:             fir.cmpc "oeq"
! CHECK:           }
! CHECK:         }
subroutine cc_complex_prefix(x, e, d, v)
  complex :: x, e, d, v
  !$omp atomic compare capture
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! complex == fail-only
! CHECK-LABEL: func.func @_QPcc_complex_failonly(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<complex<f32>> {
! CHECK:             fir.cmpc "oeq"
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<complex<f32>>, !fir.ref<complex<f32>>, complex<f32>
! CHECK:         } {fail_only}
subroutine cc_complex_failonly(x, e, d, v)
  complex :: x, e, d, v
  !$omp atomic compare capture
  if (x == e) then
    x = d
  else
    v = x
  end if
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! integer min (x > e) postfix
! CHECK-LABEL: func.func @_QPcc_int_min(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<i32> {
! CHECK:             arith.cmpi sgt
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:         }
subroutine cc_int_min(x, e, v)
  integer :: x, e, v
  !$omp atomic compare capture
  if (x > e) x = e
  v = x
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! integer max (x < e) postfix
! CHECK-LABEL: func.func @_QPcc_int_max(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<i32> {
! CHECK:             arith.cmpi slt
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:         }
subroutine cc_int_max(x, e, v)
  integer :: x, e, v
  !$omp atomic compare capture
  if (x < e) x = e
  v = x
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! real min (x > e) postfix
! CHECK-LABEL: func.func @_QPcc_real_min(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<f32> {
! CHECK:             arith.cmpf ogt
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<f32>, !fir.ref<f32>, f32
! CHECK:         }
subroutine cc_real_min(x, e, v)
  real :: x, e, v
  !$omp atomic compare capture
  if (x > e) x = e
  v = x
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! real max (x < e) postfix
! CHECK-LABEL: func.func @_QPcc_real_max(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<f32> {
! CHECK:             arith.cmpf olt
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<f32>, !fir.ref<f32>, f32
! CHECK:         }
subroutine cc_real_max(x, e, v)
  real :: x, e, v
  !$omp atomic compare capture
  if (x < e) x = e
  v = x
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! weak clause (prefix form)
! CHECK-LABEL: func.func @_QPcc_weak_prefix(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.read %{{.*}}#0 = %[[X:.*]]#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:           omp.atomic.compare %[[X]]#0 : !fir.ref<i32> {
! CHECK:             arith.cmpi eq
! CHECK:           } {weak}
! CHECK:         }
subroutine cc_weak_prefix(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture weak
  v = x
  if (x == e) x = d
  !$omp end atomic
end subroutine

! ---------------------------------------------------------------------------
! weak clause (postfix form)
! CHECK-LABEL: func.func @_QPcc_weak_postfix(
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %{{.*}}#0 : !fir.ref<i32> {
! CHECK:             arith.cmpi eq
! CHECK:           } {weak}
! CHECK:           omp.atomic.read %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:         }
subroutine cc_weak_postfix(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture weak
  if (x == e) x = d
  v = x
  !$omp end atomic
end subroutine
