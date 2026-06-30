! This test checks lowering of atomic compare constructs.
! RUN: bbc %openmp_flags -fopenmp-version=51 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPatomic_compare_int_eq(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[E:.*]]: !fir.ref<i32> {fir.bindc_name = "e"},
! CHECK-SAME:    %[[D:.*]]: !fir.ref<i32> {fir.bindc_name = "d"})
! CHECK:         %[[D_DECL:.*]]:2 = hlfir.declare %[[D]] {{.*}}
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:         %[[DVAL:.*]] = fir.load %[[D_DECL]]#0 : !fir.ref<i32>
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: i32):
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[XVAL]], %[[EVAL]] : i32
! CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[DVAL]], %[[XVAL]] : i32
! CHECK:           omp.yield(%[[SEL]] : i32)
! CHECK:         }
subroutine atomic_compare_int_eq(x, e, d)
  integer :: x, e, d
  !$omp atomic compare
  if (x .eq. e) x = d
end

! CHECK-LABEL: func.func @_QPatomic_compare_float_eq(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<f32> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[E:.*]]: !fir.ref<f32> {fir.bindc_name = "e"},
! CHECK-SAME:    %[[D:.*]]: !fir.ref<f32> {fir.bindc_name = "d"})
! CHECK:         %[[D_DECL:.*]]:2 = hlfir.declare %[[D]] {{.*}}
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
! CHECK:         %[[DVAL:.*]] = fir.load %[[D_DECL]]#0 : !fir.ref<f32>
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<f32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: f32):
! CHECK:           %[[CMP:.*]] = arith.cmpf oeq, %[[XVAL]], %[[EVAL]] fastmath<contract> : f32
! CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[DVAL]], %[[XVAL]] : f32
! CHECK:           omp.yield(%[[SEL]] : f32)
! CHECK:         }
subroutine atomic_compare_float_eq(x, e, d)
  real :: x, e, d
  !$omp atomic compare
  if (x .eq. e) x = d
end

! CHECK-LABEL: func.func @_QPatomic_compare_complex_eq(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[E:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "e"},
! CHECK-SAME:    %[[D:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "d"})
! CHECK:         %[[D_DECL:.*]]:2 = hlfir.declare %[[D]] {{.*}}
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<complex<f32>>
! CHECK:         %[[DVAL:.*]] = fir.load %[[D_DECL]]#0 : !fir.ref<complex<f32>>
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<complex<f32>> {
! CHECK:         ^bb0(%[[XVAL:.*]]: complex<f32>):
! CHECK:           %[[CMP:.*]] = fir.cmpc "oeq", %[[XVAL]], %[[EVAL]] {fastmath = #arith.fastmath<contract>} : complex<f32>
! CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[DVAL]], %[[XVAL]] : complex<f32>
! CHECK:           omp.yield(%[[SEL]] : complex<f32>)
! CHECK:         }
subroutine atomic_compare_complex_eq(x, e, d)
  complex :: x, e, d
  !$omp atomic compare
  if (x .eq. e) x = d
end

! CHECK-LABEL: func.func @_QPatomic_compare_int_lt(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[E:.*]]: !fir.ref<i32> {fir.bindc_name = "e"})
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:         %[[EVAL2:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: i32):
! CHECK:           %[[CMP:.*]] = arith.cmpi slt, %[[XVAL]], %[[EVAL]] : i32
! CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[EVAL2]], %[[XVAL]] : i32
! CHECK:           omp.yield(%[[SEL]] : i32)
! CHECK:         }
subroutine atomic_compare_int_lt(x, e)
  integer :: x, e
  !$omp atomic compare
  if (x .lt. e) x = e
end

! CHECK-LABEL: func.func @_QPatomic_compare_int_gt(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[E:.*]]: !fir.ref<i32> {fir.bindc_name = "e"})
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:         %[[EVAL2:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: i32):
! CHECK:           %[[CMP:.*]] = arith.cmpi sgt, %[[XVAL]], %[[EVAL]] : i32
! CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[EVAL2]], %[[XVAL]] : i32
! CHECK:           omp.yield(%[[SEL]] : i32)
! CHECK:         }
subroutine atomic_compare_int_gt(x, e)
  integer :: x, e
  !$omp atomic compare
  if (x .gt. e) x = e
end

! CHECK-LABEL: func.func @_QPatomic_compare_float_lt(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<f32> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[E:.*]]: !fir.ref<f32> {fir.bindc_name = "e"})
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
! CHECK:         %[[EVAL2:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<f32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: f32):
! CHECK:           %[[CMP:.*]] = arith.cmpf olt, %[[XVAL]], %[[EVAL]] fastmath<contract> : f32
! CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[EVAL2]], %[[XVAL]] : f32
! CHECK:           omp.yield(%[[SEL]] : f32)
! CHECK:         }
subroutine atomic_compare_float_lt(x, e)
  real :: x, e
  !$omp atomic compare
  if (x .lt. e) x = e
end

! CHECK-LABEL: func.func @_QPatomic_compare_float_gt(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<f32> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[E:.*]]: !fir.ref<f32> {fir.bindc_name = "e"})
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
! CHECK:         %[[EVAL2:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<f32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: f32):
! CHECK:           %[[CMP:.*]] = arith.cmpf ogt, %[[XVAL]], %[[EVAL]] fastmath<contract> : f32
! CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[EVAL2]], %[[XVAL]] : f32
! CHECK:           omp.yield(%[[SEL]] : f32)
! CHECK:         }
subroutine atomic_compare_float_gt(x, e)
  real :: x, e
  !$omp atomic compare
  if (x .gt. e) x = e
end

! CHECK-LABEL: func.func @_QPatomic_compare_int_eq_weak(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<i32> {fir.bindc_name = "x"}
! CHECK-SAME:    %[[E:.*]]: !fir.ref<i32> {fir.bindc_name = "e"}
! CHECK-SAME:    %[[D:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}
! CHECK:         %[[D_DECL:.*]]:2 = hlfir.declare %[[D]] {{.*}}
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:         %[[DVAL:.*]] = fir.load %[[D_DECL]]#0 : !fir.ref<i32>
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: i32):
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[XVAL]], %[[EVAL]] : i32
! CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[DVAL]], %[[XVAL]] : i32
! CHECK:           omp.yield(%[[SEL]] : i32)
! CHECK:         } {{.*}}weak{{.*}}
subroutine atomic_compare_int_eq_weak(x, e, d)
  integer :: x, e, d
  !$omp atomic compare weak
  if (x .eq. e) x = d
end

! CHECK-LABEL: func.func @_QPatomic_compare_capture_int_eq(
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
! CHECK:           omp.atomic.read %[[V_DECL]]#0 = %[[X_DECL]]#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:           omp.atomic.compare %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK:           ^bb0(%[[XVAL:.*]]: i32):
! CHECK:             %[[CMP:.*]] = arith.cmpi eq, %[[XVAL]], %[[EVAL]] : i32
! CHECK:             %[[SEL:.*]] = arith.select %[[CMP]], %[[DVAL]], %[[XVAL]] : i32
! CHECK:             omp.yield(%[[SEL]] : i32)
! CHECK:           }
! CHECK:         }
subroutine atomic_compare_capture_int_eq(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
    v = x
    if (x .eq. e) x = d
  !$omp end atomic
end

! CHECK-LABEL: func.func @_QPatomic_compare_capture_int_gt(
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
! CHECK:           omp.atomic.read %[[V_DECL]]#0 = %[[X_DECL]]#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:           omp.atomic.compare %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK:           ^bb0(%[[XVAL:.*]]: i32):
! CHECK:             %[[CMP:.*]] = arith.cmpi sgt, %[[XVAL]], %[[EVAL]] : i32
! CHECK:             %[[SEL:.*]] = arith.select %[[CMP]], %[[DVAL]], %[[XVAL]] : i32
! CHECK:             omp.yield(%[[SEL]] : i32)
! CHECK:           } {{.*}}weak{{.*}}
! CHECK:         }
subroutine atomic_compare_capture_int_gt(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture weak
    v = x
    if (x > e) x = d
  !$omp end atomic
end

! CHECK-LABEL: func.func @_QPatomic_compare_capture_postfix(
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
subroutine atomic_compare_capture_postfix(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
    if (x .eq. e) x = d
    v = x
  !$omp end atomic
end

! CHECK-LABEL: func.func @_QPatomic_compare_capture_fail_only(
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
! CHECK:         } {fail_only}
subroutine atomic_compare_capture_fail_only(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
    if (x .eq. e) then
      x = d
    else
      v = x
    end if
  !$omp end atomic
end

! CHECK-LABEL: func.func @_QPatomic_compare_capture_real(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<f32> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[E:.*]]: !fir.ref<f32> {fir.bindc_name = "e"},
! CHECK-SAME:    %[[D:.*]]: !fir.ref<f32> {fir.bindc_name = "d"},
! CHECK-SAME:    %[[V:.*]]: !fir.ref<f32> {fir.bindc_name = "v"})
! CHECK:         %[[D_DECL:.*]]:2 = hlfir.declare %[[D]] {{.*}}
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[V_DECL:.*]]:2 = hlfir.declare %[[V]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
! CHECK:         %[[DVAL:.*]] = fir.load %[[D_DECL]]#0 : !fir.ref<f32>
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %[[X_DECL]]#0 : !fir.ref<f32> {
! CHECK:           ^bb0(%[[XVAL:.*]]: f32):
! CHECK:             %[[CMP:.*]] = arith.cmpf oeq, %[[XVAL]], %[[EVAL]] fastmath<contract> : f32
! CHECK:             %[[SEL:.*]] = arith.select %[[CMP]], %[[DVAL]], %[[XVAL]] : f32
! CHECK:             omp.yield(%[[SEL]] : f32)
! CHECK:           }
! CHECK:           omp.atomic.read %[[V_DECL]]#0 = %[[X_DECL]]#0 : !fir.ref<f32>, !fir.ref<f32>, f32
! CHECK:         }
subroutine atomic_compare_capture_real(x, e, d, v)
  real :: x, e, d, v
  !$omp atomic compare capture
    if (x == e) x = d
    v = x
  !$omp end atomic
end

! CHECK-LABEL: func.func @_QPatomic_compare_capture_logical(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[E:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "e"},
! CHECK-SAME:    %[[D:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "d"},
! CHECK-SAME:    %[[V:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "v"})
! CHECK:         %[[D_DECL:.*]]:2 = hlfir.declare %[[D]] {{.*}}
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[V_DECL:.*]]:2 = hlfir.declare %[[V]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         %[[ELOAD:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:         %[[XCONV:.*]] = fir.convert %[[X_DECL]]#0 : (!fir.ref<!fir.logical<4>>) -> !fir.ref<i32>
! CHECK:         %[[ECONV:.*]] = fir.convert %[[ELOAD]] : (!fir.logical<4>) -> i32
! CHECK:         %[[DLOAD:.*]] = fir.load %[[D_DECL]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:         %[[DCONV:.*]] = fir.convert %[[DLOAD]] : (!fir.logical<4>) -> i32
! CHECK:         omp.atomic.capture memory_order(relaxed) {
! CHECK:           omp.atomic.compare %[[XCONV]] : !fir.ref<i32> {
! CHECK:           ^bb0(%[[XVAL:.*]]: i32):
! CHECK:             %[[CMP:.*]] = arith.cmpi eq, %[[XVAL]], %[[ECONV]] : i32
! CHECK:             %[[SEL:.*]] = arith.select %[[CMP]], %[[DCONV]], %[[XVAL]] : i32
! CHECK:             omp.yield(%[[SEL]] : i32)
! CHECK:           }
! CHECK:           omp.atomic.read %{{.*}} = %[[XCONV]] : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:         }
subroutine atomic_compare_capture_logical(x, e, d, v)
  logical :: x, e, d, v
  !$omp atomic compare capture
    if (x .eqv. e) x = d
    v = x
  !$omp end atomic
end
