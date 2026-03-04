! REQUIRES: openmp_runtime

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
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: i32):
! CHECK:           %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[XVAL]], %[[EVAL]] : i32
! CHECK:           %[[DVAL:.*]] = fir.load %[[D_DECL]]#0 : !fir.ref<i32>
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
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<f32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: f32):
! CHECK:           %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
! CHECK:           %[[CMP:.*]] = arith.cmpf oeq, %[[XVAL]], %[[EVAL]] fastmath<contract> : f32
! CHECK:           %[[DVAL:.*]] = fir.load %[[D_DECL]]#0 : !fir.ref<f32>
! CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[DVAL]], %[[XVAL]] : f32
! CHECK:           omp.yield(%[[SEL]] : f32)
! CHECK:         }
subroutine atomic_compare_float_eq(x, e, d)
  real :: x, e, d
  !$omp atomic compare
  if (x .eq. e) x = d
end

! CHECK-LABEL: func.func @_QPatomic_compare_int_lt(
! CHECK-SAME:    %[[X:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[E:.*]]: !fir.ref<i32> {fir.bindc_name = "e"})
! CHECK:         %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] {{.*}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {{.*}}
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: i32):
! CHECK:           %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:           %[[CMP:.*]] = arith.cmpi slt, %[[XVAL]], %[[EVAL]] : i32
! CHECK:           %[[EVAL2:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
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
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: i32):
! CHECK:           %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK:           %[[CMP:.*]] = arith.cmpi sgt, %[[XVAL]], %[[EVAL]] : i32
! CHECK:           %[[EVAL2:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
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
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<f32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: f32):
! CHECK:           %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
! CHECK:           %[[CMP:.*]] = arith.cmpf olt, %[[XVAL]], %[[EVAL]] fastmath<contract> : f32
! CHECK:           %[[EVAL2:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
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
! CHECK:         omp.atomic.compare memory_order(relaxed) %[[X_DECL]]#0 : !fir.ref<f32> {
! CHECK:         ^bb0(%[[XVAL:.*]]: f32):
! CHECK:           %[[EVAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
! CHECK:           %[[CMP:.*]] = arith.cmpf ogt, %[[XVAL]], %[[EVAL]] fastmath<contract> : f32
! CHECK:           %[[EVAL2:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<f32>
! CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[EVAL2]], %[[XVAL]] : f32
! CHECK:           omp.yield(%[[SEL]] : f32)
! CHECK:         }
subroutine atomic_compare_float_gt(x, e)
  real :: x, e
  !$omp atomic compare
  if (x .gt. e) x = e
end
