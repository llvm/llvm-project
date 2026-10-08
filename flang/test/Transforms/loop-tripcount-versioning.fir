// RUN: fir-opt --loop-tripcount-versioning %s | FileCheck %s
// Every versioned loop below has a declared extent of 4: a bound of 4 must
// accept it and a bound of 3 must not. A bound of 0 disables the pass.
// RUN: fir-opt --loop-tripcount-versioning="max-section-extent=4" %s | FileCheck %s
// RUN: fir-opt --loop-tripcount-versioning="max-section-extent=3" %s | FileCheck %s --check-prefix=OFF
// RUN: fir-opt --loop-tripcount-versioning="max-section-extent=0" %s | FileCheck %s --check-prefix=OFF

// Array sections with a runtime extent on an explicit-shape array. The pass
// versions the innermost loop on `extent == declared extent` so that one arm
// has a constant trip count.

// subroutine zero(tau, ncol)
//   real(8), intent(out) :: tau(4, 26)
//   integer, intent(in) :: ncol
//   tau(1:ncol, 1:26) = 0.0d0
func.func @zero(%arg0: !fir.ref<!fir.array<4x26xf64>>, %arg1: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4, %c26 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%0) uniq_name("tau") : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x26xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.maxsi %3, %c0 : index
  %5 = fir.slice %c1, %3, %c1, %c1, %c26, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %arg2 = %c1 to %c26 step %c1 unordered {
    fir.do_loop %arg3 = %c1 to %4 step %c1 unordered {
      %6 = fir.array_coor %1(%0) [%5] %arg3, %arg2 : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      fir.store %cst to %6 : !fir.ref<f64>
    }
  }
  return
}
// CHECK-LABEL: func.func @zero(
// CHECK:         %[[UB:.*]] = arith.maxsi %{{.*}}, %{{.*}} : index
// CHECK-NOT:     fir.if
// CHECK:         fir.do_loop %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[UB]], %[[C4]] : index
// CHECK:           fir.if %[[CMP]] {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[C4]] step %{{.*}} unordered {
// CHECK:               fir.array_coor
// CHECK:               fir.store
// CHECK:             }
// CHECK:           } else {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[UB]] step %{{.*}} unordered {
// CHECK:               fir.array_coor
// CHECK:               fir.store
// CHECK:             }
// CHECK:           }
// CHECK:         }

// OFF-LABEL: func.func @zero(
// OFF-NOT:     fir.if
// OFF:         return

// An unsliced access: the induction variable indexes the array directly, so
// there is no triplet to check. (The assignment conforms only when n == 4,
// which is exactly the case the fast arm is for.)
// subroutine unsliced(a, b, n)
//   real(8) :: a(4, 26), b(8, 26)
//   a = b(1:n, :)
func.func @unsliced(%arg0: !fir.ref<!fir.array<4x26xf64>>, %arg1: !fir.ref<!fir.array<8x26xf64>>, %arg2: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %0 = fir.shape %c4, %c26 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%0) uniq_name("a") : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x26xf64>>
  %2 = fir.shape %c8, %c26 : (index, index) -> !fir.shape<2>
  %3 = fir.declare %arg1(%2) uniq_name("b") : (!fir.ref<!fir.array<8x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<8x26xf64>>
  %4 = fir.load %arg2 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.maxsi %5, %c0 : index
  %7 = fir.slice %c1, %5, %c1, %c1, %c26, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %arg3 = %c1 to %c26 step %c1 unordered {
    fir.do_loop %arg4 = %c1 to %6 step %c1 unordered {
      %8 = fir.array_coor %3(%2) [%7] %arg4, %arg3 : (!fir.ref<!fir.array<8x26xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      %9 = fir.load %8 : !fir.ref<f64>
      %10 = fir.array_coor %1(%0) %arg4, %arg3 : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>, index, index) -> !fir.ref<f64>
      fir.store %9 to %10 : !fir.ref<f64>
    }
  }
  return
}
// CHECK-LABEL: func.func @unsliced(
// CHECK:         %[[UB:.*]] = arith.maxsi %{{.*}}, %{{.*}} : index
// CHECK:         fir.do_loop
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[UB]], %[[C4]] : index
// CHECK:           fir.if %[[CMP]] {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[C4]] step
// CHECK:           } else {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[UB]] step
// CHECK:           }

// A constant section bound already gives LLVM a constant trip count; there is
// nothing to version on.
// subroutine constant_ub(tau)
//   real(8) :: tau(4, 26)
//   tau(1:4, :) = 0.0d0
func.func @constant_ub(%arg0: !fir.ref<!fir.array<4x26xf64>>) {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4, %c26 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%0) uniq_name("tau") : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x26xf64>>
  %2 = fir.slice %c1, %c4, %c1, %c1, %c26, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %arg1 = %c1 to %c26 step %c1 unordered {
    fir.do_loop %arg2 = %c1 to %c4 step %c1 unordered {
      %3 = fir.array_coor %1(%0) [%2] %arg2, %arg1 : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      fir.store %cst to %3 : !fir.ref<f64>
    }
  }
  return
}
// CHECK-LABEL: func.func @constant_ub(
// CHECK-NOT:     fir.if
// CHECK:         return

// Declared extent 1 is not versioned on: legacy code declares dummies as
// `a(1)` and indexes past them, so `n == 1` would almost never hold.
// subroutine extent_one(a, n)
//   real(8) :: a(1)
//   a(1:n) = 0.0d0
func.func @extent_one(%arg0: !fir.ref<!fir.array<1xf64>>, %arg1: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %0 = fir.shape %c1 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%0) uniq_name("a") : (!fir.ref<!fir.array<1xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.maxsi %3, %c0 : index
  %5 = fir.slice %c1, %3, %c1 : (index, index, index) -> !fir.slice<1>
  fir.do_loop %arg2 = %c1 to %4 step %c1 unordered {
    %6 = fir.array_coor %1(%0) [%5] %arg2 : (!fir.ref<!fir.array<1xf64>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f64>
    fir.store %cst to %6 : !fir.ref<f64>
  }
  return
}
// CHECK-LABEL: func.func @extent_one(
// CHECK-NOT:     fir.if
// CHECK:         return

// A section that does not start at 1. Hand-written so that the loop and the
// slice share the same bound; only the slice lower bound differs from the
// accepted form, so this is the check being exercised.
func.func @slice_lb(%arg0: !fir.ref<!fir.array<4x26xf64>>, %arg1: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4, %c26 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%0) uniq_name("tau") : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x26xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.maxsi %3, %c0 : index
  %5 = fir.slice %c2, %3, %c1, %c1, %c26, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %arg2 = %c1 to %c26 step %c1 unordered {
    fir.do_loop %arg3 = %c1 to %4 step %c1 unordered {
      %6 = fir.array_coor %1(%0) [%5] %arg3, %arg2 : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      fir.store %cst to %6 : !fir.ref<f64>
    }
  }
  return
}
// CHECK-LABEL: func.func @slice_lb(
// CHECK-NOT:     fir.if
// CHECK:         return

// A strided section. Hand-written as above; only the slice stride differs
// from the accepted form.
func.func @slice_stride(%arg0: !fir.ref<!fir.array<4x26xf64>>, %arg1: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4, %c26 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%0) uniq_name("tau") : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x26xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.maxsi %3, %c0 : index
  %5 = fir.slice %c1, %3, %c2, %c1, %c26, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %arg2 = %c1 to %c26 step %c1 unordered {
    fir.do_loop %arg3 = %c1 to %4 step %c1 unordered {
      %6 = fir.array_coor %1(%0) [%5] %arg3, %arg2 : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      fir.store %cst to %6 : !fir.ref<f64>
    }
  }
  return
}
// CHECK-LABEL: func.func @slice_stride(
// CHECK-NOT:     fir.if
// CHECK:         return

// The two sides of an assignment have different section bounds. The loop runs
// over the left-hand side's bound, so only arrays whose slice shares that bound
// take part: `b` is left out and the loop is versioned on 4, not 2. (The
// program conforms only when n == m.)
// subroutine slice_ub_mismatch(a, b, n, m)
//   real(8) :: a(4, 26), b(2, 26)
//   a(1:n, :) = b(1:m, :)
func.func @slice_ub_mismatch(%arg0: !fir.ref<!fir.array<4x26xf64>>, %arg1: !fir.ref<!fir.array<2x26xf64>>, %arg2: !fir.ref<i32>, %arg3: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4, %c26 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%0) uniq_name("a") : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x26xf64>>
  %2 = fir.shape %c2, %c26 : (index, index) -> !fir.shape<2>
  %3 = fir.declare %arg1(%2) uniq_name("b") : (!fir.ref<!fir.array<2x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<2x26xf64>>
  %4 = fir.load %arg2 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.maxsi %5, %c0 : index
  %7 = fir.slice %c1, %5, %c1, %c1, %c26, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  %8 = fir.load %arg3 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = fir.slice %c1, %9, %c1, %c1, %c26, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %arg4 = %c1 to %c26 step %c1 unordered {
    fir.do_loop %arg5 = %c1 to %6 step %c1 unordered {
      %11 = fir.array_coor %3(%2) [%10] %arg5, %arg4 : (!fir.ref<!fir.array<2x26xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      %12 = fir.load %11 : !fir.ref<f64>
      %13 = fir.array_coor %1(%0) [%7] %arg5, %arg4 : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      fir.store %12 to %13 : !fir.ref<f64>
    }
  }
  return
}
// CHECK-LABEL: func.func @slice_ub_mismatch(
// CHECK:         %[[UB:.*]] = arith.maxsi %{{.*}}, %{{.*}} : index
// CHECK:         fir.do_loop
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[UB]], %[[C4]] : index
// CHECK:           fir.if %[[CMP]] {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[C4]] step
// CHECK:           } else {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[UB]] step
// CHECK:           }

// The loop bound is not clamped with arith.maxsi; the slice bound is compared
// against the loop bound directly.
func.func @unclamped(%arg0: !fir.ref<!fir.array<4x26xf64>>, %arg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4, %c26 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%0) uniq_name("tau") : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x26xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = fir.slice %c1, %3, %c1, %c1, %c26, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %arg2 = %c1 to %c26 step %c1 unordered {
    fir.do_loop %arg3 = %c1 to %3 step %c1 unordered {
      %5 = fir.array_coor %1(%0) [%4] %arg3, %arg2 : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      fir.store %cst to %5 : !fir.ref<f64>
    }
  }
  return
}
// CHECK-LABEL: func.func @unclamped(
// CHECK:         %[[UB:.*]] = fir.convert %{{.*}} : (i32) -> index
// CHECK:         fir.do_loop
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[UB]], %[[C4]] : index
// CHECK:           fir.if %[[CMP]] {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[C4]] step
// CHECK:           } else {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[UB]] step
// CHECK:           }

// The loop does not start at 1, so `ub == 4` does not mean four iterations
// over the whole dimension.
// subroutine loop_lb(a, n)
//   real(8) :: a(4)
//   do i = 2, n
//     a(i) = 0.0d0
//   end do
func.func @loop_lb(%arg0: !fir.ref<!fir.array<4xf64>>, %arg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%0) uniq_name("a") : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  fir.do_loop %arg2 = %c2 to %3 step %c1 {
    %4 = fir.array_coor %1(%0) %arg2 : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
    fir.store %cst to %4 : !fir.ref<f64>
  }
  return
}
// CHECK-LABEL: func.func @loop_lb(
// CHECK-NOT:     fir.if
// CHECK:         return

// Non-unit declared lower bound: the slice triplet is the accepted
// (1, ncol, 1), but position i of the section is element i of a dimension
// that starts at 0, so `ncol == 4` would not mean the whole dimension.
// subroutine nonunit_lb(tau, ncol)
//   real(8), intent(out) :: tau(0:3, 26)
//   tau(1:ncol, :) = 0.0d0
func.func @nonunit_lb(%arg0: !fir.ref<!fir.array<4x26xf64>>, %arg1: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %0 = fir.shape_shift %c0, %c4, %c1, %c26 : (index, index, index, index) -> !fir.shapeshift<2>
  %1 = fir.declare %arg0(%0) uniq_name("tau") : (!fir.ref<!fir.array<4x26xf64>>, !fir.shapeshift<2>) -> !fir.ref<!fir.array<4x26xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.maxsi %3, %c0 : index
  %5 = fir.slice %c1, %3, %c1, %c1, %c26, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %arg2 = %c1 to %c26 step %c1 unordered {
    fir.do_loop %arg3 = %c1 to %4 step %c1 unordered {
      %6 = fir.array_coor %1(%0) [%5] %arg3, %arg2 : (!fir.ref<!fir.array<4x26xf64>>, !fir.shapeshift<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      fir.store %cst to %6 : !fir.ref<f64>
    }
  }
  return
}
// CHECK-LABEL: func.func @nonunit_lb(
// CHECK-NOT:     fir.if
// CHECK:         return

// Two arrays with different declared extents in the same dimension: version
// on the smaller one, the only extent the section bound can equal in a
// conforming program.
// subroutine mixed_extents(a, b, n)
//   real(8) :: a(4, 26), b(8, 26)
//   a(1:n, :) = b(1:n, :)
func.func @mixed_extents(%arg0: !fir.ref<!fir.array<4x26xf64>>, %arg1: !fir.ref<!fir.array<8x26xf64>>, %arg2: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %0 = fir.shape %c4, %c26 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%0) uniq_name("a") : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x26xf64>>
  %2 = fir.shape %c8, %c26 : (index, index) -> !fir.shape<2>
  %3 = fir.declare %arg1(%2) uniq_name("b") : (!fir.ref<!fir.array<8x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<8x26xf64>>
  %4 = fir.load %arg2 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.maxsi %5, %c0 : index
  %7 = fir.slice %c1, %5, %c1, %c1, %c26, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %arg3 = %c1 to %c26 step %c1 unordered {
    fir.do_loop %arg4 = %c1 to %6 step %c1 unordered {
      %8 = fir.array_coor %3(%2) [%7] %arg4, %arg3 : (!fir.ref<!fir.array<8x26xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      %9 = fir.load %8 : !fir.ref<f64>
      %10 = fir.array_coor %1(%0) [%7] %arg4, %arg3 : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      fir.store %9 to %10 : !fir.ref<f64>
    }
  }
  return
}
// CHECK-LABEL: func.func @mixed_extents(
// CHECK:         %[[UB:.*]] = arith.maxsi %{{.*}}, %{{.*}} : index
// CHECK:         fir.do_loop
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[UB]], %[[C4]] : index
// CHECK:           fir.if %[[CMP]] {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[C4]] step
// CHECK:           } else {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[UB]] step
// CHECK:           }

// The loop variable is not the leading subscript: only the leading dimension
// is versioned.
// subroutine nonleading(a, k, n)
//   real(8) :: a(26, 4)
//   a(k, 1:n) = 0.0d0
func.func @nonleading(%arg0: !fir.ref<!fir.array<26x4xf64>>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c26, %c4 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%0) uniq_name("a") : (!fir.ref<!fir.array<26x4xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<26x4xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = fir.load %arg2 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.maxsi %5, %c0 : index
  %7 = fir.slice %3, %3, %c1, %c1, %5, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %arg3 = %c1 to %6 step %c1 unordered {
    %8 = fir.array_coor %1(%0) [%7] %c1, %arg3 : (!fir.ref<!fir.array<26x4xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
    fir.store %cst to %8 : !fir.ref<f64>
  }
  return
}
// CHECK-LABEL: func.func @nonleading(
// CHECK-NOT:     fir.if
// CHECK:         return

// A section that reaches the loop through a temporary box (fir.box_addr of a
// fir.embox) is not recognized: the declared shape is read from the memref
// type only, and here that type has an unknown extent.
// subroutine boxed(fsds, ncol)
//   real(8) :: fsds(4)
//   fsds(1:ncol) = 0.0d0
func.func @boxed(%arg0: !fir.ref<!fir.array<4xf64>>, %arg1: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%0) uniq_name("fsds") : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.maxsi %3, %c0 : index
  %5 = fir.slice %c1, %3, %c1 : (index, index, index) -> !fir.slice<1>
  %6 = fir.embox %1(%0) [%5] : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf64>>
  %7 = fir.box_addr %6 : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
  %8 = fir.shape %4 : (index) -> !fir.shape<1>
  fir.do_loop %arg2 = %c1 to %4 step %c1 unordered {
    %9 = fir.array_coor %7(%8) %arg2 : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
    fir.store %cst to %9 : !fir.ref<f64>
  }
  return
}
// CHECK-LABEL: func.func @boxed(
// CHECK-NOT:     fir.if
// CHECK:         return

// A loop whose body contains an OpenMP region is not versioned: the region is
// outlined, so a constant trip count cannot help and cloning would only
// duplicate the outlined code.
func.func @contains_omp(%arg0: !fir.ref<!fir.array<4xf64>>, %arg1: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%0) uniq_name("a") : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.maxsi %3, %c0 : index
  %5 = fir.slice %c1, %3, %c1 : (index, index, index) -> !fir.slice<1>
  fir.do_loop %arg2 = %c1 to %4 step %c1 unordered {
    %6 = fir.array_coor %1(%0) [%5] %arg2 : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f64>
    omp.parallel {
      fir.store %cst to %6 : !fir.ref<f64>
      omp.terminator
    }
  }
  return
}
// CHECK-LABEL: func.func @contains_omp(
// CHECK-NOT:     fir.if
// CHECK:         omp.parallel
// CHECK:         return

// A loop containing a loop that is not a fir.do_loop is not innermost and is
// not versioned.
func.func @contains_iterate_while(%arg0: !fir.ref<!fir.array<4xf64>>, %arg1: !fir.ref<i32>, %arg2: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%0) uniq_name("a") : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.maxsi %3, %c0 : index
  %5 = fir.slice %c1, %3, %c1 : (index, index, index) -> !fir.slice<1>
  fir.do_loop %arg3 = %c1 to %4 step %c1 unordered {
    %6 = fir.array_coor %1(%0) [%5] %arg3 : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f64>
    %7 = fir.iterate_while (%arg4 = %c1 to %c4 step %c1) and (%arg5 = %arg2) {
      fir.store %cst to %6 : !fir.ref<f64>
      fir.result %arg5 : i1
    }
  }
  return
}
// CHECK-LABEL: func.func @contains_iterate_while(
// CHECK-NOT:     fir.if
// CHECK:         fir.iterate_while
// CHECK:         return

// An explicit DO nest in the opposite order to array-section lowering: the
// outer loop variable is the leading subscript of a 4-extent array, but the
// loop contains another loop, so it is not versioned; the inner loop indexes
// the second dimension, so neither is it.
// subroutine outer_qualifies(tau, n)
//   real(8) :: tau(4, 26)
//   do i = 1, n
//     do j = 1, 26
//       tau(i, j) = 0.0d0
//     end do
//   end do
func.func @outer_qualifies(%arg0: !fir.ref<!fir.array<4x26xf64>>, %arg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4, %c26 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%0) uniq_name("tau") : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x26xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  fir.do_loop %arg2 = %c1 to %3 step %c1 {
    fir.do_loop %arg3 = %c1 to %c26 step %c1 {
      %4 = fir.array_coor %1(%0) %arg2, %arg3 : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>, index, index) -> !fir.ref<f64>
      fir.store %cst to %4 : !fir.ref<f64>
    }
  }
  return
}
// CHECK-LABEL: func.func @outer_qualifies(
// CHECK-NOT:     fir.if
// CHECK:         return

// Two candidates in one function: both are versioned.
// subroutine two_loops(a, b, n)
//   real(8) :: a(4), b(4)
//   a(1:n) = 0.0d0
//   b(1:n) = 1.0d0
func.func @two_loops(%arg0: !fir.ref<!fir.array<4xf64>>, %arg1: !fir.ref<!fir.array<4xf64>>, %arg2: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst0 = arith.constant 0.000000e+00 : f64
  %cst1 = arith.constant 1.000000e+00 : f64
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%0) uniq_name("a") : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
  %2 = fir.declare %arg1(%0) uniq_name("b") : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
  %3 = fir.load %arg2 : !fir.ref<i32>
  %4 = fir.convert %3 : (i32) -> index
  %5 = arith.maxsi %4, %c0 : index
  %6 = fir.slice %c1, %4, %c1 : (index, index, index) -> !fir.slice<1>
  fir.do_loop %arg3 = %c1 to %5 step %c1 unordered {
    %7 = fir.array_coor %1(%0) [%6] %arg3 : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f64>
    fir.store %cst0 to %7 : !fir.ref<f64>
  }
  fir.do_loop %arg3 = %c1 to %5 step %c1 unordered {
    %7 = fir.array_coor %2(%0) [%6] %arg3 : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f64>
    fir.store %cst1 to %7 : !fir.ref<f64>
  }
  return
}
// CHECK-LABEL: func.func @two_loops(
// CHECK:         %[[UB:.*]] = arith.maxsi %{{.*}}, %{{.*}} : index
// CHECK:         %[[C4A:.*]] = arith.constant 4 : index
// CHECK:         %[[CMPA:.*]] = arith.cmpi eq, %[[UB]], %[[C4A]] : index
// CHECK:         fir.if %[[CMPA]] {
// CHECK:           fir.do_loop %{{.*}} = %{{.*}} to %[[C4A]] step
// CHECK:         } else {
// CHECK:           fir.do_loop %{{.*}} = %{{.*}} to %[[UB]] step
// CHECK:         }
// CHECK:         %[[C4B:.*]] = arith.constant 4 : index
// CHECK:         %[[CMPB:.*]] = arith.cmpi eq, %[[UB]], %[[C4B]] : index
// CHECK:         fir.if %[[CMPB]] {
// CHECK:           fir.do_loop %{{.*}} = %{{.*}} to %[[C4B]] step
// CHECK:         } else {
// CHECK:           fir.do_loop %{{.*}} = %{{.*}} to %[[UB]] step
// CHECK:         }
// CHECK:         return

// A loop carrying results: fir.result must forward them through both arms.
// Hand-written; array-section lowering does not produce iter_args, but the
// rewrite must handle them.
func.func @with_results(%arg0: !fir.ref<!fir.array<4xf64>>, %arg1: !fir.ref<i32>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%0) uniq_name("a") : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.maxsi %3, %c0 : index
  %5 = fir.slice %c1, %3, %c1 : (index, index, index) -> !fir.slice<1>
  %6 = fir.do_loop %arg2 = %c1 to %4 step %c1 iter_args(%arg3 = %cst) -> (f64) {
    %7 = fir.array_coor %1(%0) [%5] %arg2 : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f64>
    %8 = fir.load %7 : !fir.ref<f64>
    %9 = arith.addf %arg3, %8 : f64
    fir.result %9 : f64
  }
  return %6 : f64
}
// CHECK-LABEL: func.func @with_results(
// CHECK:         %[[UB:.*]] = arith.maxsi
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[CMP:.*]] = arith.cmpi eq, %[[UB]], %[[C4]] : index
// CHECK:         %[[RES:.*]] = fir.if %[[CMP]] -> (f64) {
// CHECK:           %[[FAST:.*]] = fir.do_loop {{.*}} to %[[C4]] step %{{.*}} iter_args({{.*}}) -> (f64) {
// CHECK:           fir.result %[[FAST]] : f64
// CHECK:         } else {
// CHECK:           %[[SLOW:.*]] = fir.do_loop {{.*}} to %[[UB]] step %{{.*}} iter_args({{.*}}) -> (f64) {
// CHECK:           fir.result %[[SLOW]] : f64
// CHECK:         }
// CHECK:         return %[[RES]] : f64
