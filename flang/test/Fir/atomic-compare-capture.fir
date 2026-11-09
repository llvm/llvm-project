// RUN: fir-opt --split-input-file --cfg-conversion --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s

// Test FIR-to-LLVM-dialect conversion of OpenMP atomic compare-capture: the
// !fir.ref operands become !llvm.ptr and the compare region's arith ops become
// their llvm-dialect equivalents, while the omp.atomic.capture / .compare /
// .read structure is preserved for the later MLIR-to-LLVM-IR translation.

//
// Integer postfix compare-capture: {compare, read}
func.func @cc_int_postfix(%x: !fir.ref<i32>, %e: !fir.ref<i32>, %d: !fir.ref<i32>, %v: !fir.ref<i32>) {
  %eval = fir.load %e : !fir.ref<i32>
  %dval = fir.load %d : !fir.ref<i32>
  omp.atomic.capture {
    omp.atomic.compare %x : !fir.ref<i32> {
    ^bb0(%xval: i32):
      %c = arith.cmpi eq, %xval, %eval : i32
      %s = arith.select %c, %dval, %xval : i32
      omp.yield(%s : i32)
    }
    omp.atomic.read %v = %x : !fir.ref<i32>, !fir.ref<i32>, i32
  }
  return
}

// CHECK-LABEL: llvm.func @cc_int_postfix(
// CHECK-SAME:  %[[X:.*]]: !llvm.ptr, %[[E:.*]]: !llvm.ptr, %[[D:.*]]: !llvm.ptr, %[[V:.*]]: !llvm.ptr) {
// CHECK:    %[[EVAL:.*]] = llvm.load %[[E]] : !llvm.ptr -> i32
// CHECK:    %[[DVAL:.*]] = llvm.load %[[D]] : !llvm.ptr -> i32
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.compare %[[X]] : !llvm.ptr {
// CHECK:      ^bb0(%[[XVAL:.*]]: i32):
// CHECK:        %[[CMP:.*]] = llvm.icmp "eq" %[[XVAL]], %[[EVAL]] : i32
// CHECK:        %[[SEL:.*]] = llvm.select %[[CMP]], %[[DVAL]], %[[XVAL]] : i1, i32
// CHECK:        omp.yield(%[[SEL]] : i32)
// CHECK:      }
// CHECK:      omp.atomic.read %[[V]] = %[[X]] : !llvm.ptr, !llvm.ptr, i32
// CHECK:    }

// -----

// Integer prefix compare-capture: {read, compare}
func.func @cc_int_prefix(%x: !fir.ref<i32>, %e: !fir.ref<i32>, %d: !fir.ref<i32>, %v: !fir.ref<i32>) {
  %eval = fir.load %e : !fir.ref<i32>
  %dval = fir.load %d : !fir.ref<i32>
  omp.atomic.capture {
    omp.atomic.read %v = %x : !fir.ref<i32>, !fir.ref<i32>, i32
    omp.atomic.compare %x : !fir.ref<i32> {
    ^bb0(%xval: i32):
      %c = arith.cmpi eq, %xval, %eval : i32
      %s = arith.select %c, %dval, %xval : i32
      omp.yield(%s : i32)
    }
  }
  return
}

// CHECK-LABEL: llvm.func @cc_int_prefix(
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.read %{{.*}} = %{{.*}} : !llvm.ptr, !llvm.ptr, i32
// CHECK:      omp.atomic.compare %{{.*}} : !llvm.ptr {
// CHECK:        llvm.icmp "eq"
// CHECK:        llvm.select
// CHECK:        omp.yield
// CHECK:      }
// CHECK:    }

// -----

// Integer fail-only compare-capture: {compare, read} with fail_only attribute.
func.func @cc_int_failonly(%x: !fir.ref<i32>, %e: !fir.ref<i32>, %d: !fir.ref<i32>, %v: !fir.ref<i32>) {
  %eval = fir.load %e : !fir.ref<i32>
  %dval = fir.load %d : !fir.ref<i32>
  omp.atomic.capture {
    omp.atomic.compare %x : !fir.ref<i32> {
    ^bb0(%xval: i32):
      %c = arith.cmpi eq, %xval, %eval : i32
      %s = arith.select %c, %dval, %xval : i32
      omp.yield(%s : i32)
    }
    omp.atomic.read %v = %x : !fir.ref<i32>, !fir.ref<i32>, i32
  } fail_only
  return
}

// CHECK-LABEL: llvm.func @cc_int_failonly(
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.compare %{{.*}} : !llvm.ptr {
// CHECK:        llvm.icmp "eq"
// CHECK:      }
// CHECK:      omp.atomic.read %{{.*}} = %{{.*}} : !llvm.ptr, !llvm.ptr, i32
// CHECK:    } fail_only

// -----

// Real postfix compare-capture: !fir.ref<f32> -> !llvm.ptr, arith.cmpf -> llvm.fcmp.
func.func @cc_real_postfix(%x: !fir.ref<f32>, %e: !fir.ref<f32>, %d: !fir.ref<f32>, %v: !fir.ref<f32>) {
  %eval = fir.load %e : !fir.ref<f32>
  %dval = fir.load %d : !fir.ref<f32>
  omp.atomic.capture {
    omp.atomic.compare %x : !fir.ref<f32> {
    ^bb0(%xval: f32):
      %c = arith.cmpf oeq, %xval, %eval : f32
      %s = arith.select %c, %dval, %xval : f32
      omp.yield(%s : f32)
    }
    omp.atomic.read %v = %x : !fir.ref<f32>, !fir.ref<f32>, f32
  }
  return
}

// CHECK-LABEL: llvm.func @cc_real_postfix(
// CHECK-SAME:  %[[X:.*]]: !llvm.ptr, %[[E:.*]]: !llvm.ptr, %[[D:.*]]: !llvm.ptr, %[[V:.*]]: !llvm.ptr) {
// CHECK:    %[[EVAL:.*]] = llvm.load %[[E]] : !llvm.ptr -> f32
// CHECK:    %[[DVAL:.*]] = llvm.load %[[D]] : !llvm.ptr -> f32
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.compare %[[X]] : !llvm.ptr {
// CHECK:      ^bb0(%[[XVAL:.*]]: f32):
// CHECK:        %[[CMP:.*]] = llvm.fcmp "oeq" %[[XVAL]], %[[EVAL]] : f32
// CHECK:        %[[SEL:.*]] = llvm.select %[[CMP]], %[[DVAL]], %[[XVAL]] : i1, f32
// CHECK:        omp.yield(%[[SEL]] : f32)
// CHECK:      }
// CHECK:      omp.atomic.read %[[V]] = %[[X]] : !llvm.ptr, !llvm.ptr, f32
// CHECK:    }

// -----

// Real prefix compare-capture.
func.func @cc_real_prefix(%x: !fir.ref<f32>, %e: !fir.ref<f32>, %d: !fir.ref<f32>, %v: !fir.ref<f32>) {
  %eval = fir.load %e : !fir.ref<f32>
  %dval = fir.load %d : !fir.ref<f32>
  omp.atomic.capture {
    omp.atomic.read %v = %x : !fir.ref<f32>, !fir.ref<f32>, f32
    omp.atomic.compare %x : !fir.ref<f32> {
    ^bb0(%xval: f32):
      %c = arith.cmpf oeq, %xval, %eval : f32
      %s = arith.select %c, %dval, %xval : f32
      omp.yield(%s : f32)
    }
  }
  return
}

// CHECK-LABEL: llvm.func @cc_real_prefix(
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.read %{{.*}} = %{{.*}} : !llvm.ptr, !llvm.ptr, f32
// CHECK:      omp.atomic.compare %{{.*}} : !llvm.ptr {
// CHECK:        llvm.fcmp "oeq"
// CHECK:      }
// CHECK:    }

// -----

// Real fail-only compare-capture.
func.func @cc_real_failonly(%x: !fir.ref<f32>, %e: !fir.ref<f32>, %d: !fir.ref<f32>, %v: !fir.ref<f32>) {
  %eval = fir.load %e : !fir.ref<f32>
  %dval = fir.load %d : !fir.ref<f32>
  omp.atomic.capture {
    omp.atomic.compare %x : !fir.ref<f32> {
    ^bb0(%xval: f32):
      %c = arith.cmpf oeq, %xval, %eval : f32
      %s = arith.select %c, %dval, %xval : f32
      omp.yield(%s : f32)
    }
    omp.atomic.read %v = %x : !fir.ref<f32>, !fir.ref<f32>, f32
  } fail_only
  return
}

// CHECK-LABEL: llvm.func @cc_real_failonly(
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.compare %{{.*}} : !llvm.ptr {
// CHECK:        llvm.fcmp "oeq"
// CHECK:      }
// CHECK:      omp.atomic.read %{{.*}} = %{{.*}} : !llvm.ptr, !llvm.ptr, f32
// CHECK:    } fail_only

// -----

// Complex postfix compare-capture: !fir.ref<complex<f32>> -> !llvm.ptr and the
// fir.cmpc is decomposed into per-field llvm.fcmp + llvm.and over the struct.
func.func @cc_complex_postfix(%x: !fir.ref<complex<f32>>, %e: !fir.ref<complex<f32>>, %d: !fir.ref<complex<f32>>, %v: !fir.ref<complex<f32>>) {
  %eval = fir.load %e : !fir.ref<complex<f32>>
  %dval = fir.load %d : !fir.ref<complex<f32>>
  omp.atomic.capture {
    omp.atomic.compare %x : !fir.ref<complex<f32>> {
    ^bb0(%xval: complex<f32>):
      %c = fir.cmpc "oeq", %xval, %eval : complex<f32>
      %s = arith.select %c, %dval, %xval : complex<f32>
      omp.yield(%s : complex<f32>)
    }
    omp.atomic.read %v = %x : !fir.ref<complex<f32>>, !fir.ref<complex<f32>>, complex<f32>
  }
  return
}

// CHECK-LABEL: llvm.func @cc_complex_postfix(
// CHECK-SAME:  %[[X:.*]]: !llvm.ptr, %[[E:.*]]: !llvm.ptr, %[[D:.*]]: !llvm.ptr, %[[V:.*]]: !llvm.ptr) {
// CHECK:    %[[EVAL:.*]] = llvm.load %[[E]] : !llvm.ptr -> !llvm.struct<(f32, f32)>
// CHECK:    %[[DVAL:.*]] = llvm.load %[[D]] : !llvm.ptr -> !llvm.struct<(f32, f32)>
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.compare %[[X]] : !llvm.ptr {
// CHECK:      ^bb0(%[[XVAL:.*]]: !llvm.struct<(f32, f32)>):
// CHECK:        llvm.extractvalue %[[XVAL]][0]
// CHECK:        llvm.fcmp "oeq"
// CHECK:        llvm.extractvalue %[[XVAL]][1]
// CHECK:        llvm.fcmp "oeq"
// CHECK:        %[[AND:.*]] = llvm.and
// CHECK:        %[[SEL:.*]] = llvm.select %[[AND]], %[[DVAL]], %[[XVAL]] : i1, !llvm.struct<(f32, f32)>
// CHECK:        omp.yield(%[[SEL]] : !llvm.struct<(f32, f32)>)
// CHECK:      }
// CHECK:      omp.atomic.read %[[V]] = %[[X]] : !llvm.ptr, !llvm.ptr, !llvm.struct<(f32, f32)>
// CHECK:    }

// -----

// Complex prefix compare-capture.
func.func @cc_complex_prefix(%x: !fir.ref<complex<f32>>, %e: !fir.ref<complex<f32>>, %d: !fir.ref<complex<f32>>, %v: !fir.ref<complex<f32>>) {
  %eval = fir.load %e : !fir.ref<complex<f32>>
  %dval = fir.load %d : !fir.ref<complex<f32>>
  omp.atomic.capture {
    omp.atomic.read %v = %x : !fir.ref<complex<f32>>, !fir.ref<complex<f32>>, complex<f32>
    omp.atomic.compare %x : !fir.ref<complex<f32>> {
    ^bb0(%xval: complex<f32>):
      %c = fir.cmpc "oeq", %xval, %eval : complex<f32>
      %s = arith.select %c, %dval, %xval : complex<f32>
      omp.yield(%s : complex<f32>)
    }
  }
  return
}

// CHECK-LABEL: llvm.func @cc_complex_prefix(
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.read %{{.*}} = %{{.*}} : !llvm.ptr, !llvm.ptr, !llvm.struct<(f32, f32)>
// CHECK:      omp.atomic.compare %{{.*}} : !llvm.ptr {
// CHECK:        llvm.fcmp "oeq"
// CHECK:        llvm.fcmp "oeq"
// CHECK:        llvm.and
// CHECK:      }
// CHECK:    }

// -----

// Complex fail-only compare-capture.
func.func @cc_complex_failonly(%x: !fir.ref<complex<f32>>, %e: !fir.ref<complex<f32>>, %d: !fir.ref<complex<f32>>, %v: !fir.ref<complex<f32>>) {
  %eval = fir.load %e : !fir.ref<complex<f32>>
  %dval = fir.load %d : !fir.ref<complex<f32>>
  omp.atomic.capture {
    omp.atomic.compare %x : !fir.ref<complex<f32>> {
    ^bb0(%xval: complex<f32>):
      %c = fir.cmpc "oeq", %xval, %eval : complex<f32>
      %s = arith.select %c, %dval, %xval : complex<f32>
      omp.yield(%s : complex<f32>)
    }
    omp.atomic.read %v = %x : !fir.ref<complex<f32>>, !fir.ref<complex<f32>>, complex<f32>
  } fail_only
  return
}

// CHECK-LABEL: llvm.func @cc_complex_failonly(
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.compare %{{.*}} : !llvm.ptr {
// CHECK:        llvm.fcmp "oeq"
// CHECK:        llvm.and
// CHECK:      }
// CHECK:      omp.atomic.read %{{.*}} = %{{.*}} : !llvm.ptr, !llvm.ptr, !llvm.struct<(f32, f32)>
// CHECK:    } fail_only

// -----

// Integer min compare-capture (x > e): the compare region uses arith.cmpi sgt.
func.func @cc_int_min(%x: !fir.ref<i32>, %e: !fir.ref<i32>, %v: !fir.ref<i32>) {
  %eval = fir.load %e : !fir.ref<i32>
  omp.atomic.capture {
    omp.atomic.compare %x : !fir.ref<i32> {
    ^bb0(%xval: i32):
      %c = arith.cmpi sgt, %xval, %eval : i32
      %s = arith.select %c, %eval, %xval : i32
      omp.yield(%s : i32)
    }
    omp.atomic.read %v = %x : !fir.ref<i32>, !fir.ref<i32>, i32
  }
  return
}

// CHECK-LABEL: llvm.func @cc_int_min(
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.compare %{{.*}} : !llvm.ptr {
// CHECK:        llvm.icmp "sgt"
// CHECK:      }
// CHECK:      omp.atomic.read %{{.*}} = %{{.*}} : !llvm.ptr, !llvm.ptr, i32
// CHECK:    }

// -----

// Real min compare-capture (x > e): the compare region uses arith.cmpf ogt.
func.func @cc_real_min(%x: !fir.ref<f32>, %e: !fir.ref<f32>, %v: !fir.ref<f32>) {
  %eval = fir.load %e : !fir.ref<f32>
  omp.atomic.capture {
    omp.atomic.compare %x : !fir.ref<f32> {
    ^bb0(%xval: f32):
      %c = arith.cmpf ogt, %xval, %eval : f32
      %s = arith.select %c, %eval, %xval : f32
      omp.yield(%s : f32)
    }
    omp.atomic.read %v = %x : !fir.ref<f32>, !fir.ref<f32>, f32
  }
  return
}

// CHECK-LABEL: llvm.func @cc_real_min(
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.compare %{{.*}} : !llvm.ptr {
// CHECK:        llvm.fcmp "ogt"
// CHECK:      }
// CHECK:      omp.atomic.read %{{.*}} = %{{.*}} : !llvm.ptr, !llvm.ptr, f32
// CHECK:    }

// -----

// Weak clause: preserved through FIR-to-LLVM conversion.
func.func @cc_weak(%x: !fir.ref<i32>, %e: !fir.ref<i32>, %d: !fir.ref<i32>, %v: !fir.ref<i32>) {
  %eval = fir.load %e : !fir.ref<i32>
  %dval = fir.load %d : !fir.ref<i32>
  omp.atomic.capture {
    omp.atomic.read %v = %x : !fir.ref<i32>, !fir.ref<i32>, i32
    omp.atomic.compare %x : !fir.ref<i32> {
    ^bb0(%xval: i32):
      %c = arith.cmpi eq, %xval, %eval : i32
      %s = arith.select %c, %dval, %xval : i32
      omp.yield(%s : i32)
    } weak
  }
  return
}

// CHECK-LABEL: llvm.func @cc_weak(
// CHECK:    omp.atomic.capture {
// CHECK:      omp.atomic.read %{{.*}} = %{{.*}} : !llvm.ptr, !llvm.ptr, i32
// CHECK:      omp.atomic.compare %{{.*}} : !llvm.ptr {
// CHECK:        llvm.icmp "eq"
// CHECK:      } weak
// CHECK:    }
