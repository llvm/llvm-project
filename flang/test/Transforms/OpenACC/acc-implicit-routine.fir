// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-routine)" -split-input-file -verify-diagnostics | FileCheck %s

// -----

// Defined procedure called from a compute region gets an implicit acc routine.
module {
  func.func @_QPcallee() {
    return
  }
  func.func @_QPtest_defined_call() {
    acc.serial {
      fir.call @_QPcallee() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: acc.routine @acc_routine_0 func(@_QPcallee) implicit
// CHECK: func.func @_QPcallee() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
// CHECK: func.func @_QPtest_defined_call

// -----

// Nested fir.call from a compute region: both callees get implicit routines.
module {
  func.func @_QPleaf() {
    return
  }
  func.func @_QPmiddle() {
    fir.call @_QPleaf() : () -> ()
    return
  }
  func.func @_QPtest_nested_calls() {
    acc.serial {
      fir.call @_QPmiddle() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: acc.routine @acc_routine_1 func(@_QPleaf) implicit
// CHECK: func.func @_QPleaf() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_1]>}
// CHECK: acc.routine @acc_routine_0 func(@_QPmiddle) implicit
// CHECK: func.func @_QPmiddle() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
// CHECK: func.func @_QPtest_nested_calls

// -----

// Fortran intrinsic procedure declarations do not need an implicit routine.
module {
  func.func private @_QPsin(!fir.ref<f32>) -> f32 attributes {fir.proc_attrs = #fir.proc_attrs<intrinsic>}
  func.func @_QPtest_intrinsic(%arg0: !fir.ref<f32>) {
    acc.serial {
      %0 = fir.call @_QPsin(%arg0) : (!fir.ref<f32>) -> f32
      acc.yield
    }
    return
  }
}

// CHECK: func.func private @_QPsin
// CHECK: fir.call @_QPsin
// CHECK-NOT: acc.routine @{{.*}} func(@_QPsin)

// -----

// BIND(C) procedure declarations do not need an implicit routine.
module {
  func.func private @c_func() attributes {fir.bindc_name = "c_func", fir.proc_attrs = #fir.proc_attrs<bind_c>}
  func.func @_QPtest_bindc() {
    acc.serial {
      fir.call @c_func() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: func.func private @c_func
// CHECK: fir.call @c_func
// CHECK-NOT: acc.routine @{{.*}} func(@c_func)

// -----

// Fortran runtime helpers are already valid device symbols.
module {
  func.func private @_FortranASumInteger4(!fir.box<none>) -> i32 attributes {fir.runtime}
  func.func @_QPtest_runtime(%arg0: !fir.box<none>) {
    acc.serial {
      %0 = fir.call @_FortranASumInteger4(%arg0) : (!fir.box<none>) -> i32
      acc.yield
    }
    return
  }
}

// CHECK: func.func private @_FortranASumInteger4
// CHECK: fir.call @_FortranASumInteger4
// CHECK-NOT: acc.routine @{{.*}} func(@_FortranASumInteger4)

// -----

// An external procedure that already has an explicit acc routine is valid.
module {
  acc.routine @r_ext func(@_QPext_with_routine) seq
  func.func private @_QPext_with_routine() attributes {acc.routine_info = #acc.routine_info<[@r_ext]>}
  func.func @_QPtest_external_with_explicit_routine() {
    acc.serial {
      fir.call @_QPext_with_routine() : () -> ()
      acc.yield
    }
    return
  }
}

// CHECK: acc.routine @r_ext func(@_QPext_with_routine) seq
// CHECK: func.func private @_QPext_with_routine
// CHECK-NOT: acc.routine @{{.*}} func(@_QPext_with_routine) implicit

// -----

// An external Fortran procedure called from a compute region cannot receive an
// implicit routine; it must already have acc routine information.
module {
  func.func private @_QPext_callee()
  func.func @_QPtest_external_in_serial() {
    acc.serial {
      // expected-error @below {{Calls in an acc compute region must be marked with acc routine: "_QPext_callee"}}
      fir.call @_QPext_callee() : () -> ()
      acc.yield
    }
    return
  }
}

// -----

// Same requirement for an external procedure called from an existing routine.
module {
  acc.routine @r_caller func(@_QProutine_caller) seq
  func.func private @_QPext_from_routine()
  func.func @_QProutine_caller() attributes {acc.routine_info = #acc.routine_info<[@r_caller]>} {
    // expected-error @below {{Calls in acc routine must also be marked with acc routine: "_QPext_from_routine"}}
    fir.call @_QPext_from_routine() : () -> ()
    return
  }
}
