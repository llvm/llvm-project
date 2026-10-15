// Given a declared (otherwise-undeclared) bind target, external-name-interop
// renames the func.func and the acc.routine bind reference together.

// RUN: fir-opt %s --external-name-interop | FileCheck %s

module {
  acc.routine @acc_routine_0 func(@_QPaclear) bind(@_QPaclear_seq) seq
  func.func private @_QPaclear(!fir.ref<f32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>}
  func.func private @_QPaclear_seq()
}

// CHECK: acc.routine @acc_routine_0 func(@aclear_) bind(@aclear_seq_) seq
// CHECK: func.func private @aclear_seq_() attributes {{{.*}}fir.internal_name = "_QPaclear_seq"{{.*}}}
