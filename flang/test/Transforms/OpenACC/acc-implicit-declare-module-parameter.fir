// An initialized linkonce_odr module PARAMETER (the lowering used for a
// separately compiled USE) must be implicitly declared copyin when its
// address is taken inside a compute region, and DEFAULT(PRESENT) must not
// emit acc.present for it. A bodyless external constant is still hoisted.
//
// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-declare,acc-implicit-data)" -split-input-file | FileCheck %s

// -----

fir.global linkonce_odr @_QMmod_paramsECarr_val(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf64>) {alignment = 64 : i64} constant : !fir.array<4xf64>

func.func @_QMkernel_modPdo_kernel(%arg0: !fir.ref<!fir.array<4xf64>> {fir.bindc_name = "x"}) {
  acc.parallel {
    %addr = fir.address_of(@_QMmod_paramsECarr_val) : !fir.ref<!fir.array<4xf64>>
    %0 = fir.load %addr : !fir.ref<!fir.array<4xf64>>
    acc.yield
  } defaultAttr(present)
  return
}

// CHECK: fir.global linkonce_odr @_QMmod_paramsECarr_val({{.*}}) {{{.*}}acc.declare = #acc.declare<dataClause = acc_copyin>{{.*}}} {{.*}}constant : !fir.array<4xf64>
// CHECK-LABEL: func.func @_QMkernel_modPdo_kernel
// CHECK: acc.parallel
// CHECK: fir.address_of(@_QMmod_paramsECarr_val)
// CHECK-NOT: acc.present{{.*}}@_QMmod_paramsECarr_val
// CHECK-NOT: acc.present{{.*}}name("arr_val")

// -----

fir.global @_QMmod_paramsECarr_val_ext constant : !fir.array<4xf64>

func.func @_QMkernel_modPdo_kernel_ext(%arg0: !fir.ref<!fir.array<4xf64>> {fir.bindc_name = "x"}) {
  acc.parallel {
    %addr = fir.address_of(@_QMmod_paramsECarr_val_ext) : !fir.ref<!fir.array<4xf64>>
    %0 = fir.load %addr : !fir.ref<!fir.array<4xf64>>
    acc.yield
  } defaultAttr(present)
  return
}

// CHECK: fir.global @_QMmod_paramsECarr_val_ext constant : !fir.array<4xf64>
// CHECK-NOT: acc.declare
// CHECK-LABEL: func.func @_QMkernel_modPdo_kernel_ext
// CHECK: fir.address_of(@_QMmod_paramsECarr_val_ext)
// CHECK: acc.present{{.*}}{acc.from_default}
// CHECK: acc.parallel
