// RUN: fir-opt --split-input-file --cuf-device-global %s | FileCheck %s
// RUN: fir-opt --split-input-file --cuf-device-global="skip-dead-declares=false" %s | FileCheck --check-prefix=PRESERVE %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module} {
  fir.global @_QMiso_c_bindingECc_alert constant : !fir.char<1> {
    %0 = fir.string_lit "\07"(1) : !fir.char<1>
    fir.has_value %0 : !fir.char<1>
  }
  func.func @_QMrhsPkernel(%arg0: !fir.ref<f64>) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
    %c1 = arith.constant 1 : index
    %0 = fir.address_of(@_QMiso_c_bindingECc_alert) : !fir.ref<!fir.char<1>>
    %1 = fir.declare %0 typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_c_bindingECc_alert"} : (!fir.ref<!fir.char<1>>, index) -> !fir.ref<!fir.char<1>>
    return
  }
  gpu.module @cuda_device_mod {
  }
}

// With default skip-dead-declares=true, the global should NOT be in gpu.module.
// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK-NOT: fir.global @_QMiso_c_bindingECc_alert

// With skip-dead-declares=false (preserveDeclare mode), the global should be copied.
// PRESERVE: fir.global @_QMiso_c_bindingECc_alert
// PRESERVE-LABEL: gpu.module @cuda_device_mod
// PRESERVE: fir.global @_QMiso_c_bindingECc_alert
