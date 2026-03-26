// RUN: fir-opt --vscale-attr %s | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: fir-opt --vscale-attr="vscale-min=4" %s | FileCheck %s --check-prefix=CHECK-MIN
// RUN: fir-opt --vscale-attr="vscale-max=16" %s | FileCheck %s --check-prefix=CHECK-MAX
// RUN: fir-opt --vscale-attr="vscale-min=8 vscale-max=16" %s | FileCheck %s --check-prefix=CHECK-BOTH
// RUN: fir-opt --vscale-attr="vscale-min=8 vscale-max=8" %s | FileCheck %s --check-prefix=CHECK-EQUAL
// RUN: not fir-opt --vscale-attr="vscale-min=0 vscale-max=4" %s 2>&1 | FileCheck %s --check-prefix=VSCALE-MIN-0
// RUN: not fir-opt --vscale-attr="vscale-min=3 vscale-max=4" %s 2>&1 | FileCheck %s --check-prefix=VSCALE-MIN-NO-PO2
// RUN: not fir-opt --vscale-attr="vscale-max=6" %s 2>&1 | FileCheck %s --check-prefix=VSCALE-MAX-NO-PO2
// RUN: not fir-opt --vscale-attr="vscale-min=16 vscale-max=8" %s 2>&1 | FileCheck %s --check-prefix=VSCALE-MIN-GREATER


// CHECK-DEFAULT: attributes {vscale_range = #llvm.vscale_range<minRange = 1 : i32, maxRange = 0 : i32>}
// CHECK-MIN: attributes {vscale_range = #llvm.vscale_range<minRange = 4 : i32, maxRange = 0 : i32>}
// CHECK-MAX: attributes {vscale_range = #llvm.vscale_range<minRange = 1 : i32, maxRange = 16 : i32>}
// CHECK-BOTH: attributes {vscale_range = #llvm.vscale_range<minRange = 8 : i32, maxRange = 16 : i32>}
// CHECK-EQUAL: attributes {vscale_range = #llvm.vscale_range<minRange = 8 : i32, maxRange = 8 : i32>}

// VSCALE-MIN-0:      VScaleAttr: vscaleMin has to be a power-of-two greater than 0
// VSCALE-MIN-NO-PO2: VScaleAttr: vscaleMin has to be a power-of-two greater than 0
// VSCALE-MAX-NO-PO2: VScaleAttr: vscaleMax has to be a power-of-two greater-than-or-equal to vscaleMin or 0 to signify an unbounded maximum
// VSCALE-MIN-GREATER: VScaleAttr: vscaleMax has to be a power-of-two greater-than-or-equal to vscaleMin or 0 to signify an unbounded maximum
func.func @_QPtest(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}) {
  return
}
