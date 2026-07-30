// Test the max-reduction-extent pass option, which bounds the constant section
// extent that promotion will accept. A section whose extent exceeds the option
// is detected but left un-promoted (no constant-shape temporary is created).
// RUN: fir-opt --split-input-file --array-section-reduction %s \
// RUN:   | FileCheck %s --check-prefix=DEFAULT
// RUN: fir-opt --split-input-file --array-section-reduction=max-reduction-extent=9 %s \
// RUN:   | FileCheck %s --check-prefix=EXT9

// Extent 8 sits exactly on the default boundary (8): promoted.
// DEFAULT-LABEL: func.func @_QPext8
// DEFAULT:         fir.alloca !fir.array<8xf64>
func.func @_QPext8(%arg0: !fir.ref<!fir.array<8xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<8xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c8 = arith.constant 8 : index
  %sh = fir.shape %c8 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFext8Ea"} : (!fir.ref<!fir.array<8xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<8xf64>>, !fir.ref<!fir.array<8xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFext8Exx"} : (!fir.ref<!fir.array<8xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<8xf64>>, !fir.ref<!fir.array<8xf64>>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFext8En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %init = hlfir.designate %a#0 (%c1:%c8:%c1)  shape %sh : (!fir.ref<!fir.array<8xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<8xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<8xf64>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %rd = hlfir.designate %a#0 (%c1:%c8:%c1)  shape %sh : (!fir.ref<!fir.array<8xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<8xf64>>
    %xxs = hlfir.designate %xx#0 (%c1:%c8:%c1)  shape %sh : (!fir.ref<!fir.array<8xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<8xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<8xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<8xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<8xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c8:%c1)  shape %sh : (!fir.ref<!fir.array<8xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<8xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<8xf64>, !fir.ref<!fir.array<8xf64>>
    hlfir.destroy %el : !hlfir.expr<8xf64>
  }
  return
}

// -----

// Extent 9 is one past the default boundary: rejected at the default (8), and
// promoted only when max-reduction-extent is raised to 9.
// DEFAULT-LABEL: func.func @_QPext9
// DEFAULT-NOT:     fir.alloca !fir.array
// EXT9-LABEL:   func.func @_QPext9
// EXT9:           fir.alloca !fir.array<9xf64>
func.func @_QPext9(%arg0: !fir.ref<!fir.array<9xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<9xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c9 = arith.constant 9 : index
  %sh = fir.shape %c9 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFext9Ea"} : (!fir.ref<!fir.array<9xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<9xf64>>, !fir.ref<!fir.array<9xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFext9Exx"} : (!fir.ref<!fir.array<9xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<9xf64>>, !fir.ref<!fir.array<9xf64>>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFext9En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %init = hlfir.designate %a#0 (%c1:%c9:%c1)  shape %sh : (!fir.ref<!fir.array<9xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<9xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<9xf64>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %rd = hlfir.designate %a#0 (%c1:%c9:%c1)  shape %sh : (!fir.ref<!fir.array<9xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<9xf64>>
    %xxs = hlfir.designate %xx#0 (%c1:%c9:%c1)  shape %sh : (!fir.ref<!fir.array<9xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<9xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<9xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<9xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<9xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c9:%c1)  shape %sh : (!fir.ref<!fir.array<9xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<9xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<9xf64>, !fir.ref<!fir.array<9xf64>>
    hlfir.destroy %el : !hlfir.expr<9xf64>
  }
  return
}
