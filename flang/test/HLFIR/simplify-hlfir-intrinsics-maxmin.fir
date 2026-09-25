// RUN: fir-opt %s --simplify-hlfir-intrinsics=fp-maxmin-behavior=portable --split-input-file | FileCheck %s --check-prefixes=ALL,PORTABLE
// RUN: fir-opt %s --simplify-hlfir-intrinsics=fp-maxmin-behavior=extremum --split-input-file | FileCheck %s --check-prefixes=ALL,UNORDERED,EXTREMUM
// RUN: fir-opt %s --simplify-hlfir-intrinsics=fp-maxmin-behavior=extremenum --split-input-file | FileCheck %s --check-prefixes=ALL,UNORDERED,EXTREMENUM

func.func @_QPtest_unmasked_max(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_unmasked_maxEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 arg 2 {uniq_name = "_QFtest_unmasked_maxEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4 = hlfir.maxval %3#0 {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<100xf32>>) -> f32
  hlfir.assign %4 to %1#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_unmasked_max(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant true
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_3:.*]] = arith.constant 100 : index
// PORTABLE:           %[[IF_0:.*]] = fir.if %[[CONSTANT_0]] -> (f32) {
// PORTABLE:           } else {
// PORTABLE:             fir.result %[[CONSTANT_1]] : f32
// PORTABLE:           }
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_2]] iter_args(%[[VAL_1:.*]] = %[[IF_0]]) -> (f32) {
// PORTABLE:             %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1:.*]], %[[VAL_1]] fastmath<contract> : f32
// PORTABLE:             %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<contract> : f32
// PORTABLE:             %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_1]], %[[LOAD_1]] fastmath<contract> : f32
// PORTABLE:             %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:             %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:             %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[LOAD_1]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// EXTREMUM:            %[[MAXIMUMF_0:.*]] = arith.maximumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<contract> : f32
// EXTREMENUM:          %[[MAXNUMF_0:.*]] = arith.maxnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<contract> : f32

// -----

func.func @_QPtest_unmasked_max(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_unmasked_maxEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 arg 2 {uniq_name = "_QFtest_unmasked_maxEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4 = hlfir.maxval %3#0 {fastmath = #arith.fastmath<nnan>} : (!fir.ref<!fir.array<100xf32>>) -> f32
  hlfir.assign %4 to %1#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_unmasked_max(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// PORTABLE:             %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan> : f32
// PORTABLE:             %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<nnan> : f32
// PORTABLE:             %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_0]], %[[LOAD_0]] fastmath<nnan> : f32
// PORTABLE:             %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:             %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:             %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[LOAD_0]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// EXTREMUM:            %[[MAXIMUMF_0:.*]] = arith.maximumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan> : f32
// EXTREMENUM:          %[[MAXNUMF_0:.*]] = arith.maxnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan> : f32

// -----

func.func @_QPtest_unmasked_max(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_unmasked_maxEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 arg 2 {uniq_name = "_QFtest_unmasked_maxEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4 = hlfir.maxval %3#0 {fastmath = #arith.fastmath<nsz>} : (!fir.ref<!fir.array<100xf32>>) -> f32
  hlfir.assign %4 to %1#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_unmasked_max(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant true
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_3:.*]] = arith.constant 100 : index
// PORTABLE:           %[[IF_0:.*]] = fir.if %[[CONSTANT_0]] -> (f32) {
// PORTABLE:           } else {
// PORTABLE:             fir.result %[[CONSTANT_1]] : f32
// PORTABLE:           }
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_2]] iter_args(%[[VAL_1:.*]] = %[[IF_0]]) -> (f32) {
// PORTABLE:             %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nsz> : f32
// PORTABLE:             %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<nsz> : f32
// PORTABLE:             %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_1]], %[[LOAD_1]] fastmath<nsz> : f32
// PORTABLE:             %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:             %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:             %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[LOAD_1]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// EXTREMUM:            %[[MAXIMUMF_0:.*]] = arith.maximumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nsz> : f32
// EXTREMENUM:          %[[MAXNUMF_0:.*]] = arith.maxnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nsz> : f32

// -----

func.func @_QPtest_unmasked_max(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_unmasked_maxEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 arg 2 {uniq_name = "_QFtest_unmasked_maxEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4 = hlfir.maxval %3#0 {fastmath = #arith.fastmath<nnan,nsz>} : (!fir.ref<!fir.array<100xf32>>) -> f32
  hlfir.assign %4 to %1#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_unmasked_max(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// PORTABLE:             %[[MAXNUMF_0:.*]] = arith.maxnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan,nsz> : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// EXTREMUM:            %[[MAXIMUMF_0:.*]] = arith.maximumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan,nsz> : f32
// EXTREMENUM:          %[[MAXNUMF_0:.*]] = arith.maxnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan,nsz> : f32

// -----

func.func @_QPtest_masked_max(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}, %arg2: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "l"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg2(%1) dummy_scope %0 arg 3 {uniq_name = "_QFtest_masked_maxEl"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_masked_maxEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.shape %c100 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg1(%4) dummy_scope %0 arg 2 {uniq_name = "_QFtest_masked_maxEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %6 = hlfir.maxval %5#0 mask %2#0 {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100x!fir.logical<4>>>) -> f32
  hlfir.assign %6 to %3#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_masked_max(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant false
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant true
// PORTABLE:           %[[CONSTANT_3:.*]] = arith.constant -3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_4:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]]:2 = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_4]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_3]], %[[VAL_2:.*]] = %[[CONSTANT_2]]) -> (f32, i1) {
// PORTABLE:             %[[IF_0:.*]]:2 = fir.if %[[CONVERT_0:.*]] -> (f32, i1) {
// PORTABLE:               %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1:.*]], %[[VAL_1]] fastmath<contract> : f32
// PORTABLE:               %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<contract> : f32
// PORTABLE:               %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_1]], %[[LOAD_1]] fastmath<contract> : f32
// PORTABLE:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:               %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:               %[[ORI_1:.*]] = arith.ori %[[ORI_0]], %[[VAL_2]] : i1
// PORTABLE:               %[[SELECT_0:.*]] = arith.select %[[ORI_1]], %[[LOAD_1]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// UNORDERED:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// EXTREMUM:                %[[MAXIMUMF_0:.*]] = arith.maximumf %[[LOAD_1:.*]], %[[VAL_1]] fastmath<contract> : f32
// EXTREMENUM:              %[[MAXNUMF_0:.*]] = arith.maxnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<contract> : f32

// -----

func.func @_QPtest_masked_max(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}, %arg2: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "l"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg2(%1) dummy_scope %0 arg 3 {uniq_name = "_QFtest_masked_maxEl"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_masked_maxEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.shape %c100 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg1(%4) dummy_scope %0 arg 2 {uniq_name = "_QFtest_masked_maxEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %6 = hlfir.maxval %5#0 mask %2#0 {fastmath = #arith.fastmath<nnan>} : (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100x!fir.logical<4>>>) -> f32
  hlfir.assign %6 to %3#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_masked_max(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// PORTABLE:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// PORTABLE:               %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nnan> : f32
// PORTABLE:               %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<nnan> : f32
// PORTABLE:               %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_1]], %[[LOAD_1]] fastmath<nnan> : f32
// PORTABLE:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:               %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:               %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[LOAD_1]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// UNORDERED:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// EXTREMUM:                %[[MAXIMUMF_0:.*]] = arith.maximumf %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nnan> : f32
// EXTREMENUM:              %[[MAXNUMF_0:.*]] = arith.maxnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan> : f32

// -----

func.func @_QPtest_masked_max(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}, %arg2: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "l"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg2(%1) dummy_scope %0 arg 3 {uniq_name = "_QFtest_masked_maxEl"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_masked_maxEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.shape %c100 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg1(%4) dummy_scope %0 arg 2 {uniq_name = "_QFtest_masked_maxEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %6 = hlfir.maxval %5#0 mask %2#0 {fastmath = #arith.fastmath<nsz>} : (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100x!fir.logical<4>>>) -> f32
  hlfir.assign %6 to %3#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_masked_max(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant false
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant true
// PORTABLE:           %[[CONSTANT_3:.*]] = arith.constant -3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_4:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]]:2 = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_4]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_3]], %[[VAL_2:.*]] = %[[CONSTANT_2]]) -> (f32, i1) {
// PORTABLE:             %[[IF_0:.*]]:2 = fir.if %[[CONVERT_0:.*]] -> (f32, i1) {
// PORTABLE:               %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nsz> : f32
// PORTABLE:               %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<nsz> : f32
// PORTABLE:               %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_1]], %[[LOAD_1]] fastmath<nsz> : f32
// PORTABLE:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:               %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:               %[[ORI_1:.*]] = arith.ori %[[ORI_0]], %[[VAL_2]] : i1
// PORTABLE:               %[[SELECT_0:.*]] = arith.select %[[ORI_1]], %[[LOAD_1]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// UNORDERED:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// EXTREMUM:                %[[MAXIMUMF_0:.*]] = arith.maximumf %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nsz> : f32
// EXTREMENUM:              %[[MAXNUMF_0:.*]] = arith.maxnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nsz> : f32

// -----

func.func @_QPtest_masked_max(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}, %arg2: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "l"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg2(%1) dummy_scope %0 arg 3 {uniq_name = "_QFtest_masked_maxEl"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_masked_maxEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.shape %c100 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg1(%4) dummy_scope %0 arg 2 {uniq_name = "_QFtest_masked_maxEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %6 = hlfir.maxval %5#0 mask %2#0 {fastmath = #arith.fastmath<nnan,nsz>} : (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100x!fir.logical<4>>>) -> f32
  hlfir.assign %6 to %3#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_masked_max(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// PORTABLE:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// PORTABLE:               %[[MAXNUMF_0:.*]] = arith.maxnumf %[[LOAD_1]], %[[VAL_1]] fastmath<nnan,nsz> : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant -3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// UNORDERED:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// EXTREMUM:                %[[MAXIMUMF_0:.*]] = arith.maximumf %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nnan,nsz> : f32
// EXTREMENUM:              %[[MAXNUMF_0:.*]] = arith.maxnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan,nsz> : f32

// -----

func.func @_QPtest_unmasked_min(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_unmasked_minEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 arg 2 {uniq_name = "_QFtest_unmasked_minEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4 = hlfir.minval %3#0 {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<100xf32>>) -> f32
  hlfir.assign %4 to %1#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_unmasked_min(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant true
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_3:.*]] = arith.constant 100 : index
// PORTABLE:           %[[IF_0:.*]] = fir.if %[[CONSTANT_0]] -> (f32) {
// PORTABLE:           } else {
// PORTABLE:             fir.result %[[CONSTANT_1]] : f32
// PORTABLE:           }
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_2]] iter_args(%[[VAL_1:.*]] = %[[IF_0]]) -> (f32) {
// PORTABLE:             %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_1:.*]], %[[VAL_1]] fastmath<contract> : f32
// PORTABLE:             %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<contract> : f32
// PORTABLE:             %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_1]], %[[LOAD_1]] fastmath<contract> : f32
// PORTABLE:             %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:             %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:             %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[LOAD_1]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// EXTREMUM:            %[[MINIMUMF_0:.*]] = arith.minimumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<contract> : f32
// EXTREMENUM:          %[[MINNUMF_0:.*]] = arith.minnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<contract> : f32

// -----

func.func @_QPtest_unmasked_min(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_unmasked_minEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 arg 2 {uniq_name = "_QFtest_unmasked_minEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4 = hlfir.minval %3#0 {fastmath = #arith.fastmath<nnan>} : (!fir.ref<!fir.array<100xf32>>) -> f32
  hlfir.assign %4 to %1#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_unmasked_min(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// PORTABLE:             %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan> : f32
// PORTABLE:             %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<nnan> : f32
// PORTABLE:             %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_0]], %[[LOAD_0]] fastmath<nnan> : f32
// PORTABLE:             %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:             %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:             %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[LOAD_0]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// EXTREMUM:            %[[MINIMUMF_0:.*]] = arith.minimumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan> : f32
// EXTREMENUM:          %[[MINNUMF_0:.*]] = arith.minnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan> : f32

// -----

func.func @_QPtest_unmasked_min(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_unmasked_minEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 arg 2 {uniq_name = "_QFtest_unmasked_minEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4 = hlfir.minval %3#0 {fastmath = #arith.fastmath<nsz>} : (!fir.ref<!fir.array<100xf32>>) -> f32
  hlfir.assign %4 to %1#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_unmasked_min(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant true
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_3:.*]] = arith.constant 100 : index
// PORTABLE:           %[[IF_0:.*]] = fir.if %[[CONSTANT_0]] -> (f32) {
// PORTABLE:           } else {
// PORTABLE:             fir.result %[[CONSTANT_1]] : f32
// PORTABLE:           }
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_2]] iter_args(%[[VAL_1:.*]] = %[[IF_0]]) -> (f32) {
// PORTABLE:             %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nsz> : f32
// PORTABLE:             %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<nsz> : f32
// PORTABLE:             %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_1]], %[[LOAD_1]] fastmath<nsz> : f32
// PORTABLE:             %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:             %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:             %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[LOAD_1]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// EXTREMUM:            %[[MINIMUMF_0:.*]] = arith.minimumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nsz> : f32
// EXTREMENUM:          %[[MINNUMF_0:.*]] = arith.minnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nsz> : f32

// -----

func.func @_QPtest_unmasked_min(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_unmasked_minEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 arg 2 {uniq_name = "_QFtest_unmasked_minEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4 = hlfir.minval %3#0 {fastmath = #arith.fastmath<nnan,nsz>} : (!fir.ref<!fir.array<100xf32>>) -> f32
  hlfir.assign %4 to %1#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_unmasked_min(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// PORTABLE:             %[[MAXNUMF_0:.*]] = arith.minnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan,nsz> : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// EXTREMUM:            %[[MINIMUMF_0:.*]] = arith.minimumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan,nsz> : f32
// EXTREMENUM:          %[[MINNUMF_0:.*]] = arith.minnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan,nsz> : f32

// -----

func.func @_QPtest_masked_min(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}, %arg2: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "l"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg2(%1) dummy_scope %0 arg 3 {uniq_name = "_QFtest_masked_minEl"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_masked_minEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.shape %c100 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg1(%4) dummy_scope %0 arg 2 {uniq_name = "_QFtest_masked_minEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %6 = hlfir.minval %5#0 mask %2#0 {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100x!fir.logical<4>>>) -> f32
  hlfir.assign %6 to %3#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_masked_min(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant false
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant true
// PORTABLE:           %[[CONSTANT_3:.*]] = arith.constant 3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_4:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]]:2 = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_4]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_3]], %[[VAL_2:.*]] = %[[CONSTANT_2]]) -> (f32, i1) {
// PORTABLE:             %[[IF_0:.*]]:2 = fir.if %[[CONVERT_0:.*]] -> (f32, i1) {
// PORTABLE:               %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_1:.*]], %[[VAL_1]] fastmath<contract> : f32
// PORTABLE:               %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<contract> : f32
// PORTABLE:               %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_1]], %[[LOAD_1]] fastmath<contract> : f32
// PORTABLE:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:               %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:               %[[ORI_1:.*]] = arith.ori %[[ORI_0]], %[[VAL_2]] : i1
// PORTABLE:               %[[SELECT_0:.*]] = arith.select %[[ORI_1]], %[[LOAD_1]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// UNORDERED:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// EXTREMUM:                %[[MINIMUMF_0:.*]] = arith.minimumf %[[LOAD_1:.*]], %[[VAL_1]] fastmath<contract> : f32
// EXTREMENUM:              %[[MINNUMF_0:.*]] = arith.minnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<contract> : f32

// -----

func.func @_QPtest_masked_min(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}, %arg2: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "l"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg2(%1) dummy_scope %0 arg 3 {uniq_name = "_QFtest_masked_minEl"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_masked_minEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.shape %c100 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg1(%4) dummy_scope %0 arg 2 {uniq_name = "_QFtest_masked_minEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %6 = hlfir.minval %5#0 mask %2#0 {fastmath = #arith.fastmath<nnan>} : (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100x!fir.logical<4>>>) -> f32
  hlfir.assign %6 to %3#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_masked_min(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// PORTABLE:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// PORTABLE:               %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nnan> : f32
// PORTABLE:               %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<nnan> : f32
// PORTABLE:               %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_1]], %[[LOAD_1]] fastmath<nnan> : f32
// PORTABLE:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:               %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:               %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[LOAD_1]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// UNORDERED:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// EXTREMUM:                %[[MINIMUMF_0:.*]] = arith.minimumf %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nnan> : f32
// EXTREMENUM:              %[[MINNUMF_0:.*]] = arith.minnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan> : f32

// -----

func.func @_QPtest_masked_min(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}, %arg2: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "l"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg2(%1) dummy_scope %0 arg 3 {uniq_name = "_QFtest_masked_minEl"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_masked_minEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.shape %c100 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg1(%4) dummy_scope %0 arg 2 {uniq_name = "_QFtest_masked_minEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %6 = hlfir.minval %5#0 mask %2#0 {fastmath = #arith.fastmath<nsz>} : (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100x!fir.logical<4>>>) -> f32
  hlfir.assign %6 to %3#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_masked_min(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant false
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant true
// PORTABLE:           %[[CONSTANT_3:.*]] = arith.constant 3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_4:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]]:2 = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_4]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_3]], %[[VAL_2:.*]] = %[[CONSTANT_2]]) -> (f32, i1) {
// PORTABLE:             %[[IF_0:.*]]:2 = fir.if %[[CONVERT_0:.*]] -> (f32, i1) {
// PORTABLE:               %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nsz> : f32
// PORTABLE:               %[[CMPF_1:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] fastmath<nsz> : f32
// PORTABLE:               %[[CMPF_2:.*]] = arith.cmpf oeq, %[[LOAD_1]], %[[LOAD_1]] fastmath<nsz> : f32
// PORTABLE:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_1]], %[[CMPF_2]] : i1
// PORTABLE:               %[[ORI_0:.*]] = arith.ori %[[CMPF_0]], %[[ANDI_0]] : i1
// PORTABLE:               %[[ORI_1:.*]] = arith.ori %[[ORI_0]], %[[VAL_2]] : i1
// PORTABLE:               %[[SELECT_0:.*]] = arith.select %[[ORI_1]], %[[LOAD_1]], %[[VAL_1]] : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// UNORDERED:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// EXTREMUM:                %[[MINIMUMF_0:.*]] = arith.minimumf %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nsz> : f32
// EXTREMENUM:              %[[MINNUMF_0:.*]] = arith.minnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nsz> : f32

// -----

func.func @_QPtest_masked_min(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "y"}, %arg2: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "l"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg2(%1) dummy_scope %0 arg 3 {uniq_name = "_QFtest_masked_minEl"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_masked_minEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.shape %c100 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg1(%4) dummy_scope %0 arg 2 {uniq_name = "_QFtest_masked_minEy"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %6 = hlfir.minval %5#0 mask %2#0 {fastmath = #arith.fastmath<nnan,nsz>} : (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100x!fir.logical<4>>>) -> f32
  hlfir.assign %6 to %3#0 : f32, !fir.ref<f32>
  return
}

// ALL-LABEL:   func.func @_QPtest_masked_min(
// PORTABLE:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// PORTABLE:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// PORTABLE:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// PORTABLE:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// PORTABLE:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// PORTABLE:               %[[MAXNUMF_0:.*]] = arith.minnumf %[[LOAD_1]], %[[VAL_1]] fastmath<nnan,nsz> : f32

// UNORDERED:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// UNORDERED:           %[[CONSTANT_1:.*]] = arith.constant 3.40282347E+38 : f32
// UNORDERED:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
// UNORDERED:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// UNORDERED:             %[[IF_0:.*]] = fir.if %[[CONVERT_0:.*]] -> (f32) {
// EXTREMUM:                %[[MINIMUMF_0:.*]] = arith.minimumf %[[LOAD_1:.*]], %[[VAL_1]] fastmath<nnan,nsz> : f32
// EXTREMENUM:              %[[MINNUMF_0:.*]] = arith.minnumf %[[LOAD_0:.*]], %[[VAL_1]] fastmath<nnan,nsz> : f32
