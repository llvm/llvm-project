// RUN: fir-opt --lower-repack-arrays %s | FileCheck %s
// Test trivial type array repacking.
func.func @_QPtest1(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap innermost : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest1Ex"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?x?xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest1(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:             %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_5]] : i1
// CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.box<!fir.array<?x?xf32>>>
// CHECK:             %[[VAL_12:.*]] = fir.is_present %[[VAL_11]] : (!fir.ref<!fir.box<!fir.array<?x?xf32>>>) -> i1
// CHECK:             %[[VAL_13:.*]] = arith.andi %[[VAL_10]], %[[VAL_12]] : i1
// CHECK:             %[[VAL_14:.*]] = fir.if %[[VAL_13]] weights([0, 1]) -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]] = fir.shape %[[VAL_15]]#1, %[[VAL_16]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_18:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_15]]#1, %[[VAL_16]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:               %[[VAL_19:.*]] = fir.declare %[[VAL_18]](%[[VAL_17]]) {uniq_name = ".repacked"} : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:               %[[VAL_25:.*]] = fir.shape_shift %[[VAL_15]]#0, %[[VAL_15]]#1, %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_20:.*]] = fir.embox %[[VAL_19]](%[[VAL_25]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:               %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_22:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_20]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_14]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = fir.declare %[[VAL_8]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest1Ex"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_28:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           fir.if %[[VAL_28]] {
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_31:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_33:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_32]] : index
// CHECK:             fir.if %[[VAL_33]] weights([0, 1]) {
// CHECK:               %[[VAL_34:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_29]] : !fir.heap<!fir.array<?x?xf32>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test 'stack whole' repacking.
func.func @_QPtest1_whole(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 stack whole : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest1_wholeEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  fir.unpack_array %1 to %arg0 stack : !fir.box<!fir.array<?x?xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest1_whole(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] whole : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:             %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_5]] : i1
// CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.box<!fir.array<?x?xf32>>>
// CHECK:             %[[VAL_12:.*]] = fir.is_present %[[VAL_11]] : (!fir.ref<!fir.box<!fir.array<?x?xf32>>>) -> i1
// CHECK:             %[[VAL_13:.*]] = arith.andi %[[VAL_10]], %[[VAL_12]] : i1
// CHECK:             %[[VAL_14:.*]] = fir.if %[[VAL_13]] weights([0, 1]) -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]] = fir.shape %[[VAL_15]]#1, %[[VAL_16]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_18:.*]] = fir.alloca !fir.array<?x?xf32>, %[[VAL_15]]#1, %[[VAL_16]]#1 {bindc_name = ".repacked"}
// CHECK:               %[[VAL_19:.*]] = fir.declare %[[VAL_18]](%[[VAL_17]]) {uniq_name = ".repacked"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<?x?xf32>>
// CHECK:               %[[VAL_25:.*]] = fir.shape_shift %[[VAL_15]]#0, %[[VAL_15]]#1, %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_20:.*]] = fir.embox %[[VAL_19]](%[[VAL_25]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:               %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_22:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_20]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_14]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = fir.declare %[[VAL_8]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest1_wholeEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_28:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           fir.if %[[VAL_28]] {
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_31:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_33:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_32]] : index
// CHECK:             fir.if %[[VAL_33]] weights([0, 1]) {
// CHECK:               %[[VAL_34:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test unpacking with no_copy.
func.func @_QPtest1_in(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap innermost : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %2 = fir.declare %1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest1_inEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  fir.unpack_array %1 to %arg0 heap no_copy : !fir.box<!fir.array<?x?xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest1_in(
// CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           %[[VAL_7:.*]] = fir.if %[[VAL_6]] -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:             %[[VAL_8:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:             %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_4]] : i1
// CHECK:             %[[VAL_10:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.box<!fir.array<?x?xf32>>>
// CHECK:             %[[VAL_11:.*]] = fir.is_present %[[VAL_10]] : (!fir.ref<!fir.box<!fir.array<?x?xf32>>>) -> i1
// CHECK:             %[[VAL_12:.*]] = arith.andi %[[VAL_9]], %[[VAL_11]] : i1
// CHECK:             %[[VAL_13:.*]] = fir.if %[[VAL_12]] weights([0, 1]) -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:               %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]] = fir.shape %[[VAL_14]]#1, %[[VAL_15]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_17:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_14]]#1, %[[VAL_15]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:               %[[VAL_18:.*]] = fir.declare %[[VAL_17]](%[[VAL_16]]) {uniq_name = ".repacked"} : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:               %[[VAL_24:.*]] = fir.shape_shift %[[VAL_14]]#0, %[[VAL_14]]#1, %[[VAL_15]]#0, %[[VAL_15]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_19:.*]] = fir.embox %[[VAL_18]](%[[VAL_24]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:               %[[VAL_20:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_22:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_19]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_13]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = fir.declare %[[VAL_7]] dummy_scope %[[VAL_5]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest1_inEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_27:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           fir.if %[[VAL_27]] {
// CHECK:             %[[VAL_28:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_30:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_32:.*]] = arith.cmpi ne, %[[VAL_29]], %[[VAL_31]] : index
// CHECK:             fir.if %[[VAL_32]] weights([0, 1]) {
// CHECK:               fir.freemem %[[VAL_28]] : !fir.heap<!fir.array<?x?xf32>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test packing with no_copy.
func.func @_QPtest1_out(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap innermost no_copy : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %2 = fir.declare %1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFtest1_outEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?x?xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest1_out(
// CHECK-SAME:                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           %[[VAL_7:.*]] = fir.if %[[VAL_6]] -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:             %[[VAL_8:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:             %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_4]] : i1
// CHECK:             %[[VAL_10:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.box<!fir.array<?x?xf32>>>
// CHECK:             %[[VAL_11:.*]] = fir.is_present %[[VAL_10]] : (!fir.ref<!fir.box<!fir.array<?x?xf32>>>) -> i1
// CHECK:             %[[VAL_12:.*]] = arith.andi %[[VAL_9]], %[[VAL_11]] : i1
// CHECK:             %[[VAL_13:.*]] = fir.if %[[VAL_12]] weights([0, 1]) -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:               %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]] = fir.shape %[[VAL_14]]#1, %[[VAL_15]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_17:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_14]]#1, %[[VAL_15]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:               %[[VAL_18:.*]] = fir.declare %[[VAL_17]](%[[VAL_16]]) {uniq_name = ".repacked"} : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:               %[[VAL_20:.*]] = fir.shape_shift %[[VAL_14]]#0, %[[VAL_14]]#1, %[[VAL_15]]#0, %[[VAL_15]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_19:.*]] = fir.embox %[[VAL_18]](%[[VAL_20]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:               fir.result %[[VAL_19]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_13]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = fir.declare %[[VAL_7]] dummy_scope %[[VAL_5]] {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFtest1_outEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_23:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           fir.if %[[VAL_23]] {
// CHECK:             %[[VAL_24:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_26:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_28:.*]] = arith.cmpi ne, %[[VAL_25]], %[[VAL_27]] : index
// CHECK:             fir.if %[[VAL_28]] weights([0, 1]) {
// CHECK:               %[[VAL_29:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_30:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_31:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_32:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_30]], %[[VAL_31]], %[[VAL_32]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_24]] : !fir.heap<!fir.array<?x?xf32>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test character array with dynamic length and heap allocation
func.func @_QPtest2(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtest2En"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.load %1 : !fir.ref<i32>
  %3 = arith.cmpi sgt, %2, %c0_i32 : i32
  %4 = arith.select %3, %2, %c0_i32 : i32
  %5 = fir.pack_array %arg1 heap innermost typeparams %4 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  %6 = fir.declare %5 typeparams %4 dummy_scope %0 {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  fir.unpack_array %5 to %arg1 heap : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest2(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:                        %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant
// CHECK:           %[[VAL_3:.*]] = arith.constant
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant false
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest2En"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_13:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_14:.*]] = fir.if %[[VAL_13]] -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:             %[[VAL_15:.*]] = fir.is_contiguous_box %[[VAL_1]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:             %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_15]], %[[VAL_6]] : i1
// CHECK:             %[[VAL_17:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,?>>>>
// CHECK:             %[[VAL_18:.*]] = fir.is_present %[[VAL_17]] : (!fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,?>>>>) -> i1
// CHECK:             %[[VAL_19:.*]] = arith.andi %[[VAL_16]], %[[VAL_18]] : i1
// CHECK:             %[[VAL_20:.*]] = fir.if %[[VAL_19]] weights([0, 1]) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:               %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_22:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_23:.*]] = fir.shape %[[VAL_21]]#1, %[[VAL_22]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_24:.*]] = fir.allocmem !fir.array<?x?x!fir.char<1,?>>(%[[VAL_12]] : i32), %[[VAL_21]]#1, %[[VAL_22]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:               %[[VAL_25:.*]] = fir.declare %[[VAL_24]](%[[VAL_23]]) typeparams %[[VAL_12]] {uniq_name = ".repacked"} : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, i32) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_31:.*]] = fir.shape_shift %[[VAL_21]]#0, %[[VAL_21]]#1, %[[VAL_22]]#0, %[[VAL_22]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_26:.*]] = fir.embox %[[VAL_25]](%[[VAL_31]]) typeparams %[[VAL_12]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>, !fir.shapeshift<2>, i32) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_27:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_28:.*]] = fir.convert %[[VAL_26]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_29:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_30:.*]] = fir.convert %[[VAL_27]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_3]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_26]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_1]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_20]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_1]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = fir.declare %[[VAL_14]] typeparams %[[VAL_12]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_34:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           fir.if %[[VAL_34]] {
// CHECK:             %[[VAL_35:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_37:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_39:.*]] = arith.cmpi ne, %[[VAL_36]], %[[VAL_38]] : index
// CHECK:             fir.if %[[VAL_39]] weights([0, 1]) {
// CHECK:               %[[VAL_40:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_41:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_42:.*]] = fir.convert %[[VAL_14]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_43:.*]] = fir.convert %[[VAL_40]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_41]], %[[VAL_42]], %[[VAL_43]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_35]] : !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test character array with dynamic length and stack allocation
func.func @_QPtest2_stack(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtest2_stackEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.load %1 : !fir.ref<i32>
  %3 = arith.cmpi sgt, %2, %c0_i32 : i32
  %4 = arith.select %3, %2, %c0_i32 : i32
  %5 = fir.pack_array %arg1 stack innermost typeparams %4 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  %6 = fir.declare %5 typeparams %4 dummy_scope %0 {uniq_name = "_QFtest2_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  fir.unpack_array %5 to %arg1 stack : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest2_stack(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:                              %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant
// CHECK:           %[[VAL_3:.*]] = arith.constant
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant false
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest2_stackEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_13:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_14:.*]] = fir.if %[[VAL_13]] -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:             %[[VAL_15:.*]] = fir.is_contiguous_box %[[VAL_1]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:             %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_15]], %[[VAL_6]] : i1
// CHECK:             %[[VAL_17:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,?>>>>
// CHECK:             %[[VAL_18:.*]] = fir.is_present %[[VAL_17]] : (!fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,?>>>>) -> i1
// CHECK:             %[[VAL_19:.*]] = arith.andi %[[VAL_16]], %[[VAL_18]] : i1
// CHECK:             %[[VAL_20:.*]] = fir.if %[[VAL_19]] weights([0, 1]) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:               %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_22:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_23:.*]] = fir.shape %[[VAL_21]]#1, %[[VAL_22]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_24:.*]] = fir.alloca !fir.array<?x?x!fir.char<1,?>>(%[[VAL_12]] : i32), %[[VAL_21]]#1, %[[VAL_22]]#1 {bindc_name = ".repacked"}
// CHECK:               %[[VAL_25:.*]] = fir.declare %[[VAL_24]](%[[VAL_23]]) typeparams %[[VAL_12]] {uniq_name = ".repacked"} : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, i32) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_31:.*]] = fir.shape_shift %[[VAL_21]]#0, %[[VAL_21]]#1, %[[VAL_22]]#0, %[[VAL_22]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_26:.*]] = fir.embox %[[VAL_25]](%[[VAL_31]]) typeparams %[[VAL_12]] : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shapeshift<2>, i32) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_27:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_28:.*]] = fir.convert %[[VAL_26]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_29:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_30:.*]] = fir.convert %[[VAL_27]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_3]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_26]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_1]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_20]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_1]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = fir.declare %[[VAL_14]] typeparams %[[VAL_12]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest2_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_34:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           fir.if %[[VAL_34]] {
// CHECK:             %[[VAL_35:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_37:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_39:.*]] = arith.cmpi ne, %[[VAL_36]], %[[VAL_38]] : index
// CHECK:             fir.if %[[VAL_39]] weights([0, 1]) {
// CHECK:               %[[VAL_40:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_41:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_42:.*]] = fir.convert %[[VAL_14]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_43:.*]] = fir.convert %[[VAL_40]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_41]], %[[VAL_42]], %[[VAL_43]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test character array with assumed length and heap allocation.
func.func @_QPtest3(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest3Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest3(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:             %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_5]] : i1
// CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,?>>>>
// CHECK:             %[[VAL_12:.*]] = fir.is_present %[[VAL_11]] : (!fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,?>>>>) -> i1
// CHECK:             %[[VAL_13:.*]] = arith.andi %[[VAL_10]], %[[VAL_12]] : i1
// CHECK:             %[[VAL_14:.*]] = fir.if %[[VAL_13]] weights([0, 1]) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:               %[[VAL_18:.*]] = fir.shape %[[VAL_15]]#1, %[[VAL_16]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_19:.*]] = fir.allocmem !fir.array<?x?x!fir.char<1,?>>(%[[VAL_17]] : index), %[[VAL_15]]#1, %[[VAL_16]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:               %[[VAL_20:.*]] = fir.declare %[[VAL_19]](%[[VAL_18]]) typeparams %[[VAL_17]] {uniq_name = ".repacked"} : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, index) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_26:.*]] = fir.shape_shift %[[VAL_15]]#0, %[[VAL_15]]#1, %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_21:.*]] = fir.embox %[[VAL_20]](%[[VAL_26]]) typeparams %[[VAL_17]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>, !fir.shapeshift<2>, index) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_22:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_21]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_14]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = fir.declare %[[VAL_8]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest3Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_29:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           fir.if %[[VAL_29]] {
// CHECK:             %[[VAL_30:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_32:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_34:.*]] = arith.cmpi ne, %[[VAL_31]], %[[VAL_33]] : index
// CHECK:             fir.if %[[VAL_34]] weights([0, 1]) {
// CHECK:               %[[VAL_35:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_38:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_36]], %[[VAL_37]], %[[VAL_38]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_30]] : !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test character array with assumed length and stack allocation.
func.func @_QPtest3_stack(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 stack innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest3_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  fir.unpack_array %1 to %arg0 stack : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest3_stack(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:             %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_5]] : i1
// CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,?>>>>
// CHECK:             %[[VAL_12:.*]] = fir.is_present %[[VAL_11]] : (!fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,?>>>>) -> i1
// CHECK:             %[[VAL_13:.*]] = arith.andi %[[VAL_10]], %[[VAL_12]] : i1
// CHECK:             %[[VAL_14:.*]] = fir.if %[[VAL_13]] weights([0, 1]) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:               %[[VAL_18:.*]] = fir.shape %[[VAL_15]]#1, %[[VAL_16]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_19:.*]] = fir.alloca !fir.array<?x?x!fir.char<1,?>>(%[[VAL_17]] : index), %[[VAL_15]]#1, %[[VAL_16]]#1 {bindc_name = ".repacked"}
// CHECK:               %[[VAL_20:.*]] = fir.declare %[[VAL_19]](%[[VAL_18]]) typeparams %[[VAL_17]] {uniq_name = ".repacked"} : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, index) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_26:.*]] = fir.shape_shift %[[VAL_15]]#0, %[[VAL_15]]#1, %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_21:.*]] = fir.embox %[[VAL_20]](%[[VAL_26]]) typeparams %[[VAL_17]] : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shapeshift<2>, index) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_22:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_21]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_14]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = fir.declare %[[VAL_8]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest3_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_29:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           fir.if %[[VAL_29]] {
// CHECK:             %[[VAL_30:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_32:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_34:.*]] = arith.cmpi ne, %[[VAL_31]], %[[VAL_33]] : index
// CHECK:             fir.if %[[VAL_34]] weights([0, 1]) {
// CHECK:               %[[VAL_35:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_38:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_36]], %[[VAL_37]], %[[VAL_38]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test character array with constant length and heap allocation.
func.func @_QPtest4(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,10>>> {fir.bindc_name = "x"}) {
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap innermost : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  %2 = fir.declare %1 typeparams %c10 dummy_scope %0 {uniq_name = "_QFtest4Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest4(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,10>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:           %[[VAL_9:.*]] = fir.if %[[VAL_8]] -> (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) {
// CHECK:             %[[VAL_10:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:             %[[VAL_11:.*]] = arith.cmpi eq, %[[VAL_10]], %[[VAL_5]] : i1
// CHECK:             %[[VAL_12:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,10>>>>
// CHECK:             %[[VAL_13:.*]] = fir.is_present %[[VAL_12]] : (!fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,10>>>>) -> i1
// CHECK:             %[[VAL_14:.*]] = arith.andi %[[VAL_11]], %[[VAL_13]] : i1
// CHECK:             %[[VAL_15:.*]] = fir.if %[[VAL_14]] weights([0, 1]) -> (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) {
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_18:.*]] = fir.shape %[[VAL_16]]#1, %[[VAL_17]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_19:.*]] = fir.allocmem !fir.array<?x?x!fir.char<1,10>>, %[[VAL_16]]#1, %[[VAL_17]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:               %[[VAL_20:.*]] = fir.declare %[[VAL_19]](%[[VAL_18]]) typeparams %[[VAL_6]] {uniq_name = ".repacked"} : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>, !fir.shape<2>, index) -> !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:               %[[VAL_26:.*]] = fir.shape_shift %[[VAL_16]]#0, %[[VAL_16]]#1, %[[VAL_17]]#0, %[[VAL_17]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_21:.*]] = fir.embox %[[VAL_20]](%[[VAL_26]]) : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:               %[[VAL_22:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_21]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_15]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = fir.declare %[[VAL_9]] typeparams %[[VAL_6]] dummy_scope %[[VAL_7]] {uniq_name = "_QFtest4Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           %[[VAL_29:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:           fir.if %[[VAL_29]] {
// CHECK:             %[[VAL_30:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             %[[VAL_32:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>) -> index
// CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>) -> index
// CHECK:             %[[VAL_34:.*]] = arith.cmpi ne, %[[VAL_31]], %[[VAL_33]] : index
// CHECK:             fir.if %[[VAL_34]] weights([0, 1]) {
// CHECK:               %[[VAL_35:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_38:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_36]], %[[VAL_37]], %[[VAL_38]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_30]] : !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test character array with constant length and stack allocation.
func.func @_QPtest4_stack(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,10>>> {fir.bindc_name = "x"}) {
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 stack innermost : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  %2 = fir.declare %1 typeparams %c10 dummy_scope %0 {uniq_name = "_QFtest4_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  fir.unpack_array %1 to %arg0 stack : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest4_stack(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,10>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:           %[[VAL_9:.*]] = fir.if %[[VAL_8]] -> (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) {
// CHECK:             %[[VAL_10:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:             %[[VAL_11:.*]] = arith.cmpi eq, %[[VAL_10]], %[[VAL_5]] : i1
// CHECK:             %[[VAL_12:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,10>>>>
// CHECK:             %[[VAL_13:.*]] = fir.is_present %[[VAL_12]] : (!fir.ref<!fir.box<!fir.array<?x?x!fir.char<1,10>>>>) -> i1
// CHECK:             %[[VAL_14:.*]] = arith.andi %[[VAL_11]], %[[VAL_13]] : i1
// CHECK:             %[[VAL_15:.*]] = fir.if %[[VAL_14]] weights([0, 1]) -> (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) {
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_18:.*]] = fir.shape %[[VAL_16]]#1, %[[VAL_17]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_19:.*]] = fir.alloca !fir.array<?x?x!fir.char<1,10>>, %[[VAL_16]]#1, %[[VAL_17]]#1 {bindc_name = ".repacked"}
// CHECK:               %[[VAL_20:.*]] = fir.declare %[[VAL_19]](%[[VAL_18]]) typeparams %[[VAL_6]] {uniq_name = ".repacked"} : (!fir.ref<!fir.array<?x?x!fir.char<1,10>>>, !fir.shape<2>, index) -> !fir.ref<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:               %[[VAL_26:.*]] = fir.shape_shift %[[VAL_16]]#0, %[[VAL_16]]#1, %[[VAL_17]]#0, %[[VAL_17]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_21:.*]] = fir.embox %[[VAL_20]](%[[VAL_26]]) : (!fir.ref<!fir.array<?x?x!fir.char<1,10>>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:               %[[VAL_22:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_21]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_15]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = fir.declare %[[VAL_9]] typeparams %[[VAL_6]] dummy_scope %[[VAL_7]] {uniq_name = "_QFtest4_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           %[[VAL_29:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:           fir.if %[[VAL_29]] {
// CHECK:             %[[VAL_30:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             %[[VAL_32:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>) -> index
// CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>) -> index
// CHECK:             %[[VAL_34:.*]] = arith.cmpi ne, %[[VAL_31]], %[[VAL_33]] : index
// CHECK:             fir.if %[[VAL_34]] weights([0, 1]) {
// CHECK:               %[[VAL_35:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_38:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_36]], %[[VAL_37]], %[[VAL_38]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test derived type array with heap allocation.
func.func @_QPtest5(%arg0: !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap innermost : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest5Ex"} : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest5(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:             %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_5]] : i1
// CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.ref<!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:             %[[VAL_12:.*]] = fir.is_present %[[VAL_11]] : (!fir.ref<!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> i1
// CHECK:             %[[VAL_13:.*]] = arith.andi %[[VAL_10]], %[[VAL_12]] : i1
// CHECK:             %[[VAL_14:.*]] = fir.if %[[VAL_13]] weights([0, 1]) -> (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]] = fir.shape %[[VAL_15]]#1, %[[VAL_16]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_18:.*]] = fir.allocmem !fir.array<?x?x!fir.type<_QMmTt>>, %[[VAL_15]]#1, %[[VAL_16]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:               %[[VAL_19:.*]] = fir.declare %[[VAL_18]](%[[VAL_17]]) {uniq_name = ".repacked"} : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shape<2>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_25:.*]] = fir.shape_shift %[[VAL_15]]#0, %[[VAL_15]]#1, %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_20:.*]] = fir.embox %[[VAL_19]](%[[VAL_25]]) source_box %[[VAL_0]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shapeshift<2>, !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[BOX:.*]] = fir.convert %[[VAL_20]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_22:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[BOX]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_14]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = fir.declare %[[VAL_8]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest5Ex"} : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_28:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           fir.if %[[VAL_28]] {
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_31:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_33:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_32]] : index
// CHECK:             fir.if %[[VAL_33]] weights([0, 1]) {
// CHECK:               %[[VAL_34:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_29]] : !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test derived type array with stack allocation.
func.func @_QPtest5_stack(%arg0: !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 stack innermost : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest5_stackEx"} : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
  fir.unpack_array %1 to %arg0 stack : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest5_stack(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:             %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_5]] : i1
// CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.ref<!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:             %[[VAL_12:.*]] = fir.is_present %[[VAL_11]] : (!fir.ref<!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> i1
// CHECK:             %[[VAL_13:.*]] = arith.andi %[[VAL_10]], %[[VAL_12]] : i1
// CHECK:             %[[VAL_14:.*]] = fir.if %[[VAL_13]] weights([0, 1]) -> (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]] = fir.shape %[[VAL_15]]#1, %[[VAL_16]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_18:.*]] = fir.alloca !fir.array<?x?x!fir.type<_QMmTt>>, %[[VAL_15]]#1, %[[VAL_16]]#1 {bindc_name = ".repacked"}
// CHECK:               %[[VAL_19:.*]] = fir.declare %[[VAL_18]](%[[VAL_17]]) {uniq_name = ".repacked"} : (!fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shape<2>) -> !fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_25:.*]] = fir.shape_shift %[[VAL_15]]#0, %[[VAL_15]]#1, %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_20:.*]] = fir.embox %[[VAL_19]](%[[VAL_25]]) source_box %[[VAL_0]] : (!fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shapeshift<2>, !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[BOX:.*]] = fir.convert %[[VAL_20]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_22:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[BOX]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_14]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = fir.declare %[[VAL_8]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest5_stackEx"} : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_28:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           fir.if %[[VAL_28]] {
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_31:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_33:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_32]] : index
// CHECK:             fir.if %[[VAL_33]] weights([0, 1]) {
// CHECK:               %[[VAL_34:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test polymorphic type array with heap allocation.
func.func @_QPtest6(%arg0: !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap innermost : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest6Ex"} : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
  fir.unpack_array %1 to %arg0 heap : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest6(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[ARG0]] innermost : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:             %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_4]] : i1
// CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.ref<!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:             %[[VAL_12:.*]] = fir.is_present %[[VAL_11]] : (!fir.ref<!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> i1
// CHECK:             %[[VAL_13:.*]] = arith.andi %[[VAL_10]], %[[VAL_12]] : i1
// CHECK:             %[[VAL_14:.*]] = fir.if %[[VAL_13]] weights([0, 1]) -> (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[ARG0]], %[[VAL_3]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[ARG0]], %[[VAL_2]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]] = fir.shape %[[VAL_15]]#1, %[[VAL_16]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_18:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_19:.*]] = fir.embox %[[VAL_18]](%[[VAL_17]]) source_box %[[ARG0]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shape<2>, !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:               fir.store %[[VAL_19]] to %[[VAL_5]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>
// CHECK:               %[[VAL_20:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:               %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_22:.*]] = fir.absent !fir.box<none>
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<none>) -> !fir.ref<i64>
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               %[[VAL_26:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_23]], %[[VAL_24]], %[[VAL_4]], %[[VAL_22]], %[[VAL_25]], %[[VAL_1]]) : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK:               %[[VAL_27:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>
// CHECK:               %[[VAL_28:.*]] = fir.declare %[[VAL_27]] {uniq_name = ".repacked"} : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:               %[[ADDR:.*]] = fir.box_addr %[[VAL_28]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_33:.*]] = fir.shape_shift %[[VAL_15]]#0, %[[VAL_15]]#1, %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_34:.*]] = fir.embox %[[ADDR]](%[[VAL_33]]) source_box %[[ARG0]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shapeshift<2>, !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_29:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_30:.*]] = fir.convert %[[VAL_34]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_31:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_32:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_30]], %[[VAL_31]], %[[VAL_32]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_34]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             } else {
// CHECK:               fir.result %[[ARG0]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_14]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           } else {
// CHECK:             fir.result %[[ARG0]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = fir.declare %[[VAL_8]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest6Ex"} : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_36:.*]] = fir.is_present %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           fir.if %[[VAL_36]] {
// CHECK:             %[[VAL_37:.*]] = fir.box_addr %[[VAL_8]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_38:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_38]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_41:.*]] = arith.cmpi ne, %[[VAL_39]], %[[VAL_40]] : index
// CHECK:             fir.if %[[VAL_41]] weights([0, 1]) {
// CHECK:               %[[VAL_42:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_43:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_44:.*]] = fir.convert %[[VAL_8]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_45:.*]] = fir.convert %[[VAL_42]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_43]], %[[VAL_44]], %[[VAL_45]], %[[VAL_0]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_37]] : !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test polymorphic type array with requested stack allocation.
// The actual allocation is done in heap memory.
func.func @_QPtest6_stack(%arg0: !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 stack innermost : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest6_stackEx"} : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
  fir.unpack_array %1 to %arg0 stack : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest6_stack(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[ARG0]] innermost : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:             %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_4]] : i1
// CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.ref<!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:             %[[VAL_12:.*]] = fir.is_present %[[VAL_11]] : (!fir.ref<!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> i1
// CHECK:             %[[VAL_13:.*]] = arith.andi %[[VAL_10]], %[[VAL_12]] : i1
// CHECK:             %[[VAL_14:.*]] = fir.if %[[VAL_13]] weights([0, 1]) -> (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[ARG0]], %[[VAL_3]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[ARG0]], %[[VAL_2]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]] = fir.shape %[[VAL_15]]#1, %[[VAL_16]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_18:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_19:.*]] = fir.embox %[[VAL_18]](%[[VAL_17]]) source_box %[[ARG0]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shape<2>, !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:               fir.store %[[VAL_19]] to %[[VAL_5]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>
// CHECK:               %[[VAL_20:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:               %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_22:.*]] = fir.absent !fir.box<none>
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<none>) -> !fir.ref<i64>
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               %[[VAL_26:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_23]], %[[VAL_24]], %[[VAL_4]], %[[VAL_22]], %[[VAL_25]], %[[VAL_1]]) : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK:               %[[VAL_27:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>
// CHECK:               %[[VAL_28:.*]] = fir.declare %[[VAL_27]] {uniq_name = ".repacked"} : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:               %[[ADDR:.*]] = fir.box_addr %[[VAL_28]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_33:.*]] = fir.shape_shift %[[VAL_15]]#0, %[[VAL_15]]#1, %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_34:.*]] = fir.embox %[[ADDR]](%[[VAL_33]]) source_box %[[ARG0]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shapeshift<2>, !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_29:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_30:.*]] = fir.convert %[[VAL_34]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_31:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_32:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_30]], %[[VAL_31]], %[[VAL_32]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_34]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             } else {
// CHECK:               fir.result %[[ARG0]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_14]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           } else {
// CHECK:             fir.result %[[ARG0]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = fir.declare %[[VAL_8]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest6_stackEx"} : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_36:.*]] = fir.is_present %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           fir.if %[[VAL_36]] {
// CHECK:             %[[VAL_37:.*]] = fir.box_addr %[[VAL_8]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_38:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_38]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_41:.*]] = arith.cmpi ne, %[[VAL_39]], %[[VAL_40]] : index
// CHECK:             fir.if %[[VAL_41]] weights([0, 1]) {
// CHECK:               %[[VAL_42:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_43:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_44:.*]] = fir.convert %[[VAL_8]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_45:.*]] = fir.convert %[[VAL_42]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_43]], %[[VAL_44]], %[[VAL_45]], %[[VAL_0]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_37]] : !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test unlimited polymorphic type array with heap allocation.
func.func @_QPtest7(%arg0: !fir.class<!fir.array<?x?xnone>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap innermost : (!fir.class<!fir.array<?x?xnone>>) -> !fir.class<!fir.array<?x?xnone>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest7Ex"} : (!fir.class<!fir.array<?x?xnone>>, !fir.dscope) -> !fir.class<!fir.array<?x?xnone>>
  fir.unpack_array %1 to %arg0 heap : !fir.class<!fir.array<?x?xnone>>
  return
}
// CHECK-LABEL:   func.func @_QPtest7(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.class<!fir.array<?x?xnone>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?xnone>>>
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.class<!fir.array<?x?xnone>>) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[ARG0]] innermost : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:             %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_4]] : i1
// CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.ref<!fir.class<!fir.array<?x?xnone>>>
// CHECK:             %[[VAL_12:.*]] = fir.is_present %[[VAL_11]] : (!fir.ref<!fir.class<!fir.array<?x?xnone>>>) -> i1
// CHECK:             %[[VAL_13:.*]] = arith.andi %[[VAL_10]], %[[VAL_12]] : i1
// CHECK:             %[[VAL_14:.*]] = fir.if %[[VAL_13]] weights([0, 1]) -> (!fir.class<!fir.array<?x?xnone>>) {
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[ARG0]], %[[VAL_3]] : (!fir.class<!fir.array<?x?xnone>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[ARG0]], %[[VAL_2]] : (!fir.class<!fir.array<?x?xnone>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]] = fir.shape %[[VAL_15]]#1, %[[VAL_16]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_18:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xnone>>
// CHECK:               %[[VAL_19:.*]] = fir.embox %[[VAL_18]](%[[VAL_17]]) source_box %[[ARG0]] : (!fir.heap<!fir.array<?x?xnone>>, !fir.shape<2>, !fir.class<!fir.array<?x?xnone>>) -> !fir.class<!fir.heap<!fir.array<?x?xnone>>>
// CHECK:               fir.store %[[VAL_19]] to %[[VAL_5]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
// CHECK:               %[[VAL_20:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:               %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_22:.*]] = fir.absent !fir.box<none>
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<none>) -> !fir.ref<i64>
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               %[[VAL_26:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_23]], %[[VAL_24]], %[[VAL_4]], %[[VAL_22]], %[[VAL_25]], %[[VAL_1]]) : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK:               %[[VAL_27:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
// CHECK:               %[[VAL_28:.*]] = fir.declare %[[VAL_27]] {uniq_name = ".repacked"} : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>) -> !fir.class<!fir.heap<!fir.array<?x?xnone>>>
// CHECK:               %[[ADDR:.*]] = fir.box_addr %[[VAL_28]] : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>) -> !fir.heap<!fir.array<?x?xnone>>
// CHECK:               %[[VAL_33:.*]] = fir.shape_shift %[[VAL_15]]#0, %[[VAL_15]]#1, %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_34:.*]] = fir.embox %[[ADDR]](%[[VAL_33]]) source_box %[[ARG0]] : (!fir.heap<!fir.array<?x?xnone>>, !fir.shapeshift<2>, !fir.class<!fir.array<?x?xnone>>) -> !fir.class<!fir.array<?x?xnone>>
// CHECK:               %[[VAL_29:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_30:.*]] = fir.convert %[[VAL_34]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_31:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_32:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_30]], %[[VAL_31]], %[[VAL_32]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_34]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:             } else {
// CHECK:               fir.result %[[ARG0]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_14]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:           } else {
// CHECK:             fir.result %[[ARG0]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = fir.declare %[[VAL_8]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest7Ex"} : (!fir.class<!fir.array<?x?xnone>>, !fir.dscope) -> !fir.class<!fir.array<?x?xnone>>
// CHECK:           %[[VAL_36:.*]] = fir.is_present %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:           fir.if %[[VAL_36]] {
// CHECK:             %[[VAL_37:.*]] = fir.box_addr %[[VAL_8]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.heap<!fir.array<?x?xnone>>
// CHECK:             %[[VAL_38:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.heap<!fir.array<?x?xnone>>
// CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (!fir.heap<!fir.array<?x?xnone>>) -> index
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_38]] : (!fir.heap<!fir.array<?x?xnone>>) -> index
// CHECK:             %[[VAL_41:.*]] = arith.cmpi ne, %[[VAL_39]], %[[VAL_40]] : index
// CHECK:             fir.if %[[VAL_41]] weights([0, 1]) {
// CHECK:               %[[VAL_42:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_43:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_44:.*]] = fir.convert %[[VAL_8]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_45:.*]] = fir.convert %[[VAL_42]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_43]], %[[VAL_44]], %[[VAL_45]], %[[VAL_0]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_37]] : !fir.heap<!fir.array<?x?xnone>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test unlimited polymorphic type array with requested stack allocation.
// The actual allocation is done in heap memory.
func.func @_QPtest7_stack(%arg0: !fir.class<!fir.array<?x?xnone>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 stack innermost : (!fir.class<!fir.array<?x?xnone>>) -> !fir.class<!fir.array<?x?xnone>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest7Ex"} : (!fir.class<!fir.array<?x?xnone>>, !fir.dscope) -> !fir.class<!fir.array<?x?xnone>>
  fir.unpack_array %1 to %arg0 stack : !fir.class<!fir.array<?x?xnone>>
  return
}
// CHECK-LABEL:   func.func @_QPtest7_stack(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.class<!fir.array<?x?xnone>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant
// CHECK:           %[[VAL_1:.*]] = arith.constant
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?xnone>>>
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:           %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.class<!fir.array<?x?xnone>>) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[ARG0]] innermost : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:             %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_4]] : i1
// CHECK:             %[[VAL_11:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.ref<!fir.class<!fir.array<?x?xnone>>>
// CHECK:             %[[VAL_12:.*]] = fir.is_present %[[VAL_11]] : (!fir.ref<!fir.class<!fir.array<?x?xnone>>>) -> i1
// CHECK:             %[[VAL_13:.*]] = arith.andi %[[VAL_10]], %[[VAL_12]] : i1
// CHECK:             %[[VAL_14:.*]] = fir.if %[[VAL_13]] weights([0, 1]) -> (!fir.class<!fir.array<?x?xnone>>) {
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[ARG0]], %[[VAL_3]] : (!fir.class<!fir.array<?x?xnone>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[ARG0]], %[[VAL_2]] : (!fir.class<!fir.array<?x?xnone>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]] = fir.shape %[[VAL_15]]#1, %[[VAL_16]]#1 : (index, index) -> !fir.shape<2>
// CHECK:               %[[VAL_18:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xnone>>
// CHECK:               %[[VAL_19:.*]] = fir.embox %[[VAL_18]](%[[VAL_17]]) source_box %[[ARG0]] : (!fir.heap<!fir.array<?x?xnone>>, !fir.shape<2>, !fir.class<!fir.array<?x?xnone>>) -> !fir.class<!fir.heap<!fir.array<?x?xnone>>>
// CHECK:               fir.store %[[VAL_19]] to %[[VAL_5]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
// CHECK:               %[[VAL_20:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:               %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_22:.*]] = fir.absent !fir.box<none>
// CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<none>) -> !fir.ref<i64>
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               %[[VAL_26:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_23]], %[[VAL_24]], %[[VAL_4]], %[[VAL_22]], %[[VAL_25]], %[[VAL_1]]) : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK:               %[[VAL_27:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
// CHECK:               %[[VAL_28:.*]] = fir.declare %[[VAL_27]] {uniq_name = ".repacked"} : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>) -> !fir.class<!fir.heap<!fir.array<?x?xnone>>>
// CHECK:               %[[ADDR:.*]] = fir.box_addr %[[VAL_28]] : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>) -> !fir.heap<!fir.array<?x?xnone>>
// CHECK:               %[[VAL_33:.*]] = fir.shape_shift %[[VAL_15]]#0, %[[VAL_15]]#1, %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:               %[[VAL_34:.*]] = fir.embox %[[ADDR]](%[[VAL_33]]) source_box %[[ARG0]] : (!fir.heap<!fir.array<?x?xnone>>, !fir.shapeshift<2>, !fir.class<!fir.array<?x?xnone>>) -> !fir.class<!fir.array<?x?xnone>>
// CHECK:               %[[VAL_29:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_30:.*]] = fir.convert %[[VAL_34]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_31:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_32:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_30]], %[[VAL_31]], %[[VAL_32]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.result %[[VAL_34]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:             } else {
// CHECK:               fir.result %[[ARG0]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:             }
// CHECK:             fir.result %[[VAL_14]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:           } else {
// CHECK:             fir.result %[[ARG0]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = fir.declare %[[VAL_8]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest7Ex"} : (!fir.class<!fir.array<?x?xnone>>, !fir.dscope) -> !fir.class<!fir.array<?x?xnone>>
// CHECK:           %[[VAL_36:.*]] = fir.is_present %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:           fir.if %[[VAL_36]] {
// CHECK:             %[[VAL_37:.*]] = fir.box_addr %[[VAL_8]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.heap<!fir.array<?x?xnone>>
// CHECK:             %[[VAL_38:.*]] = fir.box_addr %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.heap<!fir.array<?x?xnone>>
// CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (!fir.heap<!fir.array<?x?xnone>>) -> index
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_38]] : (!fir.heap<!fir.array<?x?xnone>>) -> index
// CHECK:             %[[VAL_41:.*]] = arith.cmpi ne, %[[VAL_39]], %[[VAL_40]] : index
// CHECK:             fir.if %[[VAL_41]] weights([0, 1]) {
// CHECK:               %[[VAL_42:.*]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:               %[[VAL_43:.*]] = fir.convert %[[ARG0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_44:.*]] = fir.convert %[[VAL_8]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_45:.*]] = fir.convert %[[VAL_42]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_43]], %[[VAL_44]], %[[VAL_45]], %[[VAL_0]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_37]] : !fir.heap<!fir.array<?x?xnone>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
