// RUN: fir-opt --lower-repack-arrays %s | FileCheck %s

// Test trivial type array repacking.
// CHECK-LABEL:   func.func @_QPtest1(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
func.func @_QPtest1(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           %[[VAL_8:.*]]:3 = fir.if %[[VAL_7]] -> (index, index, i1) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:             %[[VAL_10:.*]]:3 = fir.if %[[VAL_9]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_13:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
// CHECK:               %[[VAL_14:.*]] = fir.is_present %[[VAL_13]] : (!fir.ref<!fir.array<?x?xf32>>) -> i1
// CHECK:               fir.result %[[VAL_11]]#1, %[[VAL_12]]#1, %[[VAL_14]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_10]]#0, %[[VAL_10]]#1, %[[VAL_10]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = fir.if %[[VAL_8]]#2 -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:             %[[VAL_18:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_8]]#0, %[[VAL_8]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:             %[[VAL_19:.*]] = fir.shape %[[VAL_8]]#0, %[[VAL_8]]#1 : (index, index) -> !fir.shape<2>
// CHECK:             %[[VAL_20:.*]] = fir.embox %[[VAL_18]](%[[VAL_19]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
// CHECK:             %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.box<none>
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_25:.*]] = fir.rebox %[[VAL_20]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:             fir.result %[[VAL_25]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           }
  %1 = fir.pack_array %arg0 heap innermost : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_26:.*]] = fir.declare %[[VAL_16]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest1Ex"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest1Ex"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_27:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           fir.if %[[VAL_27]] {
// CHECK:             %[[VAL_28:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_28]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_32:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_31]] : index
// CHECK:             fir.if %[[VAL_32]] {
// CHECK:               %[[VAL_33:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_34:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_28]] : !fir.heap<!fir.array<?x?xf32>>
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?x?xf32>>
  return
}

// Test 'stack whole' repacking.
// CHECK-LABEL:   func.func @_QPtest1_whole(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
func.func @_QPtest1_whole(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           %[[VAL_8:.*]]:3 = fir.if %[[VAL_7]] -> (index, index, i1) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] whole : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:             %[[VAL_10:.*]]:3 = fir.if %[[VAL_9]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_13:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
// CHECK:               %[[VAL_14:.*]] = fir.is_present %[[VAL_13]] : (!fir.ref<!fir.array<?x?xf32>>) -> i1
// CHECK:               fir.result %[[VAL_11]]#1, %[[VAL_12]]#1, %[[VAL_14]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_10]]#0, %[[VAL_10]]#1, %[[VAL_10]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = fir.alloca !fir.array<?x?xf32>, %[[VAL_8]]#0, %[[VAL_8]]#1 {bindc_name = ".repacked"}
// CHECK:           %[[VAL_18:.*]] = fir.shape %[[VAL_8]]#0, %[[VAL_8]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_16]](%[[VAL_18]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_20:.*]] = fir.if %[[VAL_8]]#2 -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:             %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_19]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_25:.*]] = fir.rebox %[[VAL_19]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:             fir.result %[[VAL_25]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           }
  %1 = fir.pack_array %arg0 stack whole : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_26:.*]] = fir.declare %[[VAL_20]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest1_wholeEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest1_wholeEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_27:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           fir.if %[[VAL_27]] {
// CHECK:             %[[VAL_28:.*]] = fir.box_addr %[[VAL_20]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_28]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_32:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_31]] : index
// CHECK:             fir.if %[[VAL_32]] {
// CHECK:               %[[VAL_33:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_34:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 stack : !fir.box<!fir.array<?x?xf32>>
  return
}

// Test unpacking with no_copy.
// CHECK-LABEL:   func.func @_QPtest1_in(
// CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
func.func @_QPtest1_in(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant false
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           %[[VAL_7:.*]]:3 = fir.if %[[VAL_6]] -> (index, index, i1) {
// CHECK:             %[[VAL_8:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:             %[[VAL_9:.*]]:3 = fir.if %[[VAL_8]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_4]], %[[VAL_4]], %[[VAL_3]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_12:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
// CHECK:               %[[VAL_13:.*]] = fir.is_present %[[VAL_12]] : (!fir.ref<!fir.array<?x?xf32>>) -> i1
// CHECK:               fir.result %[[VAL_10]]#1, %[[VAL_11]]#1, %[[VAL_13]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_9]]#0, %[[VAL_9]]#1, %[[VAL_9]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_4]], %[[VAL_4]], %[[VAL_3]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = fir.if %[[VAL_7]]#2 -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:             %[[VAL_17:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_7]]#0, %[[VAL_7]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:             %[[VAL_18:.*]] = fir.shape %[[VAL_7]]#0, %[[VAL_7]]#1 : (index, index) -> !fir.shape<2>
// CHECK:             %[[VAL_19:.*]] = fir.embox %[[VAL_17]](%[[VAL_18]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
// CHECK:             %[[VAL_20:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.box<none>
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_24:.*]] = fir.rebox %[[VAL_19]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:             fir.result %[[VAL_24]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           }
  %1 = fir.pack_array %arg0 heap innermost : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_25:.*]] = fir.declare %[[VAL_15]] dummy_scope %[[VAL_5]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest1_inEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %2 = fir.declare %1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest1_inEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_26:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           fir.if %[[VAL_26]] {
// CHECK:             %[[VAL_27:.*]] = fir.box_addr %[[VAL_15]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_27]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_28:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_28]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_31:.*]] = arith.cmpi ne, %[[VAL_29]], %[[VAL_30]] : index
// CHECK:             fir.if %[[VAL_31]] {
// CHECK:               fir.freemem %[[VAL_27]] : !fir.heap<!fir.array<?x?xf32>>
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 heap no_copy : !fir.box<!fir.array<?x?xf32>>
  return
}

// Test packing with no_copy.
// CHECK-LABEL:   func.func @_QPtest1_out(
// CHECK-SAME:                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
func.func @_QPtest1_out(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant false
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           %[[VAL_7:.*]]:3 = fir.if %[[VAL_6]] -> (index, index, i1) {
// CHECK:             %[[VAL_8:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:             %[[VAL_9:.*]]:3 = fir.if %[[VAL_8]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_4]], %[[VAL_4]], %[[VAL_3]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_12:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
// CHECK:               %[[VAL_13:.*]] = fir.is_present %[[VAL_12]] : (!fir.ref<!fir.array<?x?xf32>>) -> i1
// CHECK:               fir.result %[[VAL_10]]#1, %[[VAL_11]]#1, %[[VAL_13]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_9]]#0, %[[VAL_9]]#1, %[[VAL_9]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_4]], %[[VAL_4]], %[[VAL_3]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = fir.if %[[VAL_7]]#2 -> (!fir.box<!fir.array<?x?xf32>>) {
// CHECK:             %[[VAL_17:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_7]]#0, %[[VAL_7]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:             %[[VAL_18:.*]] = fir.shape %[[VAL_7]]#0, %[[VAL_7]]#1 : (index, index) -> !fir.shape<2>
// CHECK:             %[[VAL_19:.*]] = fir.embox %[[VAL_17]](%[[VAL_18]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
// CHECK:             %[[VAL_20:.*]] = fir.rebox %[[VAL_19]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:             fir.result %[[VAL_20]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>
// CHECK:           }
  %1 = fir.pack_array %arg0 heap innermost no_copy : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_21:.*]] = fir.declare %[[VAL_15]] dummy_scope %[[VAL_5]] {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFtest1_outEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %2 = fir.declare %1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFtest1_outEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_22:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> i1
// CHECK:           fir.if %[[VAL_22]] {
// CHECK:             %[[VAL_23:.*]] = fir.box_addr %[[VAL_15]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_23]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_24:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (!fir.heap<!fir.array<?x?xf32>>) -> index
// CHECK:             %[[VAL_27:.*]] = arith.cmpi ne, %[[VAL_25]], %[[VAL_26]] : index
// CHECK:             fir.if %[[VAL_27]] {
// CHECK:               %[[VAL_28:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_29:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_30:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:               %[[VAL_31:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_29]], %[[VAL_30]], %[[VAL_31]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_23]] : !fir.heap<!fir.array<?x?xf32>>
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?x?xf32>>
  return
}

// Test character array with dynamic length and heap allocation
// CHECK-LABEL:   func.func @_QPtest2(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:                        %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
func.func @_QPtest2(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest2En"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtest2En"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
  %2 = fir.load %1 : !fir.ref<i32>
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_7]] : i32
  %3 = arith.cmpi sgt, %2, %c0_i32 : i32
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_7]] : i32
  %4 = arith.select %3, %2, %c0_i32 : i32
// CHECK:           %[[VAL_13:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_14:.*]]:3 = fir.if %[[VAL_13]] -> (index, index, i1) {
// CHECK:             %[[VAL_15:.*]] = fir.is_contiguous_box %[[VAL_1]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:             %[[VAL_16:.*]]:3 = fir.if %[[VAL_15]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_6]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_19:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_20:.*]] = fir.is_present %[[VAL_19]] : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:               fir.result %[[VAL_17]]#1, %[[VAL_18]]#1, %[[VAL_20]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_16]]#0, %[[VAL_16]]#1, %[[VAL_16]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = fir.if %[[VAL_14]]#2 -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:             %[[VAL_24:.*]] = fir.allocmem !fir.array<?x?x!fir.char<1,?>>(%[[VAL_12]] : i32), %[[VAL_14]]#0, %[[VAL_14]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:             %[[VAL_25:.*]] = fir.shape %[[VAL_14]]#0, %[[VAL_14]]#1 : (index, index) -> !fir.shape<2>
// CHECK:             %[[VAL_26:.*]] = fir.embox %[[VAL_24]](%[[VAL_25]]) typeparams %[[VAL_12]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, i32) -> !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>
// CHECK:             %[[VAL_27:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_26]] : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>) -> !fir.box<none>
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_27]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_3]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_31:.*]] = fir.rebox %[[VAL_26]] : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             fir.result %[[VAL_31]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_1]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           }
  %5 = fir.pack_array %arg1 heap innermost typeparams %4 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_32:.*]] = fir.declare %[[VAL_22]] typeparams %[[VAL_12]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  %6 = fir.declare %5 typeparams %4 dummy_scope %0 {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_33:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           fir.if %[[VAL_33]] {
// CHECK:             %[[VAL_34:.*]] = fir.box_addr %[[VAL_22]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_34]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_35:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_35]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_38:.*]] = arith.cmpi ne, %[[VAL_36]], %[[VAL_37]] : index
// CHECK:             fir.if %[[VAL_38]] {
// CHECK:               %[[VAL_39:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_40:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_41:.*]] = fir.convert %[[VAL_22]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_42:.*]] = fir.convert %[[VAL_39]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_40]], %[[VAL_41]], %[[VAL_42]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_34]] : !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             }
// CHECK:           }
  fir.unpack_array %5 to %arg1 heap : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  return
}

// Test character array with dynamic length and stack allocation
// CHECK-LABEL:   func.func @_QPtest2_stack(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:                              %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
func.func @_QPtest2_stack(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest2_stackEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtest2_stackEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
  %2 = fir.load %1 : !fir.ref<i32>
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_7]] : i32
  %3 = arith.cmpi sgt, %2, %c0_i32 : i32
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_7]] : i32
  %4 = arith.select %3, %2, %c0_i32 : i32
// CHECK:           %[[VAL_13:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_14:.*]]:3 = fir.if %[[VAL_13]] -> (index, index, i1) {
// CHECK:             %[[VAL_15:.*]] = fir.is_contiguous_box %[[VAL_1]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:             %[[VAL_16:.*]]:3 = fir.if %[[VAL_15]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_6]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_19:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_20:.*]] = fir.is_present %[[VAL_19]] : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:               fir.result %[[VAL_17]]#1, %[[VAL_18]]#1, %[[VAL_20]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_16]]#0, %[[VAL_16]]#1, %[[VAL_16]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = fir.alloca !fir.array<?x?x!fir.char<1,?>>(%[[VAL_12]] : i32), %[[VAL_14]]#0, %[[VAL_14]]#1 {bindc_name = ".repacked"}
// CHECK:           %[[VAL_24:.*]] = fir.shape %[[VAL_14]]#0, %[[VAL_14]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_25:.*]] = fir.embox %[[VAL_22]](%[[VAL_24]]) typeparams %[[VAL_12]] : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, i32) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_26:.*]] = fir.if %[[VAL_14]]#2 -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:             %[[VAL_27:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_25]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_27]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_3]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_31:.*]] = fir.rebox %[[VAL_25]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             fir.result %[[VAL_31]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_1]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           }
  %5 = fir.pack_array %arg1 stack innermost typeparams %4 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_32:.*]] = fir.declare %[[VAL_26]] typeparams %[[VAL_12]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest2_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  %6 = fir.declare %5 typeparams %4 dummy_scope %0 {uniq_name = "_QFtest2_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_33:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           fir.if %[[VAL_33]] {
// CHECK:             %[[VAL_34:.*]] = fir.box_addr %[[VAL_26]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_34]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_35:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_35]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_38:.*]] = arith.cmpi ne, %[[VAL_36]], %[[VAL_37]] : index
// CHECK:             fir.if %[[VAL_38]] {
// CHECK:               %[[VAL_39:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_40:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_41:.*]] = fir.convert %[[VAL_26]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_42:.*]] = fir.convert %[[VAL_39]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_40]], %[[VAL_41]], %[[VAL_42]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
  fir.unpack_array %5 to %arg1 stack : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  return
}

// Test character array with assumed length and heap allocation.
// CHECK-LABEL:   func.func @_QPtest3(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
func.func @_QPtest3(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_8:.*]]:4 = fir.if %[[VAL_7]] -> (index, index, index, i1) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:             %[[VAL_10:.*]]:4 = fir.if %[[VAL_9]] -> (index, index, index, i1) {
// CHECK:               fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_13:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:               %[[VAL_14:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_15:.*]] = fir.is_present %[[VAL_14]] : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:               fir.result %[[VAL_11]]#1, %[[VAL_12]]#1, %[[VAL_13]], %[[VAL_15]] : index, index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_10]]#0, %[[VAL_10]]#1, %[[VAL_10]]#2, %[[VAL_10]]#3 : index, index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = fir.if %[[VAL_8]]#3 -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:             %[[VAL_19:.*]] = fir.allocmem !fir.array<?x?x!fir.char<1,?>>(%[[VAL_8]]#2 : index), %[[VAL_8]]#0, %[[VAL_8]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:             %[[VAL_20:.*]] = fir.shape %[[VAL_8]]#0, %[[VAL_8]]#1 : (index, index) -> !fir.shape<2>
// CHECK:             %[[VAL_21:.*]] = fir.embox %[[VAL_19]](%[[VAL_20]]) typeparams %[[VAL_8]]#2 : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, index) -> !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>
// CHECK:             %[[VAL_22:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>) -> !fir.box<none>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_26:.*]] = fir.rebox %[[VAL_21]] : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             fir.result %[[VAL_26]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           }
  %1 = fir.pack_array %arg0 heap innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_27:.*]] = fir.declare %[[VAL_17]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest3Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest3Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_28:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           fir.if %[[VAL_28]] {
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_30:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_33:.*]] = arith.cmpi ne, %[[VAL_31]], %[[VAL_32]] : index
// CHECK:             fir.if %[[VAL_33]] {
// CHECK:               %[[VAL_34:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_17]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_29]] : !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  return
}

// Test character array with assumed length and stack allocation.
// CHECK-LABEL:   func.func @_QPtest3_stack(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
func.func @_QPtest3_stack(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_8:.*]]:4 = fir.if %[[VAL_7]] -> (index, index, index, i1) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:             %[[VAL_10:.*]]:4 = fir.if %[[VAL_9]] -> (index, index, index, i1) {
// CHECK:               fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_13:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:               %[[VAL_14:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:               %[[VAL_15:.*]] = fir.is_present %[[VAL_14]] : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:               fir.result %[[VAL_11]]#1, %[[VAL_12]]#1, %[[VAL_13]], %[[VAL_15]] : index, index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_10]]#0, %[[VAL_10]]#1, %[[VAL_10]]#2, %[[VAL_10]]#3 : index, index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = fir.alloca !fir.array<?x?x!fir.char<1,?>>(%[[VAL_8]]#2 : index), %[[VAL_8]]#0, %[[VAL_8]]#1 {bindc_name = ".repacked"}
// CHECK:           %[[VAL_19:.*]] = fir.shape %[[VAL_8]]#0, %[[VAL_8]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_20:.*]] = fir.embox %[[VAL_17]](%[[VAL_19]]) typeparams %[[VAL_8]]#2 : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, index) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_21:.*]] = fir.if %[[VAL_8]]#3 -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:             %[[VAL_22:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_26:.*]] = fir.rebox %[[VAL_20]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             fir.result %[[VAL_26]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           }
  %1 = fir.pack_array %arg0 stack innermost : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_27:.*]] = fir.declare %[[VAL_21]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest3_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest3_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_28:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> i1
// CHECK:           fir.if %[[VAL_28]] {
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_21]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_30:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>) -> index
// CHECK:             %[[VAL_33:.*]] = arith.cmpi ne, %[[VAL_31]], %[[VAL_32]] : index
// CHECK:             fir.if %[[VAL_33]] {
// CHECK:               %[[VAL_34:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 stack : !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  return
}

// Test character array with constant length and heap allocation.
// CHECK-LABEL:   func.func @_QPtest4(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,10>>> {fir.bindc_name = "x"}) {
func.func @_QPtest4(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,10>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:           %[[VAL_9:.*]]:3 = fir.if %[[VAL_8]] -> (index, index, i1) {
// CHECK:             %[[VAL_10:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:             %[[VAL_11:.*]]:3 = fir.if %[[VAL_10]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_14:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:               %[[VAL_15:.*]] = fir.is_present %[[VAL_14]] : (!fir.ref<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:               fir.result %[[VAL_12]]#1, %[[VAL_13]]#1, %[[VAL_15]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_11]]#0, %[[VAL_11]]#1, %[[VAL_11]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = fir.if %[[VAL_9]]#2 -> (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) {
// CHECK:             %[[VAL_19:.*]] = fir.allocmem !fir.array<?x?x!fir.char<1,10>>, %[[VAL_9]]#0, %[[VAL_9]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:             %[[VAL_20:.*]] = fir.shape %[[VAL_9]]#0, %[[VAL_9]]#1 : (index, index) -> !fir.shape<2>
// CHECK:             %[[VAL_21:.*]] = fir.embox %[[VAL_19]](%[[VAL_20]]) : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,10>>>>
// CHECK:             %[[VAL_22:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,10>>>>) -> !fir.box<none>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_26:.*]] = fir.rebox %[[VAL_21]] : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,10>>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             fir.result %[[VAL_26]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           }
  %1 = fir.pack_array %arg0 heap innermost : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           %[[VAL_27:.*]] = fir.declare %[[VAL_17]] typeparams %[[VAL_6]] dummy_scope %[[VAL_7]] {uniq_name = "_QFtest4Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  %2 = fir.declare %1 typeparams %c10 dummy_scope %0 {uniq_name = "_QFtest4Ex"} : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           %[[VAL_28:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:           fir.if %[[VAL_28]] {
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>) -> index
// CHECK:             %[[VAL_30:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>) -> index
// CHECK:             %[[VAL_33:.*]] = arith.cmpi ne, %[[VAL_31]], %[[VAL_32]] : index
// CHECK:             fir.if %[[VAL_33]] {
// CHECK:               %[[VAL_34:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_17]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_29]] : !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  return
}

// Test character array with constant length and stack allocation.
// CHECK-LABEL:   func.func @_QPtest4_stack(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.char<1,10>>> {fir.bindc_name = "x"}) {
func.func @_QPtest4_stack(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,10>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:           %[[VAL_9:.*]]:3 = fir.if %[[VAL_8]] -> (index, index, i1) {
// CHECK:             %[[VAL_10:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:             %[[VAL_11:.*]]:3 = fir.if %[[VAL_10]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_14:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:               %[[VAL_15:.*]] = fir.is_present %[[VAL_14]] : (!fir.ref<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:               fir.result %[[VAL_12]]#1, %[[VAL_13]]#1, %[[VAL_15]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_11]]#0, %[[VAL_11]]#1, %[[VAL_11]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = fir.alloca !fir.array<?x?x!fir.char<1,10>>, %[[VAL_9]]#0, %[[VAL_9]]#1 {bindc_name = ".repacked"}
// CHECK:           %[[VAL_19:.*]] = fir.shape %[[VAL_9]]#0, %[[VAL_9]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_20:.*]] = fir.embox %[[VAL_17]](%[[VAL_19]]) : (!fir.ref<!fir.array<?x?x!fir.char<1,10>>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           %[[VAL_21:.*]] = fir.if %[[VAL_9]]#2 -> (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) {
// CHECK:             %[[VAL_22:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_26:.*]] = fir.rebox %[[VAL_20]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             fir.result %[[VAL_26]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           }
  %1 = fir.pack_array %arg0 stack innermost : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           %[[VAL_27:.*]] = fir.declare %[[VAL_21]] typeparams %[[VAL_6]] dummy_scope %[[VAL_7]] {uniq_name = "_QFtest4_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  %2 = fir.declare %1 typeparams %c10 dummy_scope %0 {uniq_name = "_QFtest4_stackEx"} : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           %[[VAL_28:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> i1
// CHECK:           fir.if %[[VAL_28]] {
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_21]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>) -> index
// CHECK:             %[[VAL_30:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<!fir.array<?x?x!fir.char<1,10>>>) -> index
// CHECK:             %[[VAL_33:.*]] = arith.cmpi ne, %[[VAL_31]], %[[VAL_32]] : index
// CHECK:             fir.if %[[VAL_33]] {
// CHECK:               %[[VAL_34:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>) -> !fir.box<none>
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 stack : !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  return
}

// Test derived type array with heap allocation.
// CHECK-LABEL:   func.func @_QPtest5(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
func.func @_QPtest5(%arg0: !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           %[[VAL_8:.*]]:3 = fir.if %[[VAL_7]] -> (index, index, i1) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:             %[[VAL_10:.*]]:3 = fir.if %[[VAL_9]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_13:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_14:.*]] = fir.is_present %[[VAL_13]] : (!fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:               fir.result %[[VAL_11]]#1, %[[VAL_12]]#1, %[[VAL_14]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_10]]#0, %[[VAL_10]]#1, %[[VAL_10]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = fir.if %[[VAL_8]]#2 -> (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:             %[[VAL_18:.*]] = fir.allocmem !fir.array<?x?x!fir.type<_QMmTt>>, %[[VAL_8]]#0, %[[VAL_8]]#1 {bindc_name = ".repacked", uniq_name = ""}
// CHECK:             %[[VAL_19:.*]] = fir.shape %[[VAL_8]]#0, %[[VAL_8]]#1 : (index, index) -> !fir.shape<2>
// CHECK:             %[[VAL_20:.*]] = fir.embox %[[VAL_18]](%[[VAL_19]]) : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:             %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> !fir.box<none>
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_25:.*]] = fir.rebox %[[VAL_20]] : (!fir.box<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             fir.result %[[VAL_25]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           }
  %1 = fir.pack_array %arg0 heap innermost : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_26:.*]] = fir.declare %[[VAL_16]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest5Ex"} : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest5Ex"} : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_27:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           fir.if %[[VAL_27]] {
// CHECK:             %[[VAL_28:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_28]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_32:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_31]] : index
// CHECK:             fir.if %[[VAL_32]] {
// CHECK:               %[[VAL_33:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_34:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_28]] : !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
  return
}

// Test derived type array with stack allocation.
// CHECK-LABEL:   func.func @_QPtest5_stack(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
func.func @_QPtest5_stack(%arg0: !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           %[[VAL_8:.*]]:3 = fir.if %[[VAL_7]] -> (index, index, i1) {
// CHECK:             %[[VAL_9:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:             %[[VAL_10:.*]]:3 = fir.if %[[VAL_9]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_13:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_14:.*]] = fir.is_present %[[VAL_13]] : (!fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:               fir.result %[[VAL_11]]#1, %[[VAL_12]]#1, %[[VAL_14]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_10]]#0, %[[VAL_10]]#1, %[[VAL_10]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = fir.alloca !fir.array<?x?x!fir.type<_QMmTt>>, %[[VAL_8]]#0, %[[VAL_8]]#1 {bindc_name = ".repacked"}
// CHECK:           %[[VAL_18:.*]] = fir.shape %[[VAL_8]]#0, %[[VAL_8]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_16]](%[[VAL_18]]) : (!fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_20:.*]] = fir.if %[[VAL_8]]#2 -> (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:             %[[VAL_21:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_19]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_25:.*]] = fir.rebox %[[VAL_19]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             fir.result %[[VAL_25]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           }
  %1 = fir.pack_array %arg0 stack innermost : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_26:.*]] = fir.declare %[[VAL_20]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest5_stackEx"} : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest5_stackEx"} : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_27:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           fir.if %[[VAL_27]] {
// CHECK:             %[[VAL_28:.*]] = fir.box_addr %[[VAL_20]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_28]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_29:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_32:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_31]] : index
// CHECK:             fir.if %[[VAL_32]] {
// CHECK:               %[[VAL_33:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_34:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 stack : !fir.box<!fir.array<?x?x!fir.type<_QMmTt>>>
  return
}

// Test polymorphic type array with heap allocation.
// CHECK-LABEL:   func.func @_QPtest6(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
func.func @_QPtest6(%arg0: !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.is_present %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           %[[VAL_10:.*]]:3 = fir.if %[[VAL_9]] -> (index, index, i1) {
// CHECK:             %[[VAL_11:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:             %[[VAL_12:.*]]:3 = fir.if %[[VAL_11]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_15:.*]] = fir.box_addr %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_16:.*]] = fir.is_present %[[VAL_15]] : (!fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:               fir.result %[[VAL_13]]#1, %[[VAL_14]]#1, %[[VAL_16]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_12]]#0, %[[VAL_12]]#1, %[[VAL_12]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = fir.if %[[VAL_10]]#2 -> (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:             %[[VAL_20:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_21:.*]] = fir.shape %[[VAL_6]], %[[VAL_6]] : (index, index) -> !fir.shape<2>
// CHECK:             %[[VAL_22:.*]] = fir.embox %[[VAL_20]](%[[VAL_21]]) : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shape<2>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:             fir.store %[[VAL_22]] to %[[VAL_7]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:             fir.call @_FortranAAllocatableApplyMold(%[[VAL_23]], %[[VAL_24]], %[[VAL_3]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_6]] : (index) -> i32
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
// CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_10]]#0 : (index) -> i64
// CHECK:             fir.call @_FortranAAllocatableSetBounds(%[[VAL_25]], %[[VAL_26]], %[[VAL_27]], %[[VAL_28]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_4]] : (index) -> i32
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_10]]#1 : (index) -> i64
// CHECK:             fir.call @_FortranAAllocatableSetBounds(%[[VAL_29]], %[[VAL_30]], %[[VAL_31]], %[[VAL_32]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
// CHECK:             %[[VAL_33:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_34:.*]] = fir.absent !fir.box<none>
// CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_37:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_35]], %[[VAL_5]], %[[VAL_34]], %[[VAL_36]], %[[VAL_2]]) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK:             %[[VAL_38:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>
// CHECK:             %[[VAL_39:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_38]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> !fir.box<none>
// CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:             %[[VAL_42:.*]] = fir.convert %[[VAL_39]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_40]], %[[VAL_41]], %[[VAL_42]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_43:.*]] = fir.rebox %[[VAL_38]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             fir.result %[[VAL_43]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           }
  %1 = fir.pack_array %arg0 heap innermost : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_44:.*]] = fir.declare %[[VAL_18]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest6Ex"} : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest6Ex"} : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_45:.*]] = fir.is_present %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           fir.if %[[VAL_45]] {
// CHECK:             %[[VAL_46:.*]] = fir.box_addr %[[VAL_18]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_46]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_47:.*]] = fir.box_addr %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_49:.*]] = fir.convert %[[VAL_47]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_50:.*]] = arith.cmpi ne, %[[VAL_48]], %[[VAL_49]] : index
// CHECK:             fir.if %[[VAL_50]] {
// CHECK:               %[[VAL_51:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_52:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_53:.*]] = fir.convert %[[VAL_18]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_51]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_52]], %[[VAL_53]], %[[VAL_54]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_46]] : !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 heap : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
  return
}

// Test polymorphic type array with stack allocation.
// CHECK-LABEL:   func.func @_QPtest6_stack(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
func.func @_QPtest6_stack(%arg0: !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.is_present %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           %[[VAL_10:.*]]:3 = fir.if %[[VAL_9]] -> (index, index, i1) {
// CHECK:             %[[VAL_11:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:             %[[VAL_12:.*]]:3 = fir.if %[[VAL_11]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_15:.*]] = fir.box_addr %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:               %[[VAL_16:.*]] = fir.is_present %[[VAL_15]] : (!fir.ref<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:               fir.result %[[VAL_13]]#1, %[[VAL_14]]#1, %[[VAL_16]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_12]]#0, %[[VAL_12]]#1, %[[VAL_12]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = fir.if %[[VAL_10]]#2 -> (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) {
// CHECK:             %[[VAL_20:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_21:.*]] = fir.shape %[[VAL_6]], %[[VAL_6]] : (index, index) -> !fir.shape<2>
// CHECK:             %[[VAL_22:.*]] = fir.embox %[[VAL_20]](%[[VAL_21]]) : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.shape<2>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>
// CHECK:             fir.store %[[VAL_22]] to %[[VAL_7]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:             fir.call @_FortranAAllocatableApplyMold(%[[VAL_23]], %[[VAL_24]], %[[VAL_3]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_6]] : (index) -> i32
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
// CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_10]]#0 : (index) -> i64
// CHECK:             fir.call @_FortranAAllocatableSetBounds(%[[VAL_25]], %[[VAL_26]], %[[VAL_27]], %[[VAL_28]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_4]] : (index) -> i32
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_10]]#1 : (index) -> i64
// CHECK:             fir.call @_FortranAAllocatableSetBounds(%[[VAL_29]], %[[VAL_30]], %[[VAL_31]], %[[VAL_32]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
// CHECK:             %[[VAL_33:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_34:.*]] = fir.absent !fir.box<none>
// CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_37:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_35]], %[[VAL_5]], %[[VAL_34]], %[[VAL_36]], %[[VAL_2]]) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK:             %[[VAL_38:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>>
// CHECK:             %[[VAL_39:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_38]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> !fir.box<none>
// CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:             %[[VAL_42:.*]] = fir.convert %[[VAL_39]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_40]], %[[VAL_41]], %[[VAL_42]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_43:.*]] = fir.rebox %[[VAL_38]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>>) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             fir.result %[[VAL_43]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           }
  %1 = fir.pack_array %arg0 stack innermost : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_44:.*]] = fir.declare %[[VAL_18]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest6_stackEx"} : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest6_stackEx"} : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>, !fir.dscope) -> !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:           %[[VAL_45:.*]] = fir.is_present %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> i1
// CHECK:           fir.if %[[VAL_45]] {
// CHECK:             %[[VAL_46:.*]] = fir.box_addr %[[VAL_18]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_46]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_47:.*]] = fir.box_addr %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>
// CHECK:             %[[VAL_49:.*]] = fir.convert %[[VAL_47]] : (!fir.heap<!fir.array<?x?x!fir.type<_QMmTt>>>) -> index
// CHECK:             %[[VAL_50:.*]] = arith.cmpi ne, %[[VAL_48]], %[[VAL_49]] : index
// CHECK:             fir.if %[[VAL_50]] {
// CHECK:               %[[VAL_51:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_52:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_53:.*]] = fir.convert %[[VAL_18]] : (!fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>) -> !fir.box<none>
// CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_51]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_52]], %[[VAL_53]], %[[VAL_54]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 stack : !fir.class<!fir.array<?x?x!fir.type<_QMmTt>>>
  return
}

// Test unlimited polymorphic type array with heap allocation.
// CHECK-LABEL:   func.func @_QPtest7(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.class<!fir.array<?x?xnone>> {fir.bindc_name = "x"}) {
func.func @_QPtest7(%arg0: !fir.class<!fir.array<?x?xnone>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?xnone>>>
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.is_present %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:           %[[VAL_10:.*]]:3 = fir.if %[[VAL_9]] -> (index, index, i1) {
// CHECK:             %[[VAL_11:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:             %[[VAL_12:.*]]:3 = fir.if %[[VAL_11]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.class<!fir.array<?x?xnone>>, index) -> (index, index, index)
// CHECK:               %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.class<!fir.array<?x?xnone>>, index) -> (index, index, index)
// CHECK:               %[[VAL_15:.*]] = fir.box_addr %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.ref<!fir.array<?x?xnone>>
// CHECK:               %[[VAL_16:.*]] = fir.is_present %[[VAL_15]] : (!fir.ref<!fir.array<?x?xnone>>) -> i1
// CHECK:               fir.result %[[VAL_13]]#1, %[[VAL_14]]#1, %[[VAL_16]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_12]]#0, %[[VAL_12]]#1, %[[VAL_12]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = fir.if %[[VAL_10]]#2 -> (!fir.class<!fir.array<?x?xnone>>) {
// CHECK:             %[[VAL_20:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xnone>>
// CHECK:             %[[VAL_21:.*]] = fir.shape %[[VAL_6]], %[[VAL_6]] : (index, index) -> !fir.shape<2>
// CHECK:             %[[VAL_22:.*]] = fir.embox %[[VAL_20]](%[[VAL_21]]) : (!fir.heap<!fir.array<?x?xnone>>, !fir.shape<2>) -> !fir.class<!fir.heap<!fir.array<?x?xnone>>>
// CHECK:             fir.store %[[VAL_22]] to %[[VAL_7]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:             fir.call @_FortranAAllocatableApplyMold(%[[VAL_23]], %[[VAL_24]], %[[VAL_3]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_6]] : (index) -> i32
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
// CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_10]]#0 : (index) -> i64
// CHECK:             fir.call @_FortranAAllocatableSetBounds(%[[VAL_25]], %[[VAL_26]], %[[VAL_27]], %[[VAL_28]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_4]] : (index) -> i32
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_10]]#1 : (index) -> i64
// CHECK:             fir.call @_FortranAAllocatableSetBounds(%[[VAL_29]], %[[VAL_30]], %[[VAL_31]], %[[VAL_32]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
// CHECK:             %[[VAL_33:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_34:.*]] = fir.absent !fir.box<none>
// CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_37:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_35]], %[[VAL_5]], %[[VAL_34]], %[[VAL_36]], %[[VAL_2]]) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK:             %[[VAL_38:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
// CHECK:             %[[VAL_39:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_38]] : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>) -> !fir.box<none>
// CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:             %[[VAL_42:.*]] = fir.convert %[[VAL_39]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_40]], %[[VAL_41]], %[[VAL_42]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_43:.*]] = fir.rebox %[[VAL_38]] : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>) -> !fir.class<!fir.array<?x?xnone>>
// CHECK:             fir.result %[[VAL_43]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:           }
  %1 = fir.pack_array %arg0 heap innermost : (!fir.class<!fir.array<?x?xnone>>) -> !fir.class<!fir.array<?x?xnone>>
// CHECK:           %[[VAL_44:.*]] = fir.declare %[[VAL_18]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest7Ex"} : (!fir.class<!fir.array<?x?xnone>>, !fir.dscope) -> !fir.class<!fir.array<?x?xnone>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest7Ex"} : (!fir.class<!fir.array<?x?xnone>>, !fir.dscope) -> !fir.class<!fir.array<?x?xnone>>
// CHECK:           %[[VAL_45:.*]] = fir.is_present %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:           fir.if %[[VAL_45]] {
// CHECK:             %[[VAL_46:.*]] = fir.box_addr %[[VAL_18]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.heap<!fir.array<?x?xnone>>
// CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_46]] : (!fir.heap<!fir.array<?x?xnone>>) -> index
// CHECK:             %[[VAL_47:.*]] = fir.box_addr %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.heap<!fir.array<?x?xnone>>
// CHECK:             %[[VAL_49:.*]] = fir.convert %[[VAL_47]] : (!fir.heap<!fir.array<?x?xnone>>) -> index
// CHECK:             %[[VAL_50:.*]] = arith.cmpi ne, %[[VAL_48]], %[[VAL_49]] : index
// CHECK:             fir.if %[[VAL_50]] {
// CHECK:               %[[VAL_51:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_52:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_53:.*]] = fir.convert %[[VAL_18]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_51]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_52]], %[[VAL_53]], %[[VAL_54]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:               fir.freemem %[[VAL_46]] : !fir.heap<!fir.array<?x?xnone>>
// CHECK:             }
// CHECK:           }
  fir.unpack_array %1 to %arg0 heap : !fir.class<!fir.array<?x?xnone>>
  return
}

// Test unlimited polymorphic type array with stack allocation.
// CHECK-LABEL:   func.func @_QPtest7_stack(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.class<!fir.array<?x?xnone>> {fir.bindc_name = "x"}) {
func.func @_QPtest7_stack(%arg0: !fir.class<!fir.array<?x?xnone>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?xnone>>>
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
  %0 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.is_present %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:           %[[VAL_10:.*]]:3 = fir.if %[[VAL_9]] -> (index, index, i1) {
// CHECK:             %[[VAL_11:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:             %[[VAL_12:.*]]:3 = fir.if %[[VAL_11]] -> (index, index, i1) {
// CHECK:               fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:             } else {
// CHECK:               %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.class<!fir.array<?x?xnone>>, index) -> (index, index, index)
// CHECK:               %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.class<!fir.array<?x?xnone>>, index) -> (index, index, index)
// CHECK:               %[[VAL_15:.*]] = fir.box_addr %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.ref<!fir.array<?x?xnone>>
// CHECK:               %[[VAL_16:.*]] = fir.is_present %[[VAL_15]] : (!fir.ref<!fir.array<?x?xnone>>) -> i1
// CHECK:               fir.result %[[VAL_13]]#1, %[[VAL_14]]#1, %[[VAL_16]] : index, index, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_12]]#0, %[[VAL_12]]#1, %[[VAL_12]]#2 : index, index, i1
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_5]] : index, index, i1
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = fir.if %[[VAL_10]]#2 -> (!fir.class<!fir.array<?x?xnone>>) {
// CHECK:             %[[VAL_20:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xnone>>
// CHECK:             %[[VAL_21:.*]] = fir.shape %[[VAL_6]], %[[VAL_6]] : (index, index) -> !fir.shape<2>
// CHECK:             %[[VAL_22:.*]] = fir.embox %[[VAL_20]](%[[VAL_21]]) : (!fir.heap<!fir.array<?x?xnone>>, !fir.shape<2>) -> !fir.class<!fir.heap<!fir.array<?x?xnone>>>
// CHECK:             fir.store %[[VAL_22]] to %[[VAL_7]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:             fir.call @_FortranAAllocatableApplyMold(%[[VAL_23]], %[[VAL_24]], %[[VAL_3]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_6]] : (index) -> i32
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
// CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_10]]#0 : (index) -> i64
// CHECK:             fir.call @_FortranAAllocatableSetBounds(%[[VAL_25]], %[[VAL_26]], %[[VAL_27]], %[[VAL_28]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_4]] : (index) -> i32
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_10]]#1 : (index) -> i64
// CHECK:             fir.call @_FortranAAllocatableSetBounds(%[[VAL_29]], %[[VAL_30]], %[[VAL_31]], %[[VAL_32]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
// CHECK:             %[[VAL_33:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_34:.*]] = fir.absent !fir.box<none>
// CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_37:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_35]], %[[VAL_5]], %[[VAL_34]], %[[VAL_36]], %[[VAL_2]]) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK:             %[[VAL_38:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
// CHECK:             %[[VAL_39:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_38]] : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>) -> !fir.box<none>
// CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:             %[[VAL_42:.*]] = fir.convert %[[VAL_39]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             fir.call @_FortranAShallowCopyDirect(%[[VAL_40]], %[[VAL_41]], %[[VAL_42]], %[[VAL_2]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             %[[VAL_43:.*]] = fir.rebox %[[VAL_38]] : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>) -> !fir.class<!fir.array<?x?xnone>>
// CHECK:             fir.result %[[VAL_43]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_0]] : !fir.class<!fir.array<?x?xnone>>
// CHECK:           }
  %1 = fir.pack_array %arg0 stack innermost : (!fir.class<!fir.array<?x?xnone>>) -> !fir.class<!fir.array<?x?xnone>>
// CHECK:           %[[VAL_44:.*]] = fir.declare %[[VAL_18]] dummy_scope %[[VAL_8]] {uniq_name = "_QFtest7Ex"} : (!fir.class<!fir.array<?x?xnone>>, !fir.dscope) -> !fir.class<!fir.array<?x?xnone>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtest7Ex"} : (!fir.class<!fir.array<?x?xnone>>, !fir.dscope) -> !fir.class<!fir.array<?x?xnone>>
// CHECK:           %[[VAL_45:.*]] = fir.is_present %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> i1
// CHECK:           fir.if %[[VAL_45]] {
// CHECK:             %[[VAL_46:.*]] = fir.box_addr %[[VAL_18]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.heap<!fir.array<?x?xnone>>
// CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_46]] : (!fir.heap<!fir.array<?x?xnone>>) -> index
// CHECK:             %[[VAL_47:.*]] = fir.box_addr %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.heap<!fir.array<?x?xnone>>
// CHECK:             %[[VAL_49:.*]] = fir.convert %[[VAL_47]] : (!fir.heap<!fir.array<?x?xnone>>) -> index
// CHECK:             %[[VAL_50:.*]] = arith.cmpi ne, %[[VAL_48]], %[[VAL_49]] : index
// CHECK:             fir.if %[[VAL_50]] {
// CHECK:               %[[VAL_51:.*]] = fir.address_of(@{{_QQcl.*}}
// CHECK:               %[[VAL_52:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_53:.*]] = fir.convert %[[VAL_18]] : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
// CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_51]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:               fir.call @_FortranAShallowCopyDirect(%[[VAL_52]], %[[VAL_53]], %[[VAL_54]], %[[VAL_1]]) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
  fir.unpack_array %1 to %arg0 stack : !fir.class<!fir.array<?x?xnone>>
  return
}
