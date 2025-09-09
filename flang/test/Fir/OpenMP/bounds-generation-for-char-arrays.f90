// RUN: fir-opt --cfg-conversion --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s

module attributes {omp.is_target_device = false} {
  func.func @_QPchar_array(%arg0 : !fir.ref<!fir.array<10x10x!fir.char<1,16>>>) {
    %c9 = arith.constant 9 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %0 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%c9 : index) extent(%c10 : index) stride(%c1 : index) start_idx(%c1 : index)
    %1 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%c9 : index) extent(%c10 : index) stride(%c1 : index) start_idx(%c1 : index)
    %2 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.array<10x10x!fir.char<1,16>>>, !fir.array<10x10x!fir.char<1,16>>) map_clauses(tofrom) capture(ByRef) bounds(%0, %1) -> !fir.ref<!fir.array<10x10x!fir.char<1,16>>> {name = ""}
    omp.target map_entries(%2 -> %arg1 : !fir.ref<!fir.array<10x10x!fir.char<1,16>>>) {
      omp.terminator
    }
    return
  }

// CHECK-LABEL:   llvm.func @_QPchar_array(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr) {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(9 : index) : i64
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(10 : index) : i64
// CHECK:           %[[VAL_4:.*]] = omp.map.bounds lower_bound(%[[VAL_1]] : i64) upper_bound(%[[VAL_0]] : i64) extent(%[[VAL_3]] : i64) stride(%[[VAL_2]] : i64) start_idx(%[[VAL_2]] : i64)
// CHECK:           %[[VAL_5:.*]] = omp.map.bounds lower_bound(%[[VAL_1]] : i64) upper_bound(%[[VAL_0]] : i64) extent(%[[VAL_3]] : i64) stride(%[[VAL_2]] : i64) start_idx(%[[VAL_2]] : i64)
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(15 : i64) : i64
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_10:.*]] = omp.map.bounds lower_bound(%[[VAL_6]] : i64) upper_bound(%[[VAL_7]] : i64) extent(%[[VAL_7]] : i64) stride(%[[VAL_8]] : i64) start_idx(%[[VAL_9]] : i64)
// CHECK:           %[[VAL_11:.*]] = omp.map.info var_ptr(%[[ARG0]] : !llvm.ptr, i8) map_clauses(tofrom) capture(ByRef) bounds(%[[VAL_4]], %[[VAL_5]], %[[VAL_10]]) -> !llvm.ptr {name = ""}
// CHECK:           omp.target map_entries(%[[VAL_11]] -> %[[VAL_12:.*]] : !llvm.ptr) {
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }

  func.func @_QPallocatable_char_array(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,16>>>>>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,16>>>>>
    %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,16>>>>, index) -> (index, index, index)
    %2 = arith.subi %1#1, %c1 : index
    %3 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%2 : index) extent(%1#1 : index) stride(%1#2 : index) start_idx(%1#0 : index) {stride_in_bytes = true}
    %4 = arith.muli %1#2, %1#1 : index
    %5:3 = fir.box_dims %0, %c1 : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,16>>>>, index) -> (index, index, index)
    %6 = arith.subi %5#1, %c1 : index
    %7 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%6 : index) extent(%5#1 : index) stride(%4 : index) start_idx(%5#0 : index) {stride_in_bytes = true}
    %8 = fir.box_offset %arg0 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,16>>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?x?x!fir.char<1,16>>>>
    %9 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,16>>>>>, !fir.char<1,16>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%8 : !fir.llvm_ptr<!fir.ref<!fir.array<?x?x!fir.char<1,16>>>>) bounds(%3, %7) -> !fir.llvm_ptr<!fir.ref<!fir.array<?x?x!fir.char<1,16>>>> {name = ""}
    %10 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,16>>>>>, !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,16>>>>) map_clauses(to) capture(ByRef) members(%9 : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?x?x!fir.char<1,16>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,16>>>>> {name = "csv_chem_list_a"}
    omp.target map_entries(%10 -> %arg1, %9 -> %arg2 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,16>>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?x?x!fir.char<1,16>>>>) {
      omp.terminator
    }
    return
  }

// CHECK-LABEL:   llvm.func @_QPallocatable_char_array(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr) {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(72 : i32) : i32
// CHECK:           "llvm.intr.memcpy"(%[[VAL_1]], %[[ARG0]], %[[VAL_4]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK:           %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_1]][0, 7, %[[VAL_3]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_1]][0, 7, %[[VAL_3]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_8:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_1]][0, 7, %[[VAL_3]], 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_11:.*]] = llvm.sub %[[VAL_8]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_12:.*]] = omp.map.bounds lower_bound(%[[VAL_3]] : i64) upper_bound(%[[VAL_11]] : i64) extent(%[[VAL_8]] : i64) stride(%[[VAL_10]] : i64) start_idx(%[[VAL_6]] : i64) {stride_in_bytes = true}
// CHECK:           %[[VAL_13:.*]] = llvm.mul %[[VAL_10]], %[[VAL_8]] : i64
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_1]][0, 7, %[[VAL_2]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_1]][0, 7, %[[VAL_2]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_17:.*]] = llvm.load %[[VAL_16]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_1]][0, 7, %[[VAL_2]], 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_19:.*]] = llvm.load %[[VAL_18]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_20:.*]] = llvm.sub %[[VAL_17]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_21:.*]] = omp.map.bounds lower_bound(%[[VAL_3]] : i64) upper_bound(%[[VAL_20]] : i64) extent(%[[VAL_17]] : i64) stride(%[[VAL_13]] : i64) start_idx(%[[VAL_15]] : i64) {stride_in_bytes = true}
// CHECK:           %[[VAL_22:.*]] = llvm.getelementptr %[[ARG0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(15 : i64) : i64
// CHECK:           %[[VAL_25:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_27:.*]] = omp.map.bounds lower_bound(%[[VAL_23]] : i64) upper_bound(%[[VAL_24]] : i64) extent(%[[VAL_24]] : i64) stride(%[[VAL_25]] : i64) start_idx(%[[VAL_26]] : i64)
// CHECK:           %[[VAL_28:.*]] = omp.map.info var_ptr(%[[ARG0]] : !llvm.ptr, i8) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[VAL_22]] : !llvm.ptr) bounds(%[[VAL_12]], %[[VAL_21]], %[[VAL_27]]) -> !llvm.ptr {name = ""}
// CHECK:           %[[VAL_29:.*]] = omp.map.info var_ptr(%[[ARG0]] : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>) map_clauses(to) capture(ByRef) members(%[[VAL_28]] : [0] : !llvm.ptr) -> !llvm.ptr {name = "csv_chem_list_a"}
// CHECK:           omp.target map_entries(%[[VAL_29]] -> %[[VAL_30:.*]], %[[VAL_28]] -> %[[VAL_31:.*]] : !llvm.ptr, !llvm.ptr) {
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }
}
