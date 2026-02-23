// RUN: fir-opt -flang-licm --split-input-file %s | FileCheck %s

// Verify that Pure fir.convert operations with !fir.ref operands
// are not hoisted by LICM out of cuf.kernel.
// CHECK-LABEL:   func.func @_QPtest(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.array<10xf32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "ra"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
// CHECK:           %[[ALLOC_0:.*]] = cuf.alloc !fir.box<!fir.ptr<!fir.array<?xf32>>> {data_attr = #cuf.cuda<device>} -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {data_attr = #cuf.cuda<device>, uniq_name = "_QFtestEa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf32>>
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtestEi"}
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtestEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[ALLOCA_1:.*]] = fir.alloca i64 {bindc_name = "pa", uniq_name = "_QFtestEpa"}
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ALLOCA_1]] {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFtestEpa"} : (!fir.ref<i64>) -> !fir.ref<i64>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ALLOC_0]] {fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFtestEra"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK:           %[[ZERO_BITS_0:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
// CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ZERO_BITS_0]](%[[SHAPE_0]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
// CHECK:           fir.store %[[EMBOX_0]] to %[[DECLARE_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[DECLARE_0]] : (!fir.ref<!fir.array<10xf32>>) -> i64
// CHECK:           fir.store %[[CONVERT_0]] to %[[DECLARE_2]] : !fir.ref<i64>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           cuf.kernel<<<*, *>>> (%[[VAL_0:.*]] : index) = (%[[CONSTANT_1]] : index) to (%[[CONSTANT_2]] : index)  step (%[[CONSTANT_1]] : index) {
// CHECK:             %[[CONVERT_1:.*]] = fir.convert %[[VAL_0]] : (index) -> i32
// CHECK:             fir.store %[[CONVERT_1]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[DECLARE_2]] : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
// CHECK:             %[[LOAD_0:.*]] = fir.load %[[CONVERT_2]] : !fir.ref<!fir.ptr<i64>>
// CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[DECLARE_3]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[CONVERT_4:.*]] = fir.convert %[[LOAD_0]] : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
// CHECK:             fir.call @_FortranAPointerAssociateScalar(%[[CONVERT_3]], %[[CONVERT_4]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[DECLARE_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_5:.*]] = fir.convert %[[LOAD_2]] : (i32) -> i64
// CHECK:             %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[LOAD_1]], %[[CONSTANT_3]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
// CHECK:             %[[SHIFT_0:.*]] = fir.shift %[[BOX_DIMS_0]]#0 : (index) -> !fir.shift<1>
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[LOAD_1]](%[[SHIFT_0]]) %[[CONVERT_5]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, i64) -> !fir.ref<f32>
// CHECK:             fir.store %[[CONSTANT_0]] to %[[ARRAY_COOR_0]] : !fir.ref<f32>
// CHECK:             "fir.end"() : () -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @_QPtest(%arg0: !fir.ref<!fir.array<10xf32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<10xf32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "ra"}) {
  %cst = arith.constant 1.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = cuf.alloc !fir.box<!fir.ptr<!fir.array<?xf32>>> {data_attr = #cuf.cuda<device>} -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %1 = fir.dummy_scope : !fir.dscope
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3 = fir.declare %arg0(%2) dummy_scope %1 arg 1 {data_attr = #cuf.cuda<device>, uniq_name = "_QFtestEa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf32>>
  %4 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtestEi"}
  %5 = fir.declare %4 {uniq_name = "_QFtestEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %6 = fir.alloca i64 {bindc_name = "pa", uniq_name = "_QFtestEpa"}
  %7 = fir.declare %6 {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFtestEpa"} : (!fir.ref<i64>) -> !fir.ref<i64>
  %8 = fir.declare %0 {fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFtestEra"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %9 = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  %10 = fir.embox %9(%2) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  fir.store %10 to %8 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %11 = fir.convert %3 : (!fir.ref<!fir.array<10xf32>>) -> i64
  fir.store %11 to %7 : !fir.ref<i64>
  cuf.kernel<<<*, *>>> (%arg2 : index) = (%c1 : index) to (%c10 : index)  step (%c1 : index) {
    %12 = fir.convert %arg2 : (index) -> i32
    fir.store %12 to %5 : !fir.ref<i32>
    %13 = fir.convert %7 : (!fir.ref<i64>) -> !fir.ref<!fir.ptr<i64>>
    %14 = fir.load %13 : !fir.ref<!fir.ptr<i64>>
    %15 = fir.convert %8 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
    %16 = fir.convert %14 : (!fir.ptr<i64>) -> !fir.llvm_ptr<i8>
    fir.call @_FortranAPointerAssociateScalar(%15, %16) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
    %17 = fir.load %8 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %18 = fir.load %5 : !fir.ref<i32>
    %19 = fir.convert %18 : (i32) -> i64
    %c0 = arith.constant 0 : index
    %20:3 = fir.box_dims %17, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
    %21 = fir.shift %20#0 : (index) -> !fir.shift<1>
    %22 = fir.array_coor %17(%21) %19 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, i64) -> !fir.ref<f32>
    fir.store %cst to %22 : !fir.ref<f32>
    "fir.end"() : () -> ()
  }
  return
}
