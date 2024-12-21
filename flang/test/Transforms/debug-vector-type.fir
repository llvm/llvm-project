// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
func.func private @foo1(%arg0: !fir.vector<20:bf16>)
// CHECK-DAG: #[[F16:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 16, encoding = DW_ATE_float>
// CHECK-DAG: #llvm.di_composite_type<tag = DW_TAG_array_type, name = "vector real (20)", baseType = #[[F16]], flags = Vector, sizeInBits = 320, elements = #llvm.di_subrange<count = 20 : i64>>

func.func private @foo2(%arg0: !fir.vector<30:f32>)
// CHECK-DAG: #[[F32:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
// CHECK-DAG: #llvm.di_composite_type<tag = DW_TAG_array_type, name = "vector real (30)", baseType = #[[F32]], flags = Vector, sizeInBits = 960, elements = #llvm.di_subrange<count = 30 : i64>>

func.func private @foo3(%arg0: !fir.vector<10:f64>)
// CHECK-DAG: #[[F64:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 64, encoding = DW_ATE_float>
// CHECK-DAG: #llvm.di_composite_type<tag = DW_TAG_array_type, name = "vector real (10)", baseType = #[[F64]], flags = Vector, sizeInBits = 640, elements = #llvm.di_subrange<count = 10 : i64>>

func.func private @foo4(%arg0: !fir.vector<5:i32>)
// CHECK-DAG: #[[I32:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
// CHECK-DAG: #llvm.di_composite_type<tag = DW_TAG_array_type, name = "vector integer (5)", baseType = #[[I32]], flags = Vector, sizeInBits = 160, elements = #llvm.di_subrange<count = 5 : i64>>

func.func private @foo5(%arg0: !fir.vector<2:i64>)
// CHECK-DAG: #[[I64:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 64, encoding = DW_ATE_signed>
// CHECK-DAG: #llvm.di_composite_type<tag = DW_TAG_array_type, name = "vector integer (2)", baseType = #[[I64]], flags = Vector, sizeInBits = 128, elements = #llvm.di_subrange<count = 2 : i64>>
}
