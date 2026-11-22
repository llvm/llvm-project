// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Test that real and complex types are named by their kind, not their size.

module {
func.func private @real(%arg0: f16, %arg1: bf16, %arg2: f32, %arg3: f64, %arg4: f80, %arg5: f128)
func.func private @cmplx(%arg0: complex<f16>, %arg1: complex<bf16>)
}

// CHECK-DAG: #[[F16:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real(kind=2)", sizeInBits = 16, encoding = DW_ATE_float>
// CHECK-DAG: #[[BF16:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real(kind=3)", sizeInBits = 16, encoding = DW_ATE_float>
// CHECK-DAG: #[[F32:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real(kind=4)", sizeInBits = 32, encoding = DW_ATE_float>
// CHECK-DAG: #[[F64:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real(kind=8)", sizeInBits = 64, encoding = DW_ATE_float>
// CHECK-DAG: #[[F80:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real(kind=10)", sizeInBits = 80, encoding = DW_ATE_float>
// CHECK-DAG: #[[F128:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real(kind=16)", sizeInBits = 128, encoding = DW_ATE_float>
// CHECK-DAG: #llvm.di_subroutine_type<{{.*}}types = #di_null_type, #[[F16]], #[[BF16]], #[[F32]], #[[F64]], #[[F80]], #[[F128]]>

// CHECK-DAG: #[[CF16:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "complex(kind=2)", sizeInBits = 32, encoding = DW_ATE_complex_float>
// CHECK-DAG: #[[CBF16:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "complex(kind=3)", sizeInBits = 32, encoding = DW_ATE_complex_float>
// CHECK-DAG: #llvm.di_subroutine_type<{{.*}}types = #di_null_type, #[[CF16]], #[[CBF16]]>
