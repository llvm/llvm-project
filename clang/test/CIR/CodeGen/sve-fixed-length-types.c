// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:   -mvscale-min=4 -mvscale-max=4 -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:   -mvscale-min=4 -mvscale-max=4 -fclangir -emit-llvm -o %t-cir.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:   -mvscale-min=4 -mvscale-max=4 -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target

// Fixed-length SVE types carry the SVE vector kind of their sizeless
// counterpart. GNU vectors of the same shape are generic.

typedef __SVInt32_t fixed_int32_t __attribute__((arm_sve_vector_bits(512)));
typedef __SVFloat64_t fixed_float64_t
    __attribute__((arm_sve_vector_bits(512)));
typedef __SVBool_t fixed_bool_t __attribute__((arm_sve_vector_bits(512)));
typedef int gnu_int32_t __attribute__((vector_size(64)));

// CIR-DAG: !rec_with_pred = !cir.struct<"with_pred" packed {data !s8i, pad !u8i, data !cir.vector<8 x !u8i, sve_predicate>}>
// LLVM-DAG: %struct.with_pred = type <{ i8, i8, <8 x i8> }>

fixed_int32_t global_int32;
// CIR-DAG: cir.global external @global_int32 = #cir.zero : !cir.vector<16 x !s32i, sve_data> align(16)
// LLVM-DAG: @global_int32 = global <16 x i32> zeroinitializer, align 16

fixed_float64_t global_float64;
// CIR-DAG: cir.global external @global_float64 = #cir.zero : !cir.vector<8 x !cir.double, sve_data> align(16)
// LLVM-DAG: @global_float64 = global <8 x double> zeroinitializer, align 16

fixed_bool_t global_bool;
// CIR-DAG: cir.global external @global_bool = #cir.zero : !cir.vector<8 x !u8i, sve_predicate> align(2)
// LLVM-DAG: @global_bool = global <8 x i8> zeroinitializer, align 2

gnu_int32_t global_gnu;
// CIR-DAG: cir.global external @global_gnu = #cir.zero : !cir.vector<16 x !s32i> align(16)
// LLVM-DAG: @global_gnu = global <16 x i32> zeroinitializer, align 16

typedef struct {
  char c;
  fixed_bool_t p;
} with_pred;

with_pred global_with_pred;
// CIR-DAG: cir.global external @global_with_pred = #cir.zero : !rec_with_pred align(2)
// LLVM-DAG: @global_with_pred = global %struct.with_pred zeroinitializer, align 2

// TODO: Add functions that take and return fixed-length SVE types once CIR
// calling-convention lowering coerces them to scalable vectors for AArch64,
// as classic codegen does.
