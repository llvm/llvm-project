// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -fexperimental-max-bitint-width=512 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -fexperimental-max-bitint-width=512 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -fexperimental-max-bitint-width=512 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

unsigned _BitInt(200) w200 = 5;
// CIR-DAG: cir.global external @w200 = #cir.int<5> : !cir.int<u, 200, bitint> {alignment = 8 : i64}
// LLVM-DAG: @w200 = global i256 5, align 8
signed _BitInt(256) w256 = -1;
// CIR-DAG: cir.global external @w256 = #cir.int<-1> : !s256i_bitint {alignment = 8 : i64}
// LLVM-DAG: @w256 = global i256 -1, align 8

signed _BitInt(129) g_neg = -1;
// CIR-DAG: cir.global external @g_neg = #cir.int<-1> : !cir.int<s, 129, bitint> {alignment = 8 : i64}
// LLVM-DAG: @g_neg = global [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF", align 8

struct HasWide129 {
  int i;
  signed _BitInt(129) bi;
};
struct HasWide129 g_struct = {7, -1};
// CIR-DAG: cir.global external @g_struct = #cir.const_record<{#cir.int<7> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<-1> : !cir.int<s, 129, bitint>}> : !rec_HasWide129 {alignment = 8 : i64}
// Note: CIR packs this, but classic codegen doesn't.  Layout is still
// identical, but there is a slight IR difference.
// LLVM-DAG: %struct.HasWide129 = type {{[<]?}}{ i32, [4 x i8], [24 x i8] }{{[>]?}}
// LLVM-DAG: @g_struct = global %struct.HasWide129 {{[<]?}}{ i32 7, [4 x i8] zeroinitializer, [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF" }{{[>]?}}, align 8

signed _BitInt(129) g_arr[2] = {-1, -2};
// Note: Alignment is 8 instead of 16 like on x86_64
// CIR-DAG: cir.global external @g_arr = #cir.const_array<[#cir.int<-1> : !cir.int<s, 129, bitint>, #cir.int<-2> : !cir.int<s, 129, bitint>]> : !cir.array<!cir.int<s, 129, bitint> x 2> {alignment = 8 : i64}
// LLVM-DAG: @g_arr = global [2 x [24 x i8]] {{\[\[}}24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF", [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FE"], align 8

struct HasWide129Array {
  int i;
  signed _BitInt(129) bi[2];
};
struct HasWide129Array g_arr2 = {7, {-1, -2}};
// CIR-DAG: cir.global external @g_arr2 = #cir.const_record<{#cir.int<7> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.const_array<[#cir.int<-1> : !cir.int<s, 129, bitint>, #cir.int<-2> : !cir.int<s, 129, bitint>]> : !cir.array<!cir.int<s, 129, bitint> x 2>}> : !rec_HasWide129Array {alignment = 8 : i64}
// LLVM-DAG: %struct.HasWide129Array = type {{[<]?}}{ i32, [4 x i8], [2 x [24 x i8]] }{{[>]?}}
// LLVM-DAG: @g_arr2 = global %struct.HasWide129Array {{[<]?}}{ i32 7, [4 x i8] zeroinitializer, [2 x [24 x i8]] {{\[\[}}24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF", [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FE"] }{{[>]?}}, align 8

signed _BitInt(320) g320 = -1;
// CIR-DAG: cir.global external @g320 = #cir.int<-1> : !cir.int<s, 320, bitint> {alignment = 8 : i64}
// LLVM-DAG: @g320 = global [40 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF", align 8

_BitInt(129) g_one = 1;
// CIR-DAG: cir.global external @g_one = #cir.int<1> : !cir.int<s, 129, bitint> {alignment = 8 : i64}
// LLVM-DAG: @g_one = global [24 x i8] c"\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01", align 8

int g_is_one(void) {
  return g_one == 1;
}
// CIR-LABEL: cir.func{{.*}} @g_is_one
// CIR: %[[GET_GLOB:.*]] = cir.get_global @g_one : !cir.ptr<!cir.int<s, 129, bitint>>
// CIR: %[[LOAD_GLOB:.*]] = cir.load align(8) %[[GET_GLOB]] : !cir.ptr<!cir.int<s, 129, bitint>>, !cir.int<s, 129, bitint>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !cir.int<s, 129, bitint>
// CIR: cir.cmp eq %[[LOAD_GLOB]], %[[ONE]] : !cir.int<s, 129, bitint>

// LLVM-LABEL: define{{.*}} @g_is_one
// LLVM: %[[LOAD_GLOB:.*]] = load i192, ptr @g_one, align 8
// LLVM: %[[TRUNC_GLOB:.*]] = trunc i192 %[[LOAD_GLOB]] to i129
// LLVM: icmp eq i129 %[[TRUNC_GLOB]], 1
