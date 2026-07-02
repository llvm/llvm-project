// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

signed _BitInt(6) s6_neg1 = -1;
unsigned _BitInt(6) u6_63 = 63;
signed _BitInt(6) s6_neg2 = -2;
signed _BitInt(2) s2_neg1 = -1;
unsigned _BitInt(4) u4_15 = 15;
signed _BitInt(17) s17_neg1 = -1;
unsigned _BitInt(17) u17_100 = 100;
signed _BitInt(128) s128 = 1234;

// CIR-DAG: cir.global external @s6_neg1 = #cir.int<-1> : !cir.int<s, 6, bitint> {alignment = 1 : i64}
// CIR-DAG: cir.global external @u6_63 = #cir.int<63> : !cir.int<u, 6, bitint> {alignment = 1 : i64}
// CIR-DAG: cir.global external @s6_neg2 = #cir.int<-2> : !cir.int<s, 6, bitint> {alignment = 1 : i64}
// CIR-DAG: cir.global external @s2_neg1 = #cir.int<-1> : !cir.int<s, 2, bitint> {alignment = 1 : i64}
// CIR-DAG: cir.global external @u4_15 = #cir.int<15> : !cir.int<u, 4, bitint> {alignment = 1 : i64}
// CIR-DAG: cir.global external @s17_neg1 = #cir.int<-1> : !cir.int<s, 17, bitint> {alignment = 4 : i64}
// CIR-DAG: cir.global external @u17_100 = #cir.int<100> : !cir.int<u, 17, bitint> {alignment = 4 : i64}
// CIR-DAG: cir.global external @s128 = #cir.int<1234> : !s128i_bitint {alignment = 8 : i64}

// A signed _BitInt is sign-extended across its padded storage integer; an
// unsigned one is zero-extended.  Storage width is the ABI size (i6 -> i8,
// i17 -> i32, i128 stays i128 with align 8).
// LLVM-DAG: @s6_neg1 = global i8 -1, align 1
// LLVM-DAG: @u6_63 = global i8 63, align 1
// LLVM-DAG: @s6_neg2 = global i8 -2, align 1
// LLVM-DAG: @s2_neg1 = global i8 -1, align 1
// LLVM-DAG: @u4_15 = global i8 15, align 1
// LLVM-DAG: @s17_neg1 = global i32 -1, align 4
// LLVM-DAG: @u17_100 = global i32 100, align 4
// LLVM-DAG: @s128 = global i128 1234, align 8

struct S { _BitInt(17) m; };
struct S gs = {-1};

// CIR-DAG: cir.global external @gs = #cir.const_record<{#cir.int<-1> : !cir.int<s, 17, bitint>}> : !rec_S {alignment = 4 : i64}
// LLVM-DAG: @gs = global %struct.S { i32 -1 }, align 4

signed _BitInt(17) garr[3] = {-1, 2, -3};

// CIR-DAG: cir.global external @garr = #cir.const_array<[#cir.int<-1> : !cir.int<s, 17, bitint>, #cir.int<2> : !cir.int<s, 17, bitint>, #cir.int<-3> : !cir.int<s, 17, bitint>]> : !cir.array<!cir.int<s, 17, bitint> x 3> {alignment = 4 : i64}
// LLVM-DAG: @garr = global [3 x i32] [i32 -1, i32 2, i32 -3], align 4

void store_load(signed _BitInt(17) *p, signed _BitInt(17) v) { *p = v; }

// CIR-LABEL: cir.func {{.*}} @store_load
// CIR: %[[V:.+]] = cir.load {{.*}} : !cir.ptr<!cir.int<s, 17, bitint>>, !cir.int<s, 17, bitint>
// CIR: cir.store {{.*}} %[[V]], {{.*}} : !cir.int<s, 17, bitint>, !cir.ptr<!cir.int<s, 17, bitint>>

// The value is sign-extended on store into i32 memory and truncated back to
// i17 on load.
// LLVM-LABEL: @store_load
// LLVM: %[[EXT:.+]] = sext i17 %{{.+}} to i32
// LLVM: store i32 %[[EXT]], ptr %{{.+}}, align 4
// LLVM: %[[LD:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM: %[[TR:.+]] = trunc i32 %[[LD]] to i17
