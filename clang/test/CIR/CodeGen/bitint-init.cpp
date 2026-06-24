// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

constexpr signed _BitInt(128) ci128 = 1234;
const signed _BitInt(128) *pci = &ci128;

struct BI { _BitInt(128) m; };
constexpr BI bi = {5678};
const BI *pbi = &bi;

unsigned _BitInt(17) f() {
  static constexpr unsigned _BitInt(17) sl = 100;
  return sl;
}

// CIR-DAG: cir.global "private" constant internal dso_local @_ZL5ci128 = #cir.int<1234> : !s128i_bitint
// CIR-DAG: cir.global external @pci = #cir.global_view<@_ZL5ci128> : !cir.ptr<!s128i_bitint>
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL2bi = #cir.const_record<{#cir.int<5678> : !s128i_bitint}> : !rec_BI
// CIR-DAG: cir.global external @pbi = #cir.global_view<@_ZL2bi> : !cir.ptr<!rec_BI>
// CIR-DAG: cir.global "private" constant internal dso_local @_ZZ1fvE2sl = #cir.int<100> : !cir.int<u, 17, bitint>

// LLVM-DAG: @_ZL5ci128 = internal constant i128 1234, align 8
// LLVM-DAG: @pci = global ptr @_ZL5ci128, align 8
// LLVM-DAG: @_ZL2bi = internal constant %struct.BI { i128 5678 }, align 8
// LLVM-DAG: @pbi = global ptr @_ZL2bi, align 8

// CIR keeps the exact width; OGCG widens i17 -> i32.
// LLVMCIR-DAG: @_ZZ1fvE2sl = internal constant i17 100, align 4
// OGCG-DAG: @_ZZ1fvE2sl = internal constant i32 100, align 4
