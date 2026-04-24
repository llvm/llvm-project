// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fclangir -emit-cir -fcxx-exceptions -fexceptions -mmlir --mlir-print-ir-before=cir-cxxabi-lowering -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR,CIR-BEFORE
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR,CIR-AFTER
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fclangir -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s --check-prefixes=LLVM

struct Base1 {
  int i, j, k;
};

struct Base2 {
  float l, m, n;
};

struct Inherits : Base1, Base2 {
  int o,p,q;
};


struct VirtualInherits : virtual Base1, virtual Base2 {
  int o,p,q;
};


// CIR: !rec_Base2 = !cir.record<struct "Base2" {!cir.float, !cir.float, !cir.float}>
// CIR: !rec_Base1 = !cir.record<struct "Base1" {!s32i, !s32i, !s32i}>
// CIR: !rec_Inherits = !cir.record<struct "Inherits" {!rec_Base1, !rec_Base2, !s32i, !s32i, !s32i}>
// CIR: !rec_VirtualInherits = !cir.record<struct "VirtualInherits" packed padded {!cir.vptr, !s32i, !s32i, !s32i, !rec_Base1, !rec_Base2, !cir.array<!u8i x 4>}>
//
// LLVM: %struct.Inherits = type { %struct.Base1, %struct.Base2, i32, i32, i32 }
// LLVM: %struct.Base1 = type { i32, i32, i32 }
// LLVM: %struct.Base2 = type { float, float, float }
// LLVM: %struct.VirtualInherits = type <{ ptr, i32, i32, i32, %struct.Base1, %struct.Base2, [4 x i8] }>
//
Inherits I;
// CIR: cir.global external @I = #cir.zero : !rec_Inherits {alignment = 4 : i64}
// LLVM: @I = global %struct.Inherits zeroinitializer, align 4

Inherits I2 {{1,2,3},{1.1, 2.2, 3.3}, 4, 5, 6};
// CIR: cir.global external @I2 = #cir.const_record<{#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.fp<1.100000e+00> : !cir.float, #cir.fp<2.200000e+00> : !cir.float, #cir.fp<3.300000e+00> : !cir.float, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i}> : !rec_anon_struct {alignment = 4 : i64}
// LLVM: @I2 = global { i32, i32, i32, float, float, float, i32, i32, i32 } { i32 1, i32 2, i32 3, float {{.*}}, float {{.*}}, float {{.*}}, i32 4, i32 5, i32 6 }, align 4

VirtualInherits VI;
// CIR-BEFORE: cir.global external @VI = ctor : !rec_VirtualInherits {
// CIR-BEFORE:   %[[GET_GLOB:.*]] = cir.get_global @VI : !cir.ptr<!rec_VirtualInherits>
// CIR-BEFORE:   cir.call @_ZN15VirtualInheritsC1Ev(%[[GET_GLOB]]) nothrow : (!cir.ptr<!rec_VirtualInherits> {llvm.align = 8 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}) -> ()
// CIR-BEFORE: } {alignment = 8 : i64, ast = #cir.var.decl.ast}
//
// CIR-AFTER: cir.global external @VI = #cir.zero : !rec_VirtualInherits {alignment = 8 : i64, ast = #cir.var.decl.ast}
// CIR-AFTER: cir.func {{.*}}@__cxx_global_var_init() {
// CIR-AFTER:   %[[GET_GLOB:.*]] = cir.get_global @VI : !cir.ptr<!rec_VirtualInherits> loc(#loc13)
// CIR-AFTER:   cir.call @_ZN15VirtualInheritsC1Ev(%[[GET_GLOB]]) nothrow : (!cir.ptr<!rec_VirtualInherits> {llvm.align = 8 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}) -> ()

// LLVM: @VI = global %struct.VirtualInherits zeroinitializer, align 8
// LLVM: define internal void @__cxx_global_var_init()
// LLVM: call void @_ZN15VirtualInheritsC1Ev(ptr noundef nonnull align 8 dereferenceable(20) @VI)
