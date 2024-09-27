// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck -check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s


void test1() {
  float f;
  double d;
  f = __builtin_huge_valf();    
  d = __builtin_huge_val();
}

// CIR-LABEL: test1
// CIR: [[F:%.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f"] {alignment = 4 : i64} 
// CIR: [[D:%.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["d"] {alignment = 8 : i64}
// CIR: [[F_VAL:%.*]] = cir.const #cir.fp<0x7F800000> : !cir.float 
// CIR: cir.store [[F_VAL]], [[F]] : !cir.float, !cir.ptr<!cir.float> 
// CIR: [[D_VAL:%.*]] = cir.const #cir.fp<0x7FF0000000000000> : !cir.double 
// CIR: cir.store [[D_VAL]], [[D]] : !cir.double, !cir.ptr<!cir.double> loc(#loc17)
// CIR: cir.return

// LLVM-LABEL: test1
// [[F:%.*]] = alloca float, align 4
// [[D:%.*]] = alloca double, align 8
// store float 0x7FF0000000000000, ptr [[F]], align 4
// store double 0x7FF0000000000000, ptr[[D]], align 8
// ret void
