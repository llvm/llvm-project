// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct {
  char x[10];
  char y[10];
  char z[10];
} literals = {"1", "", "\00"};

// CIR-LABEL: @literals
// CIR:  #cir.const_struct<{
// CIR:     #cir.const_array<"1" : !cir.array<!s8i x 1>, trailing_zeros> : !cir.array<!s8i x 10>,
// CIR:     #cir.zero : !cir.array<!s8i x 10>,
// CIR:     #cir.zero : !cir.array<!s8i x 10>
// CIR:  }> 

// LLVM-LABEL: @literals
// LLVM:  global %struct.anon.1 {
// LLVM:    [10 x i8] c"1\00\00\00\00\00\00\00\00\00",
// LLVM:    [10 x i8] zeroinitializer,
// LLVM:    [10 x i8] zeroinitializer
// LLVM:  }
