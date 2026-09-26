// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:   -fexperimental-max-bitint-width=8388608 -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:   -fexperimental-max-bitint-width=8388608 -emit-llvm %s -o - \
// RUN:   | FileCheck %s

signed _BitInt(129) one = 1;
signed _BitInt(129) minus_two = -2;

// CHECK-DAG: @one = global [24 x i8] c"\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01", align 8
// CHECK-DAG: @minus_two = global [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FE", align 8
