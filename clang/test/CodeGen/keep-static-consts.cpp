// RUN: %clang_cc1 -fkeep-static-consts -emit-llvm %s -o - -triple=x86_64-unknown-linux-gnu | FileCheck %s

// CHECK: @_ZL7srcvers = internal constant [4 x i8] c"xyz\00", align 1
// CHECK: @llvm.used = appending global [1 x i8*] [i8* getelementptr inbounds ([4 x i8], [4 x i8]* @_ZL7srcvers, i32 0, i32 0)], section "llvm.metadata"
static const char srcvers[] = "xyz";
extern const int b = 1;
