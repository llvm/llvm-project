// RUN: %clang_cc1 -fkeep-static-consts -emit-llvm %s -o - -triple=x86_64-unknown-linux-gnu | FileCheck %s

// CHECK: @_ZL7srcvers = internal constant [4 x i8] c"xyz\00", align 1
// CHECK: @_ZL8srcvers2 = internal constant [4 x i8] c"abc\00", align 1
// CHECK: @_ZL1N = internal constant i32 2, align 4
// CHECK: @llvm.used = appending global [4 x i8*] [i8* getelementptr inbounds ([4 x i8], [4 x i8]* @_ZL7srcvers, i32 0, i32 0), i8* bitcast (i32* @b to i8*), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @_ZL8srcvers2, i32 0, i32 0), i8* bitcast (i32* @_ZL1N to i8*)], section "llvm.metadata"

static const char srcvers[] = "xyz";
extern const int b = 1;
const char srcvers2[] = "abc";
constexpr int N = 2;
