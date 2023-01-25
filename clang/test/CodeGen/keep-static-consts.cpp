// RUN: %clang_cc1 -fkeep-static-consts -emit-llvm %s -o - -triple=x86_64-unknown-linux-gnu | FileCheck %s

// CHECK: @_ZL7srcvers = internal constant [4 x i8] c"xyz\00", align 1
// CHECK: @_ZL8srcvers2 = internal constant [4 x i8] c"abc\00", align 1
// CHECK: @_ZL1N = internal constant i32 2, align 4
// CHECK: @llvm.compiler.used = appending global [4 x ptr] [ptr @_ZL7srcvers, ptr @b, ptr @_ZL8srcvers2, ptr @_ZL1N], section "llvm.metadata"

static const char srcvers[] = "xyz";
extern const int b = 1;
const char srcvers2[] = "abc";
constexpr int N = 2;
