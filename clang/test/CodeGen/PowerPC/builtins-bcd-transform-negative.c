// Testfile for negative condition for BCD builtins national2packed, packed2zoned and zoned2packed.
// REQUIRES: powerpc-registered-target
// RUN: not %clang_cc1 -triple powerpc64le-unknown-unknown -O2 -target-cpu pwr9 \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64-unknown-unknown -O2 -target-cpu pwr9 \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc-unknown-unknown -O2 -target-cpu pwr9 \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s

// CHECK: builtins-bcd-transform-negative.c:12:12: error: argument value 2 is outside the valid range [0, 1]
vector unsigned char tBcd_National2packed_imm2(vector unsigned char a) {
    return __builtin_ppc_national2packed (a,'\2'); 
}

// CHECK: builtins-bcd-transform-negative.c:17:12: error: argument value 2 is outside the valid range [0, 1]
vector unsigned char tBcd_Packed2zoned_imm2(vector unsigned char a) {
    return __builtin_ppc_packed2zoned (a,'\2'); 
}

// CHECK: builtins-bcd-transform-negative.c:22:12: error: argument value 2 is outside the valid range [0, 1]
vector unsigned char tBcd_Zoned2packed_imm2(vector unsigned char a) {
    return __builtin_ppc_zoned2packed (a,'\2'); 
}