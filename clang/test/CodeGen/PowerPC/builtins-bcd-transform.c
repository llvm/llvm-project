// Testfile that verifies positive cases (0 or 1 only) for BCD builtins national2packed, packed2zoned and zoned2packed.
// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -O2 -target-cpu pwr9 \
// RUN:   -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -O2 -target-cpu pwr9 \
// RUN:   -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-unknown -O2 -target-cpu pwr9 \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: tBcd_National2packed_imm1
// CHECK: [[TMP0:%.*]] = tail call <16 x i8> @llvm.ppc.national2packed(<16 x i8> %a, i32 1)
// CHECK-NEXT: ret <16 x i8> [[TMP0]]
vector unsigned char tBcd_National2packed_imm1(vector unsigned char a) {
    return __builtin_ppc_national2packed (a,'\1'); 
}

// CHECK-LABEL: tBcd_National2packed_imm0
// CHECK: [[TMP0:%.*]] = tail call <16 x i8> @llvm.ppc.national2packed(<16 x i8> %a, i32 0)
// CHECK-NEXT: ret <16 x i8> [[TMP0]]
vector unsigned char tBcd_National2packed_imm0(vector unsigned char a) {
    return __builtin_ppc_national2packed (a,'\0');
}

// CHECK-LABEL: tBcd_Packed2national
// CHECK: [[TMP0:%.*]] = tail call <16 x i8> @llvm.ppc.packed2national(<16 x i8> %a)
// CHECK-NEXT: ret <16 x i8> [[TMP0]]
vector unsigned char tBcd_Packed2national(vector unsigned char a){
    return __builtin_ppc_packed2national(a);
}

// CHECK-LABEL: tBcd_Packed2zoned_imm0
// CHECK: [[TMP0:%.*]] = tail call <16 x i8> @llvm.ppc.packed2zoned(<16 x i8> %a, i32 0)
// CHECK-NEXT: ret <16 x i8> [[TMP0]]
vector unsigned char tBcd_Packed2zoned_imm0(vector unsigned char a){
    return __builtin_ppc_packed2zoned(a,'\0');
}

// CHECK-LABEL: tBcd_Packed2zoned_imm1
// CHECK: [[TMP0:%.*]] = tail call <16 x i8> @llvm.ppc.packed2zoned(<16 x i8> %a, i32 1)
// CHECK-NEXT: ret <16 x i8> [[TMP0]]
vector unsigned char tBcd_Packed2zoned_imm1(vector unsigned char a){
    return __builtin_ppc_packed2zoned(a,'\1');
}

// CHECK-LABEL: tBcd_Zoned2packed_imm0
// CHECK: [[TMP0:%.*]] = tail call <16 x i8> @llvm.ppc.zoned2packed(<16 x i8> %a, i32 0)
// CHECK-NEXT: ret <16 x i8> [[TMP0]]
vector unsigned char tBcd_Zoned2packed_imm0(vector unsigned char a){
    return __builtin_ppc_zoned2packed(a,'\0');
}

// CHECK-LABEL: tBcd_Zoned2packed_imm1
// CHECK: [[TMP0:%.*]] = tail call <16 x i8> @llvm.ppc.zoned2packed(<16 x i8> %a, i32 1)
// CHECK-NEXT: ret <16 x i8> [[TMP0]]
vector unsigned char tBcd_Zoned2packed_imm1(vector unsigned char a){
    return __builtin_ppc_zoned2packed(a,'\1');
}