// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-O0-optnone -emit-llvm \
// RUN:   %s -o - | opt -S -passes=mem2reg | FileCheck %s

// CHECK-LABEL: define dso_local half @test_convert_from_bf16_to_fp16(
// CHECK-SAME: bfloat noundef [[A:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext bfloat [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to half
// CHECK-NEXT:    ret half [[FPTRUNC]]
//
_Float16 test_convert_from_bf16_to_fp16(__bf16 a) {
    return (_Float16)a;
}

// CHECK-LABEL: define dso_local bfloat @test_convert_from_fp16_to_bf16(
// CHECK-SAME: half noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext half [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to bfloat
// CHECK-NEXT:    ret bfloat [[FPTRUNC]]
//
__bf16 test_convert_from_fp16_to_bf16(_Float16 a) {
    return (__bf16)a;
}

