// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void foo() {}

//      CHECK: module @"{{.*}}dlti.c" attributes {
//  CHECK-DAG: cir.sob = #cir.signed_overflow_behavior<undefined>,
//  CHECK-DAG: cir.type_size_info =
//  CHECK-DAG:   #cir.type_size_info<
//  CHECK-DAG:     char = 8,
//  CHECK-DAG:     int = {{16|32}},
//  CHECK-DAG:     size_t = {{32|64}}
//  CHECK-DAG: >
//  CHECK-DAG: dlti.dl_spec =
//  CHECK-DAG:   #dlti.dl_spec<
//  CHECK-DAG:     i16 = dense<16> : vector<2xi64>,
//  CHECK-DAG:     i32 = dense<32> : vector<2xi64>,
//  CHECK-DAG:     i8 = dense<8> : vector<2xi64>,
//  CHECK-DAG:     i1 = dense<8> : vector<2xi64>,
//  CHECK-DAG:     !llvm.ptr = dense<64> : vector<4xi64>,
//  CHECK-DAG:     f80 = dense<128> : vector<2xi64>,
//  CHECK-DAG:     i128 = dense<128> : vector<2xi64>,
//  CHECK-DAG:     !llvm.ptr<272> = dense<64> : vector<4xi64>,
//  CHECK-DAG:     i64 = dense<64> : vector<2xi64>,
//  CHECK-DAG:     !llvm.ptr<270> = dense<32> : vector<4xi64>,
//  CHECK-DAG:     !llvm.ptr<271> = dense<32> : vector<4xi64>,
//  CHECK-DAG:     f128 = dense<128> : vector<2xi64>,
//  CHECK-DAG:     f16 = dense<16> : vector<2xi64>,
//  CHECK-DAG:     f64 = dense<64> : vector<2xi64>,
//  CHECK-DAG:     "dlti.stack_alignment" = 128 : i64
//  CHECK-DAG:     "dlti.endianness" = "little"
//  >,
