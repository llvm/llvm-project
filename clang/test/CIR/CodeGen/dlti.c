// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=LITTLE

void foo() {}

//  LITTLE-DAG: dlti.dl_spec =
//  LITTLE-DAG:   #dlti.dl_spec<
//  LITTLE-DAG:     i16 = dense<16> : vector<2xi64>,
//  LITTLE-DAG:     i32 = dense<32> : vector<2xi64>,
//  LITTLE-DAG:     i8 = dense<8> : vector<2xi64>,
//  LITTLE-DAG:     i1 = dense<8> : vector<2xi64>,
//  LITTLE-DAG:     !llvm.ptr = dense<64> : vector<4xi64>,
//  LITTLE-DAG:     f80 = dense<128> : vector<2xi64>,
//  LITTLE-DAG:     i128 = dense<128> : vector<2xi64>,
//  LITTLE-DAG:     !llvm.ptr<272> = dense<64> : vector<4xi64>,
//  LITTLE-DAG:     i64 = dense<64> : vector<2xi64>,
//  LITTLE-DAG:     !llvm.ptr<270> = dense<32> : vector<4xi64>,
//  LITTLE-DAG:     !llvm.ptr<271> = dense<32> : vector<4xi64>,
//  LITTLE-DAG:     f128 = dense<128> : vector<2xi64>,
//  LITTLE-DAG:     f16 = dense<16> : vector<2xi64>,
//  LITTLE-DAG:     f64 = dense<64> : vector<2xi64>,
//  LITTLE-DAG:     "dlti.stack_alignment" = 128 : i64
//  LITTLE-DAG:     "dlti.endianness" = "little"
