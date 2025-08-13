// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=BIG

void foo() {}

// BIG-DAG: dlti.dl_spec =
// BIG-DAG:   #dlti.dl_spec<
// BIG-DAG:     i16 = dense<[16, 32]> : vector<2xi64>,
// BIG-DAG:     i32 = dense<32> : vector<2xi64>,
// BIG-DAG:     i8 = dense<[8, 32]> : vector<2xi64>,
// BIG-DAG:     i1 = dense<8> : vector<2xi64>,
// BIG-DAG:     !llvm.ptr = dense<64> : vector<4xi64>,
// BIG-DAG:     i128 = dense<128> : vector<2xi64>,
// BIG-DAG:     !llvm.ptr<272> = dense<64> : vector<4xi64>,
// BIG-DAG:     i64 = dense<64> : vector<2xi64>,
// BIG-DAG:     !llvm.ptr<270> = dense<32> : vector<4xi64>,
// BIG-DAG:     !llvm.ptr<271> = dense<32> : vector<4xi64>,
// BIG-DAG:     f128 = dense<128> : vector<2xi64>,
// BIG-DAG:     f16 = dense<16> : vector<2xi64>,
// BIG-DAG:     f64 = dense<64> : vector<2xi64>,
// BIG-DAG:     "dlti.stack_alignment" = 128 : i64
// BIG-DAG:     "dlti.endianness" = "big"
