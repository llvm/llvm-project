// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

void foo() {}

//      CHECK: module attributes {
//  CHECK-DAG: cir.sob = #cir.signed_overflow_behavior<undefined>,
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
//  CHECK-DAG: llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

