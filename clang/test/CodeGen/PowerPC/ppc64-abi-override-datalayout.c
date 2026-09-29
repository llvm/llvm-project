// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -target-abi elfv2 %s -o - -emit-llvm | FileCheck %s

// REQUIRES: powerpc-registered-target

// Make sure that overriding the ABI to ELFv2 on a target that defaults to
// ELFv1 changes the data layout:

// CHECK: target datalayout = "E-m:e-Fn32-i64:64-i128:128-n32:64-S128-v256:256:256-v512:512:512"
