// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN: -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme \
// RUN: -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// REQUIRES: aarch64-registered-target
// expected-no-diagnostics

#include <arm_sve.h>

__attribute__((target("sme2p2")))
void test_svcompact(svbool_t pg, svfloat32_t op) __arm_streaming{
  svcompact(pg, op);
}

void test_svcompact_nofeature(svbool_t pg, svfloat32_t op) __arm_streaming{
  // expected-error@+1 {{'svcompact' needs target feature (sve)|(sme, sme2p2)}}
  svcompact(pg, op);
}