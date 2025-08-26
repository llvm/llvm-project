// Regression test for compiler crash
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +sve -target-feature +sve2 -emit-obj -fsanitize=address -fsanitize-address-use-after-scope %s -o - | llvm-objdump -d - | FileCheck %s
// REQUIRES: aarch64-registered-target

#include <arm_sve.h>
int biz(svfloat64_t*);
int foo(){
    svfloat64_t a,b,c;
    return biz(&a)+biz(&b)+biz(&c);
}

//CHECK: 0000000000000000 <_Z3foov>: