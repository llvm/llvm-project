// REQUIRES: amdgpu-registered-target
// RUN: not %clang_cc1 -debug-info-kind=standalone -x cl -emit-codegen-only -triple=amdgcn-- < %s 2>&1 | FileCheck %s

// CHECK: error: <stdin>:[[@LINE+1]]:0: local memory (480000) exceeds limit (32768) in function 'use_huge_lds'
kernel void use_huge_lds() {
    volatile local int huge[120000];
    huge[0] = 2;
}
