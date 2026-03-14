// REQUIRES: amdgpu-registered-target
// RUN: not %clang_cc1 -x cl -emit-codegen-only -triple=amdgcn-- < %s 2>&1 | FileCheck %s

// CHECK: <stdin>:[[@LINE+1]]:13: error: local memory (480000) exceeds limit (32768) in 'use_huge_lds'
kernel void use_huge_lds() {
    volatile local int huge[120000];
    huge[0] = 2;
}
