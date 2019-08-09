// RUN: not %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:  -emit-llvm -o - %s -debug-info-kind=limited 2>&1 | FileCheck %s

// Placeholder test for SVE types

// CHECK: cannot yet generate code for SVE type '__SVInt8_t'
// CHECK: cannot yet generate debug info for SVE type '__SVInt8_t'

__SVInt8_t *ptr;
