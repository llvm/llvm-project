// REQUIRES: nvptx-registered-target

// Check that the CUDA driver translates the legacy Clang option to the
// shortptr ABI.
// RUN: %clang --target=x86_64-linux-gnu -x cuda --cuda-device-only \
// RUN:   -fcuda-short-ptr -nocudainc -S -o /dev/null %s

// Check that the NVPTX shortptr ABI can be selected directly.
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-abi shortptr \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=SHORT-DL
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-abi shortptr \
// RUN:   -S -o - %s | FileCheck %s --check-prefix=PTX

// SHORT-DL: target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
// PTX: .address_size 64
// PTX: .visible .func f(
// PTX: .param .b32 f_param_0

void f(__attribute__((address_space(3))) int *p) { *p = 0; }
