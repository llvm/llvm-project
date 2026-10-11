#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=HOST --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=HOST --input-file=%t.ll %s

// Host shadows of device variables are undef for every type.

struct S { float x, y; };

// HOST: @arr = internal global [4 x %struct.S] undef
__constant__ S arr[4];

// HOST: @ptr = internal global ptr undef
__device__ S *ptr;

// HOST: @dm = internal global i64 undef
__device__ int S::*dm;

// HOST: @mf = internal global { i64, i64 } undef
__device__ float (S::*mf)();
