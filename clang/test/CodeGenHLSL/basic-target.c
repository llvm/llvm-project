// RUN: %clang -target dxil-pc-shadermodel6.0-pixel -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -target dxil-pc-shadermodel6.0-vertex -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -target dxil-pc-shadermodel6.0-compute -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -target dxil-pc-shadermodel6.0-library -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -target dxil-pc-shadermodel6.0-hull -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -target dxil-pc-shadermodel6.0-domain -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -target dxil-pc-shadermodel6.0-geometry -S -emit-llvm -o - %s | FileCheck %s

// CHECK: target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
// CHECK: target triple = "dxil-pc-shadermodel6.0-{{[a-z]+}}"
