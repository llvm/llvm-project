// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

RWBuffer<float> Buf;

// CHECK: define linkonce_odr noundef ptr @"??0?$RWBuffer@M@hlsl@@QAA@XZ"
// CHECK-NEXT: entry:

// CHECK: %[[HandleRes:[0-9]+]] = call ptr @llvm.dx.create.handle(i8 1)
// CHECK: store ptr %[[HandleRes]], ptr %h, align 4
