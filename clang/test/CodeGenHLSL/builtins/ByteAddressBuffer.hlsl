// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm %s -o - | FileCheck %s

// CHECK: class.hlsl::ByteAddressBuffer" = type <{ target("dx.RawBuffer", i8, 1, 1)
// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_ByteAddressBuffer.hlsl, ptr null }]

ByteAddressBuffer Buffer;

//CHECK: define internal void @_GLOBAL__sub_I_ByteAddressBuffer.hlsl()
