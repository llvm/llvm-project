// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute \
// RUN:   -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:   FileCheck %s

// Preserve byte-address semantics when the storage layout matches another
// buffer type.
// CHECK: %"class.hlsl::ByteAddressBuffer" = type {
// CHECK-SAME: target("spirv.VulkanBuffer", [0 x i8], 12, 0, 1) }
// CHECK: %"class.hlsl::RWByteAddressBuffer" = type {
// CHECK-SAME: target("spirv.VulkanBuffer", [0 x i8], 12, 1, 1) }
// CHECK: %"class.hlsl::StructuredBuffer" = type {
// CHECK-SAME: target("spirv.VulkanBuffer", [0 x i32], 12, 0) }

ByteAddressBuffer ByteBuffer;
RWByteAddressBuffer RWByteBuffer;
StructuredBuffer<uint> Structured;

[numthreads(1, 1, 1)]
void main() {}
