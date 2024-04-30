// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -O3 -o - | FileCheck %s

#ifdef __HLSL_ENABLE_16_BIT
// CHECK: define noundef i16 @
// CHECK: call i16 @llvm.bitreverse.i16(
uint16_t test_bitreverse_ushort(uint16_t p0)
{
	return reversebits(p0);
}
// CHECK: define noundef <2 x i16> @
// CHECK: call <2 x i16> @llvm.bitreverse.v2i16
uint16_t2 test_bitreverse_ushort2(uint16_t2 p0)
{
	return reversebits(p0);
}
// CHECK: define noundef <3 x i16> @
// CHECK: call <3 x i16> @llvm.bitreverse.v3i16
uint16_t3 test_bitreverse_ushort3(uint16_t3 p0)
{
	return reversebits(p0);
}
// CHECK: define noundef <4 x i16> @
// CHECK: call <4 x i16> @llvm.bitreverse.v4i16
uint16_t4 test_bitreverse_ushort4(uint16_t4 p0)
{
	return reversebits(p0);
}
#endif

// CHECK: define noundef i32 @
// CHECK: call i32 @llvm.bitreverse.i32(
int test_bitreverse_uint(uint p0)
{
	return reversebits(p0);
}
// CHECK: define noundef <2 x i32> @
// CHECK: call <2 x i32> @llvm.bitreverse.v2i32
uint2 test_bitreverse_uint2(uint2 p0)
{
	return reversebits(p0);
}
// CHECK: define noundef <3 x i32> @
// CHECK: call <3 x i32> @llvm.bitreverse.v3i32
uint3 test_bitreverse_uint3(uint3 p0)
{
	return reversebits(p0);
}
// CHECK: define noundef <4 x i32> @
// CHECK: call <4 x i32> @llvm.bitreverse.v4i32
uint4 test_bitreverse_uint4(uint4 p0)
{
	return reversebits(p0);
}

// CHECK: define noundef i64 @
// CHECK: call i64 @llvm.bitreverse.i64(
uint64_t test_bitreverse_long(uint64_t p0)
{
	return reversebits(p0);
}
// CHECK: define noundef <2 x i64> @
// CHECK: call <2 x i64> @llvm.bitreverse.v2i64
uint64_t2 test_bitreverse_long2(uint64_t2 p0)
{
	return reversebits(p0);
}
// CHECK: define noundef <3 x i64> @
// CHECK: call <3 x i64> @llvm.bitreverse.v3i64
uint64_t3 test_bitreverse_long3(uint64_t3 p0)
{
	return reversebits(p0);
}
// CHECK: define noundef <4 x i64> @
// CHECK: call <4 x i64> @llvm.bitreverse.v4i64
uint64_t4 test_bitreverse_long4(uint64_t4 p0)
{
	return reversebits(p0);
}
