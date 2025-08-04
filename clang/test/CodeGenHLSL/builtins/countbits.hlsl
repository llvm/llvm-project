// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -O3 -o - | FileCheck %s

#ifdef __HLSL_ENABLE_16_BIT
// CHECK-LABEL: test_countbits_ushort
// CHECK: [[A:%.*]] = call i16 @llvm.ctpop.i16
// CHECK-NEXT: zext i16 [[A]] to i32
uint test_countbits_ushort(uint16_t p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_short
// CHECK: [[A:%.*]] = call i16 @llvm.ctpop.i16
// CHECK-NEXT: sext i16 [[A]] to i32
uint test_countbits_short(int16_t p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_ushort2
// CHECK: [[A:%.*]] = call <2 x i16> @llvm.ctpop.v2i16
// CHECK-NEXT: zext <2 x i16> [[A]] to <2 x i32>
uint2 test_countbits_ushort2(uint16_t2 p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_ushort3
// CHECK: [[A:%.*]] = call <3 x i16> @llvm.ctpop.v3i16
// CHECK-NEXT: zext <3 x i16> [[A]] to <3 x i32>
uint3 test_countbits_ushort3(uint16_t3 p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_ushort4
// CHECK: [[A:%.*]] = call <4 x i16> @llvm.ctpop.v4i16
// CHECK-NEXT: zext <4 x i16> [[A]] to <4 x i32>
uint4 test_countbits_ushort4(uint16_t4 p0)
{
	return countbits(p0);
}
#endif

// CHECK-LABEL: test_countbits_uint
// CHECK: call i32 @llvm.ctpop.i32
uint test_countbits_uint(uint p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_int
// CHECK: call i32 @llvm.ctpop.i32
uint test_countbits_int(int p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_uint2
// CHECK: call <2 x i32> @llvm.ctpop.v2i32
uint2 test_countbits_uint2(uint2 p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_uint3
// CHECK: call <3 x i32> @llvm.ctpop.v3i32
uint3 test_countbits_uint3(uint3 p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_uint4
// CHECK: call <4 x i32> @llvm.ctpop.v4i32
uint4 test_countbits_uint4(uint4 p0)
{
	return countbits(p0);
}

// CHECK-LABEL: test_countbits_long
// CHECK: [[A:%.*]] = call i64 @llvm.ctpop.i64
// CHECK-NEXT: trunc i64 [[A]] to i32
uint test_countbits_long(uint64_t p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_slong
// CHECK: [[A:%.*]] = call i64 @llvm.ctpop.i64
// CHECK-NEXT: trunc i64 [[A]] to i32
uint test_countbits_slong(int64_t p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_long2
// CHECK: [[A:%.*]] = call <2 x i64> @llvm.ctpop.v2i64
// CHECK-NEXT: trunc <2 x i64> [[A]] to <2 x i32>
uint2 test_countbits_long2(uint64_t2 p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_long3
// CHECK: [[A:%.*]] = call <3 x i64> @llvm.ctpop.v3i64
// CHECK-NEXT: trunc <3 x i64> [[A]] to <3 x i32>
uint3 test_countbits_long3(uint64_t3 p0)
{
	return countbits(p0);
}
// CHECK-LABEL: test_countbits_long4
// CHECK: [[A:%.*]] = call <4 x i64> @llvm.ctpop.v4i64
// CHECK-NEXT: trunc <4 x i64> [[A]] to <4 x i32>
uint4 test_countbits_long4(uint64_t4 p0)
{
	return countbits(p0);
}
