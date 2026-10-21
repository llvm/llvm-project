// REQUIRES: nvptx-registered-target
// RUN: %clangxx -O1 -S --cuda-device-only --offload-arch=sm_32 -nocudalib -nocudainc -target x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefixes=CHECK,LINUX
// RUN: %clangxx -O1 -S --cuda-device-only --offload-arch=sm_32 -nocudalib -nocudainc -target x86_64-pc-windows-msvc %s -o - | FileCheck %s --check-prefixes=CHECK,WINDOWS


#define __device__ __attribute__((device))
#define warpSize 32
#define memcpy __builtin_memcpy

// Define missing types for standalone testing
struct char2 { char x, y; };
struct char4 { char x, y, z, w; };
struct short2 { short x, y; };
struct short4 { short x, y, z, w; };
struct int2 { int x, y; };
struct int4 { int x, y, z, w; };
struct longlong2 { long long x, y; };
struct uchar2 { unsigned char x, y; };
struct uchar4 { unsigned char x, y, z, w; };
struct ushort2 { unsigned short x, y; };
struct ushort4 { unsigned short x, y, z, w; };
struct uint2 { unsigned int x, y; };
struct uint4 { unsigned int x, y, z, w; };
struct ulonglong2 { unsigned long long x, y; };
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct double2 { double x, y; };

#include "__clang_cuda_intrinsics.h"

__device__ void test_loads_cg(void *ptr) {
  // CHECK-LABEL: .func _Z13test_loads_cgPv
  // CHECK: ld.global.cg.s8
  __ldcg(((const char *)ptr) + 0);
  // CHECK: ld.global.cg.s32
  __ldcg(((const int *)ptr) + 1);
  // LINUX: ld.global.cg.s64
  // WINDOWS: ld.global.cg.s32
  __ldcg(((const long *)ptr) + 2);
  // LINUX: ld.global.cg.u64
  // WINDOWS: ld.global.cg.u32
  __ldcg(((const unsigned long *)ptr) + 3);
  // CHECK: ld.global.cg.s64
  __ldcg(((const long long *)ptr) + 4);
  // CHECK: ld.global.cg.f32
  __ldcg(((const float *)ptr) + 5);
  // CHECK: ld.global.cg.f64
  __ldcg(((const double *)ptr) + 6);
  // CHECK: ld.global.cg.v2.s8
  __ldcg(((const char2 *)ptr) + 7);
  // CHECK: ld.global.cg.v4.s8
  __ldcg(((const char4 *)ptr) + 8);
  // CHECK: ld.global.cg.v2.s16
  __ldcg(((const short2 *)ptr) + 9);
  // CHECK: ld.global.cg.v4.s16
  __ldcg(((const short4 *)ptr) + 10);
  // CHECK: ld.global.cg.v2.s32
  __ldcg(((const int2 *)ptr) + 11);
  // CHECK: ld.global.cg.v4.s32
  __ldcg(((const int4 *)ptr) + 12);
  // CHECK: ld.global.cg.v2.s64
  __ldcg(((const longlong2 *)ptr) + 13);
  // CHECK: ld.global.cg.v2.u8
  __ldcg(((const uchar2 *)ptr) + 14);
  // CHECK: ld.global.cg.v4.u8
  __ldcg(((const uchar4 *)ptr) + 15);
  // CHECK: ld.global.cg.v2.u16
  __ldcg(((const ushort2 *)ptr) + 16);
  // CHECK: ld.global.cg.v4.u16
  __ldcg(((const ushort4 *)ptr) + 17);
  // CHECK: ld.global.cg.v2.u32
  __ldcg(((const uint2 *)ptr) + 18);
  // CHECK: ld.global.cg.v4.u32
  __ldcg(((const uint4 *)ptr) + 19);
  // CHECK: ld.global.cg.v2.u64
  __ldcg(((const ulonglong2 *)ptr) + 20);
  // CHECK: ld.global.cg.v2.f32
  __ldcg(((const float2 *)ptr) + 21);
  // CHECK: ld.global.cg.v4.f32
  __ldcg(((const float4 *)ptr) + 22);
  // CHECK: ld.global.cg.v2.f64
  __ldcg(((const double2 *)ptr) + 23);
}
__device__ void test_loads_cv(void *ptr) {
  // CHECK-LABEL: .func _Z13test_loads_cvPv
  // CHECK: ld.global.cv.s8
  volatile char v_0 = __ldcv(((const char *)ptr) + 0);
  // CHECK: ld.global.cv.s32
  volatile int v_1 = __ldcv(((const int *)ptr) + 1);
  // LINUX: ld.global.cv.s64
  // WINDOWS: ld.global.cv.s32
  volatile long v_2 = __ldcv(((const long *)ptr) + 2);
  // LINUX: ld.global.cv.u64
  // WINDOWS: ld.global.cv.u32
  volatile unsigned long v_3 = __ldcv(((const unsigned long *)ptr) + 3);
  // CHECK: ld.global.cv.s64
  volatile long long v_4 = __ldcv(((const long long *)ptr) + 4);
  // CHECK: ld.global.cv.f32
  volatile float v_5 = __ldcv(((const float *)ptr) + 5);
  // CHECK: ld.global.cv.f64
  volatile double v_6 = __ldcv(((const double *)ptr) + 6);
  // CHECK: ld.global.cv.v2.s8
  volatile char2 v_7 = __ldcv(((const char2 *)ptr) + 7);
  // CHECK: ld.global.cv.v4.s8
  volatile char4 v_8 = __ldcv(((const char4 *)ptr) + 8);
  // CHECK: ld.global.cv.v2.s16
  volatile short2 v_9 = __ldcv(((const short2 *)ptr) + 9);
  // CHECK: ld.global.cv.v4.s16
  volatile short4 v_10 = __ldcv(((const short4 *)ptr) + 10);
  // CHECK: ld.global.cv.v2.s32
  volatile int2 v_11 = __ldcv(((const int2 *)ptr) + 11);
  // CHECK: ld.global.cv.v4.s32
  volatile int4 v_12 = __ldcv(((const int4 *)ptr) + 12);
  // CHECK: ld.global.cv.v2.s64
  volatile longlong2 v_13 = __ldcv(((const longlong2 *)ptr) + 13);
  // CHECK: ld.global.cv.v2.u8
  volatile uchar2 v_14 = __ldcv(((const uchar2 *)ptr) + 14);
  // CHECK: ld.global.cv.v4.u8
  volatile uchar4 v_15 = __ldcv(((const uchar4 *)ptr) + 15);
  // CHECK: ld.global.cv.v2.u16
  volatile ushort2 v_16 = __ldcv(((const ushort2 *)ptr) + 16);
  // CHECK: ld.global.cv.v4.u16
  volatile ushort4 v_17 = __ldcv(((const ushort4 *)ptr) + 17);
  // CHECK: ld.global.cv.v2.u32
  volatile uint2 v_18 = __ldcv(((const uint2 *)ptr) + 18);
  // CHECK: ld.global.cv.v4.u32
  volatile uint4 v_19 = __ldcv(((const uint4 *)ptr) + 19);
  // CHECK: ld.global.cv.v2.u64
  volatile ulonglong2 v_20 = __ldcv(((const ulonglong2 *)ptr) + 20);
  // CHECK: ld.global.cv.v2.f32
  volatile float2 v_21 = __ldcv(((const float2 *)ptr) + 21);
  // CHECK: ld.global.cv.v4.f32
  volatile float4 v_22 = __ldcv(((const float4 *)ptr) + 22);
  // CHECK: ld.global.cv.v2.f64
  volatile double2 v_23 = __ldcv(((const double2 *)ptr) + 23);
}
__device__ void test_loads_cs(void *ptr) {
  // CHECK-LABEL: .func _Z13test_loads_csPv
  // CHECK: ld.global.cs.s8
  __ldcs(((const char *)ptr) + 0);
  // CHECK: ld.global.cs.s32
  __ldcs(((const int *)ptr) + 1);
  // LINUX: ld.global.cs.s64
  // WINDOWS: ld.global.cs.s32
  __ldcs(((const long *)ptr) + 2);
  // LINUX: ld.global.cs.u64
  // WINDOWS: ld.global.cs.u32
  __ldcs(((const unsigned long *)ptr) + 3);
  // CHECK: ld.global.cs.s64
  __ldcs(((const long long *)ptr) + 4);
  // CHECK: ld.global.cs.f32
  __ldcs(((const float *)ptr) + 5);
  // CHECK: ld.global.cs.f64
  __ldcs(((const double *)ptr) + 6);
  // CHECK: ld.global.cs.v2.s8
  __ldcs(((const char2 *)ptr) + 7);
  // CHECK: ld.global.cs.v4.s8
  __ldcs(((const char4 *)ptr) + 8);
  // CHECK: ld.global.cs.v2.s16
  __ldcs(((const short2 *)ptr) + 9);
  // CHECK: ld.global.cs.v4.s16
  __ldcs(((const short4 *)ptr) + 10);
  // CHECK: ld.global.cs.v2.s32
  __ldcs(((const int2 *)ptr) + 11);
  // CHECK: ld.global.cs.v4.s32
  __ldcs(((const int4 *)ptr) + 12);
  // CHECK: ld.global.cs.v2.s64
  __ldcs(((const longlong2 *)ptr) + 13);
  // CHECK: ld.global.cs.v2.u8
  __ldcs(((const uchar2 *)ptr) + 14);
  // CHECK: ld.global.cs.v4.u8
  __ldcs(((const uchar4 *)ptr) + 15);
  // CHECK: ld.global.cs.v2.u16
  __ldcs(((const ushort2 *)ptr) + 16);
  // CHECK: ld.global.cs.v4.u16
  __ldcs(((const ushort4 *)ptr) + 17);
  // CHECK: ld.global.cs.v2.u32
  __ldcs(((const uint2 *)ptr) + 18);
  // CHECK: ld.global.cs.v4.u32
  __ldcs(((const uint4 *)ptr) + 19);
  // CHECK: ld.global.cs.v2.u64
  __ldcs(((const ulonglong2 *)ptr) + 20);
  // CHECK: ld.global.cs.v2.f32
  __ldcs(((const float2 *)ptr) + 21);
  // CHECK: ld.global.cs.v4.f32
  __ldcs(((const float4 *)ptr) + 22);
  // CHECK: ld.global.cs.v2.f64
  __ldcs(((const double2 *)ptr) + 23);
}
__device__ void test_loads_ca(void *ptr) {
  // CHECK-LABEL: .func _Z13test_loads_caPv
  // CHECK: ld.global.ca.s8
  __ldca(((const char *)ptr) + 0);
  // CHECK: ld.global.ca.s32
  __ldca(((const int *)ptr) + 1);
  // LINUX: ld.global.ca.s64
  // WINDOWS: ld.global.ca.s32
  __ldca(((const long *)ptr) + 2);
  // LINUX: ld.global.ca.u64
  // WINDOWS: ld.global.ca.u32
  __ldca(((const unsigned long *)ptr) + 3);
  // CHECK: ld.global.ca.s64
  __ldca(((const long long *)ptr) + 4);
  // CHECK: ld.global.ca.f32
  __ldca(((const float *)ptr) + 5);
  // CHECK: ld.global.ca.f64
  __ldca(((const double *)ptr) + 6);
  // CHECK: ld.global.ca.v2.s8
  __ldca(((const char2 *)ptr) + 7);
  // CHECK: ld.global.ca.v4.s8
  __ldca(((const char4 *)ptr) + 8);
  // CHECK: ld.global.ca.v2.s16
  __ldca(((const short2 *)ptr) + 9);
  // CHECK: ld.global.ca.v4.s16
  __ldca(((const short4 *)ptr) + 10);
  // CHECK: ld.global.ca.v2.s32
  __ldca(((const int2 *)ptr) + 11);
  // CHECK: ld.global.ca.v4.s32
  __ldca(((const int4 *)ptr) + 12);
  // CHECK: ld.global.ca.v2.s64
  __ldca(((const longlong2 *)ptr) + 13);
  // CHECK: ld.global.ca.v2.u8
  __ldca(((const uchar2 *)ptr) + 14);
  // CHECK: ld.global.ca.v4.u8
  __ldca(((const uchar4 *)ptr) + 15);
  // CHECK: ld.global.ca.v2.u16
  __ldca(((const ushort2 *)ptr) + 16);
  // CHECK: ld.global.ca.v4.u16
  __ldca(((const ushort4 *)ptr) + 17);
  // CHECK: ld.global.ca.v2.u32
  __ldca(((const uint2 *)ptr) + 18);
  // CHECK: ld.global.ca.v4.u32
  __ldca(((const uint4 *)ptr) + 19);
  // CHECK: ld.global.ca.v2.u64
  __ldca(((const ulonglong2 *)ptr) + 20);
  // CHECK: ld.global.ca.v2.f32
  __ldca(((const float2 *)ptr) + 21);
  // CHECK: ld.global.ca.v4.f32
  __ldca(((const float4 *)ptr) + 22);
  // CHECK: ld.global.ca.v2.f64
  __ldca(((const double2 *)ptr) + 23);
}
__device__ void test_loads_lu(void *ptr) {
  // CHECK-LABEL: .func _Z13test_loads_luPv
  // CHECK: ld.global.lu.s8
  volatile char v_0 = __ldlu(((const char *)ptr) + 0);
  // CHECK: ld.global.lu.s32
  volatile int v_1 = __ldlu(((const int *)ptr) + 1);
  // LINUX: ld.global.lu.s64
  // WINDOWS: ld.global.lu.s32
  volatile long v_2 = __ldlu(((const long *)ptr) + 2);
  // LINUX: ld.global.lu.u64
  // WINDOWS: ld.global.lu.u32
  volatile unsigned long v_3 = __ldlu(((const unsigned long *)ptr) + 3);
  // CHECK: ld.global.lu.s64
  volatile long long v_4 = __ldlu(((const long long *)ptr) + 4);
  // CHECK: ld.global.lu.f32
  volatile float v_5 = __ldlu(((const float *)ptr) + 5);
  // CHECK: ld.global.lu.f64
  volatile double v_6 = __ldlu(((const double *)ptr) + 6);
  // CHECK: ld.global.lu.v2.s8
  volatile char2 v_7 = __ldlu(((const char2 *)ptr) + 7);
  // CHECK: ld.global.lu.v4.s8
  volatile char4 v_8 = __ldlu(((const char4 *)ptr) + 8);
  // CHECK: ld.global.lu.v2.s16
  volatile short2 v_9 = __ldlu(((const short2 *)ptr) + 9);
  // CHECK: ld.global.lu.v4.s16
  volatile short4 v_10 = __ldlu(((const short4 *)ptr) + 10);
  // CHECK: ld.global.lu.v2.s32
  volatile int2 v_11 = __ldlu(((const int2 *)ptr) + 11);
  // CHECK: ld.global.lu.v4.s32
  volatile int4 v_12 = __ldlu(((const int4 *)ptr) + 12);
  // CHECK: ld.global.lu.v2.s64
  volatile longlong2 v_13 = __ldlu(((const longlong2 *)ptr) + 13);
  // CHECK: ld.global.lu.v2.u8
  volatile uchar2 v_14 = __ldlu(((const uchar2 *)ptr) + 14);
  // CHECK: ld.global.lu.v4.u8
  volatile uchar4 v_15 = __ldlu(((const uchar4 *)ptr) + 15);
  // CHECK: ld.global.lu.v2.u16
  volatile ushort2 v_16 = __ldlu(((const ushort2 *)ptr) + 16);
  // CHECK: ld.global.lu.v4.u16
  volatile ushort4 v_17 = __ldlu(((const ushort4 *)ptr) + 17);
  // CHECK: ld.global.lu.v2.u32
  volatile uint2 v_18 = __ldlu(((const uint2 *)ptr) + 18);
  // CHECK: ld.global.lu.v4.u32
  volatile uint4 v_19 = __ldlu(((const uint4 *)ptr) + 19);
  // CHECK: ld.global.lu.v2.u64
  volatile ulonglong2 v_20 = __ldlu(((const ulonglong2 *)ptr) + 20);
  // CHECK: ld.global.lu.v2.f32
  volatile float2 v_21 = __ldlu(((const float2 *)ptr) + 21);
  // CHECK: ld.global.lu.v4.f32
  volatile float4 v_22 = __ldlu(((const float4 *)ptr) + 22);
  // CHECK: ld.global.lu.v2.f64
  volatile double2 v_23 = __ldlu(((const double2 *)ptr) + 23);
}
__device__ void test_stores_wt(void *ptr, int val) {
  // CHECK-LABEL: .func _Z14test_stores_wtPvi
  // CHECK: st.global.wt.s8
  __stwt(((char *)ptr) + 0, (char)val);
  // CHECK: st.global.wt.s32
  __stwt(((int *)ptr) + 1, (int)val);
  // LINUX: st.global.wt.s64
  // WINDOWS: st.global.wt.s32
  __stwt(((long *)ptr) + 2, (long)val);
  // LINUX: st.global.wt.u64
  // WINDOWS: st.global.wt.u32
  __stwt(((unsigned long *)ptr) + 3, (unsigned long)val);
  // CHECK: st.global.wt.s64
  __stwt(((long long *)ptr) + 4, (long long)val);
  // CHECK: st.global.wt.f32
  __stwt(((float *)ptr) + 5, (float)val);
  // CHECK: st.global.wt.f64
  __stwt(((double *)ptr) + 6, (double)val);
  // CHECK: st.global.wt.v2.s8
  { char2 v = {(char)val, (char)val}; __stwt(((char2 *)ptr) + 7, v); }
  // CHECK: st.global.wt.v4.s8
  { char4 v = {(char)val, (char)val, (char)val, (char)val}; __stwt(((char4 *)ptr) + 8, v); }
  // CHECK: st.global.wt.v2.s16
  { short2 v = {(short)val, (short)val}; __stwt(((short2 *)ptr) + 9, v); }
  // CHECK: st.global.wt.v4.s16
  { short4 v = {(short)val, (short)val, (short)val, (short)val}; __stwt(((short4 *)ptr) + 10, v); }
  // CHECK: st.global.wt.v2.s32
  { int2 v = {(int)val, (int)val}; __stwt(((int2 *)ptr) + 11, v); }
  // CHECK: st.global.wt.v4.s32
  { int4 v = {(int)val, (int)val, (int)val, (int)val}; __stwt(((int4 *)ptr) + 12, v); }
  // CHECK: st.global.wt.v2.s64
  { longlong2 v = {(long long)val, (long long)val}; __stwt(((longlong2 *)ptr) + 13, v); }
  // CHECK: st.global.wt.v2.u8
  { uchar2 v = {(unsigned char)val, (unsigned char)val}; __stwt(((uchar2 *)ptr) + 14, v); }
  // CHECK: st.global.wt.v4.u8
  { uchar4 v = {(unsigned char)val, (unsigned char)val, (unsigned char)val, (unsigned char)val}; __stwt(((uchar4 *)ptr) + 15, v); }
  // CHECK: st.global.wt.v2.u16
  { ushort2 v = {(unsigned short)val, (unsigned short)val}; __stwt(((ushort2 *)ptr) + 16, v); }
  // CHECK: st.global.wt.v4.u16
  { ushort4 v = {(unsigned short)val, (unsigned short)val, (unsigned short)val, (unsigned short)val}; __stwt(((ushort4 *)ptr) + 17, v); }
  // CHECK: st.global.wt.v2.u32
  { uint2 v = {(unsigned int)val, (unsigned int)val}; __stwt(((uint2 *)ptr) + 18, v); }
  // CHECK: st.global.wt.v4.u32
  { uint4 v = {(unsigned int)val, (unsigned int)val, (unsigned int)val, (unsigned int)val}; __stwt(((uint4 *)ptr) + 19, v); }
  // CHECK: st.global.wt.v2.u64
  { ulonglong2 v = {(unsigned long long)val, (unsigned long long)val}; __stwt(((ulonglong2 *)ptr) + 20, v); }
  // CHECK: st.global.wt.v2.f32
  { float2 v = {(float)val, (float)val}; __stwt(((float2 *)ptr) + 21, v); }
  // CHECK: st.global.wt.v4.f32
  { float4 v = {(float)val, (float)val, (float)val, (float)val}; __stwt(((float4 *)ptr) + 22, v); }
  // CHECK: st.global.wt.v2.f64
  { double2 v = {(double)val, (double)val}; __stwt(((double2 *)ptr) + 23, v); }
}
__device__ void test_stores_wb(void *ptr, int val) {
  // CHECK-LABEL: .func _Z14test_stores_wbPvi
  // CHECK: st.global.wb.s8
  __stwb(((char *)ptr) + 0, (char)val);
  // CHECK: st.global.wb.s32
  __stwb(((int *)ptr) + 1, (int)val);
  // LINUX: st.global.wb.s64
  // WINDOWS: st.global.wb.s32
  __stwb(((long *)ptr) + 2, (long)val);
  // LINUX: st.global.wb.u64
  // WINDOWS: st.global.wb.u32
  __stwb(((unsigned long *)ptr) + 3, (unsigned long)val);
  // CHECK: st.global.wb.s64
  __stwb(((long long *)ptr) + 4, (long long)val);
  // CHECK: st.global.wb.f32
  __stwb(((float *)ptr) + 5, (float)val);
  // CHECK: st.global.wb.f64
  __stwb(((double *)ptr) + 6, (double)val);
  // CHECK: st.global.wb.v2.s8
  { char2 v = {(char)val, (char)val}; __stwb(((char2 *)ptr) + 7, v); }
  // CHECK: st.global.wb.v4.s8
  { char4 v = {(char)val, (char)val, (char)val, (char)val}; __stwb(((char4 *)ptr) + 8, v); }
  // CHECK: st.global.wb.v2.s16
  { short2 v = {(short)val, (short)val}; __stwb(((short2 *)ptr) + 9, v); }
  // CHECK: st.global.wb.v4.s16
  { short4 v = {(short)val, (short)val, (short)val, (short)val}; __stwb(((short4 *)ptr) + 10, v); }
  // CHECK: st.global.wb.v2.s32
  { int2 v = {(int)val, (int)val}; __stwb(((int2 *)ptr) + 11, v); }
  // CHECK: st.global.wb.v4.s32
  { int4 v = {(int)val, (int)val, (int)val, (int)val}; __stwb(((int4 *)ptr) + 12, v); }
  // CHECK: st.global.wb.v2.s64
  { longlong2 v = {(long long)val, (long long)val}; __stwb(((longlong2 *)ptr) + 13, v); }
  // CHECK: st.global.wb.v2.u8
  { uchar2 v = {(unsigned char)val, (unsigned char)val}; __stwb(((uchar2 *)ptr) + 14, v); }
  // CHECK: st.global.wb.v4.u8
  { uchar4 v = {(unsigned char)val, (unsigned char)val, (unsigned char)val, (unsigned char)val}; __stwb(((uchar4 *)ptr) + 15, v); }
  // CHECK: st.global.wb.v2.u16
  { ushort2 v = {(unsigned short)val, (unsigned short)val}; __stwb(((ushort2 *)ptr) + 16, v); }
  // CHECK: st.global.wb.v4.u16
  { ushort4 v = {(unsigned short)val, (unsigned short)val, (unsigned short)val, (unsigned short)val}; __stwb(((ushort4 *)ptr) + 17, v); }
  // CHECK: st.global.wb.v2.u32
  { uint2 v = {(unsigned int)val, (unsigned int)val}; __stwb(((uint2 *)ptr) + 18, v); }
  // CHECK: st.global.wb.v4.u32
  { uint4 v = {(unsigned int)val, (unsigned int)val, (unsigned int)val, (unsigned int)val}; __stwb(((uint4 *)ptr) + 19, v); }
  // CHECK: st.global.wb.v2.u64
  { ulonglong2 v = {(unsigned long long)val, (unsigned long long)val}; __stwb(((ulonglong2 *)ptr) + 20, v); }
  // CHECK: st.global.wb.v2.f32
  { float2 v = {(float)val, (float)val}; __stwb(((float2 *)ptr) + 21, v); }
  // CHECK: st.global.wb.v4.f32
  { float4 v = {(float)val, (float)val, (float)val, (float)val}; __stwb(((float4 *)ptr) + 22, v); }
  // CHECK: st.global.wb.v2.f64
  { double2 v = {(double)val, (double)val}; __stwb(((double2 *)ptr) + 23, v); }
}
__device__ void test_stores_cg(void *ptr, int val) {
  // CHECK-LABEL: .func _Z14test_stores_cgPvi
  // CHECK: st.global.cg.s8
  __stcg(((char *)ptr) + 0, (char)val);
  // CHECK: st.global.cg.s32
  __stcg(((int *)ptr) + 1, (int)val);
  // LINUX: st.global.cg.s64
  // WINDOWS: st.global.cg.s32
  __stcg(((long *)ptr) + 2, (long)val);
  // LINUX: st.global.cg.u64
  // WINDOWS: st.global.cg.u32
  __stcg(((unsigned long *)ptr) + 3, (unsigned long)val);
  // CHECK: st.global.cg.s64
  __stcg(((long long *)ptr) + 4, (long long)val);
  // CHECK: st.global.cg.f32
  __stcg(((float *)ptr) + 5, (float)val);
  // CHECK: st.global.cg.f64
  __stcg(((double *)ptr) + 6, (double)val);
  // CHECK: st.global.cg.v2.s8
  { char2 v = {(char)val, (char)val}; __stcg(((char2 *)ptr) + 7, v); }
  // CHECK: st.global.cg.v4.s8
  { char4 v = {(char)val, (char)val, (char)val, (char)val}; __stcg(((char4 *)ptr) + 8, v); }
  // CHECK: st.global.cg.v2.s16
  { short2 v = {(short)val, (short)val}; __stcg(((short2 *)ptr) + 9, v); }
  // CHECK: st.global.cg.v4.s16
  { short4 v = {(short)val, (short)val, (short)val, (short)val}; __stcg(((short4 *)ptr) + 10, v); }
  // CHECK: st.global.cg.v2.s32
  { int2 v = {(int)val, (int)val}; __stcg(((int2 *)ptr) + 11, v); }
  // CHECK: st.global.cg.v4.s32
  { int4 v = {(int)val, (int)val, (int)val, (int)val}; __stcg(((int4 *)ptr) + 12, v); }
  // CHECK: st.global.cg.v2.s64
  { longlong2 v = {(long long)val, (long long)val}; __stcg(((longlong2 *)ptr) + 13, v); }
  // CHECK: st.global.cg.v2.u8
  { uchar2 v = {(unsigned char)val, (unsigned char)val}; __stcg(((uchar2 *)ptr) + 14, v); }
  // CHECK: st.global.cg.v4.u8
  { uchar4 v = {(unsigned char)val, (unsigned char)val, (unsigned char)val, (unsigned char)val}; __stcg(((uchar4 *)ptr) + 15, v); }
  // CHECK: st.global.cg.v2.u16
  { ushort2 v = {(unsigned short)val, (unsigned short)val}; __stcg(((ushort2 *)ptr) + 16, v); }
  // CHECK: st.global.cg.v4.u16
  { ushort4 v = {(unsigned short)val, (unsigned short)val, (unsigned short)val, (unsigned short)val}; __stcg(((ushort4 *)ptr) + 17, v); }
  // CHECK: st.global.cg.v2.u32
  { uint2 v = {(unsigned int)val, (unsigned int)val}; __stcg(((uint2 *)ptr) + 18, v); }
  // CHECK: st.global.cg.v4.u32
  { uint4 v = {(unsigned int)val, (unsigned int)val, (unsigned int)val, (unsigned int)val}; __stcg(((uint4 *)ptr) + 19, v); }
  // CHECK: st.global.cg.v2.u64
  { ulonglong2 v = {(unsigned long long)val, (unsigned long long)val}; __stcg(((ulonglong2 *)ptr) + 20, v); }
  // CHECK: st.global.cg.v2.f32
  { float2 v = {(float)val, (float)val}; __stcg(((float2 *)ptr) + 21, v); }
  // CHECK: st.global.cg.v4.f32
  { float4 v = {(float)val, (float)val, (float)val, (float)val}; __stcg(((float4 *)ptr) + 22, v); }
  // CHECK: st.global.cg.v2.f64
  { double2 v = {(double)val, (double)val}; __stcg(((double2 *)ptr) + 23, v); }
}
__device__ void test_stores_cs(void *ptr, int val) {
  // CHECK-LABEL: .func _Z14test_stores_csPvi
  // CHECK: st.global.cs.s8
  __stcs(((char *)ptr) + 0, (char)val);
  // CHECK: st.global.cs.s32
  __stcs(((int *)ptr) + 1, (int)val);
  // LINUX: st.global.cs.s64
  // WINDOWS: st.global.cs.s32
  __stcs(((long *)ptr) + 2, (long)val);
  // LINUX: st.global.cs.u64
  // WINDOWS: st.global.cs.u32
  __stcs(((unsigned long *)ptr) + 3, (unsigned long)val);
  // CHECK: st.global.cs.s64
  __stcs(((long long *)ptr) + 4, (long long)val);
  // CHECK: st.global.cs.f32
  __stcs(((float *)ptr) + 5, (float)val);
  // CHECK: st.global.cs.f64
  __stcs(((double *)ptr) + 6, (double)val);
  // CHECK: st.global.cs.v2.s8
  { char2 v = {(char)val, (char)val}; __stcs(((char2 *)ptr) + 7, v); }
  // CHECK: st.global.cs.v4.s8
  { char4 v = {(char)val, (char)val, (char)val, (char)val}; __stcs(((char4 *)ptr) + 8, v); }
  // CHECK: st.global.cs.v2.s16
  { short2 v = {(short)val, (short)val}; __stcs(((short2 *)ptr) + 9, v); }
  // CHECK: st.global.cs.v4.s16
  { short4 v = {(short)val, (short)val, (short)val, (short)val}; __stcs(((short4 *)ptr) + 10, v); }
  // CHECK: st.global.cs.v2.s32
  { int2 v = {(int)val, (int)val}; __stcs(((int2 *)ptr) + 11, v); }
  // CHECK: st.global.cs.v4.s32
  { int4 v = {(int)val, (int)val, (int)val, (int)val}; __stcs(((int4 *)ptr) + 12, v); }
  // CHECK: st.global.cs.v2.s64
  { longlong2 v = {(long long)val, (long long)val}; __stcs(((longlong2 *)ptr) + 13, v); }
  // CHECK: st.global.cs.v2.u8
  { uchar2 v = {(unsigned char)val, (unsigned char)val}; __stcs(((uchar2 *)ptr) + 14, v); }
  // CHECK: st.global.cs.v4.u8
  { uchar4 v = {(unsigned char)val, (unsigned char)val, (unsigned char)val, (unsigned char)val}; __stcs(((uchar4 *)ptr) + 15, v); }
  // CHECK: st.global.cs.v2.u16
  { ushort2 v = {(unsigned short)val, (unsigned short)val}; __stcs(((ushort2 *)ptr) + 16, v); }
  // CHECK: st.global.cs.v4.u16
  { ushort4 v = {(unsigned short)val, (unsigned short)val, (unsigned short)val, (unsigned short)val}; __stcs(((ushort4 *)ptr) + 17, v); }
  // CHECK: st.global.cs.v2.u32
  { uint2 v = {(unsigned int)val, (unsigned int)val}; __stcs(((uint2 *)ptr) + 18, v); }
  // CHECK: st.global.cs.v4.u32
  { uint4 v = {(unsigned int)val, (unsigned int)val, (unsigned int)val, (unsigned int)val}; __stcs(((uint4 *)ptr) + 19, v); }
  // CHECK: st.global.cs.v2.u64
  { ulonglong2 v = {(unsigned long long)val, (unsigned long long)val}; __stcs(((ulonglong2 *)ptr) + 20, v); }
  // CHECK: st.global.cs.v2.f32
  { float2 v = {(float)val, (float)val}; __stcs(((float2 *)ptr) + 21, v); }
  // CHECK: st.global.cs.v4.f32
  { float4 v = {(float)val, (float)val, (float)val, (float)val}; __stcs(((float4 *)ptr) + 22, v); }
  // CHECK: st.global.cs.v2.f64
  { double2 v = {(double)val, (double)val}; __stcs(((double2 *)ptr) + 23, v); }
}
