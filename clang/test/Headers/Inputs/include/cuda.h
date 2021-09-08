/* Minimal declarations for CUDA support.  Testing purposes only. */
#pragma once

#include <stddef.h>

// Make this file work with nvcc, for testing compatibility.

#ifndef __NVCC__
#define __constant__ __attribute__((constant))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __managed__ __attribute__((managed))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ dim3(unsigned x, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};

// Host- and device-side placement new overloads.
void *operator new(__SIZE_TYPE__, void *p) { return p; }
void *operator new[](__SIZE_TYPE__, void *p) { return p; }
__device__ void *operator new(__SIZE_TYPE__, void *p) { return p; }
__device__ void *operator new[](__SIZE_TYPE__, void *p) { return p; }

#define CUDA_VERSION 10100

struct char2 {
  char x, y;
  __host__ __device__ char2(char x = 0, char y = 0) : x(x), y(y) {}
};
struct char4 {
  char x, y, z, w;
  __host__ __device__ char4(char x = 0, char y = 0, char z = 0, char w = 0) : x(x), y(y), z(z), w(w) {}
};

struct uchar2 {
  unsigned char x, y;
  __host__ __device__ uchar2(unsigned char x = 0, unsigned char y = 0) : x(x), y(y) {}
};
struct uchar4 {
  unsigned char x, y, z, w;
  __host__ __device__ uchar4(unsigned char x = 0, unsigned char y = 0, unsigned char z = 0, unsigned char w = 0) : x(x), y(y), z(z), w(w) {}
};

struct short2 {
  short x, y;
  __host__ __device__ short2(short x = 0, short y = 0) : x(x), y(y) {}
};
struct short4 {
  short x, y, z, w;
  __host__ __device__ short4(short x = 0, short y = 0, short z = 0, short w = 0) : x(x), y(y), z(z), w(w) {}
};

struct ushort2 {
  unsigned short x, y;
  __host__ __device__ ushort2(unsigned short x = 0, unsigned short y = 0) : x(x), y(y) {}
};
struct ushort4 {
  unsigned short x, y, z, w;
  __host__ __device__ ushort4(unsigned short x = 0, unsigned short y = 0, unsigned short z = 0, unsigned short w = 0) : x(x), y(y), z(z), w(w) {}
};

struct int2 {
  int x, y;
  __host__ __device__ int2(int x = 0, int y = 0) : x(x), y(y) {}
};
struct int4 {
  int x, y, z, w;
  __host__ __device__ int4(int x = 0, int y = 0, int z = 0, int w = 0) : x(x), y(y), z(z), w(w) {}
};

struct uint2 {
  unsigned x, y;
  __host__ __device__ uint2(unsigned x = 0, unsigned y = 0) : x(x), y(y) {}
};
struct uint3 {
  unsigned x, y, z;
  __host__ __device__ uint3(unsigned x = 0, unsigned y = 0, unsigned z = 0) : x(x), y(y), z(z) {}
};
struct uint4 {
  unsigned x, y, z, w;
  __host__ __device__ uint4(unsigned x = 0, unsigned y = 0, unsigned z = 0, unsigned w = 0) : x(x), y(y), z(z), w(w) {}
};

struct longlong2 {
  long long x, y;
  __host__ __device__ longlong2(long long x = 0, long long y = 0) : x(x), y(y) {}
};
struct longlong4 {
  long long x, y, z, w;
  __host__ __device__ longlong4(long long x = 0, long long y = 0, long long z = 0, long long w = 0) : x(x), y(y), z(z), w(w) {}
};

struct ulonglong2 {
  unsigned long long x, y;
  __host__ __device__ ulonglong2(unsigned long long x = 0, unsigned long long y = 0) : x(x), y(y) {}
};
struct ulonglong4 {
  unsigned long long x, y, z, w;
  __host__ __device__ ulonglong4(unsigned long long x = 0, unsigned long long y = 0, unsigned long long z = 0, unsigned long long w = 0) : x(x), y(y), z(z), w(w) {}
};

struct float2 {
  float x, y;
  __host__ __device__ float2(float x = 0, float y = 0) : x(x), y(y) {}
};
struct float4 {
  float x, y, z, w;
  __host__ __device__ float4(float x = 0, float y = 0, float z = 0, float w = 0) : x(x), y(y), z(z), w(w) {}
};

struct double2 {
  double x, y;
  __host__ __device__ double2(double x = 0, double y = 0) : x(x), y(y) {}
};
struct double4 {
  double x, y, z, w;
  __host__ __device__ double4(double x = 0, double y = 0, double z = 0, double w = 0) : x(x), y(y), z(z), w(w) {}
};

typedef unsigned long long cudaTextureObject_t;

enum cudaTextureReadMode {
  cudaReadModeNormalizedFloat,
  cudaReadModeElementType
};

enum {
  cudaTextureType1D,
  cudaTextureType2D,
  cudaTextureType3D,
  cudaTextureTypeCubemap,
  cudaTextureType1DLayered,
  cudaTextureType2DLayered,
  cudaTextureTypeCubemapLayered
};

struct textureReference {};
template <class T, int texType = cudaTextureType1D,
          enum cudaTextureReadMode mode = cudaReadModeElementType>
struct __attribute__((device_builtin_texture_type)) texture
    : public textureReference {};

#endif // !__NVCC__
