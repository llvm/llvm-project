/* Minimal declarations for CUDA support.  Testing purposes only. */

#include <stddef.h>

#if __HIP__ || __CUDA__
#define __constant__ __attribute__((constant))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#if __HIP__
#define __managed__ __attribute__((managed))
#endif
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __grid_constant__ __attribute__((grid_constant))
#define __cluster_dims__(...) __attribute__((cluster_dims(__VA_ARGS__)))
#define __no_cluster__ __attribute__((no_cluster))
#else
#define __constant__
#define __device__
#define __global__
#define __host__
#define __shared__
#define __managed__
#define __launch_bounds__(...)
#define __grid_constant__
#define __cluster_dims__(...)
#define __no_cluster__
#endif

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ dim3(unsigned x, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};

#if __HIP__ || HIP_PLATFORM
typedef struct hipStream *hipStream_t;
typedef enum hipError {} hipError_t;
int hipConfigureCall(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0,
                     hipStream_t stream = 0);
extern "C" hipError_t __hipPushCallConfiguration(dim3 gridSize, dim3 blockSize,
                                                 size_t sharedSize = 0,
                                                 hipStream_t stream = 0);
#ifndef __HIP_API_PER_THREAD_DEFAULT_STREAM__
extern "C" hipError_t hipLaunchKernel(const void *func, dim3 gridDim,
                                      dim3 blockDim, void **args,
                                      size_t sharedMem,
                                      hipStream_t stream);
#else
extern "C" hipError_t hipLaunchKernel_spt(const void *func, dim3 gridDim,
                                      dim3 blockDim, void **args,
                                      size_t sharedMem,
                                      hipStream_t stream);
#endif // __HIP_API_PER_THREAD_DEFAULT_STREAM__
#elif __OFFLOAD_VIA_LLVM__
extern "C" unsigned __llvmPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem = 0, void *stream = 0);
extern "C" unsigned llvmLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                          void **args, size_t sharedMem = 0, void *stream = 0);
#else
typedef struct cudaStream *cudaStream_t;
typedef enum cudaError {} cudaError_t;
extern "C" int cudaConfigureCall(dim3 gridSize, dim3 blockSize,
                                 size_t sharedSize = 0,
                                 cudaStream_t stream = 0);
extern "C" int __cudaPushCallConfiguration(dim3 gridSize, dim3 blockSize,
                                           size_t sharedSize = 0,
                                           cudaStream_t stream = 0);
extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem, cudaStream_t stream);
extern "C" cudaError_t cudaLaunchKernel_ptsz(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem, cudaStream_t stream);

#endif

extern "C" __device__ int printf(const char*, ...);

struct char1 {
  char x;
  __host__ __device__ char1(char x = 0) : x(x) {}
};
struct char2 {
  char x, y;
  __host__ __device__ char2(char x = 0, char y = 0) : x(x), y(y) {}
};
struct char4 {
  char x, y, z, w;
  __host__ __device__ char4(char x = 0, char y = 0, char z = 0, char w = 0) : x(x), y(y), z(z), w(w) {}
};

struct uchar1 {
  unsigned char x;
  __host__ __device__ uchar1(unsigned char x = 0) : x(x) {}
};
struct uchar2 {
  unsigned char x, y;
  __host__ __device__ uchar2(unsigned char x = 0, unsigned char y = 0) : x(x), y(y) {}
};
struct uchar4 {
  unsigned char x, y, z, w;
  __host__ __device__ uchar4(unsigned char x = 0, unsigned char y = 0, unsigned char z = 0, unsigned char w = 0) : x(x), y(y), z(z), w(w) {}
};

struct short1 {
  short x;
  __host__ __device__ short1(short x = 0) : x(x) {}
};
struct short2 {
  short x, y;
  __host__ __device__ short2(short x = 0, short y = 0) : x(x), y(y) {}
};
struct short4 {
  short x, y, z, w;
  __host__ __device__ short4(short x = 0, short y = 0, short z = 0, short w = 0) : x(x), y(y), z(z), w(w) {}
};

struct ushort1 {
  unsigned short x;
  __host__ __device__ ushort1(unsigned short x = 0) : x(x) {}
};
struct ushort2 {
  unsigned short x, y;
  __host__ __device__ ushort2(unsigned short x = 0, unsigned short y = 0) : x(x), y(y) {}
};
struct ushort4 {
  unsigned short x, y, z, w;
  __host__ __device__ ushort4(unsigned short x = 0, unsigned short y = 0, unsigned short z = 0, unsigned short w = 0) : x(x), y(y), z(z), w(w) {}
};

struct int1 {
  int x;
  __host__ __device__ int1(int x = 0) : x(x) {}
};
struct int2 {
  int x, y;
  __host__ __device__ int2(int x = 0, int y = 0) : x(x), y(y) {}
};
struct int4 {
  int x, y, z, w;
  __host__ __device__ int4(int x = 0, int y = 0, int z = 0, int w = 0) : x(x), y(y), z(z), w(w) {}
};

struct uint1 {
  unsigned x;
  __host__ __device__ uint1(unsigned x = 0) : x(x) {}
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

struct longlong1 {
  long long x;
  __host__ __device__ longlong1(long long x = 0) : x(x) {}
};
struct longlong2 {
  long long x, y;
  __host__ __device__ longlong2(long long x = 0, long long y = 0) : x(x), y(y) {}
};
struct longlong4 {
  long long x, y, z, w;
  __host__ __device__ longlong4(long long x = 0, long long y = 0, long long z = 0, long long w = 0) : x(x), y(y), z(z), w(w) {}
};

struct ulonglong1 {
  unsigned long long x;
  __host__ __device__ ulonglong1(unsigned long long x = 0) : x(x) {}
};
struct ulonglong2 {
  unsigned long long x, y;
  __host__ __device__ ulonglong2(unsigned long long x = 0, unsigned long long y = 0) : x(x), y(y) {}
};
struct ulonglong4 {
  unsigned long long x, y, z, w;
  __host__ __device__ ulonglong4(unsigned long long x = 0, unsigned long long y = 0, unsigned long long z = 0, unsigned long long w = 0) : x(x), y(y), z(z), w(w) {}
};

struct float1 {
  float x;
  __host__ __device__ float1(float x = 0) : x(x) {}
};
struct float2 {
  float x, y;
  __host__ __device__ float2(float x = 0, float y = 0) : x(x), y(y) {}
};
struct float4 {
  float x, y, z, w;
  __host__ __device__ float4(float x = 0, float y = 0, float z = 0, float w = 0) : x(x), y(y), z(z), w(w) {}
};

struct double1 {
  double x;
  __host__ __device__ double1(double x = 0) : x(x) {}
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
typedef unsigned long long cudaSurfaceObject_t;

enum cudaTextureReadMode {
  cudaReadModeNormalizedFloat,
  cudaReadModeElementType
};

enum cudaSurfaceBoundaryMode {
  cudaBoundaryModeZero,
  cudaBoundaryModeClamp,
  cudaBoundaryModeTrap
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

struct textureReference { };
template <class T, int texType = cudaTextureType1D,
          enum cudaTextureReadMode mode = cudaReadModeElementType>
struct __attribute__((device_builtin_texture_type)) texture
    : public textureReference {};

struct surfaceReference { int desc; };

template <typename T, int dim = 1>
struct __attribute__((device_builtin_surface_type)) surface : public surfaceReference {};
