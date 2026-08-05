// HIP on AMDGPU.
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -x hip -fcuda-is-device -target-cpu gfx90a -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_device_functions.h
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -x hip -fcuda-is-device -target-cpu gfx1100 -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_device_functions.h

// HIP on SPIR-V.
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple spirv64-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -x hip -fcuda-is-device -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_device_functions.h

// CUDA on NVPTX.
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-unknown \
// RUN:   -x cuda -fcuda-is-device -target-cpu sm_70 -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_device_functions.h

// HIP host compilation.
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple x86_64-unknown-unknown -aux-triple amdgcn-amd-amdhsa \
// RUN:   -aux-target-cpu gfx90a -x hip -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_device_functions.h
//
// HIP host compilation with a SPIR-V device.
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple x86_64-unknown-unknown -aux-triple spirv64-amd-amdhsa \
// RUN:   -x hip -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_device_functions.h
//
// CUDA host compilation.
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple x86_64-unknown-unknown -aux-triple nvptx64-nvidia-cuda \
// RUN:   -aux-target-cpu sm_70 -x cuda -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_device_functions.h

// expected-no-diagnostics

__global__ void test_kernel(int *p, float *f, double *d, unsigned *u) {
  unsigned i = __popc(*u) + __popcll(*u);
  i += __clz(*p) + __clzll(*p);
  i += __ffs(*p) + __ffs(*u) + __ffsll((long long)*p) + __ffsll((unsigned long long)*u);
  i += __brev(*u) + (unsigned)__brevll(*u);
  i += __mul24(*p, *p) + __umul24(*u, *u) + __mulhi(*p, *p) + __umulhi(*u, *u);
  i += (unsigned)__mul64hi(*p, *p) + (unsigned)__umul64hi(*u, *u);
  i += __sad(*p, *p, *u) + __usad(*u, *u, *u);
  i += __hadd(*p, *p) + __rhadd(*p, *p) + __uhadd(*u, *u) + __urhadd(*u, *u);
  i += __byte_perm(*u, *u, *u);
  i += __funnelshift_l(*u, *u, *u) + __funnelshift_lc(*u, *u, *u) +
       __funnelshift_r(*u, *u, *u) + __funnelshift_rc(*u, *u, *u);

  i += __lastbit_u32_u64(*u);
  i += __bitextract_u32(*u, *u, *u) + (unsigned)__bitextract_u64(*u, *u, *u);
  i += __bitinsert_u32(*u, *u, *u, *u) + (unsigned)__bitinsert_u64(*u, *u, *u, *u);

  i += __float_as_int(*f) + __float_as_uint(*f);
  *f = __int_as_float(*p) + __uint_as_float(*u);
  i += (unsigned)__double_as_longlong(*d);
  *d = __longlong_as_double((long long)*u);
  i += __double2hiint(*d) + __double2loint(*d);
  *d = __hiloint2double(*p, *p);

  i += __double2int_rd(*d) + __double2int_rn(*d) + __double2int_ru(*d) +
       __double2int_rz(*d);
  i += __double2uint_rd(*d) + __double2uint_rn(*d) + __double2uint_ru(*d) +
       __double2uint_rz(*d);
  i += (unsigned)(__double2ll_rd(*d) + __double2ll_rn(*d) + __double2ll_ru(*d) +
                  __double2ll_rz(*d));
  i += (unsigned)(__double2ull_rd(*d) + __double2ull_rn(*d) +
                  __double2ull_ru(*d) + __double2ull_rz(*d));
  i += __float2int_rd(*f) + __float2int_rn(*f) + __float2int_ru(*f) +
       __float2int_rz(*f);
  i += __float2uint_rd(*f) + __float2uint_rn(*f) + __float2uint_ru(*f) +
       __float2uint_rz(*f);
  i += (unsigned)(__float2ll_rd(*f) + __float2ll_rn(*f) + __float2ll_ru(*f) +
                  __float2ll_rz(*f));
  i += (unsigned)(__float2ull_rd(*f) + __float2ull_rn(*f) + __float2ull_ru(*f) +
                  __float2ull_rz(*f));
  *f = __double2float_rd(*d) + __double2float_rn(*d) + __double2float_ru(*d) +
       __double2float_rz(*d);
  *f = __int2float_rd(*p) + __int2float_rn(*p) + __int2float_ru(*p) +
       __int2float_rz(*p);
  *f = __uint2float_rd(*u) + __uint2float_rn(*u) + __uint2float_ru(*u) +
       __uint2float_rz(*u);
  *f = __ll2float_rd(*p) + __ll2float_rn(*p) + __ll2float_ru(*p) +
       __ll2float_rz(*p);
  *f = __ull2float_rd(*u) + __ull2float_rn(*u) + __ull2float_ru(*u) +
       __ull2float_rz(*u);
  *d = __ll2double_rd(*p) + __ll2double_rn(*p) + __ll2double_ru(*p) +
       __ll2double_rz(*p);
  *d = __ull2double_rd(*u) + __ull2double_rn(*u) + __ull2double_ru(*u) +
       __ull2double_rz(*u);
  *d = __int2double_rn(*p) + __uint2double_rn(*u);

  i += threadIdx.x + threadIdx.y + threadIdx.z;
  i += blockIdx.x + blockIdx.y + blockIdx.z;
  i += blockDim.x + blockDim.y + blockDim.z;
  i += gridDim.x + gridDim.y + gridDim.z;
  i += __lane_id();
  unsigned long long m = __ballot(*p) + __ballot64(*p) + __activemask();
  int v = __all(*p) + __any(*p);
  v += __fns(m, 0, 1) + __fns32(m, 0, 1) + __fns64(m, __lane_id(), -1);

  int s = __shfl(*p, 1) + __shfl_up(*p, 1) + __shfl_down(*p, 1) + __shfl_xor(*p, 1);
  s += (int)__shfl(*u, 1, 32) + (int)__shfl_down(*f, 1) + (int)__shfl_xor(*d, 1);
  s += (int)__shfl((long long)*p, 0) + (int)__shfl((unsigned long long)*u, 0);

  unsigned long long mask = ~0ull;
  s += __shfl_sync(mask, *p, 1) + __shfl_up_sync(mask, *p, 1) +
       __shfl_down_sync(mask, *p, 1) + __shfl_xor_sync(mask, *p, 1);
  s += __all_sync(mask, *p) + __any_sync(mask, *p) + (int)__ballot_sync(mask, *p);
  int pred;
  s += (int)__match_any(*p) + (int)__match_any_sync(mask, *p) +
       (int)__match_all_sync(mask, *p, &pred);
  s += __reduce_add_sync(mask, *p) + __reduce_min_sync(mask, *p) +
       __reduce_max_sync(mask, *p);
  s += (int)(__reduce_add_sync(mask, *u) + __reduce_min_sync(mask, *u) +
             __reduce_max_sync(mask, *u) + __reduce_and_sync(mask, *u) +
             __reduce_or_sync(mask, *u) + __reduce_xor_sync(mask, *u));

  __syncthreads();
  s += __syncthreads_count(*p) + __syncthreads_and(*p) + __syncthreads_or(*p);
  __syncwarp();
  __syncwarp(mask);
  __threadfence();
  __threadfence_block();
  __threadfence_system();
  long long c = clock() + clock64() + __clock() + __clock64() + wall_clock64();

  *p = (int)i + v + s + (int)c + warpSize;
}
