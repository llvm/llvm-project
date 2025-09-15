// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -x hip -fcuda-is-device -emit-llvm  %s \
// RUN:   -o - | FileCheck %s

// CHECK: define dso_local amdgpu_kernel void @_Z13shufflekernelv()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP1:%.*]] = alloca i32, align 4, addrspace(5)
// CHECK-NEXT:    [[TMP2:%.*]] = alloca i32, align 4, addrspace(5)
// CHECK-NEXT:    [[TMP3:%.*]] = addrspacecast ptr addrspace(5) [[TMP1:%.*]] to ptr
// CHECK-NEXT:    [[TMP4:%.*]] = addrspacecast ptr addrspace(5) [[TMP2:%.*]] to ptr
// CHECK-NEXT:    [[TMP5:%.*]] = load i32, ptr [[TMP3:%.*]], align 4
// CHECK-NEXT:    [[TMP6:%.*]] = freeze i32 [[TMP5:%.*]]
// CHECK-NEXT:    %call = call noundef i32 @_Z11__shfl_synciii(i32 noundef [[TMP6:%.*]], i32 noundef 64, i32 noundef 0) #4
// CHECK-NEXT:    store i32 %call, ptr [[TMP4:%.*]], align 4
// CHECK-NEXT:  ret void

// CHECK: define linkonce_odr noundef i32 @_Z11__shfl_synciii(i32 noundef [[TMP1:%.*]], i32 noundef [[TMP2:%.*]], i32 noundef [[TMP3:%.*]])

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __maybe_undef __attribute__((maybe_undef))
#define WARP_SIZE 64

static constexpr int warpSize = __AMDGCN_WAVEFRONT_SIZE__;

__device__ static inline unsigned int __lane_id() {
    return  __builtin_amdgcn_mbcnt_hi(
        -1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}

__device__
inline
int __shfl_sync(int __maybe_undef var, int src_lane, int width = warpSize) {
    int self = __lane_id();
    int index = src_lane + (self & ~(width-1));
    return __builtin_amdgcn_ds_bpermute(index<<2, var);
}

__global__ void
shufflekernel()
{
    int t;
    int res;
    res = __shfl_sync(t, WARP_SIZE, 0);
}
