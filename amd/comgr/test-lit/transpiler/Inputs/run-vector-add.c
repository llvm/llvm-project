//===- run-vector-add.c - Execute a raised vector add ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Usage: run-vector-add <code-object> <workgroup-size>
// Checks every sum, the holes between partial workgroups, and output guards.
//
//===----------------------------------------------------------------------===//

#define __HIP_PLATFORM_AMD__

#include <hip/hip_runtime_api.h>

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_HIP(Expr)                                                        \
  do {                                                                         \
    hipError_t Status = (Expr);                                                \
    if (Status != hipSuccess) {                                                \
      fprintf(stderr, "%s failed: %s\n", #Expr, hipGetErrorString(Status));    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Source kernargs are three global pointers at byte offsets 0, 8, and 16.
struct VectorAddArgs {
  void *A;
  void *B;
  void *C;
};
_Static_assert(sizeof(struct VectorAddArgs) == 24 &&
                   offsetof(struct VectorAddArgs, B) == 8 &&
                   offsetof(struct VectorAddArgs, C) == 16,
               "unexpected kernarg layout");

int main(int Argc, char **Argv) {
  if (Argc != 3) {
    fprintf(stderr, "usage: %s <code-object> <workgroup-size>\n", Argv[0]);
    return 2;
  }
  // The source computes workgroup_id_x * 256 + workitem_id_x.
  enum { Workgroups = 16, Stride = 256, Guard = 64 };
  enum { Elements = Workgroups * Stride, Total = Elements + 2 * Guard };
  char *End;
  long WorkgroupSize = strtol(Argv[2], &End, 10);
  if (*End || WorkgroupSize <= 0 || WorkgroupSize > Stride) {
    fprintf(stderr, "workgroup size must be between 1 and %d\n", Stride);
    return 2;
  }

  int DeviceCount;
  CHECK_HIP(hipGetDeviceCount(&DeviceCount));
  int Device;
  for (Device = 0; Device < DeviceCount; ++Device) {
    hipDeviceProp_t Properties;
    CHECK_HIP(hipGetDeviceProperties(&Properties, Device));
    if (strncmp(Properties.gcnArchName, "gfx942", 6) == 0 &&
        (Properties.gcnArchName[6] == ':' || Properties.gcnArchName[6] == '\0'))
      break;
  }
  if (Device == DeviceCount) {
    fprintf(stderr, "no gfx942 HIP device found\n");
    return 1;
  }
  CHECK_HIP(hipSetDevice(Device));

  float HostA[Total], HostB[Total], HostC[Total];
  const float Sentinel = -1.0f;
  // Exact, distinct sums expose wrong indices across workgroups.
  for (int I = 0; I < Total; ++I) {
    HostA[I] = (float)I;
    HostB[I] = (float)(2 * I + 1);
    HostC[I] = Sentinel;
  }

  hipModule_t Module;
  hipFunction_t Kernel;
  CHECK_HIP(hipModuleLoad(&Module, Argv[1]));
  CHECK_HIP(hipModuleGetFunction(&Kernel, Module, "vector_add"));

  void *DeviceA, *DeviceB, *DeviceC;
  CHECK_HIP(hipMalloc(&DeviceA, sizeof(HostA)));
  CHECK_HIP(hipMalloc(&DeviceB, sizeof(HostB)));
  CHECK_HIP(hipMalloc(&DeviceC, sizeof(HostC)));
  CHECK_HIP(hipMemcpy(DeviceA, HostA, sizeof(HostA), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(DeviceB, HostB, sizeof(HostB), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(DeviceC, HostC, sizeof(HostC), hipMemcpyHostToDevice));

  struct VectorAddArgs Args = {(float *)DeviceA + Guard,
                               (float *)DeviceB + Guard,
                               (float *)DeviceC + Guard};
  size_t ArgsSize = sizeof(Args);
  void *Config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &Args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &ArgsSize,
                    HIP_LAUNCH_PARAM_END};
  CHECK_HIP(hipModuleLaunchKernel(Kernel, Workgroups, 1, 1,
                                  (unsigned)WorkgroupSize, 1, 1, 0, NULL, NULL,
                                  Config));
  CHECK_HIP(hipStreamSynchronize(NULL));
  CHECK_HIP(hipMemcpy(HostC, DeviceC, sizeof(HostC), hipMemcpyDeviceToHost));

  int Mismatches = 0;
  for (int I = 0; I < Total; ++I) {
    int Index = I - Guard;
    float Expected = Sentinel;
    if (Index >= 0 && Index < Elements && Index % Stride < WorkgroupSize)
      Expected = HostA[I] + HostB[I];
    if (HostC[I] == Expected)
      continue;
    if (Mismatches < 8)
      printf("mismatch at %d: got %f, expected %f\n", Index, HostC[I],
             Expected);
    ++Mismatches;
  }
  CHECK_HIP(hipFree(DeviceA));
  CHECK_HIP(hipFree(DeviceB));
  CHECK_HIP(hipFree(DeviceC));
  CHECK_HIP(hipModuleUnload(Module));

  long Sums = Workgroups * WorkgroupSize;
  printf("RESULT: %s (%ld sums, %ld untouched elements)\n",
         Mismatches ? "FAIL" : "PASS", Sums, Total - Sums);
  return Mismatches != 0;
}
