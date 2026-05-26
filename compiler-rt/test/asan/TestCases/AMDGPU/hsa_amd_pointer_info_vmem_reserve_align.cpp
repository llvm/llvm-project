// RUN: %clangxx_asan -O0 -isystem %rocm_include %s -o %t -L%rocm_lib -lhsa-runtime64 \
// RUN:   -Wl,-rpath,%rocm_lib -Wl,-rpath,%compiler_rt_libdir
// RUN: %run %t 2>&1 | FileCheck %s
//
// hsa_amd_pointer_info on vmem reserved via reserve_align() must report the
// user-visible base and size, not the internal HSA backing mapping. A fixed
// +page offset is wrong when alignment is larger than the page size.
//
// REQUIRES: sanitizer-amdgpu, linux, stable-runtime, rocm
// UNSUPPORTED: android

#include "hsa_amd_test_helpers.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int main() {
  if (hsa_amd_test_require_init())
    return 1;

  const size_t kSize = 4096;
  const uint64_t kAlign = 65536;
  void *mem = nullptr;
  if (hsa_amd_vmem_address_reserve_align(&mem, kSize, /*address=*/0, kAlign,
                                         /*flags=*/0) != HSA_STATUS_SUCCESS ||
      !mem) {
    fprintf(stderr, "hsa_amd_vmem_address_reserve_align failed\n");
    return 1;
  }

  const uintptr_t user = reinterpret_cast<uintptr_t>(mem);
  if (user % kAlign != 0) {
    fprintf(stderr, "reserved address not %" PRIu64 "-byte aligned\n", kAlign);
    return 1;
  }

  hsa_amd_pointer_info_t info = {};
  info.size = sizeof(hsa_amd_pointer_info_t);
  if (hsa_amd_pointer_info(mem, &info, nullptr, nullptr, nullptr) !=
      HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_amd_pointer_info failed\n");
    return 1;
  }

  if (info.agentBaseAddress != mem ||
      reinterpret_cast<uintptr_t>(info.agentBaseAddress) + info.sizeInBytes !=
          user + kSize) {
    fprintf(stderr,
            "pointer_info mismatch: user=%p begin=%p size=%zu (expected %zu)\n",
            mem, info.agentBaseAddress, info.sizeInBytes, kSize);
    return 1;
  }

  printf("pointer_info_vmem sizeInBytes: %zu\n", info.sizeInBytes);
  printf("pointer_info_vmem begin: %p\n", info.agentBaseAddress);

  if (hsa_amd_vmem_address_free(mem, kSize) != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_amd_vmem_address_free failed\n");
    return 1;
  }
  return 0;
}

// CHECK: pointer_info_vmem sizeInBytes: 4096
// CHECK-NEXT: pointer_info_vmem begin: 0x{{[0-9a-f]+}}
