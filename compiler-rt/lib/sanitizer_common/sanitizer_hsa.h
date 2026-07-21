//===-- sanitizer_hsa.h ----------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Umbrella include for in-tree HSA stub headers (host AMDHSA sanitizer builds).
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_HSA_H_
#define SANITIZER_HSA_H_

// Reuse the AMDGPU plugin's vendored HSA headers so compiler-rt and offload
// cannot drift. Monorepo-relative path for build flexibility. No local
// fallback: a non-monorepo checkout is unsupported for AMDHSA and must fail.
#if defined(__has_include) && \
    __has_include("../../../offload/plugins-nextgen/amdgpu/dynamic_hsa/hsa.h")
#  include "../../../offload/plugins-nextgen/amdgpu/dynamic_hsa/hsa.h"
#  include "../../../offload/plugins-nextgen/amdgpu/dynamic_hsa/hsa_ext_amd.h"
#else
#  error \
      "AMDHSA sanitizer support requires the AMDGPU plugin's HSA headers (offload/plugins-nextgen/amdgpu/dynamic_hsa/): build compiler-rt within the LLVM monorepo, or disable AMDHSA support (SANITIZER_AMDHSA=0)."
#endif

#endif  // SANITIZER_HSA_H_
