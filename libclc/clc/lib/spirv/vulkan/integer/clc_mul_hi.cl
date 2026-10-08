//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Opt-out of libclc mul_hi implementation for Vulkan SPIR-V targets.
// clspv has an internal implementation that does not require using a bigger
// data size. That implementation is based on OpMulExtended which is SPIR-V
// specific, thus it cannot be written in OpenCL-C.
