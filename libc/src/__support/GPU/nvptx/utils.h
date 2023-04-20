//===-------------- NVPTX implementation of GPU utils -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_GPU_NVPTX_IO_H
#define LLVM_LIBC_SRC_SUPPORT_GPU_NVPTX_IO_H

#include "src/__support/common.h"

#include <stdint.h>

namespace __llvm_libc {

LIBC_INLINE uint32_t get_block_id_x() { return __nvvm_read_ptx_sreg_ctaid_x(); }

} // namespace __llvm_libc

#endif
