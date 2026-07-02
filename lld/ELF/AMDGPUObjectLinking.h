//===- AMDGPUObjectLinking.h - AMDGPU link-time resolution ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements link-time resolution for AMDGPU object linking. This includes
// LDS (Local Data Share) offset assignment across translation units and
// resource usage propagation through the cross-TU call graph.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_AMDGPUOBJECTLINKING_H
#define LLD_ELF_AMDGPUOBJECTLINKING_H

namespace lld::elf {
struct Ctx;

template <class ELFT> void resolveAMDGPUObjectLinking(Ctx &ctx);

} // namespace lld::elf

#endif // LLD_ELF_AMDGPUOBJECTLINKING_H
