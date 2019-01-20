//===-- cache_frag.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of EfficiencySanitizer, a family of performance tuners.
//
// Header for cache-fragmentation-specific code.
//===----------------------------------------------------------------------===//

#ifndef CACHE_FRAG_H
#define CACHE_FRAG_H

namespace __esan {

void processCacheFragCompilationUnitInit(void *Ptr);
void processCacheFragCompilationUnitExit(void *Ptr);

void initializeCacheFrag();
int finalizeCacheFrag();
void reportCacheFrag();

} // namespace __esan

#endif  // CACHE_FRAG_H
