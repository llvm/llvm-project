/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#ifndef COMGR_CACHE_H
#define COMGR_CACHE_H

#include "amd_comgr.h"
#include "comgr-cache-command.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CachePruning.h>
#include <llvm/Support/MemoryBuffer.h>

#include <functional>
#include <memory>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace COMGR {
class CommandCache {
  std::string CacheDir;
  llvm::CachePruningPolicy Policy;

  CommandCache(llvm::StringRef CacheDir,
               const llvm::CachePruningPolicy &Policy);

  static std::optional<llvm::CachePruningPolicy>
  getPolicyFromEnv(llvm::raw_ostream &LogS);

public:
  static std::unique_ptr<CommandCache> get(llvm::raw_ostream &);

  ~CommandCache();
  void prune();

  /// Checks if the Command C is cached.
  /// If it is the case, it replaces its output and logs its error-stream.
  /// Otherwise it executes C through the callback Execute
  amd_comgr_status_t execute(CachedCommandAdaptor &C, llvm::raw_ostream &LogS);
};
} // namespace COMGR

#endif
