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

#ifndef COMGR_ENV_H
#define COMGR_ENV_H

#include "llvm/ADT/StringRef.h"

namespace COMGR {
namespace env {

/// Return whether the environment requests temps be saved.
bool shouldSaveTemps();

/// If the environment requests logs be redirected, return the string identifier
/// of where to redirect. Otherwise return @p None.
std::optional<llvm::StringRef> getRedirectLogs();

/// Return whether the environment requests verbose logging.
bool shouldEmitVerboseLogs();

/// Return whether the environment requests time statistics collection.
bool needTimeStatistics();

/// If environment variable ROCM_PATH is set, return the environment varaible,
/// otherwise return the default ROCM path.
llvm::StringRef getROCMPath();

/// If environment variable HIP_PATH is set, return the environment variable,
/// otherwise return the default HIP path.
llvm::StringRef getHIPPath();

/// If environment variable LLVM_PATH is set, return the environment variable,
/// otherwise return the default LLVM path.
llvm::StringRef getLLVMPath();

} // namespace env
} // namespace COMGR

#endif // COMGR_ENV_H
