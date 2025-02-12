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

#include "comgr-env.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <fstream>
#include <memory>
#include <stdlib.h>

using namespace llvm;

namespace COMGR {
namespace env {

bool shouldSaveTemps() {
  static char *SaveTemps = getenv("AMD_COMGR_SAVE_TEMPS");
  return SaveTemps && StringRef(SaveTemps) != "0";
}

std::optional<StringRef> getRedirectLogs() {
  static char *RedirectLogs = getenv("AMD_COMGR_REDIRECT_LOGS");
  if (!RedirectLogs || StringRef(RedirectLogs) == "0") {
    return std::nullopt;
  }
  return StringRef(RedirectLogs);
}

bool needTimeStatistics() {
  static char *TimeStatistics = getenv("AMD_COMGR_TIME_STATISTICS");
  return TimeStatistics && StringRef(TimeStatistics) != "0";
}

bool shouldEmitVerboseLogs() {
  static char *VerboseLogs = getenv("AMD_COMGR_EMIT_VERBOSE_LOGS");
  return VerboseLogs && StringRef(VerboseLogs) != "0";
}

llvm::StringRef getLLVMPath() {
  static const char *EnvLLVMPath = std::getenv("LLVM_PATH");
  return EnvLLVMPath;
}

} // namespace env
} // namespace COMGR
