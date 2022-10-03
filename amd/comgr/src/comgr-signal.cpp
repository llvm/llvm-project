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

#include "comgr-signal.h"
#include "llvm/ADT/STLExtras.h"
#include <csignal>

namespace COMGR {
namespace signal {

#ifndef _MSC_VER
static const int Signals[] = {SIGHUP,
                              SIGINT,
                              SIGPIPE,
                              SIGTERM,
                              SIGUSR1,
                              SIGUSR2,
                              SIGILL,
                              SIGTRAP,
                              SIGABRT,
                              SIGFPE,
                              SIGBUS,
                              SIGSEGV,
                              SIGQUIT
#ifdef SIGSYS
                              ,
                              SIGSYS
#endif
#ifdef SIGXCPU
                              ,
                              SIGXCPU
#endif
#ifdef SIGXFSZ
                              ,
                              SIGXFSZ
#endif
#ifdef SIGEMT
                              ,
                              SIGEMT
#endif
#ifdef SIGINFO
                              ,
                              SIGINFO
#endif
};

static const unsigned NumSigs = std::size(Signals);

static struct sigaction SigActions[NumSigs];
#endif // _MSC_VER

amd_comgr_status_t saveHandlers() {
#ifndef _MSC_VER
  for (unsigned I = 0; I < NumSigs; ++I) {
    int Status = sigaction(Signals[I], nullptr, &SigActions[I]);

    if (Status) {
      return AMD_COMGR_STATUS_ERROR;
    }
  }
#endif
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t restoreHandlers() {
#ifndef _MSC_VER
  for (unsigned I = 0; I < NumSigs; ++I) {
    int Status = sigaction(Signals[I], &SigActions[I], nullptr);

    if (Status) {
      return AMD_COMGR_STATUS_ERROR;
    }
  }
#endif
  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace signal
} // namespace COMGR
