//===- comgr-signal.cpp - Save and restore signal handlers ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the interception, saving, and restoring of OS signals.
/// These are invoked during Comgr Action invocations to avoid conflicts with
/// LLVM-installed signal handlers.
///
//===----------------------------------------------------------------------===//

#include "comgr-signal.h"
#include "llvm/ADT/STLExtras.h"
#include <csignal>

namespace COMGR {
namespace signal {

namespace {
#ifndef _MSC_VER
const int Signals[] = {SIGHUP,
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

const unsigned NumSigs = std::size(Signals);

struct sigaction SigActions[NumSigs];
#endif // _MSC_VER

} // namespace

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
