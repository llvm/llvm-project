//===- comgr-signal.h - Save and restore signal handlers ------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_SIGNAL_H
#define COMGR_SIGNAL_H

#include "comgr.h"

namespace COMGR {
namespace signal {

/// Save all signal handlers which are currently registered.
amd_comgr_status_t saveHandlers();

/// Restore all saved signal handlers.
amd_comgr_status_t restoreHandlers();

} // namespace signal
} // namespace COMGR

#endif // COMGR_SIGNAL_H
