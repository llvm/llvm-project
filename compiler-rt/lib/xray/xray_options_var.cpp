/*===----- xray_options_var.cpp - XRay option variable setup  -------------===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

#include "xray_flags.h"

// FIXME: Generalize these. See lib/profile/InstrProfilingPort.h and
// include/profile/InstrProfData.inc
#define COMPILER_RT_VISIBILITY __attribute__((visibility("hidden")))
#define COMPILER_RT_WEAK __attribute__((weak))

extern "C" {
/* char __llvm_xray_options[1]
 *
 * The runtime should only provide its own definition of this symbol when the
 * user has not specified one. Set this up by moving the runtime's copy of this
 * symbol to an object file within the archive.
 */
COMPILER_RT_WEAK COMPILER_RT_VISIBILITY char XRAY_OPTIONS_VAR[1] = {0};
}
