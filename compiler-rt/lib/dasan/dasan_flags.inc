//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Device address sanitizer runtime flags.
///
//===----------------------------------------------------------------------===//

#ifndef DASAN_FLAG
#  error "Define DASAN_FLAG prior to including this file!"
#endif

DASAN_FLAG(
    bool, alias_globals, true,
    "Place each loaded code object in the tracked window so its globals are "
    "checked. The loader has to be able to load at an address. Off leaves "
    "globals at the loader's addresses, where the check cannot see them.")

DASAN_FLAG(int, quarantine_size_mb, 256,
           "How much freed device memory to hold back, across all devices, "
           "before its chunks may be handed out again. A use after free is "
           "only reported while the chunk it names is still in here.")

DASAN_FLAG(
    int, halt_on_error, -1,
    "Stop the program at the first report. The default of -1 follows "
    "the build: an access compiled with -fsanitize-recover=daddress "
    "keeps going, and 0 or 1 overrides that. An access compiled without "
    "recover cannot resume, so 0 leaves that wave's work lost and warns.")

DASAN_FLAG(bool, print_stats, false,
           "Print how many allocations were placed, and why the rest were not, "
           "when the device runtime shuts down.")
