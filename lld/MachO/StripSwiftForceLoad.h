//===- StripSwiftForceLoad.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_STRIP_SWIFT_FORCE_LOAD_H
#define LLD_MACHO_STRIP_SWIFT_FORCE_LOAD_H

namespace lld::macho {

// Drop `__DATA,__const` sections that exist only to force-load Swift overlays.
//
// The Swift compiler emits `__swift_FORCE_LOAD_$_swift<overlay>` pointers in
// `__DATA,__const` that bind to the imported symbol of the same name in a
// dependent overlay dylib. These pointers are never dereferenced -- they exist
// only to keep the overlay dylib linked in. When a section consists entirely of
// such pointers (and is not anchored by any externally visible symbol clients
// might link against), this pass drops the whole section, reclaiming its bytes
// and removing the dyld binds.
//
// The imported FORCE_LOAD DylibSymbols are intentionally left referenced, so
// the overlays' LC_LOAD_DYLIB dependencies are preserved.
//
// NOTE: Must be run after markLive().
void stripSwiftForceLoadFixups();

} // namespace lld::macho

#endif
