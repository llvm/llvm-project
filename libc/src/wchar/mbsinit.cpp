//===-- Implementation of mbsinit -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/mbsinit.h"

#include "hdr/types/mbstate_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, mbsinit, (mbstate_t * ps)) {
  if (ps == nullptr)
    return true;
  internal::CharacterConverter cr(reinterpret_cast<internal::mbstate *>(ps));
  return cr.isValidState() && cr.isEmpty();
}

} // namespace LIBC_NAMESPACE_DECL
