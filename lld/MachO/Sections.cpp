//===- Sections.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Sections.h"
#include "InputSection.h"
#include "OutputSegment.h"

#include "llvm/ADT/StringSwitch.h"

using namespace llvm;
using namespace llvm::MachO;

namespace lld::macho::sections {
bool isCodeSection(StringRef name, StringRef segName, uint32_t flags) {
  uint32_t type = sectionType(flags);
  if (type != S_REGULAR && type != S_COALESCED)
    return false;

  uint32_t attr = flags & SECTION_ATTRIBUTES_USR;
  if (attr == S_ATTR_PURE_INSTRUCTIONS)
    return true;

  if (segName == segment_names::text)
    return StringSwitch<bool>(name)
        .Cases(section_names::textCoalNt, section_names::staticInit, true)
        .Default(false);

  return false;
}

} // namespace lld::macho::sections
