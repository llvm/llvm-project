//===- DWARF.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dwarf.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSegment.h"

#include <memory>

using namespace lld;
using namespace lld::macho;
using namespace llvm;

std::unique_ptr<DwarfObject> DwarfObject::create(ObjFile *obj) {
  auto dObj = std::make_unique<DwarfObject>();
  bool hasDwarfInfo = false;
  for (SubsectionMap subsecMap : obj->subsections) {
    for (auto it : subsecMap) {
      InputSection *isec = it.second;
      if (!(isDebugSection(isec->flags) &&
            isec->segname == segment_names::dwarf))
        continue;

      if (isec->name == "__debug_info") {
        dObj->infoSection.Data = toStringRef(isec->data);
        hasDwarfInfo = true;
        continue;
      }

      if (StringRef *s = StringSwitch<StringRef *>(isec->name)
                             .Case("__debug_abbrev", &dObj->abbrevSection)
                             .Case("__debug_str", &dObj->strSection)
                             .Default(nullptr)) {
        *s = toStringRef(isec->data);
        hasDwarfInfo = true;
      }
    }
  }

  if (hasDwarfInfo)
    return dObj;
  return nullptr;
}
