//===- ObjC.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjC.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSegment.h"
#include "Target.h"

#include "llvm/BinaryFormat/MachO.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

template <class LP> static bool hasObjCSection(MemoryBufferRef mb) {
  using Section = typename LP::section;

  auto *hdr =
      reinterpret_cast<const typename LP::mach_header *>(mb.getBufferStart());
  if (hdr->magic != LP::magic)
    return false;

  if (const auto *c =
          findCommand<typename LP::segment_command>(hdr, LP::segmentLCType)) {
    auto sectionHeaders =
        ArrayRef<Section>{reinterpret_cast<const Section *>(c + 1), c->nsects};
    for (const Section &sec : sectionHeaders) {
      StringRef sectname(sec.sectname,
                         strnlen(sec.sectname, sizeof(sec.sectname)));
      StringRef segname(sec.segname, strnlen(sec.segname, sizeof(sec.segname)));
      if ((segname == segment_names::data &&
           sectname == section_names::objcCatList) ||
          (segname == segment_names::text &&
           sectname == section_names::swift)) {
        return true;
      }
    }
  }
  return false;
}

bool macho::hasObjCSection(MemoryBufferRef mb) {
  if (target->wordSize == 8)
    return ::hasObjCSection<LP64>(mb);
  else
    return ::hasObjCSection<ILP32>(mb);
}
