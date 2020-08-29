//===- ObjC.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjC.h"
#include "InputFiles.h"
#include "OutputSegment.h"

#include "llvm/BinaryFormat/MachO.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;

bool macho::hasObjCSection(MemoryBufferRef mb) {
  auto *hdr = reinterpret_cast<const mach_header_64 *>(mb.getBufferStart());
  if (const load_command *cmd = findCommand(hdr, LC_SEGMENT_64)) {
    auto *c = reinterpret_cast<const segment_command_64 *>(cmd);
    auto sectionHeaders = ArrayRef<section_64>{
        reinterpret_cast<const section_64 *>(c + 1), c->nsects};
    for (const section_64 &sec : sectionHeaders) {
      StringRef sectname(sec.sectname,
                         strnlen(sec.sectname, sizeof(sec.sectname)));
      StringRef segname(sec.segname, strnlen(sec.segname, sizeof(sec.segname)));
      if ((segname == segment_names::data && sectname == "__objc_catlist") ||
          (segname == segment_names::text && sectname == "__swift")) {
        return true;
      }
    }
  }
  return false;
}
