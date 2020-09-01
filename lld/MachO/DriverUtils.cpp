//===- DriverUtils.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DriverUtils.h"
#include "InputFiles.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/Support/Path.h"
#include "llvm/TextAPI/MachO/TextAPIReader.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

Optional<std::string> macho::resolveDylibPath(StringRef path) {
  // TODO: if a tbd and dylib are both present, we should check to make sure
  // they are consistent.
  if (fs::exists(path))
    return std::string(path);

  SmallString<261> location = path;
  path::replace_extension(location, ".tbd");
  if (fs::exists(location))
    return std::string(location);

  return {};
}

Optional<DylibFile *> macho::makeDylibFromTAPI(MemoryBufferRef mbref,
                                               DylibFile *umbrella) {
  Expected<std::unique_ptr<InterfaceFile>> result = TextAPIReader::get(mbref);
  if (!result) {
    error("could not load TAPI file at " + mbref.getBufferIdentifier() + ": " +
          toString(result.takeError()));
    return {};
  }
  return make<DylibFile>(**result, umbrella);
}
