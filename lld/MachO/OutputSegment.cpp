//===- OutputSegment.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSegment.h"
#include "lld/Common/Memory.h"

using namespace llvm;
using namespace lld;
using namespace lld::macho;

std::vector<OutputSegment *> macho::outputSegments;

OutputSegment *macho::getOrCreateOutputSegment(StringRef name, uint32_t perms) {
  for (OutputSegment *os : outputSegments)
    if (os->name == name)
      // TODO: assert that os->perms == perms, once we figure out what to do
      // about default-created segments.
      return os;

  auto *os = make<OutputSegment>();
  os->name = name;
  os->perms = perms;
  outputSegments.push_back(os);
  return os;
}
