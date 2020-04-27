//===- OutputSegment.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSegment.h"
#include "InputSection.h"

#include "lld/Common/Memory.h"
#include "llvm/BinaryFormat/MachO.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

static uint32_t initProt(StringRef name) {
  if (name == segment_names::text)
    return VM_PROT_READ | VM_PROT_EXECUTE;
  if (name == segment_names::pageZero)
    return 0;
  if (name == segment_names::linkEdit)
    return VM_PROT_READ;
  return VM_PROT_READ | VM_PROT_WRITE;
}

static uint32_t maxProt(StringRef name) {
  if (name == segment_names::pageZero)
    return 0;
  return VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE;
}

void OutputSegment::addSection(InputSection *isec) {
  isec->parent = this;
  std::vector<InputSection *> &vec = sections[isec->name];
  if (vec.empty() && !isec->isHidden()) {
    ++numNonHiddenSections;
  }
  vec.push_back(isec);
}

static llvm::DenseMap<StringRef, OutputSegment *> nameToOutputSegment;
std::vector<OutputSegment *> macho::outputSegments;

OutputSegment *macho::getOutputSegment(StringRef name) {
  return nameToOutputSegment.lookup(name);
}

OutputSegment *macho::getOrCreateOutputSegment(StringRef name) {
  OutputSegment *&segRef = nameToOutputSegment[name];
  if (segRef != nullptr)
    return segRef;

  segRef = make<OutputSegment>();
  segRef->name = name;
  segRef->maxProt = maxProt(name);
  segRef->initProt = initProt(name);

  outputSegments.push_back(segRef);
  return segRef;
}
