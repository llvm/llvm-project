//===- OutputSegment.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSegment.h"
#include "InputSection.h"
#include "MergedOutputSection.h"
#include "SyntheticSections.h"

#include "lld/Common/ErrorHandler.h"
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
  assert(config->arch != AK_i386 &&
         "TODO: i386 has different maxProt requirements");
  return initProt(name);
}

size_t OutputSegment::numNonHiddenSections() const {
  size_t count = 0;
  for (const OutputSection *osec : sections) {
    count += (!osec->isHidden() ? 1 : 0);
  }
  return count;
}

void OutputSegment::addOutputSection(OutputSection *osec) {
  osec->parent = this;
  sections.push_back(osec);
}

static llvm::DenseMap<StringRef, OutputSegment *> nameToOutputSegment;
std::vector<OutputSegment *> macho::outputSegments;

OutputSegment *macho::getOrCreateOutputSegment(StringRef name) {
  OutputSegment *&segRef = nameToOutputSegment[name];
  if (segRef)
    return segRef;

  segRef = make<OutputSegment>();
  segRef->name = name;
  segRef->maxProt = maxProt(name);
  segRef->initProt = initProt(name);

  outputSegments.push_back(segRef);
  return segRef;
}
