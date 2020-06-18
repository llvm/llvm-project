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
  if (name == segment_names::pageZero)
    return 0;
  return VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE;
}

size_t OutputSegment::numNonHiddenSections() const {
  size_t count = 0;
  for (const OutputSegment::SectionMapEntry &i : sections) {
    OutputSection *os = i.second;
    count += (!os->isHidden() ? 1 : 0);
  }
  return count;
}

void OutputSegment::addOutputSection(OutputSection *os) {
  os->parent = this;
  std::pair<SectionMap::iterator, bool> result =
      sections.insert(SectionMapEntry(os->name, os));
  if (!result.second) {
    llvm_unreachable("Attempted to set section, but a section with the same "
                     "name already exists");
  }
}

OutputSection *OutputSegment::getOrCreateOutputSection(StringRef name) {
  OutputSegment::SectionMap::iterator i = sections.find(name);
  if (i != sections.end()) {
    return i->second;
  }

  auto *os = make<MergedOutputSection>(name);
  addOutputSection(os);
  return os;
}

void OutputSegment::sortOutputSections(OutputSegmentComparator *comparator) {
  llvm::stable_sort(sections, *comparator->sectionComparator(this));
}

void OutputSegment::removeUnneededSections() {
  sections.remove_if([](const std::pair<StringRef, OutputSection *> &p) {
    return !p.second->isNeeded();
  });
}

OutputSegmentComparator::OutputSegmentComparator() {
  // This defines the order of segments and the sections within each segment.
  // Segments that are not mentioned here will end up at defaultPosition;
  // sections that are not mentioned will end up at the end of the section
  // list for their given segment.
  std::vector<std::pair<StringRef, std::vector<StringRef>>> ordering{
      {segment_names::pageZero, {}},
      {segment_names::text, {section_names::header}},
      {defaultPosition, {}},
      // Make sure __LINKEDIT is the last segment (i.e. all its hidden
      // sections must be ordered after other sections).
      {segment_names::linkEdit,
       {
           section_names::binding,
           section_names::export_,
           section_names::symbolTable,
           section_names::stringTable,
       }},
  };

  for (uint32_t i = 0, n = ordering.size(); i < n; ++i) {
    auto &p = ordering[i];
    StringRef segname = p.first;
    const std::vector<StringRef> &sectOrdering = p.second;
    orderMap.insert(std::pair<StringRef, OutputSectionComparator>(
        segname, OutputSectionComparator(i, sectOrdering)));
  }

  // Cache the position for the default comparator since this is the likely
  // scenario.
  defaultPositionComparator = &orderMap.find(defaultPosition)->second;
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
