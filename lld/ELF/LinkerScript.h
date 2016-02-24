//===- LinkerScript.h -------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_LINKER_SCRIPT_H
#define LLD_ELF_LINKER_SCRIPT_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/MemoryBuffer.h"

namespace lld {
namespace elf2 {

class ScriptParser;
template <class ELFT> class InputSectionBase;

// This class represents each rule in SECTIONS command.
class SectionRule {
public:
  SectionRule(StringRef D, StringRef S, bool Keep)
      : Dest(D), Keep(Keep), SectionPattern(S) {}

  // Returns true if S should be in Dest section.
  template <class ELFT> bool match(InputSectionBase<ELFT> *S);

  StringRef Dest;

  // KEEP command saves unused sections even if --gc-sections is specified.
  bool Keep = false;

private:
  StringRef SectionPattern;
};

// This is a runner of the linker script.
class LinkerScript {
  friend class ScriptParser;

public:
  // Parses a linker script. Calling this function may update
  // this object and Config.
  void read(MemoryBufferRef MB);

  template <class ELFT> StringRef getOutputSection(InputSectionBase<ELFT> *S);
  template <class ELFT> bool isDiscarded(InputSectionBase<ELFT> *S);
  template <class ELFT> bool shouldKeep(InputSectionBase<ELFT> *S);
  int compareSections(StringRef A, StringRef B);

private:
  template <class ELFT> SectionRule *find(InputSectionBase<ELFT> *S);

  // SECTIONS commands.
  std::vector<SectionRule> Sections;

  // Output sections are sorted by this order.
  std::vector<StringRef> SectionOrder;

  llvm::BumpPtrAllocator Alloc;
};

extern LinkerScript *Script;

} // namespace elf2
} // namespace lld

#endif
