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
  SectionRule(StringRef D, StringRef S) : Dest(D), SectionPattern(S) {}

  // Returns true if S should be in Dest section.
  template <class ELFT> bool match(InputSectionBase<ELFT> *S);

  StringRef Dest;

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
  int compareSections(StringRef A, StringRef B);

private:
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
