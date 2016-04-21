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
namespace elf {

// Parses a linker script. Calling this function updates
// Config and ScriptConfig.
void readLinkerScript(MemoryBufferRef MB);

class ScriptParser;
template <class ELFT> class InputSectionBase;
template <class ELFT> class OutputSectionBase;

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

// This enum represents what we can observe in SECTIONS tag of script:
// ExprKind is a location counter change, like ". = . + 0x1000"
// SectionKind is a description of output section, like ".data :..."
enum SectionsCommandKind { ExprKind, SectionKind };

struct SectionsCommand {
  SectionsCommandKind Kind;
  std::vector<StringRef> Expr;
  StringRef SectionName;
};

// ScriptConfiguration holds linker script parse results.
struct ScriptConfiguration {
  // SECTIONS commands.
  std::vector<SectionRule> Sections;

  // Output sections are sorted by this order.
  std::vector<StringRef> SectionOrder;

  // Section fill attribute for each section.
  llvm::StringMap<std::vector<uint8_t>> Filler;

  // Used to assign addresses to sections.
  std::vector<SectionsCommand> Commands;

  bool DoLayout = false;

  llvm::BumpPtrAllocator Alloc;
};

extern ScriptConfiguration *ScriptConfig;

// This is a runner of the linker script.
template <class ELFT> class LinkerScript {
public:
  StringRef getOutputSection(InputSectionBase<ELFT> *S);
  ArrayRef<uint8_t> getFiller(StringRef Name);
  bool isDiscarded(InputSectionBase<ELFT> *S);
  bool shouldKeep(InputSectionBase<ELFT> *S);
  void assignAddresses(std::vector<OutputSectionBase<ELFT> *> &S);
  int compareSections(StringRef A, StringRef B);

private:
  SectionRule *find(InputSectionBase<ELFT> *S);

  ScriptConfiguration &Opt = *ScriptConfig;
};

// Variable template is a C++14 feature, so we can't template
// a global variable. Use a struct to workaround.
template <class ELFT> struct Script { static LinkerScript<ELFT> *X; };
template <class ELFT> LinkerScript<ELFT> *Script<ELFT>::X;

} // namespace elf
} // namespace lld

#endif
