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

#include "Writer.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/MemoryBuffer.h"

namespace lld {
namespace elf {
template <class ELFT> class InputSectionBase;
template <class ELFT> class OutputSectionBase;
template <class ELFT> class OutputSectionFactory;

// Parses a linker script. Calling this function updates
// Config and ScriptConfig.
void readLinkerScript(MemoryBufferRef MB);

class ScriptParser;
template <class ELFT> class InputSectionBase;
template <class ELFT> class OutputSectionBase;

// This class represents each rule in SECTIONS command.
struct SectionRule {
  SectionRule(StringRef D, StringRef S)
      : Dest(D), SectionPattern(S) {}

  StringRef Dest;

  StringRef SectionPattern;
};

// This enum represents what we can observe in SECTIONS tag of script.
// Each sections-command may of be one of the following:
// (https://sourceware.org/binutils/docs/ld/SECTIONS.html#SECTIONS)
// * An ENTRY command.
// * A symbol assignment.
// * An output section description.
// * An overlay description.
// We support only AssignmentKind and OutputSectionKind for now.
enum SectionsCommandKind { AssignmentKind, OutputSectionKind };

struct BaseCommand {
  BaseCommand(int K) : Kind(K) {}
  virtual ~BaseCommand() {}
  int Kind;
};

struct SymbolAssignment : BaseCommand {
  SymbolAssignment(StringRef Name, std::vector<StringRef> &Expr)
      : BaseCommand(AssignmentKind), Name(Name), Expr(std::move(Expr)) {}
  static bool classof(const BaseCommand *C);
  StringRef Name;
  std::vector<StringRef> Expr;
};

struct OutputSectionCommand : BaseCommand {
  OutputSectionCommand(StringRef Name)
      : BaseCommand(OutputSectionKind), Name(Name) {}
  static bool classof(const BaseCommand *C);
  StringRef Name;
  std::vector<StringRef> Phdrs;
  std::vector<uint8_t> Filler;
};

struct PhdrsCommand {
  StringRef Name;
  unsigned Type;
  bool HasFilehdr;
  bool HasPhdrs;
};

// ScriptConfiguration holds linker script parse results.
struct ScriptConfiguration {
  // SECTIONS commands.
  std::vector<SectionRule> Sections;

  // Section fill attribute for each section.
  llvm::StringMap<std::vector<uint8_t>> Filler;

  // Used to assign addresses to sections.
  std::vector<std::unique_ptr<BaseCommand>> Commands;

  // Used to assign sections to headers.
  std::vector<PhdrsCommand> PhdrsCommands;

  bool DoLayout = false;

  llvm::BumpPtrAllocator Alloc;

  // List of section patterns specified with KEEP commands. They will
  // be kept even if they are unused and --gc-sections is specified.
  std::vector<StringRef> KeptSections;
};

extern ScriptConfiguration *ScriptConfig;

// This is a runner of the linker script.
template <class ELFT> class LinkerScript {
  typedef typename ELFT::uint uintX_t;

public:
  typedef PhdrEntry<ELFT> Phdr;

  std::vector<OutputSectionBase<ELFT> *>
  createSections(OutputSectionFactory<ELFT> &Factory);

  StringRef getOutputSection(InputSectionBase<ELFT> *S);
  ArrayRef<uint8_t> getFiller(StringRef Name);
  bool isDiscarded(InputSectionBase<ELFT> *S);
  bool shouldKeep(InputSectionBase<ELFT> *S);
  void assignAddresses(ArrayRef<OutputSectionBase<ELFT> *> S);
  int compareSections(StringRef A, StringRef B);
  void addScriptedSymbols();
  std::vector<Phdr> createPhdrs(ArrayRef<OutputSectionBase<ELFT> *> S);
  bool hasPhdrsCommands();

private:
  // "ScriptConfig" is a bit too long, so define a short name for it.
  ScriptConfiguration &Opt = *ScriptConfig;

  int getSectionIndex(StringRef Name);
  std::vector<size_t> getPhdrIndicesForSection(StringRef Name);

  uintX_t Dot;
};

// Variable template is a C++14 feature, so we can't template
// a global variable. Use a struct to workaround.
template <class ELFT> struct Script { static LinkerScript<ELFT> *X; };
template <class ELFT> LinkerScript<ELFT> *Script<ELFT>::X;

} // namespace elf
} // namespace lld

#endif
