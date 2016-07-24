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

// This enum is used to implement linker script SECTIONS command.
// https://sourceware.org/binutils/docs/ld/SECTIONS.html#SECTIONS
enum SectionsCommandKind {
  AssignmentKind,
  OutputSectionKind,
  InputSectionKind
};

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
  bool Provide = false;
  // Hidden and Ignore can be true, only if Provide is true
  bool Hidden = false;
  bool Ignore = false;
};

// Linker scripts allow additional constraints to be put on ouput sections.
// An output section will only be created if all of its input sections are
// read-only
// or all of its input sections are read-write by using the keyword ONLY_IF_RO
// and ONLY_IF_RW respectively.
enum ConstraintKind { NoConstraint, ReadOnly, ReadWrite };

struct OutputSectionCommand : BaseCommand {
  OutputSectionCommand(StringRef Name)
      : BaseCommand(OutputSectionKind), Name(Name) {}
  static bool classof(const BaseCommand *C);
  StringRef Name;
  std::vector<std::unique_ptr<BaseCommand>> Commands;
  std::vector<StringRef> Phdrs;
  std::vector<uint8_t> Filler;
  ConstraintKind Constraint = NoConstraint;
};

struct InputSectionDescription : BaseCommand {
  InputSectionDescription() : BaseCommand(InputSectionKind) {}
  static bool classof(const BaseCommand *C);
  std::vector<StringRef> Patterns;
};

struct PhdrsCommand {
  StringRef Name;
  unsigned Type;
  bool HasFilehdr;
  bool HasPhdrs;
  unsigned Flags;
};

// ScriptConfiguration holds linker script parse results.
struct ScriptConfiguration {
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
  std::vector<OutputSectionBase<ELFT> *>
  createSections(OutputSectionFactory<ELFT> &Factory);

  std::vector<PhdrEntry<ELFT>>
  createPhdrs(ArrayRef<OutputSectionBase<ELFT> *> S);

  ArrayRef<uint8_t> getFiller(StringRef Name);
  bool shouldKeep(InputSectionBase<ELFT> *S);
  void assignAddresses(ArrayRef<OutputSectionBase<ELFT> *> S);
  int compareSections(StringRef A, StringRef B);
  void addScriptedSymbols();
  bool hasPhdrsCommands();

private:
  // "ScriptConfig" is a bit too long, so define a short name for it.
  ScriptConfiguration &Opt = *ScriptConfig;

  std::vector<OutputSectionBase<ELFT> *>
  filter(std::vector<OutputSectionBase<ELFT> *> &Sections);

  int getSectionIndex(StringRef Name);
  std::vector<size_t> getPhdrIndicesForSection(StringRef Name);
  void dispatchAssignment(SymbolAssignment *Cmd);

  uintX_t Dot;
};

// Variable template is a C++14 feature, so we can't template
// a global variable. Use a struct to workaround.
template <class ELFT> struct Script { static LinkerScript<ELFT> *X; };
template <class ELFT> LinkerScript<ELFT> *Script<ELFT>::X;

} // namespace elf
} // namespace lld

#endif
