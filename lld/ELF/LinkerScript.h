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

#include "Strings.h"
#include "Writer.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"
#include <functional>

namespace lld {
namespace elf {
class DefinedCommon;
class ScriptParser;
class SymbolBody;
template <class ELFT> class InputSectionBase;
template <class ELFT> class OutputSectionBase;
template <class ELFT> class OutputSectionFactory;
class InputSectionData;

typedef std::function<uint64_t(uint64_t)> Expr;

// Parses a linker script. Calling this function updates
// Config and ScriptConfig.
void readLinkerScript(MemoryBufferRef MB);

void readVersionScript(MemoryBufferRef MB);

// This enum is used to implement linker script SECTIONS command.
// https://sourceware.org/binutils/docs/ld/SECTIONS.html#SECTIONS
enum SectionsCommandKind {
  AssignmentKind,
  OutputSectionKind,
  InputSectionKind,
  AssertKind
};

struct BaseCommand {
  BaseCommand(int K) : Kind(K) {}
  virtual ~BaseCommand() {}
  int Kind;
};

struct SymbolAssignment : BaseCommand {
  SymbolAssignment(StringRef Name, Expr E, bool IsAbsolute)
      : BaseCommand(AssignmentKind), Name(Name), Expression(E),
        IsAbsolute(IsAbsolute) {}
  static bool classof(const BaseCommand *C);

  // The LHS of an expression. Name is either a symbol name or ".".
  StringRef Name;
  SymbolBody *Sym = nullptr;

  // The RHS of an expression.
  Expr Expression;

  // Command attributes for PROVIDE, HIDDEN and PROVIDE_HIDDEN.
  bool Provide = false;
  bool Hidden = false;
  bool IsAbsolute;
  InputSectionData *GoesAfter = nullptr;
};

// Linker scripts allow additional constraints to be put on ouput sections.
// An output section will only be created if all of its input sections are
// read-only
// or all of its input sections are read-write by using the keyword ONLY_IF_RO
// and ONLY_IF_RW respectively.
enum class ConstraintKind { NoConstraint, ReadOnly, ReadWrite };

struct OutputSectionCommand : BaseCommand {
  OutputSectionCommand(StringRef Name)
      : BaseCommand(OutputSectionKind), Name(Name) {}
  static bool classof(const BaseCommand *C);
  StringRef Name;
  Expr AddrExpr;
  Expr AlignExpr;
  Expr LmaExpr;
  Expr SubalignExpr;
  std::vector<std::unique_ptr<BaseCommand>> Commands;
  std::vector<StringRef> Phdrs;
  std::vector<uint8_t> Filler;
  ConstraintKind Constraint = ConstraintKind::NoConstraint;
};

enum SortKind { SortNone, SortByPriority, SortByName, SortByAlignment };

struct InputSectionDescription : BaseCommand {
  InputSectionDescription(StringRef FilePattern)
      : BaseCommand(InputSectionKind),
        FileRe(compileGlobPatterns({FilePattern})) {}
  static bool classof(const BaseCommand *C);
  llvm::Regex FileRe;
  SortKind SortOuter = SortNone;
  SortKind SortInner = SortNone;
  llvm::Regex ExcludedFileRe;
  llvm::Regex SectionRe;
};

struct AssertCommand : BaseCommand {
  AssertCommand(Expr E) : BaseCommand(AssertKind), Expression(E) {}
  static bool classof(const BaseCommand *C);
  Expr Expression;
};

struct PhdrsCommand {
  StringRef Name;
  unsigned Type;
  bool HasFilehdr;
  bool HasPhdrs;
  unsigned Flags;
  Expr LMAExpr;
};

class LinkerScriptBase {
protected:
  ~LinkerScriptBase() = default;

public:
  virtual uint64_t getOutputSectionAddress(StringRef Name) = 0;
  virtual uint64_t getOutputSectionSize(StringRef Name) = 0;
  virtual uint64_t getOutputSectionAlign(StringRef Name) = 0;
  virtual uint64_t getHeaderSize() = 0;
  virtual uint64_t getSymbolValue(StringRef S) = 0;
};

// ScriptConfiguration holds linker script parse results.
struct ScriptConfiguration {
  // Used to create symbol assignments outside SECTIONS command.
  std::vector<std::unique_ptr<SymbolAssignment>> Assignments;
  // Used to assign addresses to sections.
  std::vector<std::unique_ptr<BaseCommand>> Commands;

  // Used to assign sections to headers.
  std::vector<PhdrsCommand> PhdrsCommands;

  bool HasSections = false;

  llvm::BumpPtrAllocator Alloc;

  // List of section patterns specified with KEEP commands. They will
  // be kept even if they are unused and --gc-sections is specified.
  std::vector<llvm::Regex *> KeptSections;
};

extern ScriptConfiguration *ScriptConfig;

// This is a runner of the linker script.
template <class ELFT> class LinkerScript final : public LinkerScriptBase {
  typedef typename ELFT::uint uintX_t;

public:
  LinkerScript();
  ~LinkerScript();
  void createAssignments();
  void createSections(OutputSectionFactory<ELFT> &Factory);

  std::vector<PhdrEntry<ELFT>> createPhdrs();
  bool ignoreInterpSection();

  ArrayRef<uint8_t> getFiller(StringRef Name);
  Expr getLma(StringRef Name);
  bool shouldKeep(InputSectionBase<ELFT> *S);
  void assignAddresses();
  int compareSections(StringRef A, StringRef B);
  bool hasPhdrsCommands();
  uint64_t getOutputSectionAddress(StringRef Name) override;
  uint64_t getOutputSectionSize(StringRef Name) override;
  uint64_t getOutputSectionAlign(StringRef Name) override;
  uint64_t getHeaderSize() override;
  uint64_t getSymbolValue(StringRef S) override;

  std::vector<OutputSectionBase<ELFT> *> *OutputSections;

private:
  std::vector<InputSectionBase<ELFT> *>
  getInputSections(const InputSectionDescription *);

  void discard(ArrayRef<InputSectionBase<ELFT> *> V);

  std::vector<InputSectionBase<ELFT> *>
  createInputSectionList(OutputSectionCommand &Cmd);

  // "ScriptConfig" is a bit too long, so define a short name for it.
  ScriptConfiguration &Opt = *ScriptConfig;

  int getSectionIndex(StringRef Name);
  std::vector<size_t> getPhdrIndices(StringRef SectionName);
  size_t getPhdrIndex(StringRef PhdrName);

  uintX_t Dot;
};

// Variable template is a C++14 feature, so we can't template
// a global variable. Use a struct to workaround.
template <class ELFT> struct Script { static LinkerScript<ELFT> *X; };
template <class ELFT> LinkerScript<ELFT> *Script<ELFT>::X;

extern LinkerScriptBase *ScriptBase;

} // namespace elf
} // namespace lld

#endif
