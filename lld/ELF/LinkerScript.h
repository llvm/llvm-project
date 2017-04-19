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

#include "Config.h"
#include "Strings.h"
#include "Writer.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace lld {
namespace elf {

class DefinedCommon;
class SymbolBody;
class InputSectionBase;
class InputSection;
class OutputSection;
class OutputSectionFactory;
class InputSectionBase;
class SectionBase;

struct ExprValue {
  SectionBase *Sec;
  uint64_t Val;
  bool ForceAbsolute;

  ExprValue(SectionBase *Sec, bool ForceAbsolute, uint64_t Val)
      : Sec(Sec), Val(Val), ForceAbsolute(ForceAbsolute) {}
  ExprValue(SectionBase *Sec, uint64_t Val) : ExprValue(Sec, false, Val) {}
  ExprValue(uint64_t Val) : ExprValue(nullptr, Val) {}
  bool isAbsolute() const { return ForceAbsolute || Sec == nullptr; }
  uint64_t getValue() const;
  uint64_t getSecAddr() const;
};

// This represents an expression in the linker script.
// ScriptParser::readExpr reads an expression and returns an Expr.
// Later, we evaluate the expression by calling the function.
typedef std::function<ExprValue()> Expr;

// This enum is used to implement linker script SECTIONS command.
// https://sourceware.org/binutils/docs/ld/SECTIONS.html#SECTIONS
enum SectionsCommandKind {
  AssignmentKind, // . = expr or <sym> = expr
  OutputSectionKind,
  InputSectionKind,
  AssertKind,   // ASSERT(expr)
  BytesDataKind // BYTE(expr), SHORT(expr), LONG(expr) or QUAD(expr)
};

struct BaseCommand {
  BaseCommand(int K) : Kind(K) {}
  int Kind;
};

// This represents ". = <expr>" or "<symbol> = <expr>".
struct SymbolAssignment : BaseCommand {
  SymbolAssignment(StringRef Name, Expr E, std::string Loc)
      : BaseCommand(AssignmentKind), Name(Name), Expression(E), Location(Loc) {}

  static bool classof(const BaseCommand *C);

  // The LHS of an expression. Name is either a symbol name or ".".
  StringRef Name;
  SymbolBody *Sym = nullptr;

  // The RHS of an expression.
  Expr Expression;

  // Command attributes for PROVIDE, HIDDEN and PROVIDE_HIDDEN.
  bool Provide = false;
  bool Hidden = false;

  // Holds file name and line number for error reporting.
  std::string Location;
};

// Linker scripts allow additional constraints to be put on ouput sections.
// If an output section is marked as ONLY_IF_RO, the section is created
// only if its input sections are read-only. Likewise, an output section
// with ONLY_IF_RW is created if all input sections are RW.
enum class ConstraintKind { NoConstraint, ReadOnly, ReadWrite };

// This struct is used to represent the location and size of regions of
// target memory. Instances of the struct are created by parsing the
// MEMORY command.
struct MemoryRegion {
  std::string Name;
  uint64_t Origin;
  uint64_t Length;
  uint64_t Offset;
  uint32_t Flags;
  uint32_t NegFlags;
};

struct OutputSectionCommand : BaseCommand {
  OutputSectionCommand(StringRef Name)
      : BaseCommand(OutputSectionKind), Name(Name) {}

  static bool classof(const BaseCommand *C);

  OutputSection *Sec = nullptr;
  MemoryRegion *MemRegion = nullptr;
  StringRef Name;
  Expr AddrExpr;
  Expr AlignExpr;
  Expr LMAExpr;
  Expr SubalignExpr;
  std::vector<BaseCommand *> Commands;
  std::vector<StringRef> Phdrs;
  llvm::Optional<uint32_t> Filler;
  ConstraintKind Constraint = ConstraintKind::NoConstraint;
  std::string Location;
  std::string MemoryRegionName;
};

// This struct represents one section match pattern in SECTIONS() command.
// It can optionally have negative match pattern for EXCLUDED_FILE command.
// Also it may be surrounded with SORT() command, so contains sorting rules.
struct SectionPattern {
  SectionPattern(StringMatcher &&Pat1, StringMatcher &&Pat2)
      : ExcludedFilePat(Pat1), SectionPat(Pat2) {}

  StringMatcher ExcludedFilePat;
  StringMatcher SectionPat;
  SortSectionPolicy SortOuter;
  SortSectionPolicy SortInner;
};

struct InputSectionDescription : BaseCommand {
  InputSectionDescription(StringRef FilePattern)
      : BaseCommand(InputSectionKind), FilePat(FilePattern) {}

  static bool classof(const BaseCommand *C);

  StringMatcher FilePat;

  // Input sections that matches at least one of SectionPatterns
  // will be associated with this InputSectionDescription.
  std::vector<SectionPattern> SectionPatterns;

  std::vector<InputSectionBase *> Sections;
};

// Represents an ASSERT().
struct AssertCommand : BaseCommand {
  AssertCommand(Expr E) : BaseCommand(AssertKind), Expression(E) {}

  static bool classof(const BaseCommand *C);

  Expr Expression;
};

// Represents BYTE(), SHORT(), LONG(), or QUAD().
struct BytesDataCommand : BaseCommand {
  BytesDataCommand(Expr E, unsigned Size)
      : BaseCommand(BytesDataKind), Expression(E), Size(Size) {}

  static bool classof(const BaseCommand *C);

  Expr Expression;
  unsigned Offset;
  unsigned Size;
};

struct PhdrsCommand {
  StringRef Name;
  unsigned Type;
  bool HasFilehdr;
  bool HasPhdrs;
  unsigned Flags;
  Expr LMAExpr;
};

// ScriptConfiguration holds linker script parse results.
struct ScriptConfiguration {
  // Used to assign addresses to sections.
  std::vector<BaseCommand *> Commands;

  // Used to assign sections to headers.
  std::vector<PhdrsCommand> PhdrsCommands;

  bool HasSections = false;

  // List of section patterns specified with KEEP commands. They will
  // be kept even if they are unused and --gc-sections is specified.
  std::vector<InputSectionDescription *> KeptSections;

  // A map from memory region name to a memory region descriptor.
  llvm::DenseMap<llvm::StringRef, MemoryRegion> MemoryRegions;

  // A list of symbols referenced by the script.
  std::vector<llvm::StringRef> ReferencedSymbols;
};

class LinkerScript {
protected:
  void assignSymbol(SymbolAssignment *Cmd, bool InSec);
  void setDot(Expr E, const Twine &Loc, bool InSec);

  std::vector<InputSectionBase *>
  computeInputSections(const InputSectionDescription *);

  std::vector<InputSectionBase *>
  createInputSectionList(OutputSectionCommand &Cmd);

  std::vector<size_t> getPhdrIndices(StringRef SectionName);
  size_t getPhdrIndex(const Twine &Loc, StringRef PhdrName);

  MemoryRegion *findMemoryRegion(OutputSectionCommand *Cmd);

  void switchTo(OutputSection *Sec);
  void flush();
  void output(InputSection *Sec);
  void process(BaseCommand &Base);

  OutputSection *Aether;
  bool ErrorOnMissingSection = false;

  uint64_t Dot;
  uint64_t ThreadBssOffset = 0;

  std::function<uint64_t()> LMAOffset;
  OutputSection *CurOutSec = nullptr;
  MemoryRegion *CurMemRegion = nullptr;

  llvm::DenseSet<OutputSection *> AlreadyOutputOS;
  llvm::DenseSet<InputSectionBase *> AlreadyOutputIS;

public:
  bool hasPhdrsCommands() { return !Opt.PhdrsCommands.empty(); }
  uint64_t getDot() { return Dot; }
  OutputSection *getOutputSection(const Twine &Loc, StringRef S);
  uint64_t getOutputSectionSize(StringRef S);
  void discard(ArrayRef<InputSectionBase *> V);

  ExprValue getSymbolValue(const Twine &Loc, StringRef S);
  bool isDefined(StringRef S);

  std::vector<OutputSection *> *OutputSections;
  void fabricateDefaultCommands(bool AllocateHeader);
  void addOrphanSections(OutputSectionFactory &Factory);
  void removeEmptyCommands();
  void adjustSectionsBeforeSorting();
  void adjustSectionsAfterSorting();

  std::vector<PhdrEntry> createPhdrs();
  bool ignoreInterpSection();

  llvm::Optional<uint32_t> getFiller(StringRef Name);
  bool hasLMA(StringRef Name);
  bool shouldKeep(InputSectionBase *S);
  void assignOffsets(OutputSectionCommand *Cmd);
  void placeOrphanSections();
  void processNonSectionCommands();
  void assignAddresses(std::vector<PhdrEntry> &Phdrs);
  int getSectionIndex(StringRef Name);

  void writeDataBytes(StringRef Name, uint8_t *Buf);
  void addSymbol(SymbolAssignment *Cmd);
  void processCommands(OutputSectionFactory &Factory);

  // Parsed linker script configurations are set to this struct.
  ScriptConfiguration Opt;
};

extern LinkerScript *Script;

} // end namespace elf
} // end namespace lld

#endif // LLD_ELF_LINKER_SCRIPT_H
