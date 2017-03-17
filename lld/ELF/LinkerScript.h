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
class ScriptParser;
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

// Parses a linker script. Calling this function updates
// Config and ScriptConfig.
void readLinkerScript(MemoryBufferRef MB);

// Parses a version script.
void readVersionScript(MemoryBufferRef MB);

void readDynamicList(MemoryBufferRef MB);

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

  virtual ~BaseCommand() = default;

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

struct OutputSectionCommand : BaseCommand {
  OutputSectionCommand(StringRef Name)
      : BaseCommand(OutputSectionKind), Name(Name) {}

  static bool classof(const BaseCommand *C);

  StringRef Name;
  Expr AddrExpr;
  Expr AlignExpr;
  Expr LMAExpr;
  Expr SubalignExpr;
  std::vector<std::unique_ptr<BaseCommand>> Commands;
  std::vector<StringRef> Phdrs;
  uint32_t Filler = 0;
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

// ScriptConfiguration holds linker script parse results.
struct ScriptConfiguration {
  // Used to assign addresses to sections.
  std::vector<std::unique_ptr<BaseCommand>> Commands;

  // Used to assign sections to headers.
  std::vector<PhdrsCommand> PhdrsCommands;

  bool HasSections = false;

  // List of section patterns specified with KEEP commands. They will
  // be kept even if they are unused and --gc-sections is specified.
  std::vector<InputSectionDescription *> KeptSections;

  // A map from memory region name to a memory region descriptor.
  llvm::DenseMap<llvm::StringRef, MemoryRegion> MemoryRegions;
};

extern ScriptConfiguration *ScriptConfig;

class LinkerScriptBase {
protected:
  ~LinkerScriptBase() = default;

  void assignSymbol(SymbolAssignment *Cmd, bool InSec = false);
  void computeInputSections(InputSectionDescription *);
  void setDot(Expr E, const Twine &Loc, bool InSec = false);

  std::vector<InputSectionBase *>
  createInputSectionList(OutputSectionCommand &Cmd);

  std::vector<size_t> getPhdrIndices(StringRef SectionName);
  size_t getPhdrIndex(const Twine &Loc, StringRef PhdrName);

  MemoryRegion *findMemoryRegion(OutputSectionCommand *Cmd, OutputSection *Sec);

  void switchTo(OutputSection *Sec);
  void flush();
  void output(InputSection *Sec);
  void process(BaseCommand &Base);

  OutputSection *Aether;
  bool ErrorOnMissingSection = false;

  // "ScriptConfig" is a bit too long, so define a short name for it.
  ScriptConfiguration &Opt = *ScriptConfig;

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

  virtual ExprValue getSymbolValue(const Twine &Loc, StringRef S) = 0;
  virtual bool isDefined(StringRef S) = 0;

  std::vector<OutputSection *> *OutputSections;
  void addOrphanSections(OutputSectionFactory &Factory);
  void removeEmptyCommands();
  void adjustSectionsBeforeSorting();
  void adjustSectionsAfterSorting();

  std::vector<PhdrEntry> createPhdrs();
  bool ignoreInterpSection();

  uint32_t getFiller(StringRef Name);
  bool hasLMA(StringRef Name);
  bool shouldKeep(InputSectionBase *S);
  void assignOffsets(OutputSectionCommand *Cmd);
  void placeOrphanSections();
  void processNonSectionCommands();
  void assignAddresses(std::vector<PhdrEntry> &Phdrs);
  int getSectionIndex(StringRef Name);
};

// This is a runner of the linker script.
template <class ELFT> class LinkerScript final : public LinkerScriptBase {
public:
  LinkerScript();
  ~LinkerScript();

  void writeDataBytes(StringRef Name, uint8_t *Buf);
  void addSymbol(SymbolAssignment *Cmd);
  void processCommands(OutputSectionFactory &Factory);

  ExprValue getSymbolValue(const Twine &Loc, StringRef S) override;
  bool isDefined(StringRef S) override;
};

// Variable template is a C++14 feature, so we can't template
// a global variable. Use a struct to workaround.
template <class ELFT> struct Script { static LinkerScript<ELFT> *X; };
template <class ELFT> LinkerScript<ELFT> *Script<ELFT>::X;

extern LinkerScriptBase *ScriptBase;

} // end namespace elf
} // end namespace lld

#endif // LLD_ELF_LINKER_SCRIPT_H
