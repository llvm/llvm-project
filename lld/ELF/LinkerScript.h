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
template <class ELFT> class InputSectionBase;
template <class ELFT> class InputSection;
class OutputSectionBase;
template <class ELFT> class OutputSectionFactory;
class InputSectionData;

// This represents an expression in the linker script.
// ScriptParser::readExpr reads an expression and returns an Expr.
// Later, we evaluate the expression by calling the function
// with the value of special context variable ".".
struct Expr {
  std::function<uint64_t(uint64_t)> Val;
  std::function<bool()> IsAbsolute;

  // If expression is section-relative the function below is used
  // to get the output section pointer.
  std::function<const OutputSectionBase *()> Section;

  uint64_t operator()(uint64_t Dot) const { return Val(Dot); }
  operator bool() const { return (bool)Val; }

  Expr(std::function<uint64_t(uint64_t)> Val, std::function<bool()> IsAbsolute,
       std::function<const OutputSectionBase *()> Section)
      : Val(Val), IsAbsolute(IsAbsolute), Section(Section) {}
  template <typename T>
  Expr(T V) : Expr(V, [] { return true; }, [] { return nullptr; }) {}
  Expr() : Expr(nullptr) {}
};

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
  SymbolAssignment(StringRef Name, Expr E)
      : BaseCommand(AssignmentKind), Name(Name), Expression(E) {}

  static bool classof(const BaseCommand *C);

  // The LHS of an expression. Name is either a symbol name or ".".
  StringRef Name;
  SymbolBody *Sym = nullptr;

  // The RHS of an expression.
  Expr Expression;

  // Command attributes for PROVIDE, HIDDEN and PROVIDE_HIDDEN.
  bool Provide = false;
  bool Hidden = false;
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

  std::vector<InputSectionData *> Sections;
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

class LinkerScriptBase {
protected:
  ~LinkerScriptBase() = default;

public:
  virtual uint64_t getHeaderSize() = 0;
  virtual uint64_t getSymbolValue(const Twine &Loc, StringRef S) = 0;
  virtual bool isDefined(StringRef S) = 0;
  virtual bool isAbsolute(StringRef S) = 0;
  virtual const OutputSectionBase *getSymbolSection(StringRef S) = 0;
  virtual const OutputSectionBase *getOutputSection(const Twine &Loc,
                                                    StringRef S) = 0;
  virtual uint64_t getOutputSectionSize(StringRef S) = 0;
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

// This is a runner of the linker script.
template <class ELFT> class LinkerScript final : public LinkerScriptBase {
  typedef typename ELFT::uint uintX_t;

public:
  LinkerScript();
  ~LinkerScript();

  void processCommands(OutputSectionFactory<ELFT> &Factory);
  void addOrphanSections(OutputSectionFactory<ELFT> &Factory);
  void removeEmptyCommands();
  void adjustSectionsBeforeSorting();
  void adjustSectionsAfterSorting();

  std::vector<PhdrEntry> createPhdrs();
  bool ignoreInterpSection();

  uint32_t getFiller(StringRef Name);
  void writeDataBytes(StringRef Name, uint8_t *Buf);
  bool hasLMA(StringRef Name);
  bool shouldKeep(InputSectionBase<ELFT> *S);
  void assignOffsets(OutputSectionCommand *Cmd);
  void placeOrphanSections();
  void assignAddresses(std::vector<PhdrEntry> &Phdrs);
  bool hasPhdrsCommands();
  uint64_t getHeaderSize() override;
  uint64_t getSymbolValue(const Twine &Loc, StringRef S) override;
  bool isDefined(StringRef S) override;
  bool isAbsolute(StringRef S) override;
  const OutputSectionBase *getSymbolSection(StringRef S) override;
  const OutputSectionBase *getOutputSection(const Twine &Loc,
                                            StringRef S) override;
  uint64_t getOutputSectionSize(StringRef S) override;

  std::vector<OutputSectionBase *> *OutputSections;

  int getSectionIndex(StringRef Name);

private:
  void computeInputSections(InputSectionDescription *);

  void addSection(OutputSectionFactory<ELFT> &Factory,
                  InputSectionBase<ELFT> *Sec, StringRef Name);
  void discard(ArrayRef<InputSectionBase<ELFT> *> V);

  std::vector<InputSectionBase<ELFT> *>
  createInputSectionList(OutputSectionCommand &Cmd);

  // "ScriptConfig" is a bit too long, so define a short name for it.
  ScriptConfiguration &Opt = *ScriptConfig;

  std::vector<size_t> getPhdrIndices(StringRef SectionName);
  size_t getPhdrIndex(const Twine &Loc, StringRef PhdrName);

  MemoryRegion *findMemoryRegion(OutputSectionCommand *Cmd,
                                 OutputSectionBase *Sec);

  uintX_t Dot;
  uintX_t LMAOffset = 0;
  OutputSectionBase *CurOutSec = nullptr;
  MemoryRegion *CurMemRegion = nullptr;
  uintX_t ThreadBssOffset = 0;
  void switchTo(OutputSectionBase *Sec);
  void flush();
  void output(InputSection<ELFT> *Sec);
  void process(BaseCommand &Base);
  llvm::DenseSet<OutputSectionBase *> AlreadyOutputOS;
  llvm::DenseSet<InputSectionData *> AlreadyOutputIS;
};

// Variable template is a C++14 feature, so we can't template
// a global variable. Use a struct to workaround.
template <class ELFT> struct Script { static LinkerScript<ELFT> *X; };
template <class ELFT> LinkerScript<ELFT> *Script<ELFT>::X;

extern LinkerScriptBase *ScriptBase;

} // end namespace elf
} // end namespace lld

#endif // LLD_ELF_LINKER_SCRIPT_H
