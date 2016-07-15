//===- LinkerScript.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the parser/evaluator of the linker script.
// It does not construct an AST but consume linker script directives directly.
// Results are written to Driver or Config object.
//
//===----------------------------------------------------------------------===//

#include "LinkerScript.h"
#include "Config.h"
#include "Driver.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "ScriptParser.h"
#include "Strings.h"
#include "Symbols.h"
#include "SymbolTable.h"
#include "Target.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace lld;
using namespace lld::elf;

ScriptConfiguration *elf::ScriptConfig;

// This is an operator-precedence parser to parse and evaluate
// a linker script expression. For each linker script arithmetic
// expression (e.g. ". = . + 0x1000"), a new instance of ExprParser
// is created and ran.
namespace {
class ExprParser : public ScriptParserBase {
public:
  ExprParser(std::vector<StringRef> &Tokens, uint64_t Dot)
      : ScriptParserBase(Tokens), Dot(Dot) {}

  uint64_t run();

private:
  uint64_t parsePrimary();
  uint64_t parseTernary(uint64_t Cond);
  uint64_t apply(StringRef Op, uint64_t L, uint64_t R);
  uint64_t parseExpr1(uint64_t Lhs, int MinPrec);
  uint64_t parseExpr();

  uint64_t Dot;
};
}

static int precedence(StringRef Op) {
  return StringSwitch<int>(Op)
      .Case("*", 4)
      .Case("/", 4)
      .Case("+", 3)
      .Case("-", 3)
      .Case("<", 2)
      .Case(">", 2)
      .Case(">=", 2)
      .Case("<=", 2)
      .Case("==", 2)
      .Case("!=", 2)
      .Case("&", 1)
      .Default(-1);
}

static uint64_t evalExpr(std::vector<StringRef> &Tokens, uint64_t Dot) {
  return ExprParser(Tokens, Dot).run();
}

uint64_t ExprParser::run() {
  uint64_t V = parseExpr();
  if (!atEOF() && !Error)
    setError("stray token: " + peek());
  return V;
}

// This is a part of the operator-precedence parser to evaluate
// arithmetic expressions in SECTIONS command. This function evaluates an
// integer literal, a parenthesized expression, the ALIGN function,
// or the special variable ".".
uint64_t ExprParser::parsePrimary() {
  StringRef Tok = next();
  if (Tok == ".")
    return Dot;
  if (Tok == "(") {
    uint64_t V = parseExpr();
    expect(")");
    return V;
  }
  if (Tok == "ALIGN") {
    expect("(");
    uint64_t V = parseExpr();
    expect(")");
    return alignTo(Dot, V);
  }
  uint64_t V = 0;
  if (Tok.getAsInteger(0, V))
    setError("malformed number: " + Tok);
  return V;
}

uint64_t ExprParser::parseTernary(uint64_t Cond) {
  next();
  uint64_t V = parseExpr();
  expect(":");
  uint64_t W = parseExpr();
  return Cond ? V : W;
}

uint64_t ExprParser::apply(StringRef Op, uint64_t L, uint64_t R) {
  if (Op == "*")
    return L * R;
  if (Op == "/") {
    if (R == 0) {
      error("division by zero");
      return 0;
    }
    return L / R;
  }
  if (Op == "+")
    return L + R;
  if (Op == "-")
    return L - R;
  if (Op == "<")
    return L < R;
  if (Op == ">")
    return L > R;
  if (Op == ">=")
    return L >= R;
  if (Op == "<=")
    return L <= R;
  if (Op == "==")
    return L == R;
  if (Op == "!=")
    return L != R;
  if (Op == "&")
    return L & R;
  llvm_unreachable("invalid operator");
}

// This is a part of the operator-precedence parser.
// This function assumes that the remaining token stream starts
// with an operator.
uint64_t ExprParser::parseExpr1(uint64_t Lhs, int MinPrec) {
  while (!atEOF()) {
    // Read an operator and an expression.
    StringRef Op1 = peek();
    if (Op1 == "?")
      return parseTernary(Lhs);
    if (precedence(Op1) < MinPrec)
      return Lhs;
    next();
    uint64_t Rhs = parsePrimary();

    // Evaluate the remaining part of the expression first if the
    // next operator has greater precedence than the previous one.
    // For example, if we have read "+" and "3", and if the next
    // operator is "*", then we'll evaluate 3 * ... part first.
    while (!atEOF()) {
      StringRef Op2 = peek();
      if (precedence(Op2) <= precedence(Op1))
        break;
      Rhs = parseExpr1(Rhs, precedence(Op2));
    }

    Lhs = apply(Op1, Lhs, Rhs);
  }
  return Lhs;
}

// Reads and evaluates an arithmetic expression.
uint64_t ExprParser::parseExpr() { return parseExpr1(parsePrimary(), 0); }

template <class ELFT>
StringRef LinkerScript<ELFT>::getOutputSection(InputSectionBase<ELFT> *S) {
  for (SectionRule &R : Opt.Sections)
    if (globMatch(R.SectionPattern, S->getSectionName()))
      return R.Dest;
  return "";
}

template <class ELFT>
bool LinkerScript<ELFT>::isDiscarded(InputSectionBase<ELFT> *S) {
  return getOutputSection(S) == "/DISCARD/";
}

template <class ELFT>
bool LinkerScript<ELFT>::shouldKeep(InputSectionBase<ELFT> *S) {
  for (StringRef Pat : Opt.KeptSections)
    if (globMatch(Pat, S->getSectionName()))
      return true;
  return false;
}

template <class ELFT>
void LinkerScript<ELFT>::assignAddresses(
    ArrayRef<OutputSectionBase<ELFT> *> Sections) {
  // Orphan sections are sections present in the input files which
  // are not explicitly placed into the output file by the linker script.
  // We place orphan sections at end of file.
  // Other linkers places them using some heuristics as described in
  // https://sourceware.org/binutils/docs/ld/Orphan-Sections.html#Orphan-Sections.
  for (OutputSectionBase<ELFT> *Sec : Sections) {
    StringRef Name = Sec->getName();
    if (getSectionIndex(Name) == INT_MAX)
      Opt.Commands.push_back({SectionKind, {}, Name});
  }

  // Assign addresses as instructed by linker script SECTIONS sub-commands.
  Dot = Out<ELFT>::ElfHeader->getSize() + Out<ELFT>::ProgramHeaders->getSize();
  uintX_t MinVA = std::numeric_limits<uintX_t>::max();
  uintX_t ThreadBssOffset = 0;

  for (SectionsCommand &Cmd : Opt.Commands) {
    if (Cmd.Kind == AssignmentKind) {
      uint64_t Val = evalExpr(Cmd.Expr, Dot);

      if (Cmd.Name == ".") {
        Dot = Val;
      } else {
        auto *D = cast<DefinedRegular<ELFT>>(Symtab<ELFT>::X->find(Cmd.Name));
        D->Value = Val;
      }
      continue;
    }

    // Find all the sections with required name. There can be more than
    // ont section with such name, if the alignment, flags or type
    // attribute differs.
    assert(Cmd.Kind == SectionKind);
    for (OutputSectionBase<ELFT> *Sec : Sections) {
      if (Sec->getName() != Cmd.Name)
        continue;

      if ((Sec->getFlags() & SHF_TLS) && Sec->getType() == SHT_NOBITS) {
        uintX_t TVA = Dot + ThreadBssOffset;
        TVA = alignTo(TVA, Sec->getAlignment());
        Sec->setVA(TVA);
        ThreadBssOffset = TVA - Dot + Sec->getSize();
        continue;
      }

      if (Sec->getFlags() & SHF_ALLOC) {
        Dot = alignTo(Dot, Sec->getAlignment());
        Sec->setVA(Dot);
        MinVA = std::min(MinVA, Dot);
        Dot += Sec->getSize();
        continue;
      }
    }
  }

  // ELF and Program headers need to be right before the first section in
  // memory.
  // Set their addresses accordingly.
  MinVA = alignDown(MinVA - Out<ELFT>::ElfHeader->getSize() -
                        Out<ELFT>::ProgramHeaders->getSize(),
                    Target->PageSize);
  Out<ELFT>::ElfHeader->setVA(MinVA);
  Out<ELFT>::ProgramHeaders->setVA(Out<ELFT>::ElfHeader->getSize() + MinVA);
}

template <class ELFT>
ArrayRef<uint8_t> LinkerScript<ELFT>::getFiller(StringRef Name) {
  auto I = Opt.Filler.find(Name);
  if (I == Opt.Filler.end())
    return {};
  return I->second;
}

// Returns the index of the given section name in linker script
// SECTIONS commands. Sections are laid out as the same order as they
// were in the script. If a given name did not appear in the script,
// it returns INT_MAX, so that it will be laid out at end of file.
template <class ELFT>
int LinkerScript<ELFT>::getSectionIndex(StringRef Name) {
  auto Begin = Opt.Commands.begin();
  auto End = Opt.Commands.end();
  auto I = std::find_if(Begin, End, [&](SectionsCommand &N) {
    return N.Kind == SectionKind && N.Name == Name;
  });
  return I == End ? INT_MAX : (I - Begin);
}

// A compartor to sort output sections. Returns -1 or 1 if
// A or B are mentioned in linker script. Otherwise, returns 0.
template <class ELFT>
int LinkerScript<ELFT>::compareSections(StringRef A, StringRef B) {
  int I = getSectionIndex(A);
  int J = getSectionIndex(B);
  if (I == INT_MAX && J == INT_MAX)
    return 0;
  return I < J ? -1 : 1;
}

template <class ELFT>
void LinkerScript<ELFT>::addScriptedSymbols() {
  for (SectionsCommand &Cmd : Opt.Commands)
    if (Cmd.Kind == AssignmentKind)
      if (Cmd.Name != "." && Symtab<ELFT>::X->find(Cmd.Name) == nullptr)
        Symtab<ELFT>::X->addAbsolute(Cmd.Name, STV_DEFAULT);
}

class elf::ScriptParser : public ScriptParserBase {
  typedef void (ScriptParser::*Handler)();

public:
  ScriptParser(StringRef S, bool B) : ScriptParserBase(S), IsUnderSysroot(B) {}

  void run();

private:
  void addFile(StringRef Path);

  void readAsNeeded();
  void readEntry();
  void readExtern();
  void readGroup();
  void readInclude();
  void readNothing() {}
  void readOutput();
  void readOutputArch();
  void readOutputFormat();
  void readSearchDir();
  void readSections();

  void readLocationCounterValue();
  void readOutputSectionDescription(StringRef OutSec);
  void readSymbolAssignment(StringRef Name);
  std::vector<StringRef> readSectionsCommandExpr();

  const static StringMap<Handler> Cmd;
  ScriptConfiguration &Opt = *ScriptConfig;
  StringSaver Saver = {ScriptConfig->Alloc};
  bool IsUnderSysroot;
};

const StringMap<elf::ScriptParser::Handler> elf::ScriptParser::Cmd = {
    {"ENTRY", &ScriptParser::readEntry},
    {"EXTERN", &ScriptParser::readExtern},
    {"GROUP", &ScriptParser::readGroup},
    {"INCLUDE", &ScriptParser::readInclude},
    {"INPUT", &ScriptParser::readGroup},
    {"OUTPUT", &ScriptParser::readOutput},
    {"OUTPUT_ARCH", &ScriptParser::readOutputArch},
    {"OUTPUT_FORMAT", &ScriptParser::readOutputFormat},
    {"SEARCH_DIR", &ScriptParser::readSearchDir},
    {"SECTIONS", &ScriptParser::readSections},
    {";", &ScriptParser::readNothing}};

void ScriptParser::run() {
  while (!atEOF()) {
    StringRef Tok = next();
    if (Handler Fn = Cmd.lookup(Tok))
      (this->*Fn)();
    else
      setError("unknown directive: " + Tok);
  }
}

void ScriptParser::addFile(StringRef S) {
  if (IsUnderSysroot && S.startswith("/")) {
    SmallString<128> Path;
    (Config->Sysroot + S).toStringRef(Path);
    if (sys::fs::exists(Path)) {
      Driver->addFile(Saver.save(Path.str()));
      return;
    }
  }

  if (sys::path::is_absolute(S)) {
    Driver->addFile(S);
  } else if (S.startswith("=")) {
    if (Config->Sysroot.empty())
      Driver->addFile(S.substr(1));
    else
      Driver->addFile(Saver.save(Config->Sysroot + "/" + S.substr(1)));
  } else if (S.startswith("-l")) {
    Driver->addLibrary(S.substr(2));
  } else if (sys::fs::exists(S)) {
    Driver->addFile(S);
  } else {
    std::string Path = findFromSearchPaths(S);
    if (Path.empty())
      setError("unable to find " + S);
    else
      Driver->addFile(Saver.save(Path));
  }
}

void ScriptParser::readAsNeeded() {
  expect("(");
  bool Orig = Config->AsNeeded;
  Config->AsNeeded = true;
  while (!Error) {
    StringRef Tok = next();
    if (Tok == ")")
      break;
    addFile(Tok);
  }
  Config->AsNeeded = Orig;
}

void ScriptParser::readEntry() {
  // -e <symbol> takes predecence over ENTRY(<symbol>).
  expect("(");
  StringRef Tok = next();
  if (Config->Entry.empty())
    Config->Entry = Tok;
  expect(")");
}

void ScriptParser::readExtern() {
  expect("(");
  while (!Error) {
    StringRef Tok = next();
    if (Tok == ")")
      return;
    Config->Undefined.push_back(Tok);
  }
}

void ScriptParser::readGroup() {
  expect("(");
  while (!Error) {
    StringRef Tok = next();
    if (Tok == ")")
      return;
    if (Tok == "AS_NEEDED") {
      readAsNeeded();
      continue;
    }
    addFile(Tok);
  }
}

void ScriptParser::readInclude() {
  StringRef Tok = next();
  auto MBOrErr = MemoryBuffer::getFile(Tok);
  if (!MBOrErr) {
    setError("cannot open " + Tok);
    return;
  }
  std::unique_ptr<MemoryBuffer> &MB = *MBOrErr;
  StringRef S = Saver.save(MB->getMemBufferRef().getBuffer());
  std::vector<StringRef> V = tokenize(S);
  Tokens.insert(Tokens.begin() + Pos, V.begin(), V.end());
}

void ScriptParser::readOutput() {
  // -o <file> takes predecence over OUTPUT(<file>).
  expect("(");
  StringRef Tok = next();
  if (Config->OutputFile.empty())
    Config->OutputFile = Tok;
  expect(")");
}

void ScriptParser::readOutputArch() {
  // Error checking only for now.
  expect("(");
  next();
  expect(")");
}

void ScriptParser::readOutputFormat() {
  // Error checking only for now.
  expect("(");
  next();
  StringRef Tok = next();
  if (Tok == ")")
   return;
  if (Tok != ",") {
    setError("unexpected token: " + Tok);
    return;
  }
  next();
  expect(",");
  next();
  expect(")");
}

void ScriptParser::readSearchDir() {
  expect("(");
  Config->SearchPaths.push_back(next());
  expect(")");
}

void ScriptParser::readSections() {
  Opt.DoLayout = true;
  expect("{");
  while (!Error && !skip("}")) {
    StringRef Tok = peek();
    if (Tok == ".") {
      readLocationCounterValue();
      continue;
    }
    next();
    if (peek() == "=")
      readSymbolAssignment(Tok);
    else
      readOutputSectionDescription(Tok);
  }
}

void ScriptParser::readLocationCounterValue() {
  expect(".");
  expect("=");
  std::vector<StringRef> Expr = readSectionsCommandExpr();
  if (Expr.empty())
    error("error in location counter expression");
  else
    Opt.Commands.push_back({AssignmentKind, std::move(Expr), "."});
}

void ScriptParser::readOutputSectionDescription(StringRef OutSec) {
  Opt.Commands.push_back({SectionKind, {}, OutSec});
  expect(":");
  expect("{");

  while (!Error && !skip("}")) {
    StringRef Tok = next();
    if (Tok == "*") {
      expect("(");
      while (!Error && !skip(")"))
        Opt.Sections.emplace_back(OutSec, next());
    } else if (Tok == "KEEP") {
      expect("(");
      expect("*");
      expect("(");
      while (!Error && !skip(")")) {
        StringRef Sec = next();
        Opt.Sections.emplace_back(OutSec, Sec);
        Opt.KeptSections.push_back(Sec);
      }
      expect(")");
    } else {
      setError("unknown command " + Tok);
    }
  }

  StringRef Tok = peek();
  if (Tok.startswith("=")) {
    if (!Tok.startswith("=0x")) {
      setError("filler should be a hexadecimal value");
      return;
    }
    Tok = Tok.substr(3);
    Opt.Filler[OutSec] = parseHex(Tok);
    next();
  }
}

void ScriptParser::readSymbolAssignment(StringRef Name) {
  expect("=");
  std::vector<StringRef> Expr = readSectionsCommandExpr();
  if (Expr.empty())
    error("error in symbol assignment expression");
  else
    Opt.Commands.push_back({AssignmentKind, std::move(Expr), Name});
}

std::vector<StringRef> ScriptParser::readSectionsCommandExpr() {
  std::vector<StringRef> Expr;
  while (!Error) {
    StringRef Tok = next();
    if (Tok == ";")
      break;
    Expr.push_back(Tok);
  }
  return Expr;
}

static bool isUnderSysroot(StringRef Path) {
  if (Config->Sysroot == "")
    return false;
  for (; !Path.empty(); Path = sys::path::parent_path(Path))
    if (sys::fs::equivalent(Config->Sysroot, Path))
      return true;
  return false;
}

// Entry point.
void elf::readLinkerScript(MemoryBufferRef MB) {
  StringRef Path = MB.getBufferIdentifier();
  ScriptParser(MB.getBuffer(), isUnderSysroot(Path)).run();
}

template class elf::LinkerScript<ELF32LE>;
template class elf::LinkerScript<ELF32BE>;
template class elf::LinkerScript<ELF64LE>;
template class elf::LinkerScript<ELF64BE>;
