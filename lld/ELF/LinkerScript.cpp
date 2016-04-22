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
#include "SymbolTable.h"
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

static bool matchStr(StringRef S, StringRef T);

static int precedence(StringRef Op) {
  return StringSwitch<int>(Op)
      .Case("*", 4)
      .Case("/", 3)
      .Case("+", 2)
      .Case("-", 2)
      .Case("&", 1)
      .Default(-1);
}

static StringRef next(ArrayRef<StringRef> &Tokens) {
  if (Tokens.empty()) {
    error("no next token");
    return "";
  }
  StringRef Tok = Tokens.front();
  Tokens = Tokens.slice(1);
  return Tok;
}

static bool expect(ArrayRef<StringRef> &Tokens, StringRef S) {
  if (Tokens.empty()) {
    error(S + " expected");
    return false;
  }
  StringRef Tok = Tokens.front();
  if (Tok != S) {
    error(S + " expected, but got " + Tok);
    return false;
  }
  Tokens = Tokens.slice(1);
  return true;
}

// This is a part of the operator-precedence parser to evaluate
// arithmetic expressions in SECTIONS command. This function evaluates an
// integer literal, a parenthesized expression, the ALIGN function,
// or the special variable ".".
template <class ELFT>
uint64_t LinkerScript<ELFT>::parsePrimary(ArrayRef<StringRef> &Tokens) {
  StringRef Tok = next(Tokens);
  if (Tok == ".")
    return Dot;
  if (Tok == "(") {
    uint64_t V = parseExpr(Tokens);
    if (!expect(Tokens, ")"))
      return 0;
    return V;
  }
  if (Tok == "ALIGN") {
    if (!expect(Tokens, "("))
      return 0;
    uint64_t V = parseExpr(Tokens);
    if (!expect(Tokens, ")"))
      return 0;
    return alignTo(Dot, V);
  }
  uint64_t V = 0;
  if (Tok.getAsInteger(0, V))
    error("malformed number: " + Tok);
  return V;
}

template <class ELFT>
uint64_t LinkerScript<ELFT>::parseTernary(ArrayRef<StringRef> &Tokens,
                                          uint64_t Cond) {
  next(Tokens);
  uint64_t V = parseExpr(Tokens);
  if (!expect(Tokens, ":"))
    return 0;
  uint64_t W = parseExpr(Tokens);
  return Cond ? V : W;
}

static uint64_t apply(StringRef Op, uint64_t L, uint64_t R) {
  if (Op == "+")
    return L + R;
  if (Op == "-")
    return L - R;
  if (Op == "*")
    return L * R;
  if (Op == "/") {
    if (R == 0) {
      error("division by zero");
      return 0;
    }
    return L / R;
  }
  if (Op == "&")
    return L & R;
  llvm_unreachable("invalid operator");
  return 0;
}

// This is an operator-precedence parser to evaluate
// arithmetic expressions in SECTIONS command.
// Tokens should start with an operator.
template <class ELFT>
uint64_t LinkerScript<ELFT>::parseExpr1(ArrayRef<StringRef> &Tokens,
                                        uint64_t Lhs, int MinPrec) {
  while (!Tokens.empty()) {
    // Read an operator and an expression.
    StringRef Op1 = Tokens.front();
    if (Op1 == "?")
      return parseTernary(Tokens, Lhs);

    if (precedence(Op1) < MinPrec)
      return Lhs;
    next(Tokens);
    uint64_t Rhs = parsePrimary(Tokens);

    // Evaluate the remaining part of the expression first if the
    // next operator has greater precedence than the previous one.
    // For example, if we have read "+" and "3", and if the next
    // operator is "*", then we'll evaluate 3 * ... part first.
    while (!Tokens.empty()) {
      StringRef Op2 = Tokens.front();
      if (precedence(Op2) <= precedence(Op1))
        break;
      Rhs = parseExpr1(Tokens, Rhs, precedence(Op2));
    }

    Lhs = apply(Op1, Lhs, Rhs);
  }
  return Lhs;
}

template <class ELFT>
uint64_t LinkerScript<ELFT>::parseExpr(ArrayRef<StringRef> &Tokens) {
  uint64_t V = parsePrimary(Tokens);
  return parseExpr1(Tokens, V, 0);
}

// Evaluates the expression given by list of tokens.
template <class ELFT>
uint64_t LinkerScript<ELFT>::evaluate(ArrayRef<StringRef> Tokens) {
  uint64_t V = parseExpr(Tokens);
  if (!Tokens.empty())
    error("stray token: " + Tokens[0]);
  return V;
}

template <class ELFT>
StringRef LinkerScript<ELFT>::getOutputSection(InputSectionBase<ELFT> *S) {
  for (SectionRule &R : Opt.Sections)
    if (matchStr(R.SectionPattern, S->getSectionName()))
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
    if (matchStr(Pat, S->getSectionName()))
      return true;
  return false;
}

template <class ELFT>
static OutputSectionBase<ELFT> *
findSection(ArrayRef<OutputSectionBase<ELFT> *> V, StringRef Name) {
  for (OutputSectionBase<ELFT> *Sec : V)
    if (Sec->getName() == Name)
      return Sec;
  return nullptr;
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
  uintX_t ThreadBssOffset = 0;

  for (SectionsCommand &Cmd : Opt.Commands) {
    if (Cmd.Kind == ExprKind) {
      Dot = evaluate(Cmd.Expr);
      continue;
    }

    OutputSectionBase<ELFT> *Sec = findSection<ELFT>(Sections, Cmd.SectionName);
    if (!Sec)
      continue;

    if ((Sec->getFlags() & SHF_TLS) && Sec->getType() == SHT_NOBITS) {
      uintX_t TVA = Dot + ThreadBssOffset;
      TVA = alignTo(TVA, Sec->getAlign());
      Sec->setVA(TVA);
      ThreadBssOffset = TVA - Dot + Sec->getSize();
      continue;
    }

    if (Sec->getFlags() & SHF_ALLOC) {
      Dot = alignTo(Dot, Sec->getAlign());
      Sec->setVA(Dot);
      Dot += Sec->getSize();
      continue;
    }
  }
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
    return N.Kind == SectionKind && N.SectionName == Name;
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

// Returns true if S matches T. S can contain glob meta-characters.
// The asterisk ('*') matches zero or more characacters, and the question
// mark ('?') matches one character.
static bool matchStr(StringRef S, StringRef T) {
  for (;;) {
    if (S.empty())
      return T.empty();
    if (S[0] == '*') {
      S = S.substr(1);
      if (S.empty())
        // Fast path. If a pattern is '*', it matches anything.
        return true;
      for (size_t I = 0, E = T.size(); I < E; ++I)
        if (matchStr(S, T.substr(I)))
          return true;
      return false;
    }
    if (T.empty() || (S[0] != T[0] && S[0] != '?'))
      return false;
    S = S.substr(1);
    T = T.substr(1);
  }
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
  void readOutputSectionDescription();
  void readSectionPatterns(StringRef OutSec);

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
    if (Tok == ".")
      readLocationCounterValue();
    else
      readOutputSectionDescription();
  }
}

void ScriptParser::readSectionPatterns(StringRef OutSec) {
  expect("(");
  while (!Error && !skip(")"))
    Opt.Sections.emplace_back(OutSec, next());
}

void ScriptParser::readLocationCounterValue() {
  expect(".");
  expect("=");
  Opt.Commands.push_back({ExprKind, {}, ""});
  SectionsCommand &Cmd = Opt.Commands.back();
  while (!Error) {
    StringRef Tok = next();
    if (Tok == ";")
      break;
    Cmd.Expr.push_back(Tok);
  }
  if (Cmd.Expr.empty())
    error("error in location counter expression");
}

void ScriptParser::readOutputSectionDescription() {
  StringRef OutSec = next();
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
