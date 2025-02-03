//===- ScriptParser.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a recursive-descendent parser for linker scripts.
// Parsed results are stored to Config and Script global objects.
//
//===----------------------------------------------------------------------===//

#include "ScriptParser.h"
#include "Config.h"
#include "Driver.h"
#include "InputFiles.h"
#include "LinkerScript.h"
#include "OutputSections.h"
#include "ScriptLexer.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "Target.h"
#include "lld/Common/CommonLinkerContext.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/TimeProfiler.h"
#include <cassert>
#include <limits>
#include <optional>
#include <vector>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::elf;

namespace {
class ScriptParser final : ScriptLexer {
public:
  ScriptParser(Ctx &ctx, MemoryBufferRef mb) : ScriptLexer(ctx, mb), ctx(ctx) {}

  void readLinkerScript();
  void readVersionScript();
  void readDynamicList();
  void readDefsym();

private:
  void addFile(StringRef path);

  void readAsNeeded();
  void readEntry();
  void readExtern();
  void readGroup();
  void readInclude();
  void readInput();
  void readMemory();
  void readOutput();
  void readOutputArch();
  void readOutputFormat();
  void readOverwriteSections();
  void readPhdrs();
  void readRegionAlias();
  void readSearchDir();
  void readSections();
  void readTarget();
  void readVersion();
  void readVersionScriptCommand();
  void readNoCrossRefs(bool to);

  StringRef readName();
  SymbolAssignment *readSymbolAssignment(StringRef name);
  ByteCommand *readByteCommand(StringRef tok);
  std::array<uint8_t, 4> readFill();
  bool readSectionDirective(OutputSection *cmd, StringRef tok);
  void readSectionAddressType(OutputSection *cmd);
  OutputDesc *readOverlaySectionDescription();
  OutputDesc *readOutputSectionDescription(StringRef outSec);
  SmallVector<SectionCommand *, 0> readOverlay();
  SectionClassDesc *readSectionClassDescription();
  StringRef readSectionClassName();
  SmallVector<StringRef, 0> readOutputSectionPhdrs();
  std::pair<uint64_t, uint64_t> readInputSectionFlags();
  InputSectionDescription *readInputSectionDescription(StringRef tok);
  StringMatcher readFilePatterns();
  SmallVector<SectionPattern, 0> readInputSectionsList();
  InputSectionDescription *readInputSectionRules(StringRef filePattern,
                                                 uint64_t withFlags,
                                                 uint64_t withoutFlags);
  unsigned readPhdrType();
  SortSectionPolicy peekSortKind();
  SortSectionPolicy readSortKind();
  SymbolAssignment *readProvideHidden(bool provide, bool hidden);
  SymbolAssignment *readAssignment(StringRef tok);
  void readSort();
  Expr readAssert();
  Expr readConstant();
  Expr getPageSize();

  Expr readMemoryAssignment(StringRef, StringRef, StringRef);
  void readMemoryAttributes(uint32_t &flags, uint32_t &invFlags,
                            uint32_t &negFlags, uint32_t &negInvFlags);

  Expr combine(StringRef op, Expr l, Expr r);
  Expr readExpr();
  Expr readExpr1(Expr lhs, int minPrec);
  StringRef readParenName();
  Expr readPrimary();
  Expr readTernary(Expr cond);
  Expr readParenExpr();

  // For parsing version script.
  SmallVector<SymbolVersion, 0> readVersionExtern();
  void readAnonymousDeclaration();
  void readVersionDeclaration(StringRef verStr);

  std::pair<SmallVector<SymbolVersion, 0>, SmallVector<SymbolVersion, 0>>
  readSymbols();

  Ctx &ctx;

  // If we are currently parsing a PROVIDE|PROVIDE_HIDDEN command,
  // then this member is set to the PROVIDE symbol name.
  std::optional<llvm::StringRef> activeProvideSym;
};
} // namespace

static StringRef unquote(StringRef s) {
  if (s.starts_with("\""))
    return s.substr(1, s.size() - 2);
  return s;
}

// Some operations only support one non absolute value. Move the
// absolute one to the right hand side for convenience.
static void moveAbsRight(LinkerScript &s, ExprValue &a, ExprValue &b) {
  if (a.sec == nullptr || (a.forceAbsolute && !b.isAbsolute()))
    std::swap(a, b);
  if (!b.isAbsolute())
    s.recordError(a.loc +
                  ": at least one side of the expression must be absolute");
}

static ExprValue add(LinkerScript &s, ExprValue a, ExprValue b) {
  moveAbsRight(s, a, b);
  return {a.sec, a.forceAbsolute, a.getSectionOffset() + b.getValue(), a.loc};
}

static ExprValue sub(ExprValue a, ExprValue b) {
  // The distance between two symbols in sections is absolute.
  if (!a.isAbsolute() && !b.isAbsolute())
    return a.getValue() - b.getValue();
  return {a.sec, false, a.getSectionOffset() - b.getValue(), a.loc};
}

static ExprValue bitAnd(LinkerScript &s, ExprValue a, ExprValue b) {
  moveAbsRight(s, a, b);
  return {a.sec, a.forceAbsolute,
          (a.getValue() & b.getValue()) - a.getSecAddr(), a.loc};
}

static ExprValue bitXor(LinkerScript &s, ExprValue a, ExprValue b) {
  moveAbsRight(s, a, b);
  return {a.sec, a.forceAbsolute,
          (a.getValue() ^ b.getValue()) - a.getSecAddr(), a.loc};
}

static ExprValue bitOr(LinkerScript &s, ExprValue a, ExprValue b) {
  moveAbsRight(s, a, b);
  return {a.sec, a.forceAbsolute,
          (a.getValue() | b.getValue()) - a.getSecAddr(), a.loc};
}

void ScriptParser::readDynamicList() {
  expect("{");
  SmallVector<SymbolVersion, 0> locals;
  SmallVector<SymbolVersion, 0> globals;
  std::tie(locals, globals) = readSymbols();
  expect(";");

  StringRef tok = peek();
  if (tok.size()) {
    setError("EOF expected, but got " + tok);
    return;
  }
  if (!locals.empty()) {
    setError("\"local:\" scope not supported in --dynamic-list");
    return;
  }

  for (SymbolVersion v : globals)
    ctx.arg.dynamicList.push_back(v);
}

void ScriptParser::readVersionScript() {
  readVersionScriptCommand();
  StringRef tok = peek();
  if (tok.size())
    setError("EOF expected, but got " + tok);
}

void ScriptParser::readVersionScriptCommand() {
  if (consume("{")) {
    readAnonymousDeclaration();
    return;
  }

  if (atEOF())
    setError("unexpected EOF");
  while (peek() != "}" && !atEOF()) {
    StringRef verStr = next();
    if (verStr == "{") {
      setError("anonymous version definition is used in "
               "combination with other version definitions");
      return;
    }
    expect("{");
    readVersionDeclaration(verStr);
  }
}

void ScriptParser::readVersion() {
  expect("{");
  readVersionScriptCommand();
  expect("}");
}

void ScriptParser::readLinkerScript() {
  while (!atEOF()) {
    StringRef tok = next();
    if (atEOF())
      break;
    if (tok == ";")
      continue;

    if (tok == "ENTRY") {
      readEntry();
    } else if (tok == "EXTERN") {
      readExtern();
    } else if (tok == "GROUP") {
      readGroup();
    } else if (tok == "INCLUDE") {
      readInclude();
    } else if (tok == "INPUT") {
      readInput();
    } else if (tok == "MEMORY") {
      readMemory();
    } else if (tok == "OUTPUT") {
      readOutput();
    } else if (tok == "OUTPUT_ARCH") {
      readOutputArch();
    } else if (tok == "OUTPUT_FORMAT") {
      readOutputFormat();
    } else if (tok == "OVERWRITE_SECTIONS") {
      readOverwriteSections();
    } else if (tok == "PHDRS") {
      readPhdrs();
    } else if (tok == "REGION_ALIAS") {
      readRegionAlias();
    } else if (tok == "SEARCH_DIR") {
      readSearchDir();
    } else if (tok == "SECTIONS") {
      readSections();
    } else if (tok == "TARGET") {
      readTarget();
    } else if (tok == "VERSION") {
      readVersion();
    } else if (tok == "NOCROSSREFS") {
      readNoCrossRefs(/*to=*/false);
    } else if (tok == "NOCROSSREFS_TO") {
      readNoCrossRefs(/*to=*/true);
    } else if (SymbolAssignment *cmd = readAssignment(tok)) {
      ctx.script->sectionCommands.push_back(cmd);
    } else {
      setError("unknown directive: " + tok);
    }
  }
}

void ScriptParser::readDefsym() {
  if (errCount(ctx))
    return;
  SaveAndRestore saved(lexState, State::Expr);
  StringRef name = readName();
  expect("=");
  Expr e = readExpr();
  if (!atEOF())
    setError("EOF expected, but got " + next());
  auto *cmd = make<SymbolAssignment>(
      name, e, 0, getCurrentMB().getBufferIdentifier().str());
  ctx.script->sectionCommands.push_back(cmd);
}

void ScriptParser::readNoCrossRefs(bool to) {
  expect("(");
  NoCrossRefCommand cmd{{}, to};
  while (auto tok = till(")"))
    cmd.outputSections.push_back(unquote(tok));
  if (cmd.outputSections.size() < 2)
    Warn(ctx) << getCurrentLocation()
              << ": ignored with fewer than 2 output sections";
  else
    ctx.script->noCrossRefs.push_back(std::move(cmd));
}

void ScriptParser::addFile(StringRef s) {
  if (curBuf.isUnderSysroot && s.starts_with("/")) {
    SmallString<128> pathData;
    StringRef path = (ctx.arg.sysroot + s).toStringRef(pathData);
    if (sys::fs::exists(path))
      ctx.driver.addFile(ctx.saver.save(path), /*withLOption=*/false);
    else
      setError("cannot find " + s + " inside " + ctx.arg.sysroot);
    return;
  }

  if (s.starts_with("/")) {
    // Case 1: s is an absolute path. Just open it.
    ctx.driver.addFile(s, /*withLOption=*/false);
  } else if (s.starts_with("=")) {
    // Case 2: relative to the sysroot.
    if (ctx.arg.sysroot.empty())
      ctx.driver.addFile(s.substr(1), /*withLOption=*/false);
    else
      ctx.driver.addFile(ctx.saver.save(ctx.arg.sysroot + "/" + s.substr(1)),
                         /*withLOption=*/false);
  } else if (s.starts_with("-l")) {
    // Case 3: search in the list of library paths.
    ctx.driver.addLibrary(s.substr(2));
  } else {
    // Case 4: s is a relative path. Search in the directory of the script file.
    std::string filename = std::string(getCurrentMB().getBufferIdentifier());
    StringRef directory = sys::path::parent_path(filename);
    if (!directory.empty()) {
      SmallString<0> path(directory);
      sys::path::append(path, s);
      if (sys::fs::exists(path)) {
        ctx.driver.addFile(path, /*withLOption=*/false);
        return;
      }
    }
    // Then search in the current working directory.
    if (sys::fs::exists(s)) {
      ctx.driver.addFile(s, /*withLOption=*/false);
    } else {
      // Finally, search in the list of library paths.
      if (std::optional<std::string> path = findFromSearchPaths(ctx, s))
        ctx.driver.addFile(ctx.saver.save(*path), /*withLOption=*/true);
      else
        setError("unable to find " + s);
    }
  }
}

void ScriptParser::readAsNeeded() {
  expect("(");
  bool orig = ctx.arg.asNeeded;
  ctx.arg.asNeeded = true;
  while (auto tok = till(")"))
    addFile(unquote(tok));
  ctx.arg.asNeeded = orig;
}

void ScriptParser::readEntry() {
  // -e <symbol> takes predecence over ENTRY(<symbol>).
  expect("(");
  StringRef name = readName();
  if (ctx.arg.entry.empty())
    ctx.arg.entry = name;
  expect(")");
}

void ScriptParser::readExtern() {
  expect("(");
  while (auto tok = till(")"))
    ctx.arg.undefined.push_back(unquote(tok));
}

void ScriptParser::readGroup() {
  SaveAndRestore saved(ctx.driver.isInGroup, true);
  readInput();
  if (!saved.get())
    ++ctx.driver.nextGroupId;
}

void ScriptParser::readInclude() {
  StringRef name = readName();
  if (!activeFilenames.insert(name).second) {
    setError("there is a cycle in linker script INCLUDEs");
    return;
  }

  if (std::optional<std::string> path = searchScript(ctx, name)) {
    if (std::optional<MemoryBufferRef> mb = readFile(ctx, *path)) {
      buffers.push_back(curBuf);
      curBuf = Buffer(ctx, *mb);
      mbs.push_back(*mb);
    }
    return;
  }
  setError("cannot find linker script " + name);
}

void ScriptParser::readInput() {
  expect("(");
  while (auto tok = till(")")) {
    if (tok == "AS_NEEDED")
      readAsNeeded();
    else
      addFile(unquote(tok));
  }
}

void ScriptParser::readOutput() {
  // -o <file> takes predecence over OUTPUT(<file>).
  expect("(");
  StringRef name = readName();
  if (ctx.arg.outputFile.empty())
    ctx.arg.outputFile = name;
  expect(")");
}

void ScriptParser::readOutputArch() {
  // OUTPUT_ARCH is ignored for now.
  expect("(");
  while (till(")"))
    ;
}

static std::pair<ELFKind, uint16_t> parseBfdName(StringRef s) {
  return StringSwitch<std::pair<ELFKind, uint16_t>>(s)
      .Case("elf32-i386", {ELF32LEKind, EM_386})
      .Case("elf32-avr", {ELF32LEKind, EM_AVR})
      .Case("elf32-iamcu", {ELF32LEKind, EM_IAMCU})
      .Case("elf32-littlearm", {ELF32LEKind, EM_ARM})
      .Case("elf32-bigarm", {ELF32BEKind, EM_ARM})
      .Case("elf32-x86-64", {ELF32LEKind, EM_X86_64})
      .Case("elf64-aarch64", {ELF64LEKind, EM_AARCH64})
      .Case("elf64-littleaarch64", {ELF64LEKind, EM_AARCH64})
      .Case("elf64-bigaarch64", {ELF64BEKind, EM_AARCH64})
      .Case("elf32-powerpc", {ELF32BEKind, EM_PPC})
      .Case("elf32-powerpcle", {ELF32LEKind, EM_PPC})
      .Case("elf64-powerpc", {ELF64BEKind, EM_PPC64})
      .Case("elf64-powerpcle", {ELF64LEKind, EM_PPC64})
      .Case("elf64-x86-64", {ELF64LEKind, EM_X86_64})
      .Cases("elf32-tradbigmips", "elf32-bigmips", {ELF32BEKind, EM_MIPS})
      .Case("elf32-ntradbigmips", {ELF32BEKind, EM_MIPS})
      .Case("elf32-tradlittlemips", {ELF32LEKind, EM_MIPS})
      .Case("elf32-ntradlittlemips", {ELF32LEKind, EM_MIPS})
      .Case("elf64-tradbigmips", {ELF64BEKind, EM_MIPS})
      .Case("elf64-tradlittlemips", {ELF64LEKind, EM_MIPS})
      .Case("elf32-littleriscv", {ELF32LEKind, EM_RISCV})
      .Case("elf64-littleriscv", {ELF64LEKind, EM_RISCV})
      .Case("elf64-sparc", {ELF64BEKind, EM_SPARCV9})
      .Case("elf32-msp430", {ELF32LEKind, EM_MSP430})
      .Case("elf32-loongarch", {ELF32LEKind, EM_LOONGARCH})
      .Case("elf64-loongarch", {ELF64LEKind, EM_LOONGARCH})
      .Case("elf64-s390", {ELF64BEKind, EM_S390})
      .Cases("elf32-hexagon", "elf32-littlehexagon", {ELF32LEKind, EM_HEXAGON})
      .Default({ELFNoneKind, EM_NONE});
}

// Parse OUTPUT_FORMAT(bfdname) or OUTPUT_FORMAT(default, big, little). Choose
// big if -EB is specified, little if -EL is specified, or default if neither is
// specified.
void ScriptParser::readOutputFormat() {
  expect("(");

  StringRef s = readName();
  if (!consume(")")) {
    expect(",");
    StringRef tmp = readName();
    if (ctx.arg.optEB)
      s = tmp;
    expect(",");
    tmp = readName();
    if (ctx.arg.optEL)
      s = tmp;
    consume(")");
  }
  // If more than one OUTPUT_FORMAT is specified, only the first is checked.
  if (!ctx.arg.bfdname.empty())
    return;
  ctx.arg.bfdname = s;

  if (s == "binary") {
    ctx.arg.oFormatBinary = true;
    return;
  }

  if (s.consume_back("-freebsd"))
    ctx.arg.osabi = ELFOSABI_FREEBSD;

  std::tie(ctx.arg.ekind, ctx.arg.emachine) = parseBfdName(s);
  if (ctx.arg.emachine == EM_NONE)
    setError("unknown output format name: " + ctx.arg.bfdname);
  if (s == "elf32-ntradlittlemips" || s == "elf32-ntradbigmips")
    ctx.arg.mipsN32Abi = true;
  if (ctx.arg.emachine == EM_MSP430)
    ctx.arg.osabi = ELFOSABI_STANDALONE;
}

void ScriptParser::readPhdrs() {
  expect("{");
  while (auto tok = till("}")) {
    PhdrsCommand cmd;
    cmd.name = tok;
    cmd.type = readPhdrType();

    while (!errCount(ctx) && !consume(";")) {
      if (consume("FILEHDR"))
        cmd.hasFilehdr = true;
      else if (consume("PHDRS"))
        cmd.hasPhdrs = true;
      else if (consume("AT"))
        cmd.lmaExpr = readParenExpr();
      else if (consume("FLAGS"))
        cmd.flags = readParenExpr()().getValue();
      else
        setError("unexpected header attribute: " + next());
    }

    ctx.script->phdrsCommands.push_back(cmd);
  }
}

void ScriptParser::readRegionAlias() {
  expect("(");
  StringRef alias = readName();
  expect(",");
  StringRef name = readName();
  expect(")");

  if (ctx.script->memoryRegions.count(alias))
    setError("redefinition of memory region '" + alias + "'");
  if (!ctx.script->memoryRegions.count(name))
    setError("memory region '" + name + "' is not defined");
  ctx.script->memoryRegions.insert({alias, ctx.script->memoryRegions[name]});
}

void ScriptParser::readSearchDir() {
  expect("(");
  StringRef name = readName();
  if (!ctx.arg.nostdlib)
    ctx.arg.searchPaths.push_back(name);
  expect(")");
}

// This reads an overlay description. Overlays are used to describe output
// sections that use the same virtual memory range and normally would trigger
// linker's sections sanity check failures.
// https://sourceware.org/binutils/docs/ld/Overlay-Description.html#Overlay-Description
SmallVector<SectionCommand *, 0> ScriptParser::readOverlay() {
  Expr addrExpr;
  if (consume(":")) {
    addrExpr = [s = ctx.script] { return s->getDot(); };
  } else {
    addrExpr = readExpr();
    expect(":");
  }
  // When AT is omitted, LMA should equal VMA. script->getDot() when evaluating
  // lmaExpr will ensure this, even if the start address is specified.
  Expr lmaExpr = consume("AT") ? readParenExpr()
                               : [s = ctx.script] { return s->getDot(); };
  expect("{");

  SmallVector<SectionCommand *, 0> v;
  OutputSection *prev = nullptr;
  while (!errCount(ctx) && !consume("}")) {
    // VA is the same for all sections. The LMAs are consecutive in memory
    // starting from the base load address specified.
    OutputDesc *osd = readOverlaySectionDescription();
    osd->osec.addrExpr = addrExpr;
    if (prev) {
      osd->osec.lmaExpr = [=] { return prev->getLMA() + prev->size; };
    } else {
      osd->osec.lmaExpr = lmaExpr;
      // Use first section address for subsequent sections as initial addrExpr
      // can be DOT. Ensure the first section, even if empty, is not discarded.
      osd->osec.usedInExpression = true;
      addrExpr = [=]() -> ExprValue { return {&osd->osec, false, 0, ""}; };
    }
    v.push_back(osd);
    prev = &osd->osec;
  }

  // According to the specification, at the end of the overlay, the location
  // counter should be equal to the overlay base address plus size of the
  // largest section seen in the overlay.
  // Here we want to create the Dot assignment command to achieve that.
  Expr moveDot = [=] {
    uint64_t max = 0;
    for (SectionCommand *cmd : v)
      max = std::max(max, cast<OutputDesc>(cmd)->osec.size);
    return addrExpr().getValue() + max;
  };
  v.push_back(make<SymbolAssignment>(".", moveDot, 0, getCurrentLocation()));
  return v;
}

SectionClassDesc *ScriptParser::readSectionClassDescription() {
  StringRef name = readSectionClassName();
  SectionClassDesc *desc = make<SectionClassDesc>(name);
  if (!ctx.script->sectionClasses.insert({CachedHashStringRef(name), desc})
           .second)
    setError("section class '" + name + "' already defined");
  expect("{");
  while (auto tok = till("}")) {
    if (tok == "(" || tok == ")") {
      setError("expected filename pattern");
    } else if (peek() == "(") {
      InputSectionDescription *isd = readInputSectionDescription(tok);
      if (!isd->classRef.empty())
        setError("section class '" + name + "' references class '" +
                 isd->classRef + "'");
      desc->sc.commands.push_back(isd);
    }
  }
  return desc;
}

StringRef ScriptParser::readSectionClassName() {
  expect("(");
  StringRef name = unquote(next());
  expect(")");
  return name;
}

void ScriptParser::readOverwriteSections() {
  expect("{");
  while (auto tok = till("}"))
    ctx.script->overwriteSections.push_back(readOutputSectionDescription(tok));
}

void ScriptParser::readSections() {
  expect("{");
  SmallVector<SectionCommand *, 0> v;
  while (auto tok = till("}")) {
    if (tok == "OVERLAY") {
      for (SectionCommand *cmd : readOverlay())
        v.push_back(cmd);
      continue;
    }
    if (tok == "CLASS") {
      v.push_back(readSectionClassDescription());
      continue;
    }
    if (tok == "INCLUDE") {
      readInclude();
      continue;
    }

    if (SectionCommand *cmd = readAssignment(tok))
      v.push_back(cmd);
    else
      v.push_back(readOutputSectionDescription(tok));
  }

  // If DATA_SEGMENT_RELRO_END is absent, for sections after DATA_SEGMENT_ALIGN,
  // the relro fields should be cleared.
  if (!ctx.script->seenRelroEnd)
    for (SectionCommand *cmd : v)
      if (auto *osd = dyn_cast<OutputDesc>(cmd))
        osd->osec.relro = false;

  ctx.script->sectionCommands.insert(ctx.script->sectionCommands.end(),
                                     v.begin(), v.end());

  if (atEOF() || !consume("INSERT")) {
    ctx.script->hasSectionsCommand = true;
    return;
  }

  bool isAfter = false;
  if (consume("AFTER"))
    isAfter = true;
  else if (!consume("BEFORE"))
    setError("expected AFTER/BEFORE, but got '" + next() + "'");
  StringRef where = readName();
  SmallVector<StringRef, 0> names;
  for (SectionCommand *cmd : v)
    if (auto *os = dyn_cast<OutputDesc>(cmd))
      names.push_back(os->osec.name);
  if (!names.empty())
    ctx.script->insertCommands.push_back({std::move(names), isAfter, where});
}

void ScriptParser::readTarget() {
  // TARGET(foo) is an alias for "--format foo". Unlike GNU linkers,
  // we accept only a limited set of BFD names (i.e. "elf" or "binary")
  // for --format. We recognize only /^elf/ and "binary" in the linker
  // script as well.
  expect("(");
  StringRef tok = readName();
  expect(")");

  if (tok.starts_with("elf"))
    ctx.arg.formatBinary = false;
  else if (tok == "binary")
    ctx.arg.formatBinary = true;
  else
    setError("unknown target: " + tok);
}

static int precedence(StringRef op) {
  return StringSwitch<int>(op)
      .Cases("*", "/", "%", 11)
      .Cases("+", "-", 10)
      .Cases("<<", ">>", 9)
      .Cases("<", "<=", ">", ">=", 8)
      .Cases("==", "!=", 7)
      .Case("&", 6)
      .Case("^", 5)
      .Case("|", 4)
      .Case("&&", 3)
      .Case("||", 2)
      .Case("?", 1)
      .Default(-1);
}

StringMatcher ScriptParser::readFilePatterns() {
  StringMatcher Matcher;
  while (auto tok = till(")"))
    Matcher.addPattern(SingleStringMatcher(tok));
  return Matcher;
}

SortSectionPolicy ScriptParser::peekSortKind() {
  return StringSwitch<SortSectionPolicy>(peek())
      .Case("REVERSE", SortSectionPolicy::Reverse)
      .Cases("SORT", "SORT_BY_NAME", SortSectionPolicy::Name)
      .Case("SORT_BY_ALIGNMENT", SortSectionPolicy::Alignment)
      .Case("SORT_BY_INIT_PRIORITY", SortSectionPolicy::Priority)
      .Case("SORT_NONE", SortSectionPolicy::None)
      .Default(SortSectionPolicy::Default);
}

SortSectionPolicy ScriptParser::readSortKind() {
  SortSectionPolicy ret = peekSortKind();
  if (ret != SortSectionPolicy::Default)
    skip();
  return ret;
}

// Reads SECTIONS command contents in the following form:
//
// <contents> ::= <elem>*
// <elem>     ::= <exclude>? <glob-pattern>
// <exclude>  ::= "EXCLUDE_FILE" "(" <glob-pattern>+ ")"
//
// For example,
//
// *(.foo EXCLUDE_FILE (a.o) .bar EXCLUDE_FILE (b.o) .baz)
//
// is parsed as ".foo", ".bar" with "a.o", and ".baz" with "b.o".
// The semantics of that is section .foo in any file, section .bar in
// any file but a.o, and section .baz in any file but b.o.
SmallVector<SectionPattern, 0> ScriptParser::readInputSectionsList() {
  SmallVector<SectionPattern, 0> ret;
  while (!errCount(ctx) && peek() != ")") {
    StringMatcher excludeFilePat;
    if (consume("EXCLUDE_FILE")) {
      expect("(");
      excludeFilePat = readFilePatterns();
    }

    StringMatcher SectionMatcher;
    // Break if the next token is ), EXCLUDE_FILE, or SORT*.
    while (!errCount(ctx) && peekSortKind() == SortSectionPolicy::Default) {
      StringRef s = peek();
      if (s == ")" || s == "EXCLUDE_FILE")
        break;
      // Detect common mistakes when certain non-wildcard meta characters are
      // used without a closing ')'.
      if (!s.empty() && strchr("(){}", s[0])) {
        skip();
        setError("section pattern is expected");
        break;
      }
      SectionMatcher.addPattern(readName());
    }

    if (!SectionMatcher.empty())
      ret.push_back({std::move(excludeFilePat), std::move(SectionMatcher)});
    else if (excludeFilePat.empty())
      break;
    else
      setError("section pattern is expected");
  }
  return ret;
}

// Reads contents of "SECTIONS" directive. That directive contains a
// list of glob patterns for input sections. The grammar is as follows.
//
// <patterns> ::= <section-list>
//              | <sort> "(" <section-list> ")"
//              | <sort> "(" <sort> "(" <section-list> ")" ")"
//
// <sort>     ::= "SORT" | "SORT_BY_NAME" | "SORT_BY_ALIGNMENT"
//              | "SORT_BY_INIT_PRIORITY" | "SORT_NONE"
//
// <section-list> is parsed by readInputSectionsList().
InputSectionDescription *
ScriptParser::readInputSectionRules(StringRef filePattern, uint64_t withFlags,
                                    uint64_t withoutFlags) {
  auto *cmd =
      make<InputSectionDescription>(filePattern, withFlags, withoutFlags);
  expect("(");

  while (peek() != ")" && !atEOF()) {
    SortSectionPolicy outer = readSortKind();
    SortSectionPolicy inner = SortSectionPolicy::Default;
    SmallVector<SectionPattern, 0> v;
    if (outer != SortSectionPolicy::Default) {
      expect("(");
      inner = readSortKind();
      if (inner != SortSectionPolicy::Default) {
        expect("(");
        v = readInputSectionsList();
        expect(")");
      } else {
        v = readInputSectionsList();
      }
      expect(")");
    } else {
      v = readInputSectionsList();
    }

    for (SectionPattern &pat : v) {
      pat.sortInner = inner;
      pat.sortOuter = outer;
    }

    std::move(v.begin(), v.end(), std::back_inserter(cmd->sectionPatterns));
  }
  expect(")");
  return cmd;
}

InputSectionDescription *
ScriptParser::readInputSectionDescription(StringRef tok) {
  // Input section wildcard can be surrounded by KEEP.
  // https://sourceware.org/binutils/docs/ld/Input-Section-Keep.html#Input-Section-Keep
  uint64_t withFlags = 0;
  uint64_t withoutFlags = 0;
  if (tok == "KEEP") {
    expect("(");
    if (consume("INPUT_SECTION_FLAGS"))
      std::tie(withFlags, withoutFlags) = readInputSectionFlags();

    tok = next();
    InputSectionDescription *cmd;
    if (tok == "CLASS")
      cmd = make<InputSectionDescription>(StringRef{}, withFlags, withoutFlags,
                                          readSectionClassName());
    else
      cmd = readInputSectionRules(tok, withFlags, withoutFlags);
    expect(")");
    ctx.script->keptSections.push_back(cmd);
    return cmd;
  }
  if (tok == "INPUT_SECTION_FLAGS") {
    std::tie(withFlags, withoutFlags) = readInputSectionFlags();
    tok = next();
  }
  if (tok == "CLASS")
    return make<InputSectionDescription>(StringRef{}, withFlags, withoutFlags,
                                         readSectionClassName());
  return readInputSectionRules(tok, withFlags, withoutFlags);
}

void ScriptParser::readSort() {
  expect("(");
  expect("CONSTRUCTORS");
  expect(")");
}

Expr ScriptParser::readAssert() {
  expect("(");
  Expr e = readExpr();
  expect(",");
  StringRef msg = readName();
  expect(")");

  return [=, s = ctx.script, &ctx = ctx]() -> ExprValue {
    if (!e().getValue())
      Err(ctx) << msg;
    return s->getDot();
  };
}

#define ECase(X)                                                               \
  { #X, X }
constexpr std::pair<const char *, unsigned> typeMap[] = {
    ECase(SHT_PROGBITS),   ECase(SHT_NOTE),       ECase(SHT_NOBITS),
    ECase(SHT_INIT_ARRAY), ECase(SHT_FINI_ARRAY), ECase(SHT_PREINIT_ARRAY),
};
#undef ECase

// Tries to read the special directive for an output section definition which
// can be one of following: "(NOLOAD)", "(COPY)", "(INFO)", "(OVERLAY)", and
// "(TYPE=<value>)".
bool ScriptParser::readSectionDirective(OutputSection *cmd, StringRef tok) {
  if (tok != "NOLOAD" && tok != "COPY" && tok != "INFO" && tok != "OVERLAY" &&
      tok != "TYPE")
    return false;

  if (consume("NOLOAD")) {
    cmd->type = SHT_NOBITS;
    cmd->typeIsSet = true;
  } else if (consume("TYPE")) {
    expect("=");
    StringRef value = peek();
    auto it = llvm::find_if(typeMap, [=](auto e) { return e.first == value; });
    if (it != std::end(typeMap)) {
      // The value is a recognized literal SHT_*.
      cmd->type = it->second;
      skip();
    } else if (value.starts_with("SHT_")) {
      setError("unknown section type " + value);
    } else {
      // Otherwise, read an expression.
      cmd->type = readExpr()().getValue();
    }
    cmd->typeIsSet = true;
  } else {
    skip(); // This is "COPY", "INFO" or "OVERLAY".
    cmd->nonAlloc = true;
  }
  expect(")");
  return true;
}

// Reads an expression and/or the special directive for an output
// section definition. Directive is one of following: "(NOLOAD)",
// "(COPY)", "(INFO)" or "(OVERLAY)".
//
// An output section name can be followed by an address expression
// and/or directive. This grammar is not LL(1) because "(" can be
// interpreted as either the beginning of some expression or beginning
// of directive.
//
// https://sourceware.org/binutils/docs/ld/Output-Section-Address.html
// https://sourceware.org/binutils/docs/ld/Output-Section-Type.html
void ScriptParser::readSectionAddressType(OutputSection *cmd) {
  if (consume("(")) {
    // Temporarily set lexState to support TYPE=<value> without spaces.
    SaveAndRestore saved(lexState, State::Expr);
    if (readSectionDirective(cmd, peek()))
      return;
    cmd->addrExpr = readExpr();
    expect(")");
  } else {
    cmd->addrExpr = readExpr();
  }

  if (consume("(")) {
    SaveAndRestore saved(lexState, State::Expr);
    StringRef tok = peek();
    if (!readSectionDirective(cmd, tok))
      setError("unknown section directive: " + tok);
  }
}

static Expr checkAlignment(Ctx &ctx, Expr e, std::string &loc) {
  return [=, &ctx] {
    uint64_t alignment = std::max((uint64_t)1, e().getValue());
    if (!isPowerOf2_64(alignment)) {
      ErrAlways(ctx) << loc << ": alignment must be power of 2";
      return (uint64_t)1; // Return a dummy value.
    }
    return alignment;
  };
}

OutputDesc *ScriptParser::readOverlaySectionDescription() {
  OutputDesc *osd =
      ctx.script->createOutputSection(readName(), getCurrentLocation());
  osd->osec.inOverlay = true;
  expect("{");
  while (auto tok = till("}")) {
    uint64_t withFlags = 0;
    uint64_t withoutFlags = 0;
    if (tok == "INPUT_SECTION_FLAGS") {
      std::tie(withFlags, withoutFlags) = readInputSectionFlags();
      tok = till("");
    }
    if (tok == "CLASS")
      osd->osec.commands.push_back(make<InputSectionDescription>(
          StringRef{}, withFlags, withoutFlags, readSectionClassName()));
    else
      osd->osec.commands.push_back(
          readInputSectionRules(tok, withFlags, withoutFlags));
  }
  osd->osec.phdrs = readOutputSectionPhdrs();
  return osd;
}

OutputDesc *ScriptParser::readOutputSectionDescription(StringRef outSec) {
  OutputDesc *cmd =
      ctx.script->createOutputSection(unquote(outSec), getCurrentLocation());
  OutputSection *osec = &cmd->osec;
  // Maybe relro. Will reset to false if DATA_SEGMENT_RELRO_END is absent.
  osec->relro = ctx.script->seenDataAlign && !ctx.script->seenRelroEnd;

  size_t symbolsReferenced = ctx.script->referencedSymbols.size();

  if (peek() != ":")
    readSectionAddressType(osec);
  expect(":");

  std::string location = getCurrentLocation();
  if (consume("AT"))
    osec->lmaExpr = readParenExpr();
  if (consume("ALIGN"))
    osec->alignExpr = checkAlignment(ctx, readParenExpr(), location);
  if (consume("SUBALIGN"))
    osec->subalignExpr = checkAlignment(ctx, readParenExpr(), location);

  // Parse constraints.
  if (consume("ONLY_IF_RO"))
    osec->constraint = ConstraintKind::ReadOnly;
  if (consume("ONLY_IF_RW"))
    osec->constraint = ConstraintKind::ReadWrite;
  expect("{");

  while (auto tok = till("}")) {
    if (tok == ";") {
      // Empty commands are allowed. Do nothing here.
    } else if (SymbolAssignment *assign = readAssignment(tok)) {
      osec->commands.push_back(assign);
    } else if (ByteCommand *data = readByteCommand(tok)) {
      osec->commands.push_back(data);
    } else if (tok == "CONSTRUCTORS") {
      // CONSTRUCTORS is a keyword to make the linker recognize C++ ctors/dtors
      // by name. This is for very old file formats such as ECOFF/XCOFF.
      // For ELF, we should ignore.
    } else if (tok == "FILL") {
      // We handle the FILL command as an alias for =fillexp section attribute,
      // which is different from what GNU linkers do.
      // https://sourceware.org/binutils/docs/ld/Output-Section-Data.html
      if (peek() != "(")
        setError("( expected, but got " + peek());
      osec->filler = readFill();
    } else if (tok == "SORT") {
      readSort();
    } else if (tok == "INCLUDE") {
      readInclude();
    } else if (tok == "(" || tok == ")") {
      setError("expected filename pattern");
    } else if (peek() == "(") {
      osec->commands.push_back(readInputSectionDescription(tok));
    } else {
      // We have a file name and no input sections description. It is not a
      // commonly used syntax, but still acceptable. In that case, all sections
      // from the file will be included.
      // FIXME: GNU ld permits INPUT_SECTION_FLAGS to be used here. We do not
      // handle this case here as it will already have been matched by the
      // case above.
      auto *isd = make<InputSectionDescription>(tok);
      isd->sectionPatterns.push_back({{}, StringMatcher("*")});
      osec->commands.push_back(isd);
    }
  }

  if (consume(">"))
    osec->memoryRegionName = std::string(readName());

  if (consume("AT")) {
    expect(">");
    osec->lmaRegionName = std::string(readName());
  }

  if (osec->lmaExpr && !osec->lmaRegionName.empty())
    ErrAlways(ctx) << "section can't have both LMA and a load region";

  osec->phdrs = readOutputSectionPhdrs();

  if (peek() == "=" || peek().starts_with("=")) {
    lexState = State::Expr;
    consume("=");
    osec->filler = readFill();
    lexState = State::Script;
  }

  // Consume optional comma following output section command.
  consume(",");

  if (ctx.script->referencedSymbols.size() > symbolsReferenced)
    osec->expressionsUseSymbols = true;
  return cmd;
}

// Reads a `=<fillexp>` expression and returns its value as a big-endian number.
// https://sourceware.org/binutils/docs/ld/Output-Section-Fill.html
// We do not support using symbols in such expressions.
//
// When reading a hexstring, ld.bfd handles it as a blob of arbitrary
// size, while ld.gold always handles it as a 32-bit big-endian number.
// We are compatible with ld.gold because it's easier to implement.
// Also, we require that expressions with operators must be wrapped into
// round brackets. We did it to resolve the ambiguity when parsing scripts like:
// SECTIONS { .foo : { ... } =120+3 /DISCARD/ : { ... } }
std::array<uint8_t, 4> ScriptParser::readFill() {
  uint64_t value = readPrimary()().val;
  if (value > UINT32_MAX)
    setError("filler expression result does not fit 32-bit: 0x" +
             Twine::utohexstr(value));

  std::array<uint8_t, 4> buf;
  write32be(buf.data(), (uint32_t)value);
  return buf;
}

SymbolAssignment *ScriptParser::readProvideHidden(bool provide, bool hidden) {
  expect("(");
  StringRef name = readName(), eq = peek();
  if (eq != "=") {
    setError("= expected, but got " + next());
    while (till(")"))
      ;
    return nullptr;
  }
  llvm::SaveAndRestore saveActiveProvideSym(activeProvideSym);
  if (provide)
    activeProvideSym = name;
  SymbolAssignment *cmd = readSymbolAssignment(name);
  cmd->provide = provide;
  cmd->hidden = hidden;
  expect(")");
  return cmd;
}

// Replace whitespace sequence (including \n) with one single space. The output
// is used by -Map.
static void squeezeSpaces(std::string &str) {
  char prev = '\0';
  auto it = str.begin();
  for (char c : str)
    if (!isSpace(c) || (c = ' ') != prev)
      *it++ = prev = c;
  str.erase(it, str.end());
}

SymbolAssignment *ScriptParser::readAssignment(StringRef tok) {
  // Assert expression returns Dot, so this is equal to ".=."
  if (tok == "ASSERT")
    return make<SymbolAssignment>(".", readAssert(), 0, getCurrentLocation());

  const char *oldS = prevTok.data();
  SymbolAssignment *cmd = nullptr;
  bool savedSeenRelroEnd = ctx.script->seenRelroEnd;
  const StringRef op = peek();
  {
    SaveAndRestore saved(lexState, State::Expr);
    if (op.starts_with("=")) {
      // Support = followed by an expression without whitespace.
      cmd = readSymbolAssignment(unquote(tok));
    } else if ((op.size() == 2 && op[1] == '=' && strchr("+-*/&^|", op[0])) ||
               op == "<<=" || op == ">>=") {
      cmd = readSymbolAssignment(unquote(tok));
    } else if (tok == "PROVIDE") {
      cmd = readProvideHidden(true, false);
    } else if (tok == "HIDDEN") {
      cmd = readProvideHidden(false, true);
    } else if (tok == "PROVIDE_HIDDEN") {
      cmd = readProvideHidden(true, true);
    }
  }

  if (cmd) {
    cmd->dataSegmentRelroEnd = !savedSeenRelroEnd && ctx.script->seenRelroEnd;
    cmd->commandString = StringRef(oldS, curTok.data() - oldS).str();
    squeezeSpaces(cmd->commandString);
    expect(";");
  }
  return cmd;
}

StringRef ScriptParser::readName() { return unquote(next()); }

SymbolAssignment *ScriptParser::readSymbolAssignment(StringRef name) {
  StringRef op = next();
  assert(op == "=" || op == "*=" || op == "/=" || op == "+=" || op == "-=" ||
         op == "&=" || op == "^=" || op == "|=" || op == "<<=" || op == ">>=");
  // Note: GNU ld does not support %=.
  Expr e = readExpr();
  if (op != "=") {
    std::string loc = getCurrentLocation();
    e = [=, s = ctx.script, c = op[0], &ctx = ctx]() -> ExprValue {
      ExprValue lhs = s->getSymbolValue(name, loc);
      switch (c) {
      case '*':
        return lhs.getValue() * e().getValue();
      case '/':
        if (uint64_t rv = e().getValue())
          return lhs.getValue() / rv;
        ErrAlways(ctx) << loc << ": division by zero";
        return 0;
      case '+':
        return add(*s, lhs, e());
      case '-':
        return sub(lhs, e());
      case '<':
        return lhs.getValue() << e().getValue() % 64;
      case '>':
        return lhs.getValue() >> e().getValue() % 64;
      case '&':
        return lhs.getValue() & e().getValue();
      case '^':
        return lhs.getValue() ^ e().getValue();
      case '|':
        return lhs.getValue() | e().getValue();
      default:
        llvm_unreachable("");
      }
    };
  }
  return make<SymbolAssignment>(name, e, ctx.scriptSymOrderCounter++,
                                getCurrentLocation());
}

// This is an operator-precedence parser to parse a linker
// script expression.
Expr ScriptParser::readExpr() {
  // Our lexer is context-aware. Set the in-expression bit so that
  // they apply different tokenization rules.
  SaveAndRestore saved(lexState, State::Expr);
  Expr e = readExpr1(readPrimary(), 0);
  return e;
}

Expr ScriptParser::combine(StringRef op, Expr l, Expr r) {
  if (op == "+")
    return [=, s = ctx.script] { return add(*s, l(), r()); };
  if (op == "-")
    return [=] { return sub(l(), r()); };
  if (op == "*")
    return [=] { return l().getValue() * r().getValue(); };
  if (op == "/") {
    std::string loc = getCurrentLocation();
    return [=, &ctx = ctx]() -> uint64_t {
      if (uint64_t rv = r().getValue())
        return l().getValue() / rv;
      ErrAlways(ctx) << loc << ": division by zero";
      return 0;
    };
  }
  if (op == "%") {
    std::string loc = getCurrentLocation();
    return [=, &ctx = ctx]() -> uint64_t {
      if (uint64_t rv = r().getValue())
        return l().getValue() % rv;
      ErrAlways(ctx) << loc << ": modulo by zero";
      return 0;
    };
  }
  if (op == "<<")
    return [=] { return l().getValue() << r().getValue() % 64; };
  if (op == ">>")
    return [=] { return l().getValue() >> r().getValue() % 64; };
  if (op == "<")
    return [=] { return l().getValue() < r().getValue(); };
  if (op == ">")
    return [=] { return l().getValue() > r().getValue(); };
  if (op == ">=")
    return [=] { return l().getValue() >= r().getValue(); };
  if (op == "<=")
    return [=] { return l().getValue() <= r().getValue(); };
  if (op == "==")
    return [=] { return l().getValue() == r().getValue(); };
  if (op == "!=")
    return [=] { return l().getValue() != r().getValue(); };
  if (op == "||")
    return [=] { return l().getValue() || r().getValue(); };
  if (op == "&&")
    return [=] { return l().getValue() && r().getValue(); };
  if (op == "&")
    return [=, s = ctx.script] { return bitAnd(*s, l(), r()); };
  if (op == "^")
    return [=, s = ctx.script] { return bitXor(*s, l(), r()); };
  if (op == "|")
    return [=, s = ctx.script] { return bitOr(*s, l(), r()); };
  llvm_unreachable("invalid operator");
}

// This is a part of the operator-precedence parser. This function
// assumes that the remaining token stream starts with an operator.
Expr ScriptParser::readExpr1(Expr lhs, int minPrec) {
  while (!atEOF() && !errCount(ctx)) {
    // Read an operator and an expression.
    StringRef op1 = peek();
    if (precedence(op1) < minPrec)
      break;
    skip();
    if (op1 == "?")
      return readTernary(lhs);
    Expr rhs = readPrimary();

    // Evaluate the remaining part of the expression first if the
    // next operator has greater precedence than the previous one.
    // For example, if we have read "+" and "3", and if the next
    // operator is "*", then we'll evaluate 3 * ... part first.
    while (!atEOF()) {
      StringRef op2 = peek();
      if (precedence(op2) <= precedence(op1))
        break;
      rhs = readExpr1(rhs, precedence(op2));
    }

    lhs = combine(op1, lhs, rhs);
  }
  return lhs;
}

Expr ScriptParser::getPageSize() {
  std::string location = getCurrentLocation();
  return [=, &ctx = this->ctx]() -> uint64_t {
    if (ctx.target)
      return ctx.arg.commonPageSize;
    ErrAlways(ctx) << location << ": unable to calculate page size";
    return 4096; // Return a dummy value.
  };
}

Expr ScriptParser::readConstant() {
  StringRef s = readParenName();
  if (s == "COMMONPAGESIZE")
    return getPageSize();
  if (s == "MAXPAGESIZE")
    return [&ctx = this->ctx] { return ctx.arg.maxPageSize; };
  setError("unknown constant: " + s);
  return [] { return 0; };
}

// Parses Tok as an integer. It recognizes hexadecimal (prefixed with
// "0x" or suffixed with "H") and decimal numbers. Decimal numbers may
// have "K" (Ki) or "M" (Mi) suffixes.
static std::optional<uint64_t> parseInt(StringRef tok) {
  // Hexadecimal
  uint64_t val;
  if (tok.starts_with_insensitive("0x")) {
    if (!to_integer(tok.substr(2), val, 16))
      return std::nullopt;
    return val;
  }
  if (tok.ends_with_insensitive("H")) {
    if (!to_integer(tok.drop_back(), val, 16))
      return std::nullopt;
    return val;
  }

  // Decimal
  if (tok.ends_with_insensitive("K")) {
    if (!to_integer(tok.drop_back(), val, 10))
      return std::nullopt;
    return val * 1024;
  }
  if (tok.ends_with_insensitive("M")) {
    if (!to_integer(tok.drop_back(), val, 10))
      return std::nullopt;
    return val * 1024 * 1024;
  }
  if (!to_integer(tok, val, 10))
    return std::nullopt;
  return val;
}

ByteCommand *ScriptParser::readByteCommand(StringRef tok) {
  int size = StringSwitch<int>(tok)
                 .Case("BYTE", 1)
                 .Case("SHORT", 2)
                 .Case("LONG", 4)
                 .Case("QUAD", 8)
                 .Default(-1);
  if (size == -1)
    return nullptr;

  const char *oldS = prevTok.data();
  Expr e = readParenExpr();
  std::string commandString = StringRef(oldS, curBuf.s.data() - oldS).str();
  squeezeSpaces(commandString);
  return make<ByteCommand>(e, size, std::move(commandString));
}

static std::optional<uint64_t> parseFlag(StringRef tok) {
  if (std::optional<uint64_t> asInt = parseInt(tok))
    return asInt;
#define CASE_ENT(enum) #enum, ELF::enum
  return StringSwitch<std::optional<uint64_t>>(tok)
      .Case(CASE_ENT(SHF_WRITE))
      .Case(CASE_ENT(SHF_ALLOC))
      .Case(CASE_ENT(SHF_EXECINSTR))
      .Case(CASE_ENT(SHF_MERGE))
      .Case(CASE_ENT(SHF_STRINGS))
      .Case(CASE_ENT(SHF_INFO_LINK))
      .Case(CASE_ENT(SHF_LINK_ORDER))
      .Case(CASE_ENT(SHF_OS_NONCONFORMING))
      .Case(CASE_ENT(SHF_GROUP))
      .Case(CASE_ENT(SHF_TLS))
      .Case(CASE_ENT(SHF_COMPRESSED))
      .Case(CASE_ENT(SHF_EXCLUDE))
      .Case(CASE_ENT(SHF_ARM_PURECODE))
      .Default(std::nullopt);
#undef CASE_ENT
}

// Reads the '(' <flags> ')' list of section flags in
// INPUT_SECTION_FLAGS '(' <flags> ')' in the
// following form:
// <flags> ::= <flag>
//           | <flags> & flag
// <flag>  ::= Recognized Flag Name, or Integer value of flag.
// If the first character of <flag> is a ! then this means without flag,
// otherwise with flag.
// Example: SHF_EXECINSTR & !SHF_WRITE means with flag SHF_EXECINSTR and
// without flag SHF_WRITE.
std::pair<uint64_t, uint64_t> ScriptParser::readInputSectionFlags() {
  uint64_t withFlags = 0;
  uint64_t withoutFlags = 0;
  expect("(");
  while (!errCount(ctx)) {
    StringRef tok = readName();
    bool without = tok.consume_front("!");
    if (std::optional<uint64_t> flag = parseFlag(tok)) {
      if (without)
        withoutFlags |= *flag;
      else
        withFlags |= *flag;
    } else {
      setError("unrecognised flag: " + tok);
    }
    if (consume(")"))
      break;
    if (!consume("&")) {
      next();
      setError("expected & or )");
    }
  }
  return std::make_pair(withFlags, withoutFlags);
}

StringRef ScriptParser::readParenName() {
  expect("(");
  auto saved = std::exchange(lexState, State::Script);
  StringRef name = readName();
  lexState = saved;
  expect(")");
  return name;
}

static void checkIfExists(LinkerScript &script, const OutputSection &osec,
                          StringRef location) {
  if (osec.location.empty() && script.errorOnMissingSection)
    script.recordError(location + ": undefined section " + osec.name);
}

static bool isValidSymbolName(StringRef s) {
  auto valid = [](char c) {
    return isAlnum(c) || c == '$' || c == '.' || c == '_';
  };
  return !s.empty() && !isDigit(s[0]) && llvm::all_of(s, valid);
}

Expr ScriptParser::readPrimary() {
  if (peek() == "(")
    return readParenExpr();

  if (consume("~")) {
    Expr e = readPrimary();
    return [=] { return ~e().getValue(); };
  }
  if (consume("!")) {
    Expr e = readPrimary();
    return [=] { return !e().getValue(); };
  }
  if (consume("-")) {
    Expr e = readPrimary();
    return [=] { return -e().getValue(); };
  }
  if (consume("+"))
    return readPrimary();

  StringRef tok = next();
  std::string location = getCurrentLocation();

  // Built-in functions are parsed here.
  // https://sourceware.org/binutils/docs/ld/Builtin-Functions.html.
  if (tok == "ABSOLUTE") {
    Expr inner = readParenExpr();
    return [=] {
      ExprValue i = inner();
      i.forceAbsolute = true;
      return i;
    };
  }
  if (tok == "ADDR") {
    StringRef name = readParenName();
    OutputSection *osec = &ctx.script->getOrCreateOutputSection(name)->osec;
    osec->usedInExpression = true;
    return [=, s = ctx.script]() -> ExprValue {
      checkIfExists(*s, *osec, location);
      return {osec, false, 0, location};
    };
  }
  if (tok == "ALIGN") {
    expect("(");
    Expr e = readExpr();
    if (consume(")")) {
      e = checkAlignment(ctx, e, location);
      return [=, s = ctx.script] {
        return alignToPowerOf2(s->getDot(), e().getValue());
      };
    }
    expect(",");
    Expr e2 = checkAlignment(ctx, readExpr(), location);
    expect(")");
    return [=] {
      ExprValue v = e();
      v.alignment = e2().getValue();
      return v;
    };
  }
  if (tok == "ALIGNOF") {
    StringRef name = readParenName();
    OutputSection *osec = &ctx.script->getOrCreateOutputSection(name)->osec;
    return [=, s = ctx.script] {
      checkIfExists(*s, *osec, location);
      return osec->addralign;
    };
  }
  if (tok == "ASSERT")
    return readAssert();
  if (tok == "CONSTANT")
    return readConstant();
  if (tok == "DATA_SEGMENT_ALIGN") {
    expect("(");
    Expr e = readExpr();
    expect(",");
    readExpr();
    expect(")");
    ctx.script->seenDataAlign = true;
    return [=, s = ctx.script] {
      uint64_t align = std::max(uint64_t(1), e().getValue());
      return (s->getDot() + align - 1) & -align;
    };
  }
  if (tok == "DATA_SEGMENT_END") {
    expect("(");
    expect(".");
    expect(")");
    return [s = ctx.script] { return s->getDot(); };
  }
  if (tok == "DATA_SEGMENT_RELRO_END") {
    // GNU linkers implements more complicated logic to handle
    // DATA_SEGMENT_RELRO_END. We instead ignore the arguments and
    // just align to the next page boundary for simplicity.
    expect("(");
    readExpr();
    expect(",");
    readExpr();
    expect(")");
    ctx.script->seenRelroEnd = true;
    return [&ctx = this->ctx] {
      return alignToPowerOf2(ctx.script->getDot(), ctx.arg.maxPageSize);
    };
  }
  if (tok == "DEFINED") {
    StringRef name = readParenName();
    // Return 1 if s is defined. If the definition is only found in a linker
    // script, it must happen before this DEFINED.
    auto order = ctx.scriptSymOrderCounter++;
    return [=, &ctx = this->ctx] {
      Symbol *s = ctx.symtab->find(name);
      return s && s->isDefined() && ctx.scriptSymOrder.lookup(s) < order ? 1
                                                                         : 0;
    };
  }
  if (tok == "LENGTH") {
    StringRef name = readParenName();
    if (ctx.script->memoryRegions.count(name) == 0) {
      setError("memory region not defined: " + name);
      return [] { return 0; };
    }
    return ctx.script->memoryRegions[name]->length;
  }
  if (tok == "LOADADDR") {
    StringRef name = readParenName();
    OutputSection *osec = &ctx.script->getOrCreateOutputSection(name)->osec;
    osec->usedInExpression = true;
    return [=, s = ctx.script] {
      checkIfExists(*s, *osec, location);
      return osec->getLMA();
    };
  }
  if (tok == "LOG2CEIL") {
    expect("(");
    Expr a = readExpr();
    expect(")");
    return [=] {
      // LOG2CEIL(0) is defined to be 0.
      return llvm::Log2_64_Ceil(std::max(a().getValue(), UINT64_C(1)));
    };
  }
  if (tok == "MAX" || tok == "MIN") {
    expect("(");
    Expr a = readExpr();
    expect(",");
    Expr b = readExpr();
    expect(")");
    if (tok == "MIN")
      return [=] { return std::min(a().getValue(), b().getValue()); };
    return [=] { return std::max(a().getValue(), b().getValue()); };
  }
  if (tok == "ORIGIN") {
    StringRef name = readParenName();
    if (ctx.script->memoryRegions.count(name) == 0) {
      setError("memory region not defined: " + name);
      return [] { return 0; };
    }
    return ctx.script->memoryRegions[name]->origin;
  }
  if (tok == "SEGMENT_START") {
    expect("(");
    skip();
    expect(",");
    Expr e = readExpr();
    expect(")");
    return [=] { return e(); };
  }
  if (tok == "SIZEOF") {
    StringRef name = readParenName();
    OutputSection *cmd = &ctx.script->getOrCreateOutputSection(name)->osec;
    // Linker script does not create an output section if its content is empty.
    // We want to allow SIZEOF(.foo) where .foo is a section which happened to
    // be empty.
    return [=] { return cmd->size; };
  }
  if (tok == "SIZEOF_HEADERS")
    return [=, &ctx = ctx] { return elf::getHeaderSize(ctx); };

  // Tok is the dot.
  if (tok == ".")
    return [=, s = ctx.script] { return s->getSymbolValue(tok, location); };

  // Tok is a literal number.
  if (std::optional<uint64_t> val = parseInt(tok))
    return [=] { return *val; };

  // Tok is a symbol name.
  if (tok.starts_with("\""))
    tok = unquote(tok);
  else if (!isValidSymbolName(tok))
    setError("malformed number: " + tok);
  if (activeProvideSym)
    ctx.script->provideMap[*activeProvideSym].push_back(tok);
  else
    ctx.script->referencedSymbols.push_back(tok);
  return [=, s = ctx.script] { return s->getSymbolValue(tok, location); };
}

Expr ScriptParser::readTernary(Expr cond) {
  Expr l = readExpr();
  expect(":");
  Expr r = readExpr();
  return [=] { return cond().getValue() ? l() : r(); };
}

Expr ScriptParser::readParenExpr() {
  expect("(");
  Expr e = readExpr();
  expect(")");
  return e;
}

SmallVector<StringRef, 0> ScriptParser::readOutputSectionPhdrs() {
  SmallVector<StringRef, 0> phdrs;
  while (!errCount(ctx) && peek().starts_with(":")) {
    StringRef tok = next();
    phdrs.push_back((tok.size() == 1) ? readName() : tok.substr(1));
  }
  return phdrs;
}

// Read a program header type name. The next token must be a
// name of a program header type or a constant (e.g. "0x3").
unsigned ScriptParser::readPhdrType() {
  StringRef tok = next();
  if (std::optional<uint64_t> val = parseInt(tok))
    return *val;

  unsigned ret = StringSwitch<unsigned>(tok)
                     .Case("PT_NULL", PT_NULL)
                     .Case("PT_LOAD", PT_LOAD)
                     .Case("PT_DYNAMIC", PT_DYNAMIC)
                     .Case("PT_INTERP", PT_INTERP)
                     .Case("PT_NOTE", PT_NOTE)
                     .Case("PT_SHLIB", PT_SHLIB)
                     .Case("PT_PHDR", PT_PHDR)
                     .Case("PT_TLS", PT_TLS)
                     .Case("PT_GNU_EH_FRAME", PT_GNU_EH_FRAME)
                     .Case("PT_GNU_STACK", PT_GNU_STACK)
                     .Case("PT_GNU_RELRO", PT_GNU_RELRO)
                     .Case("PT_OPENBSD_MUTABLE", PT_OPENBSD_MUTABLE)
                     .Case("PT_OPENBSD_RANDOMIZE", PT_OPENBSD_RANDOMIZE)
                     .Case("PT_OPENBSD_SYSCALLS", PT_OPENBSD_SYSCALLS)
                     .Case("PT_OPENBSD_WXNEEDED", PT_OPENBSD_WXNEEDED)
                     .Case("PT_OPENBSD_BOOTDATA", PT_OPENBSD_BOOTDATA)
                     .Default(-1);

  if (ret == (unsigned)-1) {
    setError("invalid program header type: " + tok);
    return PT_NULL;
  }
  return ret;
}

// Reads an anonymous version declaration.
void ScriptParser::readAnonymousDeclaration() {
  SmallVector<SymbolVersion, 0> locals;
  SmallVector<SymbolVersion, 0> globals;
  std::tie(locals, globals) = readSymbols();
  for (const SymbolVersion &pat : locals)
    ctx.arg.versionDefinitions[VER_NDX_LOCAL].localPatterns.push_back(pat);
  for (const SymbolVersion &pat : globals)
    ctx.arg.versionDefinitions[VER_NDX_GLOBAL].nonLocalPatterns.push_back(pat);

  expect(";");
}

// Reads a non-anonymous version definition,
// e.g. "VerStr { global: foo; bar; local: *; };".
void ScriptParser::readVersionDeclaration(StringRef verStr) {
  // Read a symbol list.
  SmallVector<SymbolVersion, 0> locals;
  SmallVector<SymbolVersion, 0> globals;
  std::tie(locals, globals) = readSymbols();

  // Create a new version definition and add that to the global symbols.
  VersionDefinition ver;
  ver.name = verStr;
  ver.nonLocalPatterns = std::move(globals);
  ver.localPatterns = std::move(locals);
  ver.id = ctx.arg.versionDefinitions.size();
  ctx.arg.versionDefinitions.push_back(ver);

  // Each version may have a parent version. For example, "Ver2"
  // defined as "Ver2 { global: foo; local: *; } Ver1;" has "Ver1"
  // as a parent. This version hierarchy is, probably against your
  // instinct, purely for hint; the runtime doesn't care about it
  // at all. In LLD, we simply ignore it.
  if (next() != ";")
    expect(";");
}

bool elf::hasWildcard(StringRef s) {
  return s.find_first_of("?*[") != StringRef::npos;
}

// Reads a list of symbols, e.g. "{ global: foo; bar; local: *; };".
std::pair<SmallVector<SymbolVersion, 0>, SmallVector<SymbolVersion, 0>>
ScriptParser::readSymbols() {
  SmallVector<SymbolVersion, 0> locals;
  SmallVector<SymbolVersion, 0> globals;
  SmallVector<SymbolVersion, 0> *v = &globals;

  while (auto tok = till("}")) {
    if (tok == "extern") {
      SmallVector<SymbolVersion, 0> ext = readVersionExtern();
      v->insert(v->end(), ext.begin(), ext.end());
    } else {
      if (tok == "local:" || (tok == "local" && consume(":"))) {
        v = &locals;
        continue;
      }
      if (tok == "global:" || (tok == "global" && consume(":"))) {
        v = &globals;
        continue;
      }
      v->push_back({unquote(tok), false, hasWildcard(tok)});
    }
    expect(";");
  }
  return {locals, globals};
}

// Reads an "extern C++" directive, e.g.,
// "extern "C++" { ns::*; "f(int, double)"; };"
//
// The last semicolon is optional. E.g. this is OK:
// "extern "C++" { ns::*; "f(int, double)" };"
SmallVector<SymbolVersion, 0> ScriptParser::readVersionExtern() {
  StringRef tok = next();
  bool isCXX = tok == "\"C++\"";
  if (!isCXX && tok != "\"C\"")
    setError("Unknown language");
  expect("{");

  SmallVector<SymbolVersion, 0> ret;
  while (auto tok = till("}")) {
    ret.push_back(
        {unquote(tok), isCXX, !tok.str.starts_with("\"") && hasWildcard(tok)});
    if (consume("}"))
      return ret;
    expect(";");
  }
  return ret;
}

Expr ScriptParser::readMemoryAssignment(StringRef s1, StringRef s2,
                                        StringRef s3) {
  if (!consume(s1) && !consume(s2) && !consume(s3)) {
    setError("expected one of: " + s1 + ", " + s2 + ", or " + s3);
    return [] { return 0; };
  }
  expect("=");
  return readExpr();
}

// Parse the MEMORY command as specified in:
// https://sourceware.org/binutils/docs/ld/MEMORY.html
//
// MEMORY { name [(attr)] : ORIGIN = origin, LENGTH = len ... }
void ScriptParser::readMemory() {
  expect("{");
  while (auto tok = till("}")) {
    if (tok == "INCLUDE") {
      readInclude();
      continue;
    }

    uint32_t flags = 0;
    uint32_t invFlags = 0;
    uint32_t negFlags = 0;
    uint32_t negInvFlags = 0;
    if (consume("(")) {
      readMemoryAttributes(flags, invFlags, negFlags, negInvFlags);
      expect(")");
    }
    expect(":");

    Expr origin = readMemoryAssignment("ORIGIN", "org", "o");
    expect(",");
    Expr length = readMemoryAssignment("LENGTH", "len", "l");

    // Add the memory region to the region map.
    MemoryRegion *mr = make<MemoryRegion>(tok, origin, length, flags, invFlags,
                                          negFlags, negInvFlags);
    if (!ctx.script->memoryRegions.insert({tok, mr}).second)
      setError("region '" + tok + "' already defined");
  }
}

// This function parses the attributes used to match against section
// flags when placing output sections in a memory region. These flags
// are only used when an explicit memory region name is not used.
void ScriptParser::readMemoryAttributes(uint32_t &flags, uint32_t &invFlags,
                                        uint32_t &negFlags,
                                        uint32_t &negInvFlags) {
  bool invert = false;

  for (char c : next().lower()) {
    if (c == '!') {
      invert = !invert;
      std::swap(flags, negFlags);
      std::swap(invFlags, negInvFlags);
      continue;
    }
    if (c == 'w')
      flags |= SHF_WRITE;
    else if (c == 'x')
      flags |= SHF_EXECINSTR;
    else if (c == 'a')
      flags |= SHF_ALLOC;
    else if (c == 'r')
      invFlags |= SHF_WRITE;
    else
      setError("invalid memory region attribute");
  }

  if (invert) {
    std::swap(flags, negFlags);
    std::swap(invFlags, negInvFlags);
  }
}

void elf::readLinkerScript(Ctx &ctx, MemoryBufferRef mb) {
  llvm::TimeTraceScope timeScope("Read linker script",
                                 mb.getBufferIdentifier());
  ScriptParser(ctx, mb).readLinkerScript();
}

void elf::readVersionScript(Ctx &ctx, MemoryBufferRef mb) {
  llvm::TimeTraceScope timeScope("Read version script",
                                 mb.getBufferIdentifier());
  ScriptParser(ctx, mb).readVersionScript();
}

void elf::readDynamicList(Ctx &ctx, MemoryBufferRef mb) {
  llvm::TimeTraceScope timeScope("Read dynamic list", mb.getBufferIdentifier());
  ScriptParser(ctx, mb).readDynamicList();
}

void elf::readDefsym(Ctx &ctx, MemoryBufferRef mb) {
  ScriptParser(ctx, mb).readDefsym();
}
