//===- Symbols.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "COFFLinkerContext.h"
#include "InputFiles.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Strings.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

using namespace lld::coff;

namespace lld {

static_assert(sizeof(SymbolUnion) <= 48,
              "symbols should be optimized for memory usage");

// Returns a symbol name for an error message.
std::string maybeDemangleSymbol(const COFFLinkerContext &ctx,
                                StringRef symName) {
  if (ctx.config.demangle) {
    std::string prefix;
    StringRef prefixless = symName;
    if (prefixless.consume_front("__imp_"))
      prefix = "__declspec(dllimport) ";
    StringRef demangleInput = prefixless;
    if (ctx.config.machine == I386)
      demangleInput.consume_front("_");
    std::string demangled = demangle(demangleInput);
    if (demangled != demangleInput)
      return prefix + demangled;
    return (prefix + prefixless).str();
  }
  return std::string(symName);
}
std::string toString(const COFFLinkerContext &ctx, coff::Symbol &b) {
  return maybeDemangleSymbol(ctx, b.getName());
}
std::string toCOFFString(const COFFLinkerContext &ctx,
                         const Archive::Symbol &b) {
  return maybeDemangleSymbol(ctx, b.getName());
}

const COFFSyncStream &
coff::operator<<(const COFFSyncStream &s,
                 const llvm::object::Archive::Symbol *sym) {
  s << maybeDemangleSymbol(s.ctx, sym->getName());
  return s;
}

const COFFSyncStream &coff::operator<<(const COFFSyncStream &s, Symbol *sym) {
  return s << maybeDemangleSymbol(s.ctx, sym->getName());
}

namespace coff {

void Symbol::computeName() {
  assert(nameData == nullptr &&
         "should only compute the name once for DefinedCOFF symbols");
  auto *d = cast<DefinedCOFF>(this);
  StringRef nameStr =
      check(cast<ObjFile>(d->file)->getCOFFObj()->getSymbolName(d->sym));
  nameData = nameStr.data();
  nameSize = nameStr.size();
  assert(nameSize == nameStr.size() && "name length truncated");
}

InputFile *Symbol::getFile() {
  if (auto *sym = dyn_cast<DefinedCOFF>(this))
    return sym->file;
  if (auto *sym = dyn_cast<LazyArchive>(this))
    return sym->file;
  if (auto *sym = dyn_cast<LazyObject>(this))
    return sym->file;
  if (auto *sym = dyn_cast<LazyDLLSymbol>(this))
    return sym->file;
  return nullptr;
}

bool Symbol::isLive() const {
  if (auto *r = dyn_cast<DefinedRegular>(this))
    return r->getChunk()->live;
  if (auto *imp = dyn_cast<DefinedImportData>(this))
    return imp->file->live;
  if (auto *imp = dyn_cast<DefinedImportThunk>(this))
    return imp->getChunk()->live;
  // Assume any other kind of symbol is live.
  return true;
}

void Symbol::replaceKeepingName(Symbol *other, size_t size) {
  StringRef origName = getName();
  memcpy(this, other, size);
  nameData = origName.data();
  nameSize = origName.size();
}

COFFSymbolRef DefinedCOFF::getCOFFSymbol() {
  size_t symSize = cast<ObjFile>(file)->getCOFFObj()->getSymbolTableEntrySize();
  if (symSize == sizeof(coff_symbol16))
    return COFFSymbolRef(reinterpret_cast<const coff_symbol16 *>(sym));
  assert(symSize == sizeof(coff_symbol32));
  return COFFSymbolRef(reinterpret_cast<const coff_symbol32 *>(sym));
}

uint64_t DefinedAbsolute::getRVA() { return va - ctx.config.imageBase; }

DefinedImportThunk::DefinedImportThunk(COFFLinkerContext &ctx, StringRef name,
                                       DefinedImportData *s,
                                       ImportThunkChunk *chunk)
    : Defined(DefinedImportThunkKind, name), wrappedSym(s), data(chunk) {}

Symbol *Undefined::getWeakAlias() {
  // A weak alias may be a weak alias to another symbol, so check recursively.
  DenseSet<Symbol *> weakChain;
  for (Symbol *a = weakAlias; a; a = cast<Undefined>(a)->weakAlias) {
    // Anti-dependency symbols can't be chained.
    if (a->isAntiDep)
      break;
    if (!isa<Undefined>(a))
      return a;
    if (!weakChain.insert(a).second)
      break; // We have a cycle.
  }
  return nullptr;
}

bool Undefined::resolveWeakAlias() {
  Defined *d = getDefinedWeakAlias();
  if (!d)
    return false;

  // We want to replace Sym with D. However, we can't just blindly
  // copy sizeof(SymbolUnion) bytes from D to Sym because D may be an
  // internal symbol, and internal symbols are stored as "unparented"
  // Symbols. For that reason we need to check which type of symbol we
  // are dealing with and copy the correct number of bytes.
  StringRef name = getName();
  bool wasAntiDep = isAntiDep;
  if (isa<DefinedRegular>(d))
    memcpy(this, d, sizeof(DefinedRegular));
  else if (isa<DefinedAbsolute>(d))
    memcpy(this, d, sizeof(DefinedAbsolute));
  else
    memcpy(this, d, sizeof(SymbolUnion));

  nameData = name.data();
  nameSize = name.size();
  isAntiDep = wasAntiDep;
  return true;
}

MemoryBufferRef LazyArchive::getMemberBuffer() {
  Archive::Child c =
      CHECK(sym.getMember(), "could not get the member for symbol " +
                                 toCOFFString(file->symtab.ctx, sym));
  return CHECK(c.getMemoryBufferRef(),
               "could not get the buffer for the member defining symbol " +
                   toCOFFString(file->symtab.ctx, sym));
}
} // namespace coff
} // namespace lld
