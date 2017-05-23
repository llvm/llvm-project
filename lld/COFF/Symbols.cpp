//===- Symbols.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "Error.h"
#include "InputFiles.h"
#include "Memory.h"
#include "Strings.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

// Returns a symbol name for an error message.
std::string lld::toString(coff::SymbolBody &B) {
  if (Optional<std::string> S = coff::demangle(B.getName()))
    return ("\"" + *S + "\" (" + B.getName() + ")").str();
  return B.getName();
}

namespace lld {
namespace coff {

StringRef SymbolBody::getName() {
  // COFF symbol names are read lazily for a performance reason.
  // Non-external symbol names are never used by the linker except for logging
  // or debugging. Their internal references are resolved not by name but by
  // symbol index. And because they are not external, no one can refer them by
  // name. Object files contain lots of non-external symbols, and creating
  // StringRefs for them (which involves lots of strlen() on the string table)
  // is a waste of time.
  if (Name.empty()) {
    auto *D = cast<DefinedCOFF>(this);
    cast<ObjectFile>(D->File)->getCOFFObj()->getSymbolName(D->Sym, Name);
  }
  return Name;
}

InputFile *SymbolBody::getFile() {
  if (auto *Sym = dyn_cast<DefinedCOFF>(this))
    return Sym->File;
  if (auto *Sym = dyn_cast<Lazy>(this))
    return Sym->File;
  return nullptr;
}

COFFSymbolRef DefinedCOFF::getCOFFSymbol() {
  size_t SymSize =
      cast<ObjectFile>(File)->getCOFFObj()->getSymbolTableEntrySize();
  if (SymSize == sizeof(coff_symbol16))
    return COFFSymbolRef(reinterpret_cast<const coff_symbol16 *>(Sym));
  assert(SymSize == sizeof(coff_symbol32));
  return COFFSymbolRef(reinterpret_cast<const coff_symbol32 *>(Sym));
}

DefinedImportThunk::DefinedImportThunk(StringRef Name, DefinedImportData *S,
                                       uint16_t Machine)
    : Defined(DefinedImportThunkKind, Name) {
  switch (Machine) {
  case AMD64: Data = make<ImportThunkChunkX64>(S); return;
  case I386:  Data = make<ImportThunkChunkX86>(S); return;
  case ARMNT: Data = make<ImportThunkChunkARM>(S); return;
  default:    llvm_unreachable("unknown machine type");
  }
}

Defined *Undefined::getWeakAlias() {
  // A weak alias may be a weak alias to another symbol, so check recursively.
  for (SymbolBody *A = WeakAlias; A; A = cast<Undefined>(A)->WeakAlias)
    if (auto *D = dyn_cast<Defined>(A))
      return D;
  return nullptr;
}
} // namespace coff
} // namespace lld
