//===- Symbols.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm::object;
using llvm::sys::fs::identify_magic;
using llvm::sys::fs::file_magic;

namespace lld {
namespace coff {

StringRef SymbolBody::getName() {
  // DefinedCOFF names are read lazily for a performance reason.
  // Non-external symbol names are never used by the linker except for logging
  // or debugging. Their internal references are resolved not by name but by
  // symbol index. And because they are not external, no one can refer them by
  // name. Object files contain lots of non-external symbols, and creating
  // StringRefs for them (which involves lots of strlen() on the string table)
  // is a waste of time.
  if (Name.empty()) {
    auto *D = cast<DefinedCOFF>(this);
    D->File->getCOFFObj()->getSymbolName(D->Sym, Name);
  }
  return Name;
}

// Returns 1, 0 or -1 if this symbol should take precedence
// over the Other, tie or lose, respectively.
int SymbolBody::compare(SymbolBody *Other) {
  Kind LK = kind(), RK = Other->kind();

  // Normalize so that the smaller kind is on the left.
  if (LK > RK)
    return -Other->compare(this);

  // First handle comparisons between two different kinds.
  if (LK != RK) {
    if (RK > LastDefinedKind) {
      if (LK == LazyKind && cast<Undefined>(Other)->WeakAlias)
        return -1;

      // The LHS is either defined or lazy and so it wins.
      assert((LK <= LastDefinedKind || LK == LazyKind) && "Bad kind!");
      return 1;
    }

    // Bitcode has special complexities.
    if (RK == DefinedBitcodeKind) {
      auto *RHS = cast<DefinedBitcode>(Other);

      switch (LK) {
      case DefinedCommonKind:
        return 1;

      case DefinedRegularKind:
        // As an approximation, regular symbols win over bitcode symbols,
        // but we definitely have a conflict if the regular symbol is not
        // replaceable and neither is the bitcode symbol. We do not
        // replicate the rest of the symbol resolution logic here; symbol
        // resolution will be done accurately after lowering bitcode symbols
        // to regular symbols in addCombinedLTOObject().
        if (cast<DefinedRegular>(this)->isCOMDAT() || RHS->IsReplaceable)
          return 1;

        // Fallthrough to the default of a tie otherwise.
      default:
        return 0;
      }
    }

    // Either of the object file kind will trump a higher kind.
    if (LK <= LastDefinedCOFFKind)
      return 1;

    // The remaining kind pairs are ties amongst defined symbols.
    return 0;
  }

  // Now handle the case where the kinds are the same.
  switch (LK) {
  case DefinedRegularKind: {
    auto *LHS = cast<DefinedRegular>(this);
    auto *RHS = cast<DefinedRegular>(Other);
    if (LHS->isCOMDAT() && RHS->isCOMDAT())
      return LHS->getFileIndex() < RHS->getFileIndex() ? 1 : -1;
    return 0;
  }

  case DefinedCommonKind: {
    auto *LHS = cast<DefinedCommon>(this);
    auto *RHS = cast<DefinedCommon>(Other);
    if (LHS->getSize() == RHS->getSize())
      return LHS->getFileIndex() < RHS->getFileIndex() ? 1 : -1;
    return LHS->getSize() > RHS->getSize() ? 1 : -1;
  }

  case DefinedBitcodeKind: {
    auto *LHS = cast<DefinedBitcode>(this);
    auto *RHS = cast<DefinedBitcode>(Other);
    // If both are non-replaceable, we have a tie.
    if (!LHS->IsReplaceable && !RHS->IsReplaceable)
      return 0;

    // Non-replaceable symbols win, but even two replaceable symboles don't
    // tie. If both symbols are replaceable, choice is arbitrary.
    if (RHS->IsReplaceable && LHS->IsReplaceable)
      return uintptr_t(LHS) < uintptr_t(RHS) ? 1 : -1;
    return LHS->IsReplaceable ? -1 : 1;
  }

  case LazyKind: {
    // Don't tie, pick the earliest.
    auto *LHS = cast<Lazy>(this);
    auto *RHS = cast<Lazy>(Other);
    return LHS->getFileIndex() < RHS->getFileIndex() ? 1 : -1;
  }

  case UndefinedKind: {
    auto *LHS = cast<Undefined>(this);
    auto *RHS = cast<Undefined>(Other);
    // Tie if both undefined symbols have different weak aliases.
    if (LHS->WeakAlias && RHS->WeakAlias) {
      if (LHS->WeakAlias->getName() != RHS->WeakAlias->getName())
        return 0;
      return uintptr_t(LHS) < uintptr_t(RHS) ? 1 : -1;
    }
    return LHS->WeakAlias ? 1 : -1;
  }

  case DefinedLocalImportKind:
  case DefinedImportThunkKind:
  case DefinedImportDataKind:
  case DefinedAbsoluteKind:
  case DefinedRelativeKind:
    // These all simply tie.
    return 0;
  }
  llvm_unreachable("unknown symbol kind");
}

std::string SymbolBody::getDebugName() {
  std::string N = getName().str();
  if (auto *D = dyn_cast<DefinedCOFF>(this)) {
    N += " ";
    N += D->File->getShortName();
  } else if (auto *D = dyn_cast<DefinedBitcode>(this)) {
    N += " ";
    N += D->File->getShortName();
  }
  return N;
}

COFFSymbolRef DefinedCOFF::getCOFFSymbol() {
  size_t SymSize = File->getCOFFObj()->getSymbolTableEntrySize();
  if (SymSize == sizeof(coff_symbol16))
    return COFFSymbolRef(reinterpret_cast<const coff_symbol16 *>(Sym));
  assert(SymSize == sizeof(coff_symbol32));
  return COFFSymbolRef(reinterpret_cast<const coff_symbol32 *>(Sym));
}

DefinedImportThunk::DefinedImportThunk(StringRef Name, DefinedImportData *S,
                                       uint16_t Machine)
    : Defined(DefinedImportThunkKind, Name) {
  switch (Machine) {
  case AMD64: Data.reset(new ImportThunkChunkX64(S)); return;
  case I386:  Data.reset(new ImportThunkChunkX86(S)); return;
  case ARMNT: Data.reset(new ImportThunkChunkARM(S)); return;
  default:    llvm_unreachable("unknown machine type");
  }
}

std::unique_ptr<InputFile> Lazy::getMember() {
  MemoryBufferRef MBRef = File->getMember(&Sym);

  // getMember returns an empty buffer if the member was already
  // read from the library.
  if (MBRef.getBuffer().empty())
    return std::unique_ptr<InputFile>(nullptr);

  file_magic Magic = identify_magic(MBRef.getBuffer());
  if (Magic == file_magic::coff_import_library)
    return std::unique_ptr<InputFile>(new ImportFile(MBRef));

  std::unique_ptr<InputFile> Obj;
  if (Magic == file_magic::coff_object)
    Obj.reset(new ObjectFile(MBRef));
  else if (Magic == file_magic::bitcode)
    Obj.reset(new BitcodeFile(MBRef));
  else
    fatal("unknown file type: " + File->getName());

  Obj->setParentName(File->getName());
  return Obj;
}

Defined *Undefined::getWeakAlias() {
  // A weak alias may be a weak alias to another symbol, so check recursively.
  for (SymbolBody *A = WeakAlias; A; A = cast<Undefined>(A)->WeakAlias)
    if (auto *D = dyn_cast<Defined>(A->repl()))
      return D;
  return nullptr;
}

} // namespace coff
} // namespace lld
