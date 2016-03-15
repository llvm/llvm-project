//===- Symbols.h ------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_SYMBOLS_H
#define LLD_COFF_SYMBOLS_H

#include "Chunks.h"
#include "Config.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include <atomic>
#include <memory>
#include <vector>

namespace lld {
namespace coff {

using llvm::object::Archive;
using llvm::object::COFFSymbolRef;
using llvm::object::coff_import_header;
using llvm::object::coff_symbol_generic;

class ArchiveFile;
class BitcodeFile;
class InputFile;
class ObjectFile;
class SymbolBody;

// A real symbol object, SymbolBody, is usually accessed indirectly
// through a Symbol. There's always one Symbol for each symbol name.
// The resolver updates SymbolBody pointers as it resolves symbols.
struct Symbol {
  explicit Symbol(SymbolBody *P) : Body(P) {}
  SymbolBody *Body;
};

// The base class for real symbol classes.
class SymbolBody {
public:
  enum Kind {
    // The order of these is significant. We start with the regular defined
    // symbols as those are the most prevelant and the zero tag is the cheapest
    // to set. Among the defined kinds, the lower the kind is preferred over
    // the higher kind when testing wether one symbol should take precedence
    // over another.
    DefinedRegularKind = 0,
    DefinedCommonKind,
    DefinedLocalImportKind,
    DefinedImportThunkKind,
    DefinedImportDataKind,
    DefinedAbsoluteKind,
    DefinedRelativeKind,
    DefinedBitcodeKind,

    UndefinedKind,
    LazyKind,

    LastDefinedCOFFKind = DefinedCommonKind,
    LastDefinedKind = DefinedBitcodeKind,
  };

  Kind kind() const { return static_cast<Kind>(SymbolKind); }

  // Returns true if this is an external symbol.
  bool isExternal() { return IsExternal; }

  // Returns the symbol name.
  StringRef getName();

  // A SymbolBody has a backreference to a Symbol. Originally they are
  // doubly-linked. A backreference will never change. But the pointer
  // in the Symbol may be mutated by the resolver. If you have a
  // pointer P to a SymbolBody and are not sure whether the resolver
  // has chosen the object among other objects having the same name,
  // you can access P->Backref->Body to get the resolver's result.
  void setBackref(Symbol *P) { Backref = P; }
  SymbolBody *repl() { return Backref ? Backref->Body : this; }

  // Decides which symbol should "win" in the symbol table, this or
  // the Other. Returns 1 if this wins, -1 if the Other wins, or 0 if
  // they are duplicate (conflicting) symbols.
  int compare(SymbolBody *Other);

  // Returns a name of this symbol including source file name.
  // Used only for debugging and logging.
  std::string getDebugName();

protected:
  explicit SymbolBody(Kind K, StringRef N = "")
      : SymbolKind(K), IsExternal(true), IsCOMDAT(false),
        IsReplaceable(false), Name(N) {}

  const unsigned SymbolKind : 8;
  unsigned IsExternal : 1;

  // This bit is used by the \c DefinedRegular subclass.
  unsigned IsCOMDAT : 1;

  // This bit is used by the \c DefinedBitcode subclass.
  unsigned IsReplaceable : 1;

  StringRef Name;
  Symbol *Backref = nullptr;
};

// The base class for any defined symbols, including absolute symbols,
// etc.
class Defined : public SymbolBody {
public:
  Defined(Kind K, StringRef N = "") : SymbolBody(K, N) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() <= LastDefinedKind;
  }

  // Returns the RVA (relative virtual address) of this symbol. The
  // writer sets and uses RVAs.
  uint64_t getRVA();

  // Returns the RVA relative to the beginning of the output section.
  // Used to implement SECREL relocation type.
  uint64_t getSecrel();

  // Returns the output section index.
  // Used to implement SECTION relocation type.
  uint64_t getSectionIndex();

  // Returns true if this symbol points to an executable (e.g. .text) section.
  // Used to implement ARM relocations.
  bool isExecutable();
};

// Symbols defined via a COFF object file.
class DefinedCOFF : public Defined {
  friend SymbolBody;
public:
  DefinedCOFF(Kind K, ObjectFile *F, COFFSymbolRef S)
      : Defined(K), File(F), Sym(S.getGeneric()) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() <= LastDefinedCOFFKind;
  }

  int getFileIndex() { return File->Index; }

  COFFSymbolRef getCOFFSymbol();

protected:
  ObjectFile *File;
  const coff_symbol_generic *Sym;
};

// Regular defined symbols read from object file symbol tables.
class DefinedRegular : public DefinedCOFF {
public:
  DefinedRegular(ObjectFile *F, COFFSymbolRef S, SectionChunk *C)
      : DefinedCOFF(DefinedRegularKind, F, S), Data(&C->Repl) {
    IsExternal = S.isExternal();
    IsCOMDAT = C->isCOMDAT();
  }

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedRegularKind;
  }

  uint64_t getRVA() { return (*Data)->getRVA() + Sym->Value; }
  bool isCOMDAT() { return IsCOMDAT; }
  SectionChunk *getChunk() { return *Data; }
  uint32_t getValue() { return Sym->Value; }

private:
  SectionChunk **Data;
};

class DefinedCommon : public DefinedCOFF {
public:
  DefinedCommon(ObjectFile *F, COFFSymbolRef S, CommonChunk *C)
      : DefinedCOFF(DefinedCommonKind, F, S), Data(C) {
    IsExternal = S.isExternal();
  }

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedCommonKind;
  }

  uint64_t getRVA() { return Data->getRVA(); }

private:
  friend SymbolBody;
  uint64_t getSize() { return Sym->Value; }
  CommonChunk *Data;
};

// Absolute symbols.
class DefinedAbsolute : public Defined {
public:
  DefinedAbsolute(StringRef N, COFFSymbolRef S)
      : Defined(DefinedAbsoluteKind, N), VA(S.getValue()) {
    IsExternal = S.isExternal();
  }

  DefinedAbsolute(StringRef N, uint64_t V)
      : Defined(DefinedAbsoluteKind, N), VA(V) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedAbsoluteKind;
  }

  uint64_t getRVA() { return VA - Config->ImageBase; }
  void setVA(uint64_t V) { VA = V; }

private:
  uint64_t VA;
};

// This is a kind of absolute symbol but relative to the image base.
// Unlike absolute symbols, relocations referring this kind of symbols
// are subject of the base relocation. This type is used rarely --
// mainly for __ImageBase.
class DefinedRelative : public Defined {
public:
  explicit DefinedRelative(StringRef Name, uint64_t V = 0)
      : Defined(DefinedRelativeKind, Name), RVA(V) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedRelativeKind;
  }

  uint64_t getRVA() { return RVA; }
  void setRVA(uint64_t V) { RVA = V; }

private:
  uint64_t RVA;
};

// This class represents a symbol defined in an archive file. It is
// created from an archive file header, and it knows how to load an
// object file from an archive to replace itself with a defined
// symbol. If the resolver finds both Undefined and Lazy for
// the same name, it will ask the Lazy to load a file.
class Lazy : public SymbolBody {
public:
  Lazy(ArchiveFile *F, const Archive::Symbol S)
      : SymbolBody(LazyKind, S.getName()), File(F), Sym(S) {}

  static bool classof(const SymbolBody *S) { return S->kind() == LazyKind; }

  // Returns an object file for this symbol, or a nullptr if the file
  // was already returned.
  std::unique_ptr<InputFile> getMember();

  int getFileIndex() { return File->Index; }

private:
  ArchiveFile *File;
  const Archive::Symbol Sym;
};

// Undefined symbols.
class Undefined : public SymbolBody {
public:
  explicit Undefined(StringRef N) : SymbolBody(UndefinedKind, N) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == UndefinedKind;
  }

  // An undefined symbol can have a fallback symbol which gives an
  // undefined symbol a second chance if it would remain undefined.
  // If it remains undefined, it'll be replaced with whatever the
  // Alias pointer points to.
  SymbolBody *WeakAlias = nullptr;

  // If this symbol is external weak, try to resolve it to a defined
  // symbol by searching the chain of fallback symbols. Returns the symbol if
  // successful, otherwise returns null.
  Defined *getWeakAlias();
};

// Windows-specific classes.

// This class represents a symbol imported from a DLL. This has two
// names for internal use and external use. The former is used for
// name resolution, and the latter is used for the import descriptor
// table in an output. The former has "__imp_" prefix.
class DefinedImportData : public Defined {
public:
  DefinedImportData(StringRef D, StringRef N, StringRef E,
                    const coff_import_header *H)
      : Defined(DefinedImportDataKind, N), DLLName(D), ExternalName(E), Hdr(H) {
  }

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedImportDataKind;
  }

  uint64_t getRVA() { return Location->getRVA(); }
  StringRef getDLLName() { return DLLName; }
  StringRef getExternalName() { return ExternalName; }
  void setLocation(Chunk *AddressTable) { Location = AddressTable; }
  uint16_t getOrdinal() { return Hdr->OrdinalHint; }

private:
  StringRef DLLName;
  StringRef ExternalName;
  const coff_import_header *Hdr;
  Chunk *Location = nullptr;
};

// This class represents a symbol for a jump table entry which jumps
// to a function in a DLL. Linker are supposed to create such symbols
// without "__imp_" prefix for all function symbols exported from
// DLLs, so that you can call DLL functions as regular functions with
// a regular name. A function pointer is given as a DefinedImportData.
class DefinedImportThunk : public Defined {
public:
  DefinedImportThunk(StringRef Name, DefinedImportData *S, uint16_t Machine);

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedImportThunkKind;
  }

  uint64_t getRVA() { return Data->getRVA(); }
  Chunk *getChunk() { return Data.get(); }

private:
  std::unique_ptr<Chunk> Data;
};

// If you have a symbol "__imp_foo" in your object file, a symbol name
// "foo" becomes automatically available as a pointer to "__imp_foo".
// This class is for such automatically-created symbols.
// Yes, this is an odd feature. We didn't intend to implement that.
// This is here just for compatibility with MSVC.
class DefinedLocalImport : public Defined {
public:
  DefinedLocalImport(StringRef N, Defined *S)
      : Defined(DefinedLocalImportKind, N), Data(S) {}

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedLocalImportKind;
  }

  uint64_t getRVA() { return Data.getRVA(); }
  Chunk *getChunk() { return &Data; }

private:
  LocalImportChunk Data;
};

class DefinedBitcode : public Defined {
  friend SymbolBody;
public:
  DefinedBitcode(BitcodeFile *F, StringRef N, bool IsReplaceable)
      : Defined(DefinedBitcodeKind, N), File(F) {
    this->IsReplaceable = IsReplaceable;
  }

  static bool classof(const SymbolBody *S) {
    return S->kind() == DefinedBitcodeKind;
  }

private:
  BitcodeFile *File;
};

inline uint64_t Defined::getRVA() {
  switch (kind()) {
  case DefinedAbsoluteKind:
    return cast<DefinedAbsolute>(this)->getRVA();
  case DefinedRelativeKind:
    return cast<DefinedRelative>(this)->getRVA();
  case DefinedImportDataKind:
    return cast<DefinedImportData>(this)->getRVA();
  case DefinedImportThunkKind:
    return cast<DefinedImportThunk>(this)->getRVA();
  case DefinedLocalImportKind:
    return cast<DefinedLocalImport>(this)->getRVA();
  case DefinedCommonKind:
    return cast<DefinedCommon>(this)->getRVA();
  case DefinedRegularKind:
    return cast<DefinedRegular>(this)->getRVA();
  case DefinedBitcodeKind:
    llvm_unreachable("There is no address for a bitcode symbol.");
  case LazyKind:
  case UndefinedKind:
    llvm_unreachable("Cannot get the address for an undefined symbol.");
  }
  llvm_unreachable("unknown symbol kind");
}

} // namespace coff
} // namespace lld

#endif
