//===- Symbols.h ------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_SYMBOLS_H
#define LLD_WASM_SYMBOLS_H

#include "lld/Common/LLVM.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Wasm.h"

using llvm::object::Archive;
using llvm::wasm::WasmSignature;

namespace lld {
namespace wasm {

class InputFile;
class InputChunk;
class InputSegment;
class InputFunction;

#define INVALID_INDEX UINT32_MAX

// The base class for real symbol classes.
class Symbol {
public:
  enum Kind {
    DefinedFunctionKind,
    DefinedGlobalKind,

    LazyKind,
    UndefinedFunctionKind,
    UndefinedGlobalKind,

    LastDefinedKind = DefinedGlobalKind,
    InvalidKind,
  };

  Kind kind() const { return SymbolKind; }

  bool isLazy() const { return SymbolKind == LazyKind; }
  bool isDefined() const { return SymbolKind <= LastDefinedKind; }
  bool isUndefined() const {
    return SymbolKind == UndefinedGlobalKind ||
           SymbolKind == UndefinedFunctionKind;
  }
  bool isLocal() const;
  bool isWeak() const;
  bool isHidden() const;

  // Returns the symbol name.
  StringRef getName() const { return Name; }

  // Returns the file from which this symbol was created.
  InputFile *getFile() const { return File; }
  InputChunk *getChunk() const;

  void setHidden(bool IsHidden);

  uint32_t getOutputIndex() const;

  // Returns true if an output index has been set for this symbol
  bool hasOutputIndex() const;

  // Set the output index of the symbol (in the function or global index
  // space of the output object.
  void setOutputIndex(uint32_t Index);

protected:
  Symbol(StringRef Name, Kind K, uint32_t Flags, InputFile *F)
      : Name(Name), SymbolKind(K), Flags(Flags), File(F) {}

  StringRef Name;
  Kind SymbolKind;
  uint32_t Flags;
  InputFile *File;
  uint32_t OutputIndex = INVALID_INDEX;
};

class FunctionSymbol : public Symbol {
public:
  static bool classof(const Symbol *S) {
    return S->kind() == DefinedFunctionKind ||
           S->kind() == UndefinedFunctionKind;
  }

  const WasmSignature *getFunctionType() const { return FunctionType; }

  uint32_t getTableIndex() const;

  // Returns true if a table index has been set for this symbol
  bool hasTableIndex() const;

  // Set the table index of the symbol
  void setTableIndex(uint32_t Index);

protected:
  FunctionSymbol(StringRef Name, Kind K, uint32_t Flags, InputFile *F,
                 const WasmSignature *Type)
      : Symbol(Name, K, Flags, F), FunctionType(Type) {}

  uint32_t TableIndex = INVALID_INDEX;

  const WasmSignature *FunctionType;
};

class DefinedFunction : public FunctionSymbol {
public:
  DefinedFunction(StringRef Name, uint32_t Flags, InputFile *F,
                  InputFunction *Function);

  DefinedFunction(StringRef Name, uint32_t Flags, const WasmSignature *Type)
      : FunctionSymbol(Name, DefinedFunctionKind, Flags, nullptr, Type) {}

  static bool classof(const Symbol *S) {
    return S->kind() == DefinedFunctionKind;
  }

  InputFunction *Function;
};

class UndefinedFunction : public FunctionSymbol {
public:
  UndefinedFunction(StringRef Name, uint32_t Flags, InputFile *File = nullptr,
                    const WasmSignature *Type = nullptr)
      : FunctionSymbol(Name, UndefinedFunctionKind, Flags, File, Type) {}

  static bool classof(const Symbol *S) {
    return S->kind() == UndefinedFunctionKind;
  }
};

class GlobalSymbol : public Symbol {
public:
  static bool classof(const Symbol *S) {
    return S->kind() == DefinedGlobalKind || S->kind() == UndefinedGlobalKind;
  }

protected:
  GlobalSymbol(StringRef Name, Kind K, uint32_t Flags, InputFile *F)
      : Symbol(Name, K, Flags, F) {}
};

class DefinedGlobal : public GlobalSymbol {
public:
  DefinedGlobal(StringRef Name, uint32_t Flags, InputFile *F = nullptr,
                InputSegment *Segment = nullptr, uint32_t Address = 0)
      : GlobalSymbol(Name, DefinedGlobalKind, Flags, F), Segment(Segment),
        VirtualAddress(Address) {}

  static bool classof(const Symbol *S) {
    return S->kind() == DefinedGlobalKind;
  }

  uint32_t getVirtualAddress() const;
  void setVirtualAddress(uint32_t VA);

  InputSegment *Segment;

protected:
  uint32_t VirtualAddress;
};

class UndefinedGlobal : public GlobalSymbol {
public:
  UndefinedGlobal(StringRef Name, uint32_t Flags, InputFile *File = nullptr)
      : GlobalSymbol(Name, UndefinedGlobalKind, Flags, File) {}
  static bool classof(const Symbol *S) {
    return S->kind() == UndefinedGlobalKind;
  }
};

class LazySymbol : public Symbol {
public:
  LazySymbol(StringRef Name, InputFile *File, const Archive::Symbol &Sym)
      : Symbol(Name, LazyKind, 0, File), ArchiveSymbol(Sym) {}

  static bool classof(const Symbol *S) { return S->kind() == LazyKind; }

  const Archive::Symbol &getArchiveSymbol() { return ArchiveSymbol; }

protected:
  Archive::Symbol ArchiveSymbol;
};

// linker-generated symbols
struct WasmSym {
  // __stack_pointer
  // Global that holds the address of the top of the explicit value stack in
  // linear memory.
  static DefinedGlobal *StackPointer;

  // __data_end
  // Symbol marking the end of the data and bss.
  static DefinedGlobal *DataEnd;

  // __heap_base
  // Symbol marking the end of the data, bss and explicit stack.  Any linear
  // memory following this address is not used by the linked code and can
  // therefore be used as a backing store for brk()/malloc() implementations.
  static DefinedGlobal *HeapBase;

  // __wasm_call_ctors
  // Function that directly calls all ctors in priority order.
  static DefinedFunction *CallCtors;

  // __dso_handle
  // Global used in calls to __cxa_atexit to determine current DLL
  static DefinedGlobal *DsoHandle;
};

// A buffer class that is large enough to hold any Symbol-derived
// object. We allocate memory using this class and instantiate a symbol
// using the placement new.
union SymbolUnion {
  alignas(DefinedFunction) char A[sizeof(DefinedFunction)];
  alignas(DefinedGlobal) char B[sizeof(DefinedGlobal)];
  alignas(LazySymbol) char C[sizeof(LazySymbol)];
  alignas(UndefinedFunction) char D[sizeof(UndefinedFunction)];
  alignas(UndefinedGlobal) char E[sizeof(UndefinedFunction)];
};

template <typename T, typename... ArgT>
T *replaceSymbol(Symbol *S, ArgT &&... Arg) {
  static_assert(std::is_trivially_destructible<T>(),
                "Symbol types must be trivially destructible");
  static_assert(sizeof(T) <= sizeof(SymbolUnion), "Symbol too small");
  static_assert(alignof(T) <= alignof(SymbolUnion),
                "SymbolUnion not aligned enough");
  assert(static_cast<Symbol *>(static_cast<T *>(nullptr)) == nullptr &&
         "Not a Symbol");
  return new (S) T(std::forward<ArgT>(Arg)...);
}

} // namespace wasm

// Returns a symbol name for an error message.
std::string toString(const wasm::Symbol &Sym);
std::string toString(wasm::Symbol::Kind Kind);

} // namespace lld

#endif
