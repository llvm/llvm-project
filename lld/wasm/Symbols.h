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

  Symbol(StringRef Name, uint32_t Flags) : Flags(Flags), Name(Name) {}

  Kind getKind() const { return SymbolKind; }

  bool isLazy() const { return SymbolKind == LazyKind; }
  bool isDefined() const { return SymbolKind <= LastDefinedKind; }
  bool isUndefined() const {
    return SymbolKind == UndefinedGlobalKind ||
           SymbolKind == UndefinedFunctionKind;
  }
  bool isFunction() const {
    return SymbolKind == DefinedFunctionKind ||
           SymbolKind == UndefinedFunctionKind;
  }
  bool isGlobal() const { return !isFunction(); }
  bool isLocal() const;
  bool isWeak() const;
  bool isHidden() const;

  // Returns the symbol name.
  StringRef getName() const { return Name; }

  // Returns the file from which this symbol was created.
  InputFile *getFile() const { return File; }
  InputChunk *getChunk() const { return Chunk; }

  bool hasFunctionType() const { return FunctionType != nullptr; }
  const WasmSignature &getFunctionType() const;
  void setFunctionType(const WasmSignature *Type);
  void setHidden(bool IsHidden);

  uint32_t getOutputIndex() const;

  // Returns true if an output index has been set for this symbol
  bool hasOutputIndex() const;

  // Set the output index of the symbol (in the function or global index
  // space of the output object.
  void setOutputIndex(uint32_t Index);

  uint32_t getTableIndex() const;

  // Returns true if a table index has been set for this symbol
  bool hasTableIndex() const;

  // Set the table index of the symbol
  void setTableIndex(uint32_t Index);

  // Returns the virtual address of a defined global.
  // Only works for globals, not functions.
  uint32_t getVirtualAddress() const;

  void setVirtualAddress(uint32_t VA);

  void update(Kind K, InputFile *F = nullptr, uint32_t Flags = 0,
              InputChunk *chunk = nullptr, uint32_t Address = UINT32_MAX);

  void setArchiveSymbol(const Archive::Symbol &Sym) { ArchiveSymbol = Sym; }
  const Archive::Symbol &getArchiveSymbol() { return ArchiveSymbol; }

protected:
  uint32_t Flags;
  uint32_t VirtualAddress = 0;

  StringRef Name;
  Archive::Symbol ArchiveSymbol = {nullptr, 0, 0};
  Kind SymbolKind = InvalidKind;
  InputFile *File = nullptr;
  InputChunk *Chunk = nullptr;
  llvm::Optional<uint32_t> OutputIndex;
  llvm::Optional<uint32_t> TableIndex;
  const WasmSignature *FunctionType = nullptr;
};

} // namespace wasm

// Returns a symbol name for an error message.
std::string toString(const wasm::Symbol &Sym);
std::string toString(wasm::Symbol::Kind Kind);

} // namespace lld

#endif
