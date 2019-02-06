//===- Symbols.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_SYMBOLS_H
#define LLD_WASM_SYMBOLS_H

#include "Config.h"
#include "lld/Common/LLVM.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Wasm.h"

namespace lld {
namespace wasm {

using llvm::wasm::WasmSymbolType;

class InputFile;
class InputChunk;
class InputSegment;
class InputFunction;
class InputGlobal;
class InputEvent;
class InputSection;

#define INVALID_INDEX UINT32_MAX

// The base class for real symbol classes.
class Symbol {
public:
  enum Kind {
    DefinedFunctionKind,
    DefinedDataKind,
    DefinedGlobalKind,
    DefinedEventKind,
    SectionKind,
    UndefinedFunctionKind,
    UndefinedDataKind,
    UndefinedGlobalKind,
    LazyKind,
  };

  Kind kind() const { return SymbolKind; }

  bool isDefined() const {
    return SymbolKind == DefinedFunctionKind || SymbolKind == DefinedDataKind ||
           SymbolKind == DefinedGlobalKind || SymbolKind == DefinedEventKind ||
           SymbolKind == SectionKind;
  }

  bool isUndefined() const {
    return SymbolKind == UndefinedFunctionKind ||
           SymbolKind == UndefinedDataKind || SymbolKind == UndefinedGlobalKind;
  }

  bool isLazy() const { return SymbolKind == LazyKind; }

  bool isLocal() const;
  bool isWeak() const;
  bool isHidden() const;

  // True if this is an undefined weak symbol. This only works once
  // all input files have been added.
  bool isUndefWeak() const {
    // See comment on lazy symbols for details.
    return isWeak() && (isUndefined() || isLazy());
  }

  // Returns the symbol name.
  StringRef getName() const { return Name; }

  // Returns the file from which this symbol was created.
  InputFile *getFile() const { return File; }

  InputChunk *getChunk() const;

  // Indicates that the section or import for this symbol will be included in
  // the final image.
  bool isLive() const;

  // Marks the symbol's InputChunk as Live, so that it will be included in the
  // final image.
  void markLive();

  void setHidden(bool IsHidden);

  // Get/set the index in the output symbol table.  This is only used for
  // relocatable output.
  uint32_t getOutputSymbolIndex() const;
  void setOutputSymbolIndex(uint32_t Index);

  WasmSymbolType getWasmType() const;
  bool isExported() const;

  // True if this symbol was referenced by a regular (non-bitcode) object.
  unsigned IsUsedInRegularObj : 1;
  unsigned ForceExport : 1;

  // True if this symbol is specified by --trace-symbol option.
  unsigned Traced : 1;

protected:
  Symbol(StringRef Name, Kind K, uint32_t Flags, InputFile *F)
      : IsUsedInRegularObj(false), ForceExport(false), Traced(false),
        Name(Name), SymbolKind(K), Flags(Flags), File(F),
        Referenced(!Config->GcSections) {}

  StringRef Name;
  Kind SymbolKind;
  uint32_t Flags;
  InputFile *File;
  uint32_t OutputSymbolIndex = INVALID_INDEX;
  bool Referenced;
};

class FunctionSymbol : public Symbol {
public:
  static bool classof(const Symbol *S) {
    return S->kind() == DefinedFunctionKind ||
           S->kind() == UndefinedFunctionKind;
  }

  // Get/set the table index
  void setTableIndex(uint32_t Index);
  uint32_t getTableIndex() const;
  bool hasTableIndex() const;

  // Get/set the function index
  uint32_t getFunctionIndex() const;
  void setFunctionIndex(uint32_t Index);
  bool hasFunctionIndex() const;

  const WasmSignature *Signature;

protected:
  FunctionSymbol(StringRef Name, Kind K, uint32_t Flags, InputFile *F,
                 const WasmSignature *Sig)
      : Symbol(Name, K, Flags, F), Signature(Sig) {}

  uint32_t TableIndex = INVALID_INDEX;
  uint32_t FunctionIndex = INVALID_INDEX;
};

class DefinedFunction : public FunctionSymbol {
public:
  DefinedFunction(StringRef Name, uint32_t Flags, InputFile *F,
                  InputFunction *Function);

  static bool classof(const Symbol *S) {
    return S->kind() == DefinedFunctionKind;
  }

  InputFunction *Function;
};

class UndefinedFunction : public FunctionSymbol {
public:
  UndefinedFunction(StringRef Name, StringRef Module, uint32_t Flags,
                    InputFile *File = nullptr,
                    const WasmSignature *Type = nullptr)
      : FunctionSymbol(Name, UndefinedFunctionKind, Flags, File, Type),
        Module(Module) {}

  static bool classof(const Symbol *S) {
    return S->kind() == UndefinedFunctionKind;
  }

  StringRef Module;
};

class SectionSymbol : public Symbol {
public:
  static bool classof(const Symbol *S) { return S->kind() == SectionKind; }

  SectionSymbol(StringRef Name, uint32_t Flags, const InputSection *S,
                InputFile *F = nullptr)
      : Symbol(Name, SectionKind, Flags, F), Section(S) {}

  const InputSection *Section;

  uint32_t getOutputSectionIndex() const;
  void setOutputSectionIndex(uint32_t Index);

protected:
  uint32_t OutputSectionIndex = INVALID_INDEX;
};

class DataSymbol : public Symbol {
public:
  static bool classof(const Symbol *S) {
    return S->kind() == DefinedDataKind || S->kind() == UndefinedDataKind;
  }

protected:
  DataSymbol(StringRef Name, Kind K, uint32_t Flags, InputFile *F)
      : Symbol(Name, K, Flags, F) {}
};

class DefinedData : public DataSymbol {
public:
  // Constructor for regular data symbols originating from input files.
  DefinedData(StringRef Name, uint32_t Flags, InputFile *F,
              InputSegment *Segment, uint32_t Offset, uint32_t Size)
      : DataSymbol(Name, DefinedDataKind, Flags, F), Segment(Segment),
        Offset(Offset), Size(Size) {}

  // Constructor for linker synthetic data symbols.
  DefinedData(StringRef Name, uint32_t Flags)
      : DataSymbol(Name, DefinedDataKind, Flags, nullptr) {}

  static bool classof(const Symbol *S) { return S->kind() == DefinedDataKind; }

  // Returns the output virtual address of a defined data symbol.
  uint32_t getVirtualAddress() const;
  void setVirtualAddress(uint32_t VA);

  // Returns the offset of a defined data symbol within its OutputSegment.
  uint32_t getOutputSegmentOffset() const;
  uint32_t getOutputSegmentIndex() const;
  uint32_t getSize() const { return Size; }

  InputSegment *Segment = nullptr;

protected:
  uint32_t Offset = 0;
  uint32_t Size = 0;
};

class UndefinedData : public DataSymbol {
public:
  UndefinedData(StringRef Name, uint32_t Flags, InputFile *File = nullptr)
      : DataSymbol(Name, UndefinedDataKind, Flags, File) {}
  static bool classof(const Symbol *S) {
    return S->kind() == UndefinedDataKind;
  }
};

class GlobalSymbol : public Symbol {
public:
  static bool classof(const Symbol *S) {
    return S->kind() == DefinedGlobalKind || S->kind() == UndefinedGlobalKind;
  }

  const WasmGlobalType *getGlobalType() const { return GlobalType; }

  // Get/set the global index
  uint32_t getGlobalIndex() const;
  void setGlobalIndex(uint32_t Index);
  bool hasGlobalIndex() const;

protected:
  GlobalSymbol(StringRef Name, Kind K, uint32_t Flags, InputFile *F,
               const WasmGlobalType *GlobalType)
      : Symbol(Name, K, Flags, F), GlobalType(GlobalType) {}

  const WasmGlobalType *GlobalType;
  uint32_t GlobalIndex = INVALID_INDEX;
};

class DefinedGlobal : public GlobalSymbol {
public:
  DefinedGlobal(StringRef Name, uint32_t Flags, InputFile *File,
                InputGlobal *Global);

  static bool classof(const Symbol *S) {
    return S->kind() == DefinedGlobalKind;
  }

  InputGlobal *Global;
};

class UndefinedGlobal : public GlobalSymbol {
public:
  UndefinedGlobal(StringRef Name, uint32_t Flags, InputFile *File = nullptr,
                  const WasmGlobalType *Type = nullptr)
      : GlobalSymbol(Name, UndefinedGlobalKind, Flags, File, Type) {}

  static bool classof(const Symbol *S) {
    return S->kind() == UndefinedGlobalKind;
  }
};

// Wasm events are features that suspend the current execution and transfer the
// control flow to a corresponding handler. Currently the only supported event
// kind is exceptions.
//
// Event tags are values to distinguish different events. For exceptions, they
// can be used to distinguish different language's exceptions, i.e., all C++
// exceptions have the same tag. Wasm can generate code capable of doing
// different handling actions based on the tag of caught exceptions.
//
// A single EventSymbol object represents a single tag. C++ exception event
// symbol is a weak symbol generated in every object file in which exceptions
// are used, and has name '__cpp_exception' for linking.
class EventSymbol : public Symbol {
public:
  static bool classof(const Symbol *S) { return S->kind() == DefinedEventKind; }

  const WasmEventType *getEventType() const { return EventType; }

  // Get/set the event index
  uint32_t getEventIndex() const;
  void setEventIndex(uint32_t Index);
  bool hasEventIndex() const;

  const WasmSignature *Signature;

protected:
  EventSymbol(StringRef Name, Kind K, uint32_t Flags, InputFile *F,
              const WasmEventType *EventType, const WasmSignature *Sig)
      : Symbol(Name, K, Flags, F), Signature(Sig), EventType(EventType) {}

  const WasmEventType *EventType;
  uint32_t EventIndex = INVALID_INDEX;
};

class DefinedEvent : public EventSymbol {
public:
  DefinedEvent(StringRef Name, uint32_t Flags, InputFile *File,
               InputEvent *Event);

  static bool classof(const Symbol *S) { return S->kind() == DefinedEventKind; }

  InputEvent *Event;
};

// LazySymbol represents a symbol that is not yet in the link, but we know where
// to find it if needed. If the resolver finds both Undefined and Lazy for the
// same name, it will ask the Lazy to load a file.
//
// A special complication is the handling of weak undefined symbols. They should
// not load a file, but we have to remember we have seen both the weak undefined
// and the lazy. We represent that with a lazy symbol with a weak binding. This
// means that code looking for undefined symbols normally also has to take lazy
// symbols into consideration.
class LazySymbol : public Symbol {
public:
  LazySymbol(StringRef Name, uint32_t Flags, InputFile *File,
             const llvm::object::Archive::Symbol &Sym)
      : Symbol(Name, LazyKind, Flags, File), ArchiveSymbol(Sym) {}

  static bool classof(const Symbol *S) { return S->kind() == LazyKind; }
  void fetch();

  // Lazy symbols can have a signature because they can replace an
  // UndefinedFunction which which case we need to be able to preserve the
  // signture.
  // TODO(sbc): This repetition of the signature field is inelegant.  Revisit
  // the use of class hierarchy to represent symbol taxonomy.
  const WasmSignature *Signature = nullptr;

private:
  llvm::object::Archive::Symbol ArchiveSymbol;
};

// linker-generated symbols
struct WasmSym {
  // __stack_pointer
  // Global that holds the address of the top of the explicit value stack in
  // linear memory.
  static GlobalSymbol *StackPointer;

  // __data_end
  // Symbol marking the end of the data and bss.
  static DefinedData *DataEnd;

  // __heap_base
  // Symbol marking the end of the data, bss and explicit stack.  Any linear
  // memory following this address is not used by the linked code and can
  // therefore be used as a backing store for brk()/malloc() implementations.
  static DefinedData *HeapBase;

  // __wasm_call_ctors
  // Function that directly calls all ctors in priority order.
  static DefinedFunction *CallCtors;

  // __dso_handle
  // Symbol used in calls to __cxa_atexit to determine current DLL
  static DefinedData *DsoHandle;

  // __table_base
  // Used in PIC code for offset of indirect function table
  static UndefinedGlobal *TableBase;

  // __memory_base
  // Used in PIC code for offset of global data
  static UndefinedGlobal *MemoryBase;
};

// A buffer class that is large enough to hold any Symbol-derived
// object. We allocate memory using this class and instantiate a symbol
// using the placement new.
union SymbolUnion {
  alignas(DefinedFunction) char A[sizeof(DefinedFunction)];
  alignas(DefinedData) char B[sizeof(DefinedData)];
  alignas(DefinedGlobal) char C[sizeof(DefinedGlobal)];
  alignas(DefinedEvent) char D[sizeof(DefinedEvent)];
  alignas(LazySymbol) char E[sizeof(LazySymbol)];
  alignas(UndefinedFunction) char F[sizeof(UndefinedFunction)];
  alignas(UndefinedData) char G[sizeof(UndefinedData)];
  alignas(UndefinedGlobal) char H[sizeof(UndefinedGlobal)];
  alignas(SectionSymbol) char I[sizeof(SectionSymbol)];
};

void printTraceSymbol(Symbol *Sym);

template <typename T, typename... ArgT>
T *replaceSymbol(Symbol *S, ArgT &&... Arg) {
  static_assert(std::is_trivially_destructible<T>(),
                "Symbol types must be trivially destructible");
  static_assert(sizeof(T) <= sizeof(SymbolUnion), "SymbolUnion too small");
  static_assert(alignof(T) <= alignof(SymbolUnion),
                "SymbolUnion not aligned enough");
  assert(static_cast<Symbol *>(static_cast<T *>(nullptr)) == nullptr &&
         "Not a Symbol");

  Symbol SymCopy = *S;

  T *S2 = new (S) T(std::forward<ArgT>(Arg)...);
  S2->IsUsedInRegularObj = SymCopy.IsUsedInRegularObj;
  S2->ForceExport = SymCopy.ForceExport;
  S2->Traced = SymCopy.Traced;

  // Print out a log message if --trace-symbol was specified.
  // This is for debugging.
  if (S2->Traced)
    printTraceSymbol(S2);

  return S2;
}

} // namespace wasm

// Returns a symbol name for an error message.
std::string toString(const wasm::Symbol &Sym);
std::string toString(wasm::Symbol::Kind Kind);
std::string maybeDemangleSymbol(StringRef Name);

} // namespace lld

#endif
