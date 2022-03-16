//===- Symbols.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various types of Symbols.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SYMBOLS_H
#define LLD_ELF_SYMBOLS_H

#include "Config.h"
#include "lld/Common/LLVM.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Object/ELF.h"
#include <tuple>

namespace lld {
namespace elf {
class Symbol;
}
// Returns a string representation for a symbol for diagnostics.
std::string toString(const elf::Symbol &);

namespace elf {
class CommonSymbol;
class Defined;
class OutputSection;
class SectionBase;
class InputSectionBase;
class SharedSymbol;
class Symbol;
class Undefined;
class LazyObject;
class InputFile;

// Some index properties of a symbol are stored separately in this auxiliary
// struct to decrease sizeof(SymbolUnion) in the majority of cases.
struct SymbolAux {
  uint32_t gotIdx = -1;
  uint32_t pltIdx = -1;
  uint32_t tlsDescIdx = -1;
  uint32_t tlsGdIdx = -1;
};

extern SmallVector<SymbolAux, 0> symAux;

// The base class for real symbol classes.
class Symbol {
public:
  enum Kind {
    PlaceholderKind,
    DefinedKind,
    CommonKind,
    SharedKind,
    UndefinedKind,
    LazyObjectKind,
  };

  Kind kind() const { return static_cast<Kind>(symbolKind); }

  // The file from which this symbol was created.
  InputFile *file;

protected:
  const char *nameData;
  // 32-bit size saves space.
  uint32_t nameSize;

public:
  // Symbol binding. This is not overwritten by replace() to track
  // changes during resolution. In particular:
  //  - An undefined weak is still weak when it resolves to a shared library.
  //  - An undefined weak will not extract archive members, but we have to
  //    remember it is weak.
  uint8_t binding;

  // The following fields have the same meaning as the ELF symbol attributes.
  uint8_t type;    // symbol type
  uint8_t stOther; // st_other field value

  uint8_t symbolKind;

  // The partition whose dynamic symbol table contains this symbol's definition.
  uint8_t partition = 1;

  // Symbol visibility. This is the computed minimum visibility of all
  // observed non-DSO symbols.
  uint8_t visibility : 2;

  // True if this symbol is preemptible at load time.
  uint8_t isPreemptible : 1;

  // True if the symbol was used for linking and thus need to be added to the
  // output file's symbol table. This is true for all symbols except for
  // unreferenced DSO symbols, lazy (archive) symbols, and bitcode symbols that
  // are unreferenced except by other bitcode objects.
  uint8_t isUsedInRegularObj : 1;

  // True if an undefined or shared symbol is used from a live section.
  //
  // NOTE: In Writer.cpp the field is used to mark local defined symbols
  // which are referenced by relocations when -r or --emit-relocs is given.
  uint8_t used : 1;

  // Used by a Defined symbol with protected or default visibility, to record
  // whether it is required to be exported into .dynsym. This is set when any of
  // the following conditions hold:
  //
  // - If there is an interposable symbol from a DSO. Note: We also do this for
  //   STV_PROTECTED symbols which can't be interposed (to match BFD behavior).
  // - If -shared or --export-dynamic is specified, any symbol in an object
  //   file/bitcode sets this property, unless suppressed by LTO
  //   canBeOmittedFromSymbolTable().
  uint8_t exportDynamic : 1;

  // True if the symbol is in the --dynamic-list file. A Defined symbol with
  // protected or default visibility with this property is required to be
  // exported into .dynsym.
  uint8_t inDynamicList : 1;

  // Used to track if there has been at least one undefined reference to the
  // symbol. For Undefined and SharedSymbol, the binding may change to STB_WEAK
  // if the first undefined reference from a non-shared object is weak.
  //
  // This is also used to retain __wrap_foo when foo is referenced.
  uint8_t referenced : 1;

  // True if this symbol is specified by --trace-symbol option.
  uint8_t traced : 1;

  // True if the name contains '@'.
  uint8_t hasVersionSuffix : 1;

  inline void replace(const Symbol &other);

  bool includeInDynsym() const;
  uint8_t computeBinding() const;
  bool isGlobal() const { return binding == llvm::ELF::STB_GLOBAL; }
  bool isWeak() const { return binding == llvm::ELF::STB_WEAK; }

  bool isUndefined() const { return symbolKind == UndefinedKind; }
  bool isCommon() const { return symbolKind == CommonKind; }
  bool isDefined() const { return symbolKind == DefinedKind; }
  bool isShared() const { return symbolKind == SharedKind; }
  bool isPlaceholder() const { return symbolKind == PlaceholderKind; }

  bool isLocal() const { return binding == llvm::ELF::STB_LOCAL; }

  bool isLazy() const { return symbolKind == LazyObjectKind; }

  // True if this is an undefined weak symbol. This only works once
  // all input files have been added.
  bool isUndefWeak() const { return isWeak() && isUndefined(); }

  StringRef getName() const { return {nameData, nameSize}; }

  void setName(StringRef s) {
    nameData = s.data();
    nameSize = s.size();
  }

  void parseSymbolVersion();

  // Get the NUL-terminated version suffix ("", "@...", or "@@...").
  //
  // For @@, the name has been truncated by insert(). For @, the name has been
  // truncated by Symbol::parseSymbolVersion().
  const char *getVersionSuffix() const { return nameData + nameSize; }

  uint32_t getGotIdx() const {
    return auxIdx == uint32_t(-1) ? uint32_t(-1) : symAux[auxIdx].gotIdx;
  }
  uint32_t getPltIdx() const {
    return auxIdx == uint32_t(-1) ? uint32_t(-1) : symAux[auxIdx].pltIdx;
  }
  uint32_t getTlsDescIdx() const {
    return auxIdx == uint32_t(-1) ? uint32_t(-1) : symAux[auxIdx].tlsDescIdx;
  }
  uint32_t getTlsGdIdx() const {
    return auxIdx == uint32_t(-1) ? uint32_t(-1) : symAux[auxIdx].tlsGdIdx;
  }

  bool isInGot() const { return getGotIdx() != uint32_t(-1); }
  bool isInPlt() const { return getPltIdx() != uint32_t(-1); }

  uint64_t getVA(int64_t addend = 0) const;

  uint64_t getGotOffset() const;
  uint64_t getGotVA() const;
  uint64_t getGotPltOffset() const;
  uint64_t getGotPltVA() const;
  uint64_t getPltVA() const;
  uint64_t getSize() const;
  OutputSection *getOutputSection() const;

  // The following two functions are used for symbol resolution.
  //
  // You are expected to call mergeProperties for all symbols in input
  // files so that attributes that are attached to names rather than
  // indivisual symbol (such as visibility) are merged together.
  //
  // Every time you read a new symbol from an input, you are supposed
  // to call resolve() with the new symbol. That function replaces
  // "this" object as a result of name resolution if the new symbol is
  // more appropriate to be included in the output.
  //
  // For example, if "this" is an undefined symbol and a new symbol is
  // a defined symbol, "this" is replaced with the new symbol.
  void mergeProperties(const Symbol &other);
  void resolve(const Symbol &other);

  // If this is a lazy symbol, extract an input file and add the symbol
  // in the file to the symbol table. Calling this function on
  // non-lazy object causes a runtime error.
  void extract() const;

  void checkDuplicate(const Defined &other) const;

private:
  void resolveUndefined(const Undefined &other);
  void resolveCommon(const CommonSymbol &other);
  void resolveDefined(const Defined &other);
  void resolveLazy(const LazyObject &other);
  void resolveShared(const SharedSymbol &other);

  bool shouldReplace(const Defined &other) const;

  inline size_t getSymbolSize() const;

protected:
  Symbol(Kind k, InputFile *file, StringRef name, uint8_t binding,
         uint8_t stOther, uint8_t type)
      : file(file), nameData(name.data()), nameSize(name.size()),
        binding(binding), type(type), stOther(stOther), symbolKind(k),
        visibility(stOther & 3), isPreemptible(false),
        isUsedInRegularObj(false), used(false), exportDynamic(false),
        inDynamicList(false), referenced(false), traced(false),
        hasVersionSuffix(false), isInIplt(false), gotInIgot(false),
        folded(false), needsTocRestore(false), scriptDefined(false),
        needsCopy(false), needsGot(false), needsPlt(false), needsTlsDesc(false),
        needsTlsGd(false), needsTlsGdToIe(false), needsGotDtprel(false),
        needsTlsIe(false), hasDirectReloc(false) {}

public:
  // True if this symbol is in the Iplt sub-section of the Plt and the Igot
  // sub-section of the .got.plt or .got.
  uint8_t isInIplt : 1;

  // True if this symbol needs a GOT entry and its GOT entry is actually in
  // Igot. This will be true only for certain non-preemptible ifuncs.
  uint8_t gotInIgot : 1;

  // True if defined relative to a section discarded by ICF.
  uint8_t folded : 1;

  // True if a call to this symbol needs to be followed by a restore of the
  // PPC64 toc pointer.
  uint8_t needsTocRestore : 1;

  // True if this symbol is defined by a symbol assignment or wrapped by --wrap.
  //
  // LTO shouldn't inline the symbol because it doesn't know the final content
  // of the symbol.
  uint8_t scriptDefined : 1;

  // True if this symbol needs a canonical PLT entry, or (during
  // postScanRelocations) a copy relocation.
  uint8_t needsCopy : 1;

  // Temporary flags used to communicate which symbol entries need PLT and GOT
  // entries during postScanRelocations();
  uint8_t needsGot : 1;
  uint8_t needsPlt : 1;
  uint8_t needsTlsDesc : 1;
  uint8_t needsTlsGd : 1;
  uint8_t needsTlsGdToIe : 1;
  uint8_t needsGotDtprel : 1;
  uint8_t needsTlsIe : 1;
  uint8_t hasDirectReloc : 1;

  // A symAux index used to access GOT/PLT entry indexes. This is allocated in
  // postScanRelocations().
  uint32_t auxIdx = -1;
  uint32_t dynsymIndex = 0;

  // This field is a index to the symbol's version definition.
  uint16_t verdefIndex = -1;

  // Version definition index.
  uint16_t versionId;

  bool needsDynReloc() const {
    return needsCopy || needsGot || needsPlt || needsTlsDesc || needsTlsGd ||
           needsTlsGdToIe || needsGotDtprel || needsTlsIe;
  }
  void allocateAux() {
    assert(auxIdx == uint32_t(-1));
    auxIdx = symAux.size();
    symAux.emplace_back();
  }

  bool isSection() const { return type == llvm::ELF::STT_SECTION; }
  bool isTls() const { return type == llvm::ELF::STT_TLS; }
  bool isFunc() const { return type == llvm::ELF::STT_FUNC; }
  bool isGnuIFunc() const { return type == llvm::ELF::STT_GNU_IFUNC; }
  bool isObject() const { return type == llvm::ELF::STT_OBJECT; }
  bool isFile() const { return type == llvm::ELF::STT_FILE; }
};

// Represents a symbol that is defined in the current output file.
class Defined : public Symbol {
public:
  Defined(InputFile *file, StringRef name, uint8_t binding, uint8_t stOther,
          uint8_t type, uint64_t value, uint64_t size, SectionBase *section)
      : Symbol(DefinedKind, file, name, binding, stOther, type), value(value),
        size(size), section(section) {
    exportDynamic = config->exportDynamic;
  }

  static bool classof(const Symbol *s) { return s->isDefined(); }

  uint64_t value;
  uint64_t size;
  SectionBase *section;
};

// Represents a common symbol.
//
// On Unix, it is traditionally allowed to write variable definitions
// without initialization expressions (such as "int foo;") to header
// files. Such definition is called "tentative definition".
//
// Using tentative definition is usually considered a bad practice
// because you should write only declarations (such as "extern int
// foo;") to header files. Nevertheless, the linker and the compiler
// have to do something to support bad code by allowing duplicate
// definitions for this particular case.
//
// Common symbols represent variable definitions without initializations.
// The compiler creates common symbols when it sees variable definitions
// without initialization (you can suppress this behavior and let the
// compiler create a regular defined symbol by -fno-common).
//
// The linker allows common symbols to be replaced by regular defined
// symbols. If there are remaining common symbols after name resolution is
// complete, they are converted to regular defined symbols in a .bss
// section. (Therefore, the later passes don't see any CommonSymbols.)
class CommonSymbol : public Symbol {
public:
  CommonSymbol(InputFile *file, StringRef name, uint8_t binding,
               uint8_t stOther, uint8_t type, uint64_t alignment, uint64_t size)
      : Symbol(CommonKind, file, name, binding, stOther, type),
        alignment(alignment), size(size) {
    exportDynamic = config->exportDynamic;
  }

  static bool classof(const Symbol *s) { return s->isCommon(); }

  uint32_t alignment;
  uint64_t size;
};

class Undefined : public Symbol {
public:
  Undefined(InputFile *file, StringRef name, uint8_t binding, uint8_t stOther,
            uint8_t type, uint32_t discardedSecIdx = 0)
      : Symbol(UndefinedKind, file, name, binding, stOther, type),
        discardedSecIdx(discardedSecIdx) {}

  static bool classof(const Symbol *s) { return s->kind() == UndefinedKind; }

  // The section index if in a discarded section, 0 otherwise.
  uint32_t discardedSecIdx;
  bool nonPrevailing = false;
};

class SharedSymbol : public Symbol {
public:
  static bool classof(const Symbol *s) { return s->kind() == SharedKind; }

  SharedSymbol(InputFile &file, StringRef name, uint8_t binding,
               uint8_t stOther, uint8_t type, uint64_t value, uint64_t size,
               uint32_t alignment)
      : Symbol(SharedKind, &file, name, binding, stOther, type), value(value),
        size(size), alignment(alignment) {
    exportDynamic = true;
    // GNU ifunc is a mechanism to allow user-supplied functions to
    // resolve PLT slot values at load-time. This is contrary to the
    // regular symbol resolution scheme in which symbols are resolved just
    // by name. Using this hook, you can program how symbols are solved
    // for you program. For example, you can make "memcpy" to be resolved
    // to a SSE-enabled version of memcpy only when a machine running the
    // program supports the SSE instruction set.
    //
    // Naturally, such symbols should always be called through their PLT
    // slots. What GNU ifunc symbols point to are resolver functions, and
    // calling them directly doesn't make sense (unless you are writing a
    // loader).
    //
    // For DSO symbols, we always call them through PLT slots anyway.
    // So there's no difference between GNU ifunc and regular function
    // symbols if they are in DSOs. So we can handle GNU_IFUNC as FUNC.
    if (this->type == llvm::ELF::STT_GNU_IFUNC)
      this->type = llvm::ELF::STT_FUNC;
  }

  uint64_t value; // st_value
  uint64_t size;  // st_size
  uint32_t alignment;
};

// LazyObject symbols represent symbols in object files between --start-lib and
// --end-lib options. LLD also handles traditional archives as if all the files
// in the archive are surrounded by --start-lib and --end-lib.
//
// A special complication is the handling of weak undefined symbols. They should
// not load a file, but we have to remember we have seen both the weak undefined
// and the lazy. We represent that with a lazy symbol with a weak binding. This
// means that code looking for undefined symbols normally also has to take lazy
// symbols into consideration.
class LazyObject : public Symbol {
public:
  LazyObject(InputFile &file)
      : Symbol(LazyObjectKind, &file, {}, llvm::ELF::STB_GLOBAL,
               llvm::ELF::STV_DEFAULT, llvm::ELF::STT_NOTYPE) {}

  static bool classof(const Symbol *s) { return s->kind() == LazyObjectKind; }
};

// Some linker-generated symbols need to be created as
// Defined symbols.
struct ElfSym {
  // __bss_start
  static Defined *bss;

  // etext and _etext
  static Defined *etext1;
  static Defined *etext2;

  // edata and _edata
  static Defined *edata1;
  static Defined *edata2;

  // end and _end
  static Defined *end1;
  static Defined *end2;

  // The _GLOBAL_OFFSET_TABLE_ symbol is defined by target convention to
  // be at some offset from the base of the .got section, usually 0 or
  // the end of the .got.
  static Defined *globalOffsetTable;

  // _gp, _gp_disp and __gnu_local_gp symbols. Only for MIPS.
  static Defined *mipsGp;
  static Defined *mipsGpDisp;
  static Defined *mipsLocalGp;

  // __rel{,a}_iplt_{start,end} symbols.
  static Defined *relaIpltStart;
  static Defined *relaIpltEnd;

  // __global_pointer$ for RISC-V.
  static Defined *riscvGlobalPointer;

  // _TLS_MODULE_BASE_ on targets that support TLSDESC.
  static Defined *tlsModuleBase;
};

// A buffer class that is large enough to hold any Symbol-derived
// object. We allocate memory using this class and instantiate a symbol
// using the placement new.
union SymbolUnion {
  alignas(Defined) char a[sizeof(Defined)];
  alignas(CommonSymbol) char b[sizeof(CommonSymbol)];
  alignas(Undefined) char c[sizeof(Undefined)];
  alignas(SharedSymbol) char d[sizeof(SharedSymbol)];
  alignas(LazyObject) char e[sizeof(LazyObject)];
};

// It is important to keep the size of SymbolUnion small for performance and
// memory usage reasons. 64 bytes is a soft limit based on the size of Defined
// on a 64-bit system.
static_assert(sizeof(SymbolUnion) <= 64, "SymbolUnion too large");

template <typename T> struct AssertSymbol {
  static_assert(std::is_trivially_destructible<T>(),
                "Symbol types must be trivially destructible");
  static_assert(sizeof(T) <= sizeof(SymbolUnion), "SymbolUnion too small");
  static_assert(alignof(T) <= alignof(SymbolUnion),
                "SymbolUnion not aligned enough");
};

static inline void assertSymbols() {
  AssertSymbol<Defined>();
  AssertSymbol<CommonSymbol>();
  AssertSymbol<Undefined>();
  AssertSymbol<SharedSymbol>();
  AssertSymbol<LazyObject>();
}

void printTraceSymbol(const Symbol &sym, StringRef name);

size_t Symbol::getSymbolSize() const {
  switch (kind()) {
  case CommonKind:
    return sizeof(CommonSymbol);
  case DefinedKind:
    return sizeof(Defined);
  case LazyObjectKind:
    return sizeof(LazyObject);
  case SharedKind:
    return sizeof(SharedSymbol);
  case UndefinedKind:
    return sizeof(Undefined);
  case PlaceholderKind:
    return sizeof(Symbol);
  }
  llvm_unreachable("unknown symbol kind");
}

// replace() replaces "this" object with a given symbol by memcpy'ing
// it over to "this". This function is called as a result of name
// resolution, e.g. to replace an undefind symbol with a defined symbol.
void Symbol::replace(const Symbol &other) {
  Symbol old = *this;
  memcpy(this, &other, other.getSymbolSize());

  // old may be a placeholder. The referenced fields must be initialized in
  // SymbolTable::insert.
  nameData = old.nameData;
  nameSize = old.nameSize;
  partition = old.partition;
  visibility = old.visibility;
  isPreemptible = old.isPreemptible;
  isUsedInRegularObj = old.isUsedInRegularObj;
  exportDynamic = old.exportDynamic;
  inDynamicList = old.inDynamicList;
  referenced = old.referenced;
  traced = old.traced;
  hasVersionSuffix = old.hasVersionSuffix;
  scriptDefined = old.scriptDefined;
  versionId = old.versionId;

  // Print out a log message if --trace-symbol was specified.
  // This is for debugging.
  if (traced)
    printTraceSymbol(*this, getName());
}

template <typename... T> Defined *makeDefined(T &&...args) {
  return new (reinterpret_cast<Defined *>(
      getSpecificAllocSingleton<SymbolUnion>().Allocate()))
      Defined(std::forward<T>(args)...);
}

void reportDuplicate(const Symbol &sym, const InputFile *newFile,
                     InputSectionBase *errSec, uint64_t errOffset);
void maybeWarnUnorderableSymbol(const Symbol *sym);
bool computeIsPreemptible(const Symbol &sym);
void reportBackrefs();

} // namespace elf
} // namespace lld

#endif
