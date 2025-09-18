//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_CONFIG_H
#define LLD_WASM_CONFIG_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/Support/CachePruning.h"
#include <optional>

namespace llvm {
enum class CodeGenOptLevel;
} // namespace llvm

namespace lld::wasm {

class InputFile;
class StubFile;
class ObjFile;
class SharedFile;
class BitcodeFile;
class InputTable;
class InputGlobal;
class InputFunction;
class Symbol;
class DefinedData;
class GlobalSymbol;
class DefinedFunction;
class UndefinedGlobal;
class TableSymbol;

// For --unresolved-symbols.
enum class UnresolvedPolicy { ReportError, Warn, Ignore, ImportDynamic };

// For --build-id.
enum class BuildIdKind { None, Fast, Sha1, Hexstring, Uuid };

// This struct contains the global configuration for the linker.
// Most fields are direct mapping from the command line options
// and such fields have the same name as the corresponding options.
// Most fields are initialized by the driver.
struct Config {
  bool allowMultipleDefinition;
  bool bsymbolic;
  bool checkFeatures;
  bool compressRelocations;
  bool demangle;
  bool disableVerify;
  bool experimentalPic;
  bool emitRelocs;
  bool exportAll;
  bool exportDynamic;
  bool exportTable;
  bool extendedConst;
  bool growableTable;
  bool gcSections;
  llvm::StringSet<> keepSections;
  std::optional<std::pair<llvm::StringRef, llvm::StringRef>> memoryImport;
  std::optional<llvm::StringRef> memoryExport;
  bool sharedMemory;
  bool importTable;
  bool importUndefined;
  std::optional<bool> is64;
  bool mergeDataSegments;
  bool noinhibitExec;
  bool pie;
  bool printGcSections;
  bool relocatable;
  bool saveTemps;
  bool shared;
  bool shlibSigCheck;
  bool stripAll;
  bool stripDebug;
  bool stackFirst;
  // Because dyamanic linking under Wasm is still experimental we default to
  // static linking
  bool isStatic = true;
  bool thinLTOEmitImportsFiles;
  bool thinLTOEmitIndexFiles;
  bool thinLTOIndexOnly;
  bool trace;
  uint64_t globalBase;
  uint64_t initialHeap;
  uint64_t initialMemory;
  uint64_t maxMemory;
  bool noGrowableMemory;
  // The table offset at which to place function addresses.  We reserve zero
  // for the null function pointer.  This gets set to 1 for executables and 0
  // for shared libraries (since they always added to a dynamic offset at
  // runtime).
  uint64_t tableBase;
  uint64_t zStackSize;
  uint64_t pageSize;
  unsigned ltoPartitions;
  unsigned ltoo;
  llvm::CodeGenOptLevel ltoCgo;
  unsigned optimize;
  bool ltoDebugPassManager;
  UnresolvedPolicy unresolvedSymbols;
  BuildIdKind buildId = BuildIdKind::None;

  llvm::StringRef entry;
  llvm::StringRef ltoObjPath;
  llvm::StringRef mapFile;
  llvm::StringRef outputFile;
  llvm::StringRef soName;
  llvm::StringRef thinLTOCacheDir;
  llvm::StringRef thinLTOJobs;
  llvm::StringRef thinLTOIndexOnlyArg;
  std::pair<llvm::StringRef, llvm::StringRef> thinLTOObjectSuffixReplace;
  llvm::StringRef thinLTOPrefixReplaceOld;
  llvm::StringRef thinLTOPrefixReplaceNew;
  llvm::StringRef thinLTOPrefixReplaceNativeObject;
  llvm::StringRef whyExtract;

  llvm::StringSet<> allowUndefinedSymbols;
  llvm::StringSet<> exportedSymbols;
  std::vector<llvm::StringRef> requiredExports;
  llvm::SmallVector<llvm::StringRef, 0> searchPaths;
  llvm::SmallVector<llvm::StringRef, 0> rpath;
  llvm::CachePruningPolicy thinLTOCachePolicy;
  std::optional<std::vector<std::string>> features;
  std::optional<std::vector<std::string>> extraFeatures;
  llvm::SmallVector<uint8_t, 0> buildIdVector;
};

// The Ctx object hold all other (non-configuration) global state.
struct Ctx {
  Config arg;

  llvm::SmallVector<ObjFile *, 0> objectFiles;
  llvm::SmallVector<StubFile *, 0> stubFiles;
  llvm::SmallVector<SharedFile *, 0> sharedFiles;
  llvm::SmallVector<BitcodeFile *, 0> bitcodeFiles;
  llvm::SmallVector<BitcodeFile *, 0> lazyBitcodeFiles;
  llvm::SmallVector<InputFunction *, 0> syntheticFunctions;
  llvm::SmallVector<InputGlobal *, 0> syntheticGlobals;
  llvm::SmallVector<InputTable *, 0> syntheticTables;

  // linker-generated symbols
  struct WasmSym {
    // __global_base
    // Symbol marking the start of the global section.
    DefinedData *globalBase;

    // __stack_pointer/__stack_low/__stack_high
    // Global that holds current value of stack pointer and data symbols marking
    // the start and end of the stack region.  stackPointer is initialized to
    // stackHigh and grows downwards towards stackLow
    GlobalSymbol *stackPointer;
    DefinedData *stackLow;
    DefinedData *stackHigh;

    // __tls_base
    // Global that holds the address of the base of the current thread's
    // TLS block.
    GlobalSymbol *tlsBase;

    // __tls_size
    // Symbol whose value is the size of the TLS block.
    GlobalSymbol *tlsSize;

    // __tls_size
    // Symbol whose value is the alignment of the TLS block.
    GlobalSymbol *tlsAlign;

    // __data_end
    // Symbol marking the end of the data and bss.
    DefinedData *dataEnd;

    // __heap_base/__heap_end
    // Symbols marking the beginning and end of the "heap". It starts at the end
    // of the data, bss and explicit stack, and extends to the end of the linear
    // memory allocated by wasm-ld. This region of memory is not used by the
    // linked code, so it may be used as a backing store for `sbrk` or `malloc`
    // implementations.
    DefinedData *heapBase;
    DefinedData *heapEnd;

    // __wasm_first_page_end
    // A symbol whose address is the end of the first page in memory (if any).
    DefinedData *firstPageEnd;

    // __wasm_init_memory_flag
    // Symbol whose contents are nonzero iff memory has already been
    // initialized.
    DefinedData *initMemoryFlag;

    // __wasm_init_memory
    // Function that initializes passive data segments during instantiation.
    DefinedFunction *initMemory;

    // __wasm_call_ctors
    // Function that directly calls all ctors in priority order.
    DefinedFunction *callCtors;

    // __wasm_call_dtors
    // Function that calls the libc/etc. cleanup function.
    DefinedFunction *callDtors;

    // __wasm_apply_global_relocs
    // Function that applies relocations to wasm globals post-instantiation.
    // Unlike __wasm_apply_data_relocs this needs to run on every thread.
    DefinedFunction *applyGlobalRelocs;

    // __wasm_apply_tls_relocs
    // Like __wasm_apply_data_relocs but for TLS section.  These must be
    // delayed until __wasm_init_tls.
    DefinedFunction *applyTLSRelocs;

    // __wasm_apply_global_tls_relocs
    // Like applyGlobalRelocs but for globals that hold TLS addresses.  These
    // must be delayed until __wasm_init_tls.
    DefinedFunction *applyGlobalTLSRelocs;

    // __wasm_init_tls
    // Function that allocates thread-local storage and initializes it.
    DefinedFunction *initTLS;

    // Pointer to the function that is to be used in the start section.
    // (normally an alias of initMemory, or applyGlobalRelocs).
    DefinedFunction *startFunction;

    // __dso_handle
    // Symbol used in calls to __cxa_atexit to determine current DLL
    DefinedData *dsoHandle;

    // __table_base
    // Used in PIC code for offset of indirect function table
    UndefinedGlobal *tableBase;
    DefinedData *definedTableBase;

    // __memory_base
    // Used in PIC code for offset of global data
    UndefinedGlobal *memoryBase;
    DefinedData *definedMemoryBase;

    // __indirect_function_table
    // Used as an address space for function pointers, with each function that
    // is used as a function pointer being allocated a slot.
    TableSymbol *indirectFunctionTable;
  };
  WasmSym sym;

  // True if we are creating position-independent code.
  bool isPic = false;

  // True if we have an MVP input that uses __indirect_function_table and which
  // requires it to be allocated to table number 0.
  bool legacyFunctionTable = false;

  // Will be set to true if bss data segments should be emitted. In most cases
  // this is not necessary.
  bool emitBssSegments = false;

  // A tuple of (reference, extractedFile, sym). Used by --why-extract=.
  llvm::SmallVector<std::tuple<std::string, const InputFile *, const Symbol &>,
                    0>
      whyExtractRecords;

  Ctx();
  void reset();
};

extern Ctx ctx;

void errorOrWarn(const llvm::Twine &msg);

} // namespace lld::wasm

#endif
