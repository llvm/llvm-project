//===- comgr-symbol.cpp - Symbol lookup -----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements helper functions for the amd_comgr_iterate_symbols()
/// and amd_comgr_symbol_lookup() APIs.
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "comgr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/SymbolSize.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support;
using namespace COMGR;

amd_comgr_status_t COMGR::setCStr(char *&Dest, StringRef Src, size_t *Size) {
  free(Dest);
  Dest = reinterpret_cast<char *>(malloc(Src.size() + 1));
  if (!Dest) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  memcpy(Dest, Src.data(), Src.size());
  Dest[Src.size()] = '\0';
  if (Size) {
    *Size = Src.size();
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

SymbolContext::SymbolContext()
    : Name(nullptr), Type(AMD_COMGR_SYMBOL_TYPE_NOTYPE), Size(0),
      Undefined(true), Value(0) {}

SymbolContext::~SymbolContext() { free(Name); }

DataSymbol::DataSymbol(SymbolContext *DataSym) : DataSym(DataSym) {}
DataSymbol::~DataSymbol() { delete DataSym; }

amd_comgr_status_t SymbolContext::setName(llvm::StringRef Name) {
  return setCStr(this->Name, Name);
}

amd_comgr_symbol_type_t
SymbolHelper::mapToComgrSymbolType(uint8_t ELFSymbolType) {
  switch (ELFSymbolType) {
  case ELF::STT_NOTYPE:
    return AMD_COMGR_SYMBOL_TYPE_NOTYPE;
  case ELF::STT_OBJECT:
    return AMD_COMGR_SYMBOL_TYPE_OBJECT;
  case ELF::STT_FUNC:
    return AMD_COMGR_SYMBOL_TYPE_FUNC;
  case ELF::STT_SECTION:
    return AMD_COMGR_SYMBOL_TYPE_SECTION;
  case ELF::STT_FILE:
    return AMD_COMGR_SYMBOL_TYPE_FILE;
  case ELF::STT_COMMON:
    return AMD_COMGR_SYMBOL_TYPE_COMMON;
  case ELF::STT_AMDGPU_HSA_KERNEL:
    return AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL;
  default:
    return AMD_COMGR_SYMBOL_TYPE_UNKNOWN;
  }
}

ObjectFile *SymbolHelper::getCachedObjectFile(DataObject *DataP) {
  std::scoped_lock<std::mutex> CacheLock(DataP->CacheMutex);

  if (!DataP->CachedBinary) {
    MemoryBufferRef MBRef(StringRef(DataP->Data, DataP->Size), "");
    Expected<std::unique_ptr<Binary>> BinOrErr =
        llvm::object::createBinary(MBRef);
    if (!BinOrErr) {
      consumeError(BinOrErr.takeError());
      return nullptr;
    }
    DataP->CachedBinary = std::move(*BinOrErr);
  }

  return dyn_cast<ObjectFile>(DataP->CachedBinary.get());
}

const StringMap<SymbolInfo> *SymbolHelper::getSymbolIndex(DataObject *DataP) {
  // Outside the lock: getCachedObjectFile takes CacheMutex itself.
  ObjectFile *Obj = getCachedObjectFile(DataP);
  if (!Obj) {
    return nullptr;
  }

  std::scoped_lock<std::mutex> CacheLock(DataP->CacheMutex);

  if (DataP->SymbolIndex) {
    return DataP->SymbolIndex.get();
  }

  auto Index = std::make_unique<StringMap<SymbolInfo>>();

  if (const auto *E = dyn_cast<ELFObjectFileBase>(Obj)) {
    auto Add = [&](const ELFSymbolRef &Sym) {
      Expected<StringRef> NameOrErr = Sym.getName();
      if (!NameOrErr) {
        consumeError(NameOrErr.takeError());
        return;
      }
      Expected<uint64_t> ValueOrErr = Sym.getValue();
      if (!ValueOrErr) {
        consumeError(ValueOrErr.takeError());
        return;
      }
      Expected<uint32_t> FlagsOrErr =
          Sym.getObject()->getSymbolFlags(Sym.getRawDataRefImpl());
      if (!FlagsOrErr) {
        consumeError(FlagsOrErr.takeError());
        return;
      }

      SymbolInfo Info;
      Info.Value = *ValueOrErr;
      Info.Size = Sym.getSize();
      Info.Type = mapToComgrSymbolType(Sym.getELFType());
      Info.Undefined = (*FlagsOrErr & SymbolRef::SF_Undefined) != 0;

      // First occurrence must win, as the linear scan this replaces did.
      Index->try_emplace(*NameOrErr, Info);
    };

    if (DataP->DataKind == AMD_COMGR_DATA_KIND_EXECUTABLE) {
      for (ELFSymbolRef Dsym : E->getDynamicSymbolIterators()) {
        Add(Dsym);
      }
    } else if (DataP->DataKind == AMD_COMGR_DATA_KIND_RELOCATABLE) {
      for (ELFSymbolRef Sym : E->symbols()) {
        Add(Sym);
      }
    }
  }

  DataP->SymbolIndex = std::move(Index);
  return DataP->SymbolIndex.get();
}

SymbolContext *SymbolHelper::createBinary(DataObject *DataP, const char *Name) {
  const StringMap<SymbolInfo> *Index = getSymbolIndex(DataP);
  if (!Index) {
    return nullptr;
  }

  auto It = Index->find(StringRef(Name));
  if (It == Index->end()) {
    return nullptr;
  }

  std::unique_ptr<SymbolContext> Symp(new (std::nothrow) SymbolContext());
  if (!Symp) {
    return nullptr;
  }

  if (Symp->setName(Name) != AMD_COMGR_STATUS_SUCCESS) {
    return nullptr;
  }

  const SymbolInfo &Info = It->second;
  Symp->Value = Info.Value;
  Symp->Size = Info.Size;
  Symp->Type = Info.Type;
  Symp->Undefined = Info.Undefined;

  return Symp.release();
}

amd_comgr_status_t SymbolHelper::iterateTable(
    DataObject *DataP,
    amd_comgr_status_t (*Callback)(amd_comgr_symbol_t, void *),
    void *UserData) {
  ObjectFile *Obj = getCachedObjectFile(DataP);
  if (!Obj) {
    return AMD_COMGR_STATUS_ERROR;
  }

  const auto *E = dyn_cast<ELFObjectFileBase>(Obj);
  if (!E) {
    return AMD_COMGR_STATUS_SUCCESS;
  }

  SmallVector<ELFSymbolRef, 32> SymbolList;
  if (DataP->DataKind == AMD_COMGR_DATA_KIND_EXECUTABLE) {
    for (ELFSymbolRef Dsym : E->getDynamicSymbolIterators()) {
      SymbolList.push_back(Dsym);
    }
  } else if (DataP->DataKind == AMD_COMGR_DATA_KIND_RELOCATABLE) {
    for (ELFSymbolRef Sym : E->symbols()) {
      SymbolList.push_back(Sym);
    }
  }

  for (const ELFSymbolRef &Symbol : SymbolList) {
    std::unique_ptr<SymbolContext> Ctxp(new (std::nothrow) SymbolContext());
    if (!Ctxp) {
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    Expected<StringRef> SymNameOrErr = Symbol.getName();
    if (!SymNameOrErr) {
      consumeError(SymNameOrErr.takeError());
      return AMD_COMGR_STATUS_ERROR;
    }
    if (Ctxp->setName(*SymNameOrErr) != AMD_COMGR_STATUS_SUCCESS) {
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    Expected<uint64_t> ValueOrErr = Symbol.getValue();
    if (!ValueOrErr) {
      consumeError(ValueOrErr.takeError());
      return AMD_COMGR_STATUS_ERROR;
    }
    Ctxp->Value = *ValueOrErr;

    Expected<uint32_t> FlagsOrErr =
        Symbol.getObject()->getSymbolFlags(Symbol.getRawDataRefImpl());
    if (!FlagsOrErr) {
      consumeError(FlagsOrErr.takeError());
      return AMD_COMGR_STATUS_ERROR;
    }

    Ctxp->Size = Symbol.getSize();
    Ctxp->Type = mapToComgrSymbolType(Symbol.getELFType());
    Ctxp->Undefined = (*FlagsOrErr & SymbolRef::SF_Undefined) != 0;

    std::unique_ptr<COMGR::DataSymbol> Symp(
        new (std::nothrow) COMGR::DataSymbol(Ctxp.release()));
    if (!Symp) {
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    (*Callback)(COMGR::DataSymbol::convert(Symp.get()), UserData);
  }

  return AMD_COMGR_STATUS_SUCCESS;
}
