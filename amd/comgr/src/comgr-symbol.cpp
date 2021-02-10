/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2003-2017 University of Illinois at Urbana-Champaign.
 * Modifications (c) 2018 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of the LLVM Team, University of Illinois at
 *       Urbana-Champaign, nor the names of its contributors may be used to
 *       endorse or promote products derived from this Software without specific
 *       prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include "amd_comgr.h"
#include "comgr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/SymbolSize.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support;
using namespace COMGR;

SymbolContext::SymbolContext()
    : Name(nullptr), Type(AMD_COMGR_SYMBOL_TYPE_NOTYPE), Size(0),
      Undefined(true), Value(0) {}

SymbolContext::~SymbolContext() { free(Name); }

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

// SymbolHelper version of createBinary, contrary to the one in Binary.cpp,
// in_text is textual input, not a filename.
Expected<OwningBinary<Binary>> SymbolHelper::createBinary(StringRef InText) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getMemBuffer(InText);
  if (std::error_code EC = BufOrErr.getError()) {
    return errorCodeToError(EC);
  }
  std::unique_ptr<MemoryBuffer> &Buffer = BufOrErr.get();

  Expected<std::unique_ptr<Binary>> BinOrErr =
      llvm::object::createBinary(Buffer->getMemBufferRef());
  if (!BinOrErr) {
    return BinOrErr.takeError();
  }
  std::unique_ptr<Binary> &Bin = BinOrErr.get();

  return OwningBinary<Binary>(std::move(Bin), std::move(Buffer));
}

SymbolContext *SymbolHelper::createBinary(StringRef Ins, const char *Name,
                                          amd_comgr_data_kind_t Kind) {
  StringRef Sname(Name);

  Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(Ins);
  if (!BinaryOrErr) {
    return NULL;
  }

  Binary &Binary = *BinaryOrErr.get().getBinary();

  if (ObjectFile *Obj = dyn_cast<ObjectFile>(&Binary)) {

    std::vector<SymbolRef> SymbolList;
    SymbolList.clear();

    // extract the symbol list from dynsymtab or symtab
    if (const auto *E = dyn_cast<ELFObjectFileBase>(Obj)) {
      if (Kind == AMD_COMGR_DATA_KIND_EXECUTABLE) {
        // executable kind, search dynsymtab
        iterator_range<elf_symbol_iterator> Dsyms =
            E->getDynamicSymbolIterators();
        for (ELFSymbolRef Dsym : Dsyms) {
          SymbolList.push_back(Dsym);
        }

      } else if (Kind == AMD_COMGR_DATA_KIND_RELOCATABLE) {
        // relocatable kind, search symtab
        auto Syms = E->symbols();
        for (ELFSymbolRef Sym : Syms) {
          SymbolList.push_back(Sym);
        }
      }
    }

    // Find symbol with specified name
    SymbolRef Fsym;
    bool Found = false;
    for (auto &Symbol : SymbolList) {
      Expected<StringRef> SymNameOrErr = Symbol.getName();
      if (!SymNameOrErr) {
        return NULL;
      }
      StringRef SymName = *SymNameOrErr;
      if (SymName.equals(Sname)) {
#if DEBUG
        outs() << "Found! " << sname.data() << "\n";
#endif
        Fsym = Symbol;
        Found = true;
        break;
      }
    }

    if (!Found) {
      return NULL;
    }

    // ATTENTION: Do not attempt to split out the above "find symbol" code
    // into a separate function returning a found SymbolRef. For some
    // unknown reason, maybe a gcc codegen bug, at the return of the
    // SymbolRef, the very beginning code "create_binary" will be called
    // again unexpectedly, corrupting memory used by the returned SymbolRef.
    // I also suspect it's the OwningBinary of create_binary causing the
    // problem, but basically the reason is unknown.

    // Found the specified symbol, fill the SymbolContext values
    std::unique_ptr<SymbolContext> Symp(new (std::nothrow) SymbolContext());
    if (!Symp) {
      return NULL;
    }

    Symp->setName(Name);
    auto ExpectedFsymValue = Fsym.getValue();
    if (!ExpectedFsymValue) {
      return NULL;
    }
    Symp->Value = ExpectedFsymValue.get();

    DataRefImpl Symb = Fsym.getRawDataRefImpl();
    auto Flags = Fsym.getObject()->getSymbolFlags(Symb);
    if (!Flags) {
      return NULL;
    }

    // symbol size
    ELFSymbolRef Esym(Fsym);
    Symp->Size = Esym.getSize();
    Symp->Type = mapToComgrSymbolType(Esym.getELFType());

    // symbol undefined?
    if (*Flags & SymbolRef::SF_Undefined) {
      Symp->Undefined = true;
    } else {
      Symp->Undefined = false;
    }

    return Symp.release();
  }

  return NULL;
}

amd_comgr_status_t SymbolHelper::iterateTable(
    StringRef Ins, amd_comgr_data_kind_t Kind,
    amd_comgr_status_t (*Callback)(amd_comgr_symbol_t, void *),
    void *UserData) {
  Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(Ins);
  if (!BinaryOrErr) {
    return AMD_COMGR_STATUS_ERROR;
  }

  Binary &Binary = *BinaryOrErr.get().getBinary();

  if (ObjectFile *Obj = dyn_cast<ObjectFile>(&Binary)) {

    std::vector<SymbolRef> SymbolList;
    SymbolList.clear();

    // extract the symbol list from dynsymtab or symtab
    if (const auto *E = dyn_cast<ELFObjectFileBase>(Obj)) {
      if (Kind == AMD_COMGR_DATA_KIND_EXECUTABLE) {
        // executable kind, search dynsymtab
        iterator_range<elf_symbol_iterator> Dsyms =
            E->getDynamicSymbolIterators();
        for (ELFSymbolRef Dsym : Dsyms) {
          SymbolList.push_back(Dsym);
        }

      } else if (Kind == AMD_COMGR_DATA_KIND_RELOCATABLE) {
        // relocatable kind, search symtab
        auto Syms = E->symbols();
        for (ELFSymbolRef Sym : Syms) {
          SymbolList.push_back(Sym);
        }
      }
    }

    for (auto &Symbol : SymbolList) {
      std::unique_ptr<SymbolContext> Ctxp(new (std::nothrow) SymbolContext());
      if (!Ctxp) {
        return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
      }

      Expected<StringRef> SymNameOrErr = Symbol.getName();
      if (!SymNameOrErr) {
        return AMD_COMGR_STATUS_ERROR;
      }
      StringRef SymName = *SymNameOrErr;
      Ctxp->setName(SymName);
      auto ExpectedSymbolValue = Symbol.getValue();
      if (!ExpectedSymbolValue) {
        return AMD_COMGR_STATUS_ERROR;
      }
      Ctxp->Value = ExpectedSymbolValue.get();

      Expected<SymbolRef::Type> TypeOrErr = Symbol.getType();
      if (!TypeOrErr) {
        return AMD_COMGR_STATUS_ERROR;
      }
      DataRefImpl Symb = Symbol.getRawDataRefImpl();
      auto Flags = Symbol.getObject()->getSymbolFlags(Symb);
      if (!Flags) {
        return AMD_COMGR_STATUS_ERROR;
      }

      ELFSymbolRef Esym(Symbol);
      Ctxp->Size = Esym.getSize();
      Ctxp->Type = mapToComgrSymbolType(Esym.getELFType());

      Ctxp->Undefined = (*Flags & SymbolRef::SF_Undefined) ? true : false;

      std::unique_ptr<COMGR::DataSymbol> Symp(
          new (std::nothrow) COMGR::DataSymbol(Ctxp.release()));
      if (!Symp) {
        return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
      }
      amd_comgr_symbol_t Symt = COMGR::DataSymbol::convert(Symp.get());

      (*Callback)(Symt, UserData);
    }

    return AMD_COMGR_STATUS_SUCCESS;
  } // ObjectFile

  return AMD_COMGR_STATUS_ERROR;
}
