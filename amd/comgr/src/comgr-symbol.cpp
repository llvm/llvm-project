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
*******************************************************************************/

#include <iostream>
#include "comgr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/SymbolSize.h"
#include "amd_comgr.h"

using namespace llvm;
using namespace llvm::support;
using namespace COMGR;

SymbolContext::SymbolContext()
  : name(nullptr),
    type(AMD_COMGR_SYMBOL_TYPE_NOTYPE),
    size(0),
    undefined(true),
    value(0) {}

SymbolContext::~SymbolContext() {
  free(name);
}

amd_comgr_status_t SymbolContext::SetName(llvm::StringRef Name) {
  return SetCStr(name, Name);
}

amd_comgr_symbol_type_t
SymbolHelper::map_to_comgr_symbol_type(SymbolRef::Type stype, uint64_t flags)
{
  amd_comgr_symbol_type_t type;

  switch (stype) {
    case SymbolRef::ST_Unknown:
         type = AMD_COMGR_SYMBOL_TYPE_NOTYPE;
         break;
    case SymbolRef::ST_File:
         type = AMD_COMGR_SYMBOL_TYPE_FILE;
         break;
    case SymbolRef::ST_Function:
         type = AMD_COMGR_SYMBOL_TYPE_FUNC;
         break;
    case SymbolRef::ST_Data:
         if (flags & SymbolRef::SF_Common)
           type = AMD_COMGR_SYMBOL_TYPE_COMMON;
         else
           type = AMD_COMGR_SYMBOL_TYPE_OBJECT;
         break;
    case SymbolRef::ST_Debug:
         type = AMD_COMGR_SYMBOL_TYPE_SECTION;
         break;
    default:
         type = AMD_COMGR_SYMBOL_TYPE_NOTYPE;
         break;  // Please Check:
                 // Actually there is a ST_Other, the API may be missing
                 // a default amd_comgr_symbol_type_t here.
  }

  return type;
}

// SymbolHelper version of createBinary, contrary to the one in Binary.cpp,
// in_text is textual input, not a filename.
Expected<OwningBinary<Binary>> SymbolHelper::create_binary(StringRef in_text) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
    MemoryBuffer::getMemBuffer(in_text);
  if (std::error_code EC = BufOrErr.getError())
    return errorCodeToError(EC);
  std::unique_ptr<MemoryBuffer> &Buffer = BufOrErr.get();

  Expected<std::unique_ptr<Binary>> BinOrErr =
      createBinary( Buffer->getMemBufferRef());
  if (!BinOrErr)
    return BinOrErr.takeError();
  std::unique_ptr<Binary> &Bin = BinOrErr.get();

  return OwningBinary<Binary>(std::move(Bin), std::move(Buffer));
}

SymbolContext*
SymbolHelper::search_symbol(StringRef ins, const char *name, amd_comgr_data_kind_t kind)
{
  StringRef sname(name);

  Expected<OwningBinary<Binary>> BinaryOrErr = create_binary(ins);
  if (!BinaryOrErr) {
    return NULL;
  }

  Binary &Binary = *BinaryOrErr.get().getBinary();

  if (ObjectFile *obj = dyn_cast<ObjectFile>(&Binary)) {

    std::vector<SymbolRef> symbol_list ;
    symbol_list.clear();

    // extract the symbol list from dynsymtab or symtab
    if (const auto *E = dyn_cast<ELFObjectFileBase>(obj)) {
      if (kind == AMD_COMGR_DATA_KIND_EXECUTABLE) {
        // executable kind, search dynsymtab
        iterator_range<elf_symbol_iterator> dsyms = E->getDynamicSymbolIterators();
        for (ELFSymbolRef dsym : dsyms)
          symbol_list.push_back(dsym);

      } else if (kind == AMD_COMGR_DATA_KIND_RELOCATABLE) {
        // relocatable kind, search symtab
        auto syms = E->symbols();
        for (ELFSymbolRef sym : syms)
          symbol_list.push_back(sym);
      }
    }

    // Find symbol with specified name
    SymbolRef fsym;
    bool found = false;
    for (auto &symbol: symbol_list) {
      Expected<StringRef> symNameOrErr = symbol.getName();
      if (!symNameOrErr)
        return NULL;
      StringRef sym_name = *symNameOrErr;
      if (sym_name.equals(sname)) {
#if DEBUG
        outs() << "Found! " << sname.data() << "\n";
#endif
        fsym = symbol;
        found = true;
        break;
      }
    }

    if (!found)
       return NULL;

    // ATTENTION: Do not attempt to split out the above "find symbol" code
    // into a separate function returning a found SymbolRef. For some
    // unknown reason, maybe a gcc codegen bug, at the return of the
    // SymbolRef, the very beginning code "create_binary" will be called
    // again unexpectedly, corrupting memory used by the returned SymbolRef.
    // I also suspect it's the OwningBinary of create_binary causing the
    // problem, but basically the reason is unknown.

    // Found the specified symbol, fill the SymbolContext values
    SymbolContext *symp = new (std::nothrow) SymbolContext();
    if (symp == NULL)
      return NULL;   // out of space

    symp->SetName(name);
    symp->value = fsym.getValue();

    Expected<SymbolRef::Type> TypeOrErr = fsym.getType();
    if (!TypeOrErr)
      return NULL;

    // get flags in symbol
    // SymbolRef does not directly use ELF::STT_<types>, it maps them to
    // SymbolRef::ST_<types> in ELFObjectFile<ELFT>::getSymbolType().
    DataRefImpl symb = fsym.getRawDataRefImpl();
    uint64_t flags = fsym.getObject()->getSymbolFlags(symb);
    symp->type = map_to_comgr_symbol_type(*TypeOrErr, flags);

    // symbol size
    ELFSymbolRef esym(fsym);
    symp->size = esym.getSize();

    // symbol undefined?
    if (flags & SymbolRef::SF_Undefined)
      symp->undefined = true;
    else
      symp->undefined = false;

    return symp;
  }

  return NULL;
}

amd_comgr_status_t SymbolHelper::iterate_table(
    StringRef ins, amd_comgr_data_kind_t kind,
    amd_comgr_status_t (*callback)(amd_comgr_symbol_t, void *),
    void *user_data) {
  Expected<OwningBinary<Binary>> BinaryOrErr = create_binary(ins);
  if (!BinaryOrErr) {
    return AMD_COMGR_STATUS_ERROR;
  }

  Binary &Binary = *BinaryOrErr.get().getBinary();

  if (ObjectFile *obj = dyn_cast<ObjectFile>(&Binary)) {

    std::vector<SymbolRef> symbol_list ;
    symbol_list.clear();

    // extract the symbol list from dynsymtab or symtab
    if (const auto *E = dyn_cast<ELFObjectFileBase>(obj)) {
      if (kind == AMD_COMGR_DATA_KIND_EXECUTABLE) {
        // executable kind, search dynsymtab
        iterator_range<elf_symbol_iterator> dsyms = E->getDynamicSymbolIterators();
        for (ELFSymbolRef dsym : dsyms)
          symbol_list.push_back(dsym);

      } else if (kind == AMD_COMGR_DATA_KIND_RELOCATABLE) {
        // relocatable kind, search symtab
        auto syms = E->symbols();
        for (ELFSymbolRef sym : syms)
          symbol_list.push_back(sym);
      }
    }

    // iterate all symbols
    for (auto &symbol: symbol_list) {
      // create symbol context
      SymbolContext *ctxp = new (std::nothrow) SymbolContext();
      if (ctxp == NULL)
        return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

      // get name
      Expected<StringRef> symNameOrErr = symbol.getName();
      if (!symNameOrErr)
        return AMD_COMGR_STATUS_ERROR;
      StringRef sym_name = *symNameOrErr;
      ctxp->SetName(sym_name);
      ctxp->value = symbol.getValue();

      // get type
      Expected<SymbolRef::Type> TypeOrErr = symbol.getType();
      if (!TypeOrErr)
        return AMD_COMGR_STATUS_ERROR;
      DataRefImpl symb = symbol.getRawDataRefImpl();
      uint64_t flags = symbol.getObject()->getSymbolFlags(symb);
      ctxp->type = map_to_comgr_symbol_type(*TypeOrErr, flags);

      // get size
      ELFSymbolRef esym(symbol);
      ctxp->size = esym.getSize();

      // set undefined
      ctxp->undefined = (flags & SymbolRef::SF_Undefined) ? true : false;

      // create amd_comgr_symbol_t
      COMGR::DataSymbol *symp = new (std::nothrow) COMGR::DataSymbol(ctxp);
      if (symp == NULL)
        return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
      amd_comgr_symbol_t symt = COMGR::DataSymbol::Convert(symp);

      // invoke callback(symbol, user_data)
      (*callback)(symt, user_data);

      // delete symt completely to avoid memory leak,
      // user needs to save if necessary in callback
      delete symp;

    } // next symbol in list

    return AMD_COMGR_STATUS_SUCCESS;
  } // ObjectFile

  return AMD_COMGR_STATUS_ERROR;
}
