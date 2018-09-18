/*******************************************************************************
*
* University of Illinois/NCSA
* Open Source License
*
* Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
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
*     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
*       contributors may be used to endorse or promote products derived from
*       this Software without specific prior written permission.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
* THE SOFTWARE.
*
*******************************************************************************/

#include "comgr-metadata.h"
#include "comgr-msgpack.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace llvm;

namespace COMGR {
namespace metadata {

template <class ELFT>
static amd_comgr_status_t
getElfMetadataRoot(const std::unique_ptr<ELFObjectFile<ELFT>> Obj,
                   DataMeta *metap) {
  const ELFFile<ELFT> *ELFFile = Obj->getELFFile();

  amd_comgr_status_t Status;
  bool Found = false;

  using Elf_Note = typename ELFT::Note;
  auto ProcessNote = [&](const Elf_Note &Note) {
    if (Note.getName() == "AMD") {
      auto DescString =
          StringRef(reinterpret_cast<const char *>(Note.getDesc().data()),
                    Note.getDesc().size());
      if (Note.getType() == ELF::NT_AMD_AMDGPU_HSA_METADATA) {
        Found = true;
        metap->node = YAML::Load(DescString);
        return AMD_COMGR_STATUS_SUCCESS;
      } else if (Note.getType() == 13) {
        Found = true;
        llvm::msgpack::Reader MPReader(DescString);
        return COMGR::msgpack::parse(MPReader, metap->msgpack_node);
      }
    }
    return AMD_COMGR_STATUS_SUCCESS;
  };

  auto ProgramHeadersOrError = ELFFile->program_headers();
  if (errorToBool(ProgramHeadersOrError.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (const auto &Phdr : *ProgramHeadersOrError) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;
    Error Err = Error::success();
    for (const auto &Note : ELFFile->notes(Phdr, Err)) {
      Status = ProcessNote(Note);
      if (Found)
        break;
    }
    if (errorToBool(std::move(Err)))
      return AMD_COMGR_STATUS_ERROR;
    if (Found)
      return Status;
  }

  auto SectionsOrError = ELFFile->sections();
  if (errorToBool(SectionsOrError.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (const auto &Shdr : *SectionsOrError) {
    if (Shdr.sh_type != ELF::SHT_NOTE)
      continue;
    Error Err = Error::success();
    for (const auto &Note : ELFFile->notes(Shdr, Err)) {
      Status = ProcessNote(Note);
      if (Found)
        break;
    }
    if (errorToBool(std::move(Err)))
      return AMD_COMGR_STATUS_ERROR;
    if (Found)
      return Status;
  }

  // We did not find the metadata note.
  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}

amd_comgr_status_t getMetadataRoot(DataObject *datap, DataMeta *metap) {
  std::unique_ptr<MemoryBuffer> Buf =
      MemoryBuffer::getMemBuffer(StringRef(datap->data, datap->size));

  Expected<std::unique_ptr<ObjectFile>> ObjOrErr =
      ObjectFile::createELFObjectFile(*Buf);

  if (errorToBool(ObjOrErr.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  std::unique_ptr<ELFObjectFileBase> Obj =
      unique_dyn_cast<ELFObjectFileBase>(std::move(*ObjOrErr));

  if (auto ELF32LE = unique_dyn_cast<ELF32LEObjectFile>(Obj))
    return getElfMetadataRoot(std::move(ELF32LE), metap);
  if (auto ELF64LE = unique_dyn_cast<ELF64LEObjectFile>(Obj))
    return getElfMetadataRoot(std::move(ELF64LE), metap);
  if (auto ELF32BE = unique_dyn_cast<ELF32BEObjectFile>(Obj))
    return getElfMetadataRoot(std::move(ELF32BE), metap);
  auto ELF64BE = unique_dyn_cast<ELF64BEObjectFile>(Obj);
  return getElfMetadataRoot(std::move(ELF64BE), metap);
}

static const char *SupportedIsas[] = {
#define HANDLE_ISA(NAME, ...) NAME,
#include "comgr-isa-metadata.def"
};

struct IsaInfo {
  bool TrapHandlerEnabled;
  unsigned LocalMemorySize;
  unsigned EUsPerCU;
  unsigned MaxWavesPerCU;
  unsigned MaxFlatWorkGroupSize;
  unsigned SGPRAllocGranule;
  unsigned TotalNumSGPRs;
  unsigned AddressableNumSGPRs;
  unsigned VGPRAllocGranule;
  unsigned TotalNumVGPRs;
  unsigned AddressableNumVGPRs;
  unsigned LDSBankCount;
} IsaInfos[] = {
#define HANDLE_ISA(NAME, TRAP_HANDLER_ENABLED, LOCAL_MEMORY_SIZE, EUS_PER_CU,  \
                   MAX_WAVES_PER_CU, MAX_FLAT_WORK_GROUP_SIZE,                 \
                   SGPR_ALLOC_GRANULE, TOTAL_NUM_SGPRS, ADDRESSABLE_NUM_SGPRS, \
                   VGPR_ALLOC_GRANULE, TOTAL_NUM_VGPRS, ADDRESSABLE_NUM_VGPRS, \
                   LDS_BANK_COUNT)                                             \
  {TRAP_HANDLER_ENABLED, LOCAL_MEMORY_SIZE,        EUS_PER_CU,                 \
   MAX_WAVES_PER_CU,     MAX_FLAT_WORK_GROUP_SIZE, SGPR_ALLOC_GRANULE,         \
   TOTAL_NUM_SGPRS,      ADDRESSABLE_NUM_SGPRS,    VGPR_ALLOC_GRANULE,         \
   TOTAL_NUM_VGPRS,      ADDRESSABLE_NUM_VGPRS,    LDS_BANK_COUNT},
#include "comgr-isa-metadata.def"
};

static_assert((sizeof(SupportedIsas) / sizeof(*SupportedIsas)) == (sizeof(IsaInfos) / sizeof(*IsaInfos)), "all SupportedIsas must have matching IsaInfos");

size_t getIsaCount() {
  return sizeof(SupportedIsas) / sizeof(*SupportedIsas);
}

const char *getIsaName(size_t Index) { return SupportedIsas[Index]; }

amd_comgr_status_t getIsaMetadata(StringRef IsaName, DataMeta *Meta) {
  auto IsaIterator = find(SupportedIsas, IsaName);
  if (IsaIterator == std::end(SupportedIsas))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  auto IsaIndex = std::distance(std::begin(SupportedIsas), IsaIterator);

  TargetIdentifier Ident;
  if (auto Status = ParseTargetIdentifier(IsaName, Ident))
    return Status;

  auto Root = std::make_shared<COMGR::msgpack::Map>();
  Root->Elements[COMGR::msgpack::String::make("Name")] =
      COMGR::msgpack::String::make(IsaName);
  Root->Elements[COMGR::msgpack::String::make("Architecture")] =
      COMGR::msgpack::String::make(Ident.Arch);
  Root->Elements[COMGR::msgpack::String::make("Vendor")] =
      COMGR::msgpack::String::make(Ident.Vendor);
  Root->Elements[COMGR::msgpack::String::make("OS")] =
      COMGR::msgpack::String::make(Ident.OS);
  Root->Elements[COMGR::msgpack::String::make("Environment")] =
      COMGR::msgpack::String::make(Ident.Environ);
  Root->Elements[COMGR::msgpack::String::make("Processor")] =
      COMGR::msgpack::String::make(Ident.Processor);
  auto XNACKEnabled = false;
  auto FeaturesNode = std::make_shared<COMGR::msgpack::List>(Ident.Features.size());
  for (size_t I = 0; I < Ident.Features.size(); ++I) {
    FeaturesNode->Elements[I].reset(new COMGR::msgpack::String(Ident.Features[I]));
    if (Ident.Features[I] == "xnack")
      XNACKEnabled = true;
  }
  Root->Elements[COMGR::msgpack::String::make("Features")] = FeaturesNode;

  Root->Elements[COMGR::msgpack::String::make("XNACKEnabled")] =
      COMGR::msgpack::String::make(std::to_string(XNACKEnabled));

  auto Info = IsaInfos[IsaIndex];
  Root->Elements[msgpack::String::make("TrapHandlerEnabled")] =
      msgpack::String::make(std::to_string(Info.TrapHandlerEnabled));
  Root->Elements[msgpack::String::make("LocalMemorySize")] =
      msgpack::String::make(std::to_string(Info.LocalMemorySize));
  Root->Elements[msgpack::String::make("EUsPerCU")] =
      msgpack::String::make(std::to_string(Info.EUsPerCU));
  Root->Elements[msgpack::String::make("MaxWavesPerCU")] =
      msgpack::String::make(std::to_string(Info.MaxWavesPerCU));
  Root->Elements[msgpack::String::make("MaxFlatWorkGroupSize")] =
      msgpack::String::make(std::to_string(Info.MaxFlatWorkGroupSize));
  Root->Elements[msgpack::String::make("SGPRAllocGranule")] =
      msgpack::String::make(std::to_string(Info.SGPRAllocGranule));
  Root->Elements[msgpack::String::make("TotalNumSGPRs")] =
      msgpack::String::make(std::to_string(Info.TotalNumSGPRs));
  Root->Elements[msgpack::String::make("AddressableNumSGPRs")] =
      msgpack::String::make(std::to_string(Info.AddressableNumSGPRs));
  Root->Elements[msgpack::String::make("VGPRAllocGranule")] =
      msgpack::String::make(std::to_string(Info.VGPRAllocGranule));
  Root->Elements[msgpack::String::make("TotalNumVGPRs")] =
      msgpack::String::make(std::to_string(Info.TotalNumVGPRs));
  Root->Elements[msgpack::String::make("AddressableNumVGPRs")] =
      msgpack::String::make(std::to_string(Info.AddressableNumVGPRs));
  Root->Elements[msgpack::String::make("LDSBankCount")] =
      msgpack::String::make(std::to_string(Info.LDSBankCount));

  Meta->msgpack_node = Root;

  return AMD_COMGR_STATUS_SUCCESS;
}

bool isValidIsaName(StringRef IsaName) {
  auto IsaIterator = find(SupportedIsas, IsaName);
  if (IsaIterator == std::end(SupportedIsas))
    return false;

  TargetIdentifier Ident;
  return ParseTargetIdentifier(IsaName, Ident) == AMD_COMGR_STATUS_SUCCESS;
}

} // namespace COMGR
} // namespace metadata
