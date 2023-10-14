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
 ******************************************************************************/

#include "comgr-metadata.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <iostream>

using namespace llvm;
using namespace llvm::object;

namespace COMGR {
namespace metadata {

template <typename ELFT> using Elf_Note = typename ELFT::Note;

static Expected<std::unique_ptr<ELFObjectFileBase>>
getELFObjectFileBase(DataObject *DataP) {
  std::unique_ptr<MemoryBuffer> Buf =
      MemoryBuffer::getMemBuffer(StringRef(DataP->Data, DataP->Size));

  Expected<std::unique_ptr<ObjectFile>> ObjOrErr =
      ObjectFile::createELFObjectFile(*Buf);

  if (auto Err = ObjOrErr.takeError()) {
    return std::move(Err);
  }

  return unique_dyn_cast<ELFObjectFileBase>(std::move(*ObjOrErr));
}

/// Process all notes in the given ELF object file, passing them each to @p
/// ProcessNote.
///
/// @p ProcessNote should return @c true when the desired note is found, which
/// signals to stop searching and return @c AMD_COMGR_STATUS_SUCCESS. It should
/// return @c false otherwise to continue iteration.
///
/// @returns @c AMD_COMGR_STATUS_ERROR if an error was encountered in parsing
/// the ELF file; @c AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT if all notes are
/// processed without @p ProcessNote returning @c true; otherwise @c
/// AMD_COMGR_STATUS_SUCCESS.
template <class ELFT, typename F>
static amd_comgr_status_t processElfNotes(const ELFObjectFile<ELFT> *Obj,
                                          F ProcessNote) {
  const ELFFile<ELFT> &ELFFile = Obj->getELFFile();

  bool Found = false;

  auto ProgramHeadersOrError = ELFFile.program_headers();
  if (errorToBool(ProgramHeadersOrError.takeError())) {
    return AMD_COMGR_STATUS_ERROR;
  }

  for (const auto &Phdr : *ProgramHeadersOrError) {
    if (Phdr.p_type != ELF::PT_NOTE) {
      continue;
    }
    Error Err = Error::success();
    for (const auto &Note : ELFFile.notes(Phdr, Err)) {
      if (ProcessNote(Note)) {
        Found = true;
        break;
      }
    }
    if (errorToBool(std::move(Err))) {
      return AMD_COMGR_STATUS_ERROR;
    }
    if (Found) {
      return AMD_COMGR_STATUS_SUCCESS;
    }
  }

  auto SectionsOrError = ELFFile.sections();
  if (errorToBool(SectionsOrError.takeError())) {
    return AMD_COMGR_STATUS_ERROR;
  }

  for (const auto &Shdr : *SectionsOrError) {
    if (Shdr.sh_type != ELF::SHT_NOTE) {
      continue;
    }
    Error Err = Error::success();
    for (const auto &Note : ELFFile.notes(Shdr, Err)) {
      if (ProcessNote(Note)) {
        Found = true;
        break;
      }
    }
    if (errorToBool(std::move(Err))) {
      return AMD_COMGR_STATUS_ERROR;
    }
    if (Found) {
      return AMD_COMGR_STATUS_SUCCESS;
    }
  }

  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}

// PAL currently produces MsgPack metadata in a note with this ID.
// FIXME: Unify with HSA note types?
#define PAL_METADATA_NOTE_TYPE 13

// Try to merge "amdhsa.kernels" from DocNode @p From to @p To.
// The merge is allowed only if
// 1. "amdhsa.printf" record is not existing in either of the nodes.
// 2. "amdhsa.version" exists and is same.
// 3. "amdhsa.kernels" exists in both nodes.
//
// If merge is possible the function merges Kernel records
// to @p To and returns @c true.
static bool mergeNoteRecords(llvm::msgpack::DocNode &From,
                             llvm::msgpack::DocNode &To,
                             const StringRef VersionStrKey,
                             const StringRef PrintfStrKey,
                             const StringRef KernelStrKey) {
  if (!From.isMap()) {
    return false;
  }

  if (To.isEmpty()) {
    To = From;
    return true;
  }

  assert(To.isMap());

  if (From.getMap().find(PrintfStrKey) != From.getMap().end()) {
    /* Check if both have Printf records */
    if (To.getMap().find(PrintfStrKey) != To.getMap().end()) {
      return false;
    }

    /* Add Printf record for 'To' */
    To.getMap()[PrintfStrKey] = From.getMap()[PrintfStrKey];
  }

  auto &FromMapNode = From.getMap();
  auto &ToMapNode = To.getMap();

  auto FromVersionArrayNode = FromMapNode.find(VersionStrKey);
  auto ToVersionArrayNode = ToMapNode.find(VersionStrKey);

  if ((FromVersionArrayNode == FromMapNode.end() ||
       !FromVersionArrayNode->second.isArray()) ||
      (ToVersionArrayNode == ToMapNode.end() ||
       !ToVersionArrayNode->second.isArray())) {
    return false;
  }

  auto FromVersionArray = FromMapNode[VersionStrKey].getArray();
  auto ToVersionArray = ToMapNode[VersionStrKey].getArray();

  if (FromVersionArray.size() != ToVersionArray.size()) {
    return false;
  }

  for (size_t I = 0, E = FromVersionArray.size(); I != E; ++I) {
    if (FromVersionArray[I] != ToVersionArray[I]) {
      return false;
    }
  }

  auto FromKernelArray = FromMapNode.find(KernelStrKey);
  auto ToKernelArray = ToMapNode.find(KernelStrKey);

  if ((FromKernelArray == FromMapNode.end() ||
       !FromKernelArray->second.isArray()) ||
      (ToKernelArray == ToMapNode.end() || !ToKernelArray->second.isArray())) {
    return false;
  }

  auto &ToKernelRecords = ToKernelArray->second.getArray();
  for (auto Kernel : FromKernelArray->second.getArray()) {
    ToKernelRecords.push_back(Kernel);
  }

  return true;
}

template <class ELFT>
static bool processNote(const Elf_Note<ELFT> &Note, DataMeta *MetaP,
                        llvm::msgpack::DocNode &Root) {
  auto DescString = Note.getDescAsStringRef(4);

  if (Note.getName() == "AMD" && Note.getType() == ELF::NT_AMD_HSA_METADATA) {

    if (!Root.isEmpty()) {
      return false;
    }

    MetaP->MetaDoc->EmitIntegerBooleans = false;
    MetaP->MetaDoc->RawDocument.clear();
    if (!MetaP->MetaDoc->Document.fromYAML(DescString)) {
      return false;
    }

    Root = MetaP->MetaDoc->Document.getRoot();
    return true;
  }
  if (((Note.getName() == "AMD" || Note.getName() == "AMDGPU") &&
       Note.getType() == PAL_METADATA_NOTE_TYPE) ||
      (Note.getName() == "AMDGPU" &&
       Note.getType() == ELF::NT_AMDGPU_METADATA)) {
    if (!Root.isEmpty() && MetaP->MetaDoc->EmitIntegerBooleans != true) {
      return false;
    }

    MetaP->MetaDoc->EmitIntegerBooleans = true;
    MetaP->MetaDoc->RawDocumentList.push_back(std::string(DescString));

    /* TODO add support for merge using readFromBlob merge function */
    auto &Document = MetaP->MetaDoc->Document;

    Document.clear();
    if (!Document.readFromBlob(MetaP->MetaDoc->RawDocumentList.back(), false)) {
      return false;
    }

    return mergeNoteRecords(Document.getRoot(), Root, "amdhsa.version",
                            "amdhsa.printf", "amdhsa.kernels");
  }
  return false;
}

template <class ELFT>
static amd_comgr_status_t getElfMetadataRoot(const ELFObjectFile<ELFT> *Obj,
                                             DataMeta *MetaP) {
  bool Found = false;
  llvm::msgpack::DocNode Root;
  const ELFFile<ELFT> &ELFFile = Obj->getELFFile();

  auto ProgramHeadersOrError = ELFFile.program_headers();
  if (errorToBool(ProgramHeadersOrError.takeError())) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  for (const auto &Phdr : *ProgramHeadersOrError) {
    if (Phdr.p_type != ELF::PT_NOTE) {
      continue;
    }
    Error Err = Error::success();
    for (const auto &Note : ELFFile.notes(Phdr, Err)) {
      if (processNote<ELFT>(Note, MetaP, Root)) {
        Found = true;
      }
    }

    if (errorToBool(std::move(Err))) {
      return AMD_COMGR_STATUS_ERROR;
    }
  }

  if (Found) {
    MetaP->MetaDoc->Document.getRoot() = Root;
    MetaP->DocNode = MetaP->MetaDoc->Document.getRoot();
    return AMD_COMGR_STATUS_SUCCESS;
  }

  auto SectionsOrError = ELFFile.sections();
  if (errorToBool(SectionsOrError.takeError())) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  for (const auto &Shdr : *SectionsOrError) {
    if (Shdr.sh_type != ELF::SHT_NOTE) {
      continue;
    }
    Error Err = Error::success();
    for (const auto &Note : ELFFile.notes(Shdr, Err)) {
      if (processNote<ELFT>(Note, MetaP, Root)) {
        Found = true;
      }
    }

    if (errorToBool(std::move(Err))) {
      return AMD_COMGR_STATUS_ERROR;
    }
  }

  if (Found) {
    MetaP->MetaDoc->Document.getRoot() = Root;
    MetaP->DocNode = MetaP->MetaDoc->Document.getRoot();
    return AMD_COMGR_STATUS_SUCCESS;
  }

  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}

amd_comgr_status_t getMetadataRoot(DataObject *DataP, DataMeta *MetaP) {
  auto ObjOrErr = getELFObjectFileBase(DataP);
  if (errorToBool(ObjOrErr.takeError())) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  auto *Obj = ObjOrErr->get();

  if (auto *ELF32LE = dyn_cast<ELF32LEObjectFile>(Obj)) {
    return getElfMetadataRoot(ELF32LE, MetaP);
  }
  if (auto *ELF64LE = dyn_cast<ELF64LEObjectFile>(Obj)) {
    return getElfMetadataRoot(ELF64LE, MetaP);
  }
  if (auto *ELF32BE = dyn_cast<ELF32BEObjectFile>(Obj)) {
    return getElfMetadataRoot(ELF32BE, MetaP);
  }
  auto *ELF64BE = dyn_cast<ELF64BEObjectFile>(Obj);
  return getElfMetadataRoot(ELF64BE, MetaP);
}

struct IsaInfo {
  const char *IsaName;
  const char *Processor;
  bool SrameccSupported;
  bool XnackSupported;
  unsigned ElfMachine;
  bool TrapHandlerEnabled;
  unsigned LDSSize;
  unsigned LDSBankCount;
  unsigned EUsPerCU;
  unsigned MaxWavesPerCU;
  unsigned MaxFlatWorkGroupSize;
  unsigned SGPRAllocGranule;
  unsigned TotalNumSGPRs;
  unsigned AddressableNumSGPRs;
  unsigned VGPRAllocGranule;
  unsigned TotalNumVGPRs;
  unsigned AddressableNumVGPRs;
} IsaInfos[] = {
#define HANDLE_ISA(TARGET_TRIPLE, PROCESSOR, SRAMECC_SUPPORTED,                \
                   XNACK_SUPPORTED, ELF_MACHINE, TRAP_HANDLER_ENABLED,         \
                   LDS_SIZE, LDS_BANK_COUNT, EUS_PER_CU, MAX_WAVES_PER_CU,     \
                   MAX_FLAT_WORK_GROUP_SIZE, SGPR_ALLOC_GRANULE,               \
                   TOTAL_NUM_SGPRS, ADDRESSABLE_NUM_SGPRS, VGPR_ALLOC_GRANULE, \
                   TOTAL_NUM_VGPRS, ADDRESSABLE_NUM_VGPRS)                     \
  {TARGET_TRIPLE "-" PROCESSOR,                                                \
   PROCESSOR,                                                                  \
   SRAMECC_SUPPORTED,                                                          \
   XNACK_SUPPORTED,                                                            \
   ELF::ELF_MACHINE,                                                           \
   TRAP_HANDLER_ENABLED,                                                       \
   LDS_SIZE,                                                                   \
   LDS_BANK_COUNT,                                                             \
   EUS_PER_CU,                                                                 \
   MAX_WAVES_PER_CU,                                                           \
   MAX_FLAT_WORK_GROUP_SIZE,                                                   \
   SGPR_ALLOC_GRANULE,                                                         \
   TOTAL_NUM_SGPRS,                                                            \
   ADDRESSABLE_NUM_SGPRS,                                                      \
   VGPR_ALLOC_GRANULE,                                                         \
   TOTAL_NUM_VGPRS,                                                            \
   ADDRESSABLE_NUM_VGPRS},
#include "comgr-isa-metadata.def"
};

size_t getIsaCount() {
  return std::distance(std::begin(IsaInfos), std::end(IsaInfos));
}

// NOLINTNEXTLINE(readability-identifier-naming)
typedef struct amdgpu_hsa_note_code_object_version_s {
  uint32_t major_version; // NOLINT(readability-identifier-naming)
  uint32_t minor_version; // NOLINT(readability-identifier-naming)
} amdgpu_hsa_note_code_object_version_t;

// NOLINTNEXTLINE(readability-identifier-naming)
typedef struct amdgpu_hsa_note_hsail_s {
  uint32_t hsail_major_version; // NOLINT(readability-identifier-naming)
  uint32_t hsail_minor_version; // NOLINT(readability-identifier-naming)
  uint8_t profile;              // NOLINT(readability-identifier-naming)
  uint8_t machine_model;        // NOLINT(readability-identifier-naming)
  uint8_t default_float_round;  // NOLINT(readability-identifier-naming)
} amdgpu_hsa_note_hsail_t;

// NOLINTNEXTLINE(readability-identifier-naming)
typedef struct amdgpu_hsa_note_isa_s {
  uint16_t vendor_name_size;            // NOLINT(readability-identifier-naming)
  uint16_t architecture_name_size;      // NOLINT(readability-identifier-naming)
  uint32_t major;                       // NOLINT(readability-identifier-naming)
  uint32_t minor;                       // NOLINT(readability-identifier-naming)
  uint32_t stepping;                    // NOLINT(readability-identifier-naming)
  char vendor_and_architecture_name[1]; // NOLINT(readability-identifier-naming)
} amdgpu_hsa_note_isa_t;

static bool getMachInfo(unsigned Mach, std::string &Processor,
                        bool &SrameccSupported, bool &XnackSupported) {
  auto *IsaIterator = std::find_if(
      std::begin(IsaInfos), std::end(IsaInfos),
      [Mach](const IsaInfo &IsaInfo) { return Mach == IsaInfo.ElfMachine; });
  if (IsaIterator == std::end(IsaInfos)) {
    return false;
  }

  Processor = IsaIterator->Processor;
  SrameccSupported = IsaIterator->SrameccSupported;
  XnackSupported = IsaIterator->XnackSupported;
  return true;
}

// This function is an exact copy of the ROCr loader function of the same name.
static std::string convertOldTargetNameToNew(const std::string &OldName,
                                             bool IsFinalizer,
                                             uint32_t EFlags) {
  assert(!OldName.empty() && "Expecting non-empty old name");

  unsigned Mach = 0;
  if (OldName == "AMD:AMDGPU:6:0:0") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX600;
  } else if (OldName == "AMD:AMDGPU:6:0:1") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX601;
  } else if (OldName == "AMD:AMDGPU:6:0:2") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX602;
  } else if (OldName == "AMD:AMDGPU:7:0:0") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX700;
  } else if (OldName == "AMD:AMDGPU:7:0:1") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX701;
  } else if (OldName == "AMD:AMDGPU:7:0:2") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX702;
  } else if (OldName == "AMD:AMDGPU:7:0:3") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX703;
  } else if (OldName == "AMD:AMDGPU:7:0:4") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX704;
  } else if (OldName == "AMD:AMDGPU:7:0:5") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX705;
  } else if (OldName == "AMD:AMDGPU:8:0:1") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX801;
  } else if (OldName == "AMD:AMDGPU:8:0:0" || OldName == "AMD:AMDGPU:8:0:2") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX802;
  } else if (OldName == "AMD:AMDGPU:8:0:3" || OldName == "AMD:AMDGPU:8:0:4") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX803;
  } else if (OldName == "AMD:AMDGPU:8:0:5") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX805;
  } else if (OldName == "AMD:AMDGPU:8:1:0") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX810;
  } else if (OldName == "AMD:AMDGPU:9:0:0" || OldName == "AMD:AMDGPU:9:0:1") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX900;
  } else if (OldName == "AMD:AMDGPU:9:0:2" || OldName == "AMD:AMDGPU:9:0:3") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX902;
  } else if (OldName == "AMD:AMDGPU:9:0:4" || OldName == "AMD:AMDGPU:9:0:5") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX904;
  } else if (OldName == "AMD:AMDGPU:9:0:6" || OldName == "AMD:AMDGPU:9:0:7") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX906;
  } else if (OldName == "AMD:AMDGPU:9:0:12") {
    Mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX90C;
  } else {
    // Code object v2 only supports asics up to gfx906. Do NOT add handling
    // of new asics into this if-else-if* block.
    return "";
  }

  std::string Name;
  bool SrameccSupported = false;
  bool XnackSupported = false;
  if (!getMachInfo(Mach, Name, SrameccSupported, XnackSupported)) {
    return "";
  }

  // Only "AMD:AMDGPU:9:0:6" and "AMD:AMDGPU:9:0:7" supports SRAMECC for
  // code object V2, and it must be OFF.
  if (SrameccSupported) {
    Name += ":sramecc-";
  }

  if (IsFinalizer) {
    if (EFlags & ELF::EF_AMDGPU_FEATURE_XNACK_V2) {
      Name += ":xnack+";
    } else if (XnackSupported) {
      Name += ":xnack-";
    }
  } else {
    if (OldName == "AMD:AMDGPU:8:0:1") {
      Name += ":xnack+";
    } else if (OldName == "AMD:AMDGPU:8:1:0") {
      Name += ":xnack+";
    } else if (OldName == "AMD:AMDGPU:9:0:1") {
      Name += ":xnack+";
    } else if (OldName == "AMD:AMDGPU:9:0:3") {
      Name += ":xnack+";
    } else if (OldName == "AMD:AMDGPU:9:0:5") {
      Name += ":xnack+";
    } else if (OldName == "AMD:AMDGPU:9:0:7") {
      Name += ":xnack+";
    } else if (XnackSupported) {
      Name += ":xnack-";
    }
  }

  return Name;
}

template <class ELFT>
static amd_comgr_status_t
getElfIsaNameFromElfNotes(const ELFObjectFile<ELFT> *Obj,
                          std::string &NoteIsaName) {

  auto ElfHeader = Obj->getELFFile().getHeader();

  // Only ELFABIVERSION_AMDGPU_HSA_V2 used note records for the isa name.
  assert(ElfHeader.e_ident[ELF::EI_ABIVERSION] ==
         ELF::ELFABIVERSION_AMDGPU_HSA_V2);

  bool IsError = false;
  bool IsCodeObjectVersion = false;
  bool IsHSAIL = false;
  bool IsIsa = false;
  uint32_t Major = 0;
  uint32_t Minor = 0;
  uint32_t Stepping = 0;
  StringRef VendorName;
  StringRef ArchitectureName;

  auto ProcessNote = [&](const Elf_Note<ELFT> &Note) {
    if (Note.getName() != "AMD") {
      return false;
    }

    switch (Note.getType()) {
    case ELF::NT_AMD_HSA_CODE_OBJECT_VERSION: {
      if (Note.getDesc(4).size() <
          sizeof(amdgpu_hsa_note_code_object_version_s)) {
        IsError = true;
        return true;
      }

      const auto *NoteCodeObjectVersion =
          reinterpret_cast<const amdgpu_hsa_note_code_object_version_s *>(
              Note.getDesc(4).data());

      // Only code objects up to version 2 used note records.
      if (NoteCodeObjectVersion->major_version > 2) {
        IsError = true;
        return true;
      }

      IsCodeObjectVersion = true;
      break;
    }

    case ELF::NT_AMD_HSA_HSAIL: {
      if (Note.getDesc(4).size() < sizeof(amdgpu_hsa_note_hsail_s)) {
        IsError = true;
        return true;
      }

      IsHSAIL = true;
      break;
    }

    case ELF::NT_AMD_HSA_ISA_VERSION: {
      if (Note.getDesc(4).size() <
          offsetof(amdgpu_hsa_note_isa_s, vendor_and_architecture_name)) {
        IsError = true;
        return true;
      }

      const auto *NoteIsa = reinterpret_cast<const amdgpu_hsa_note_isa_s *>(
          Note.getDesc(4).data());

      if (!NoteIsa->vendor_name_size || !NoteIsa->architecture_name_size) {
        IsError = true;
        return true;
      }

      if (Note.getDesc(4).size() <
          offsetof(amdgpu_hsa_note_isa_s, vendor_and_architecture_name) +
              NoteIsa->vendor_name_size + NoteIsa->architecture_name_size) {
        IsError = true;
        return true;
      }

      Major = NoteIsa->major;
      Minor = NoteIsa->minor;
      Stepping = NoteIsa->stepping;
      VendorName = StringRef(NoteIsa->vendor_and_architecture_name,
                             NoteIsa->vendor_name_size - 1);
      ArchitectureName = StringRef(NoteIsa->vendor_and_architecture_name +
                                       NoteIsa->vendor_name_size,
                                   NoteIsa->architecture_name_size - 1);

      IsIsa = true;
      break;
    }
    }

    // Only stop searching when found all the possible note records needed.
    return IsCodeObjectVersion && IsHSAIL && IsIsa;
  };

  if ((processElfNotes(Obj, ProcessNote) == AMD_COMGR_STATUS_ERROR) ||
      IsError) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Code objects up to V2 must have both code object version and isa note
  // records.
  if (!(IsCodeObjectVersion && IsIsa)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  std::string OldName;
  OldName += VendorName;
  OldName += ":";
  OldName += ArchitectureName;
  OldName += ":";
  OldName += std::to_string(Major);
  OldName += ":";
  OldName += std::to_string(Minor);
  OldName += ":";
  OldName += std::to_string(Stepping);

  NoteIsaName = convertOldTargetNameToNew(OldName, IsHSAIL, ElfHeader.e_flags);
  if (NoteIsaName.empty()) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  NoteIsaName = "amdgcn-amd-amdhsa--" + NoteIsaName;

  return AMD_COMGR_STATUS_SUCCESS;
}

template <class ELFT>
static amd_comgr_status_t
getElfIsaNameFromElfHeader(const ELFObjectFile<ELFT> *Obj,
                           std::string &ElfIsaName) {
  auto ElfHeader = Obj->getELFFile().getHeader();

  switch (ElfHeader.e_ident[ELF::EI_CLASS]) {
  case ELF::ELFCLASSNONE:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  case ELF::ELFCLASS32:
    ElfIsaName += "r600";
    break;
  case ELF::ELFCLASS64:
    ElfIsaName += "amdgcn";
    break;
  }

  if (ElfHeader.e_machine != ELF::EM_AMDGPU) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ElfIsaName += "-amd-";

  switch (ElfHeader.e_ident[ELF::EI_OSABI]) {
  case ELF::ELFOSABI_NONE:
    ElfIsaName += "unknown";
    break;
  case ELF::ELFOSABI_AMDGPU_HSA:
    ElfIsaName += "amdhsa";
    break;
  case ELF::ELFOSABI_AMDGPU_PAL:
    ElfIsaName += "amdpal";
    break;
  case ELF::ELFOSABI_AMDGPU_MESA3D:
    ElfIsaName += "mesa3d";
    break;
  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ElfIsaName += "--";

  std::string Processor;
  bool SrameccSupported, XnackSupported;
  if (!getMachInfo(ElfHeader.e_flags & ELF::EF_AMDGPU_MACH, Processor,
                   SrameccSupported, XnackSupported)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ElfIsaName += Processor;

  switch (ElfHeader.e_ident[ELF::EI_ABIVERSION]) {
  case ELF::ELFABIVERSION_AMDGPU_HSA_V2: {
    // ELFABIVERSION_AMDGPU_HSA_V2 uses ELF note records and is not supported.
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  case ELF::ELFABIVERSION_AMDGPU_HSA_V3: {
    if (SrameccSupported) {
      if (ElfHeader.e_flags & ELF::EF_AMDGPU_FEATURE_SRAMECC_V3) {
        ElfIsaName += ":sramecc+";
      } else {
        ElfIsaName += ":sramecc-";
      }
    }
    if (XnackSupported) {
      if (ElfHeader.e_flags & ELF::EF_AMDGPU_FEATURE_XNACK_V3) {
        ElfIsaName += ":xnack+";
      } else {
        ElfIsaName += ":xnack-";
      }
    }
    break;
  }

  case ELF::ELFABIVERSION_AMDGPU_HSA_V4:
  case ELF::ELFABIVERSION_AMDGPU_HSA_V5: {
    switch (ElfHeader.e_flags & ELF::EF_AMDGPU_FEATURE_SRAMECC_V4) {
    case ELF::EF_AMDGPU_FEATURE_SRAMECC_OFF_V4:
      ElfIsaName += ":sramecc-";
      break;
    case ELF::EF_AMDGPU_FEATURE_SRAMECC_ON_V4:
      ElfIsaName += ":sramecc+";
      break;
    }
    switch (ElfHeader.e_flags & ELF::EF_AMDGPU_FEATURE_XNACK_V4) {
    case ELF::EF_AMDGPU_FEATURE_XNACK_OFF_V4:
      ElfIsaName += ":xnack-";
      break;
    case ELF::EF_AMDGPU_FEATURE_XNACK_ON_V4:
      ElfIsaName += ":xnack+";
      break;
    }
    break;
  }

  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

template <class ELFT>
static amd_comgr_status_t getElfIsaNameImpl(const ELFObjectFile<ELFT> *Obj,
                                            std::string &IsaName) {
  auto ElfHeader = Obj->getELFFile().getHeader();

  if (ElfHeader.e_ident[ELF::EI_ABIVERSION] ==
      ELF::ELFABIVERSION_AMDGPU_HSA_V2) {
    return getElfIsaNameFromElfNotes(Obj, IsaName);
  }

  return getElfIsaNameFromElfHeader(Obj, IsaName);
}

amd_comgr_status_t getElfIsaName(DataObject *DataP, std::string &IsaName) {
  auto ObjOrErr = getELFObjectFileBase(DataP);
  if (errorToBool(ObjOrErr.takeError())) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  auto *Obj = ObjOrErr->get();

  if (auto *ELF32LE = dyn_cast<ELF32LEObjectFile>(Obj)) {
    return getElfIsaNameImpl(ELF32LE, IsaName);
  }
  if (auto *ELF64LE = dyn_cast<ELF64LEObjectFile>(Obj)) {
    return getElfIsaNameImpl(ELF64LE, IsaName);
  }
  if (auto *ELF32BE = dyn_cast<ELF32BEObjectFile>(Obj)) {
    return getElfIsaNameImpl(ELF32BE, IsaName);
  }
  auto *ELF64BE = dyn_cast<ELF64BEObjectFile>(Obj);
  return getElfIsaNameImpl(ELF64BE, IsaName);
}

amd_comgr_status_t getIsaIndex(StringRef IsaString, size_t &Index) {
  auto IsaName = IsaString.take_until([](char C) { return C == ':'; });
  auto *IsaIterator = std::find_if(
      std::begin(IsaInfos), std::end(IsaInfos),
      [&](const IsaInfo &IsaInfo) { return IsaName == IsaInfo.IsaName; });
  if (IsaIterator == std::end(IsaInfos)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  Index = std::distance(std::begin(IsaInfos), IsaIterator);

  return AMD_COMGR_STATUS_SUCCESS;
}

bool isSupportedFeature(size_t IsaIndex, StringRef Feature) {
  if (Feature.empty() ||
      (Feature.take_back() != "+" && Feature.take_back() != "-")) {
    return false;
  }

  return (Feature.drop_back() == "xnack" &&
          IsaInfos[IsaIndex].XnackSupported) ||
         (Feature.drop_back() == "sramecc" &&
          IsaInfos[IsaIndex].SrameccSupported);
}

const char *getIsaName(size_t Index) { return IsaInfos[Index].IsaName; }

amd_comgr_status_t getIsaMetadata(StringRef IsaName,
                                  llvm::msgpack::Document &Doc) {
  amd_comgr_status_t Status;

  size_t IsaIndex;
  Status = getIsaIndex(IsaName, IsaIndex);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    return Status;
  }

  TargetIdentifier Ident;
  Status = parseTargetIdentifier(IsaName, Ident);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    return Status;
  }

  auto Root = Doc.getRoot().getMap(/*Convert=*/true);

  Root["Name"] = Doc.getNode(IsaName, /*Copy=*/true);
  Root["Architecture"] = Doc.getNode(Ident.Arch, /*Copy=*/true);
  Root["Vendor"] = Doc.getNode(Ident.Vendor, /*Copy=*/true);
  Root["OS"] = Doc.getNode(Ident.OS, /*Copy=*/true);
  Root["Environment"] = Doc.getNode(Ident.Environ, /*Copy=*/true);
  Root["Processor"] = Doc.getNode(Ident.Processor, /*Copy=*/true);
  Root["Version"] = Doc.getNode("1.0.0", /*Copy=*/true);

  auto FeaturesNode = Doc.getMapNode();
  if (IsaInfos[IsaIndex].XnackSupported) {
    FeaturesNode["xnack"] = Doc.getNode("any", /*Copy=*/true);
  }
  if (IsaInfos[IsaIndex].SrameccSupported) {
    FeaturesNode["sramecc"] = Doc.getNode("any", /*Copy=*/true);
  }

  for (size_t I = 0; I < Ident.Features.size(); ++I) {
    if (FeaturesNode.find(Ident.Features[I].drop_back()) ==
        FeaturesNode.end()) {
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }

    auto State = Ident.Features[I].take_back();
    if (State == "+") {
      FeaturesNode[Ident.Features[I].drop_back()] =
          Doc.getNode("on", /*Copy=*/true);
    } else if (State == "-") {
      FeaturesNode[Ident.Features[I].drop_back()] =
          Doc.getNode("off", /*Copy=*/true);
    } else {
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  Root["Features"] = FeaturesNode;

  auto Info = IsaInfos[IsaIndex];
  Root["TrapHandlerEnabled"] =
      Doc.getNode(std::to_string(Info.TrapHandlerEnabled), /*Copy=*/true);
  Root["LocalMemorySize"] =
      Doc.getNode(std::to_string(Info.LDSSize), /*Copy=*/true);
  Root["EUsPerCU"] = Doc.getNode(std::to_string(Info.EUsPerCU), /*Copy=*/true);
  Root["MaxWavesPerCU"] =
      Doc.getNode(std::to_string(Info.MaxWavesPerCU), /*Copy=*/true);
  Root["MaxFlatWorkGroupSize"] =
      Doc.getNode(std::to_string(Info.MaxFlatWorkGroupSize), /*Copy=*/true);
  Root["SGPRAllocGranule"] =
      Doc.getNode(std::to_string(Info.SGPRAllocGranule), /*Copy=*/true);
  Root["TotalNumSGPRs"] =
      Doc.getNode(std::to_string(Info.TotalNumSGPRs), /*Copy=*/true);
  Root["AddressableNumSGPRs"] =
      Doc.getNode(std::to_string(Info.AddressableNumSGPRs), /*Copy=*/true);
  Root["VGPRAllocGranule"] =
      Doc.getNode(std::to_string(Info.VGPRAllocGranule), /*Copy=*/true);
  Root["TotalNumVGPRs"] =
      Doc.getNode(std::to_string(Info.TotalNumVGPRs), /*Copy=*/true);
  Root["AddressableNumVGPRs"] =
      Doc.getNode(std::to_string(Info.AddressableNumVGPRs), /*Copy=*/true);
  Root["LDSBankCount"] =
      Doc.getNode(std::to_string(Info.LDSBankCount), /*Copy=*/true);

  return AMD_COMGR_STATUS_SUCCESS;
}

bool isValidIsaName(StringRef IsaString) {
  TargetIdentifier Ident;
  return parseTargetIdentifier(IsaString, Ident) == AMD_COMGR_STATUS_SUCCESS;
}

static size_t constexpr strLiteralLength(char const *str) {
  size_t I = 0;
  while (str[I]) {
    ++I;
  }
  return I;
}

static constexpr const char *OFFLOAD_KIND_HIP = "hip";
static constexpr const char *OFFLOAD_KIND_HIPV4 = "hipv4";
static constexpr const char *OFFLOAD_KIND_HCC = "hcc";
static constexpr const char *CLANG_OFFLOAD_BUNDLER_MAGIC =
    "__CLANG_OFFLOAD_BUNDLE__";
static constexpr size_t OffloadBundleMagicLen =
    strLiteralLength(CLANG_OFFLOAD_BUNDLER_MAGIC);

bool isCompatibleIsaName(StringRef IsaName, StringRef CodeObjectIsaName) {
  if (IsaName == CodeObjectIsaName) {
    return true;
  }

  TargetIdentifier CodeObjectIdent;
  if (parseTargetIdentifier(CodeObjectIsaName, CodeObjectIdent)) {
    return false;
  }

  TargetIdentifier IsaIdent;
  if (parseTargetIdentifier(IsaName, IsaIdent)) {
    return false;
  }

  if (CodeObjectIdent.Processor != IsaIdent.Processor) {
    return false;
  }

  char CodeObjectXnack = ' ', CodeObjectSramecc = ' ';
  for (auto Feature : CodeObjectIdent.Features) {
    if (Feature.drop_back() == "xnack") {
      CodeObjectXnack = Feature.take_back()[0];
    }

    if (Feature.drop_back() == "sramecc") {
      CodeObjectSramecc = Feature.take_back()[0];
    }
  }

  char IsaXnack = ' ', IsaSramecc = ' ';
  for (auto Feature : IsaIdent.Features) {
    if (Feature.drop_back() == "xnack") {
      IsaXnack = Feature.take_back()[0];
    }
    if (Feature.drop_back() == "sramecc") {
      IsaSramecc = Feature.take_back()[0];
    }
  }

  if (CodeObjectXnack != ' ') {
    if (CodeObjectXnack != IsaXnack) {
      return false;
    }
  }

  if (CodeObjectSramecc != ' ') {
    if (CodeObjectSramecc != IsaSramecc) {
      return false;
    }
  }
  return true;
}

amd_comgr_status_t
lookUpCodeObjectInSharedObject(DataObject *DataP,
                               amd_comgr_code_object_info_t *QueryList,
                               size_t QueryListSize) {
  for (uint64_t I = 0; I < QueryListSize; I++) {
    QueryList[I].offset = 0;
    QueryList[I].size = 0;
  }

  std::string IsaName;
  amd_comgr_status_t Status = getElfIsaName(DataP, IsaName);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    return Status;
  }

  for (unsigned J = 0; J < QueryListSize; J++) {
    if (isCompatibleIsaName(QueryList[J].isa, IsaName)) {
      QueryList[J].offset = 0;
      QueryList[J].size = DataP->Size;
      break;
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t lookUpCodeObject(DataObject *DataP,
                                    amd_comgr_code_object_info_t *QueryList,
                                    size_t QueryListSize) {

  if (DataP->DataKind == AMD_COMGR_DATA_KIND_EXECUTABLE) {
    return lookUpCodeObjectInSharedObject(DataP, QueryList, QueryListSize);
  }

  int Seen = 0;
  BinaryStreamReader Reader(StringRef(DataP->Data, DataP->Size),
                            llvm::endianness::little);

  StringRef Magic;
  if (auto EC = Reader.readFixedString(Magic, OffloadBundleMagicLen)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  if (Magic != CLANG_OFFLOAD_BUNDLER_MAGIC) {
    if (DataP->DataKind == AMD_COMGR_DATA_KIND_BYTES) {
      return lookUpCodeObjectInSharedObject(DataP, QueryList, QueryListSize);
    }
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  uint64_t NumOfCodeObjects;
  if (auto EC = Reader.readInteger(NumOfCodeObjects)) {
    return AMD_COMGR_STATUS_ERROR;
  }

  for (uint64_t I = 0; I < QueryListSize; I++) {
    QueryList[I].offset = 0;
    QueryList[I].size = 0;
  }

  // For each code object, extract BundleEntryID information, and check that
  // against each ISA in the QueryList
  for (uint64_t I = 0; I < NumOfCodeObjects; I++) {
    uint64_t BundleEntryCodeObjectSize;
    uint64_t BundleEntryCodeObjectOffset;
    uint64_t BundleEntryIDSize;
    StringRef BundleEntryID;

    if (auto EC = Reader.readInteger(BundleEntryCodeObjectOffset)) {
      return AMD_COMGR_STATUS_ERROR;
    }

    if (auto Status = Reader.readInteger(BundleEntryCodeObjectSize)) {
      return AMD_COMGR_STATUS_ERROR;
    }

    if (auto Status = Reader.readInteger(BundleEntryIDSize)) {
      return AMD_COMGR_STATUS_ERROR;
    }

    if (Reader.readFixedString(BundleEntryID, BundleEntryIDSize)) {
      return AMD_COMGR_STATUS_ERROR;
    }

    const auto OffloadAndTargetId = BundleEntryID.split('-');
    if (OffloadAndTargetId.first != OFFLOAD_KIND_HIP &&
        OffloadAndTargetId.first != OFFLOAD_KIND_HIPV4 &&
        OffloadAndTargetId.first != OFFLOAD_KIND_HCC) {
      continue;
    }

    for (unsigned J = 0; J < QueryListSize; J++) {
      // If this QueryList item has already been found to be compatible with
      // another BundleEntryID, no need to check against the current
      // BundleEntryID
      if (QueryList[J].size != 0) {
        continue;
      }

      // If the QueryList Isa is compatible with the BundleEntryID, set the
      // QueryList offset/size to this BundleEntryID
      if (isCompatibleIsaName(QueryList[J].isa, OffloadAndTargetId.second)) {
        QueryList[J].offset = BundleEntryCodeObjectOffset;
        QueryList[J].size = BundleEntryCodeObjectSize;
        Seen++;
        break;
      }
    }

    // Stop iterating over BundleEntryIDs once we have populated the entire
    // QueryList
    if (Seen == (int) QueryListSize) {
      break;
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace metadata
} // namespace COMGR
