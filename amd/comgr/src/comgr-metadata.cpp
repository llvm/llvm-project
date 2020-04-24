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
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
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

  if (auto Err = ObjOrErr.takeError())
    return std::move(Err);

  return unique_dyn_cast<ELFObjectFileBase>(std::move(*ObjOrErr));
}

/// Process all notes in the given ELF object file, passing them each to @p
/// ProcessNote.
///
/// @p ProcessNote should return @c true when the desired note is found, which
/// signals to stop searching and return @c AMD_COMGR_STATUS_SUCCESS. It should
/// return @c false otherwise to continue iteration.
///
/// @returns @c AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT If all notes are
/// processed without @p ProcessNote returning @c true, otherwise
/// AMD_COMGR_STATUS_SUCCESS.
template <class ELFT, typename F>
static amd_comgr_status_t processElfNotes(const ELFObjectFile<ELFT> *Obj,
                                          F ProcessNote) {
  const ELFFile<ELFT> *ELFFile = Obj->getELFFile();

  bool Found = false;

  auto ProgramHeadersOrError = ELFFile->program_headers();
  if (errorToBool(ProgramHeadersOrError.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (const auto &Phdr : *ProgramHeadersOrError) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;
    Error Err = Error::success();
    for (const auto &Note : ELFFile->notes(Phdr, Err))
      if (ProcessNote(Note)) {
        Found = true;
        break;
      }
    if (errorToBool(std::move(Err)))
      return AMD_COMGR_STATUS_ERROR;
    if (Found)
      return AMD_COMGR_STATUS_SUCCESS;
  }

  auto SectionsOrError = ELFFile->sections();
  if (errorToBool(SectionsOrError.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (const auto &Shdr : *SectionsOrError) {
    if (Shdr.sh_type != ELF::SHT_NOTE)
      continue;
    Error Err = Error::success();
    for (const auto &Note : ELFFile->notes(Shdr, Err))
      if (ProcessNote(Note)) {
        Found = true;
        break;
      }
    if (errorToBool(std::move(Err)))
      return AMD_COMGR_STATUS_ERROR;
    if (Found)
      return AMD_COMGR_STATUS_SUCCESS;
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
  if (!From.isMap())
    return false;

  if (To.isEmpty()) {
    To = From;
    return true;
  }

  assert(To.isMap());

  if (From.getMap().find(PrintfStrKey) != From.getMap().end()) {
    /* Check if both have Printf records */
    if (To.getMap().find(PrintfStrKey) != To.getMap().end())
      return false;

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
       !ToVersionArrayNode->second.isArray()))
    return false;

  auto FromVersionArray = FromMapNode[VersionStrKey].getArray();
  auto ToVersionArray = ToMapNode[VersionStrKey].getArray();

  if (FromVersionArray.size() != ToVersionArray.size())
    return false;

  for (size_t i = 0, e = FromVersionArray.size(); i != e; ++i) {
    if (FromVersionArray[i] != ToVersionArray[i])
      return false;
  }

  auto FromKernelArray = FromMapNode.find(KernelStrKey);
  auto ToKernelArray = ToMapNode.find(KernelStrKey);

  if ((FromKernelArray == FromMapNode.end() ||
       !FromKernelArray->second.isArray()) ||
      (ToKernelArray == ToMapNode.end() || !ToKernelArray->second.isArray()))
    return false;

  auto &ToKernelRecords = ToKernelArray->second.getArray();
  for (auto Kernel : FromKernelArray->second.getArray())
    ToKernelRecords.push_back(Kernel);

  return true;
}

template <class ELFT>
static bool processNote(const Elf_Note<ELFT> &Note, DataMeta *MetaP,
                        llvm::msgpack::DocNode &Root) {
  auto DescString = Note.getDescAsStringRef();

  if (Note.getName() == "AMD" &&
      Note.getType() == ELF::NT_AMD_AMDGPU_HSA_METADATA) {

    if (!Root.isEmpty())
      return false;

    MetaP->MetaDoc->EmitIntegerBooleans = false;
    MetaP->MetaDoc->RawDocument.clear();
    if (!MetaP->MetaDoc->Document.fromYAML(DescString))
      return false;

    Root = MetaP->MetaDoc->Document.getRoot();
    return true;
  } else if (((Note.getName() == "AMD" || Note.getName() == "AMDGPU") &&
              Note.getType() == PAL_METADATA_NOTE_TYPE) ||
             (Note.getName() == "AMDGPU" &&
              Note.getType() == ELF::NT_AMDGPU_METADATA)) {
    if (!Root.isEmpty() && MetaP->MetaDoc->EmitIntegerBooleans != true)
      return false;

    MetaP->MetaDoc->EmitIntegerBooleans = true;
    MetaP->MetaDoc->RawDocumentList.push_back(std::string(DescString));

    /* TODO add support for merge using readFromBlob merge function */
    auto &Document = MetaP->MetaDoc->Document;

    Document.clear();
    if (!Document.readFromBlob(MetaP->MetaDoc->RawDocumentList.back(), false))
      return false;

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
  const ELFFile<ELFT> *ELFFile = Obj->getELFFile();

  auto ProgramHeadersOrError = ELFFile->program_headers();
  if (errorToBool(ProgramHeadersOrError.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (const auto &Phdr : *ProgramHeadersOrError) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;
    Error Err = Error::success();
    for (const auto &Note : ELFFile->notes(Phdr, Err))
      if (processNote<ELFT>(Note, MetaP, Root))
        Found = true;

    if (errorToBool(std::move(Err)))
      return AMD_COMGR_STATUS_ERROR;
  }

  if (Found) {
    MetaP->MetaDoc->Document.getRoot() = Root;
    MetaP->DocNode = MetaP->MetaDoc->Document.getRoot();
    return AMD_COMGR_STATUS_SUCCESS;
  }

  auto SectionsOrError = ELFFile->sections();
  if (errorToBool(SectionsOrError.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (const auto &Shdr : *SectionsOrError) {
    if (Shdr.sh_type != ELF::SHT_NOTE)
      continue;
    Error Err = Error::success();
    for (const auto &Note : ELFFile->notes(Shdr, Err))
      if (processNote<ELFT>(Note, MetaP, Root))
        Found = true;

    if (errorToBool(std::move(Err)))
      return AMD_COMGR_STATUS_ERROR;
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
  if (errorToBool(ObjOrErr.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  auto Obj = ObjOrErr->get();

  if (auto ELF32LE = dyn_cast<ELF32LEObjectFile>(Obj))
    return getElfMetadataRoot(ELF32LE, MetaP);
  if (auto ELF64LE = dyn_cast<ELF64LEObjectFile>(Obj))
    return getElfMetadataRoot(ELF64LE, MetaP);
  if (auto ELF32BE = dyn_cast<ELF32BEObjectFile>(Obj))
    return getElfMetadataRoot(ELF32BE, MetaP);
  auto ELF64BE = dyn_cast<ELF64BEObjectFile>(Obj);
  return getElfMetadataRoot(ELF64BE, MetaP);
}

#define NT_AMDGPU_HSA_ISA 3
// NOLINTNEXTLINE(readability-identifier-naming)
typedef struct amdgpu_hsa_note_isa_s {
  uint16_t vendor_name_size;            // NOLINT(readability-identifier-naming)
  uint16_t architecture_name_size;      // NOLINT(readability-identifier-naming)
  uint32_t major;                       // NOLINT(readability-identifier-naming)
  uint32_t minor;                       // NOLINT(readability-identifier-naming)
  uint32_t stepping;                    // NOLINT(readability-identifier-naming)
  char vendor_and_architecture_name[1]; // NOLINT(readability-identifier-naming)
} amdgpu_hsa_note_isa_t;

static amd_comgr_status_t
getNoteIsaName(StringRef VendorName, StringRef ArchitectureName,
               uint32_t MajorVersion, uint32_t MinorVersion, uint32_t Stepping,
               uint32_t EFlags, std::string &NoteIsaName) {
  std::string OldName;
  OldName += VendorName;
  OldName += ":";
  OldName += ArchitectureName;
  OldName += ":";
  OldName += std::to_string(MajorVersion);
  OldName += ":";
  OldName += std::to_string(MinorVersion);
  OldName += ":";
  OldName += std::to_string(Stepping);

  if (OldName == "AMD:AMDGPU:7:0:0")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx700";
  else if (OldName == "AMD:AMDGPU:7:0:1")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx701";
  else if (OldName == "AMD:AMDGPU:7:0:2")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx702";
  else if (OldName == "AMD:AMDGPU:7:0:3")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx703";
  else if (OldName == "AMD:AMDGPU:7:0:4")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx704";
  else if (OldName == "AMD:AMDGPU:8:0:1")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx801";
  else if (OldName == "AMD:AMDGPU:8:0:2")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx802";
  else if (OldName == "AMD:AMDGPU:8:0:3")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx803";
  else if (OldName == "AMD:AMDGPU:8:1:0")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx810";
  else if (OldName == "AMD:AMDGPU:9:0:0")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx900";
  else if (OldName == "AMD:AMDGPU:9:0:2")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx902";
  else if (OldName == "AMD:AMDGPU:9:0:4")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx904";
  else if (OldName == "AMD:AMDGPU:9:0:6")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx906";
  else if (OldName == "AMD:AMDGPU:9:0:8")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx908";
  else if (OldName == "AMD:AMDGPU:9:0:9")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx909";
  else if (OldName == "AMD:AMDGPU:10:1:0")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx1010";
  else if (OldName == "AMD:AMDGPU:10:1:1")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx1011";
  else if (OldName == "AMD:AMDGPU:10:1:2")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx1012";
  else if (OldName == "AMD:AMDGPU:10:3:0")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx1030";
  else if (OldName == "AMD:AMDGPU:10:3:1")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx1031";
  else if (OldName == "AMD:AMDGPU:10:3:2")
    NoteIsaName = "amdgcn-amd-amdhsa--gfx1032";
  else
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (EFlags & ELF::EF_AMDGPU_XNACK)
    NoteIsaName = NoteIsaName + "+xnack";
  if (EFlags & ELF::EF_AMDGPU_SRAM_ECC)
    NoteIsaName = NoteIsaName + "+sram-ecc";

  return AMD_COMGR_STATUS_SUCCESS;
}

template <class ELFT>
static amd_comgr_status_t getElfIsaNameV2(const ELFObjectFile<ELFT> *Obj,
                                          size_t *Size, char *IsaName) {
  amd_comgr_status_t NoteStatus = AMD_COMGR_STATUS_SUCCESS;
  auto ProcessNote = [&](const Elf_Note<ELFT> &Note) {
    if (Note.getName() == "AMD" && Note.getType() == NT_AMDGPU_HSA_ISA) {
      if (Note.getDesc().size() <= sizeof(amdgpu_hsa_note_isa_s)) {
        NoteStatus = AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
        return true;
      }

      auto NoteIsa = reinterpret_cast<const amdgpu_hsa_note_isa_s *>(
          Note.getDesc().data());

      if (!NoteIsa->vendor_name_size || !NoteIsa->architecture_name_size) {
        NoteStatus = AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
        return true;
      }

      auto VendorName = StringRef(NoteIsa->vendor_and_architecture_name,
                                  NoteIsa->vendor_name_size - 1);
      auto ArchitectureName = StringRef(NoteIsa->vendor_and_architecture_name +
                                            NoteIsa->vendor_name_size,
                                        NoteIsa->architecture_name_size - 1);
      auto EFlags = Obj->getELFFile()->getHeader().e_flags;

      std::string NoteIsaName;
      NoteStatus = getNoteIsaName(VendorName, ArchitectureName, NoteIsa->major,
                                  NoteIsa->minor, NoteIsa->stepping, EFlags,
                                  NoteIsaName);
      if (NoteStatus)
        return true;

      if (IsaName)
        memcpy(IsaName, NoteIsaName.c_str(), *Size);
      else
        *Size = NoteIsaName.size() + 1;

      return true;
    }
    return false;
  };
  if (auto ElfStatus = processElfNotes(Obj, ProcessNote))
    return ElfStatus;
  if (NoteStatus)
    return NoteStatus;
  return AMD_COMGR_STATUS_SUCCESS;
}

template <class ELFT>
static amd_comgr_status_t getElfIsaNameV3(const ELFObjectFile<ELFT> *Obj,
                                          size_t *Size, char *IsaName) {
  auto EHdr = Obj->getELFFile()->getHeader();

  std::string ElfIsaName;

  switch (EHdr.e_ident[ELF::EI_CLASS]) {
  case ELF::ELFCLASSNONE:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  case ELF::ELFCLASS32:
    ElfIsaName += "r600";
    break;
  case ELF::ELFCLASS64:
    ElfIsaName += "amdgcn";
    break;
  }

  ElfIsaName += "-amd-";

  switch (EHdr.e_ident[ELF::EI_OSABI]) {
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

  switch (EHdr.e_flags & ELF::EF_AMDGPU_MACH) {
  case ELF::EF_AMDGPU_MACH_NONE:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  case ELF::EF_AMDGPU_MACH_R600_R600:
    ElfIsaName += "r600";
    break;
  case ELF::EF_AMDGPU_MACH_R600_R630:
    ElfIsaName += "r630";
    break;
  case ELF::EF_AMDGPU_MACH_R600_RS880:
    ElfIsaName += "rs880";
    break;
  case ELF::EF_AMDGPU_MACH_R600_RV670:
    ElfIsaName += "rv670";
    break;
  case ELF::EF_AMDGPU_MACH_R600_RV710:
    ElfIsaName += "rv710";
    break;
  case ELF::EF_AMDGPU_MACH_R600_RV730:
    ElfIsaName += "rv730";
    break;
  case ELF::EF_AMDGPU_MACH_R600_RV770:
    ElfIsaName += "rv770";
    break;
  case ELF::EF_AMDGPU_MACH_R600_CEDAR:
    ElfIsaName += "cedar";
    break;
  case ELF::EF_AMDGPU_MACH_R600_CYPRESS:
    ElfIsaName += "cypress";
    break;
  case ELF::EF_AMDGPU_MACH_R600_JUNIPER:
    ElfIsaName += "juniper";
    break;
  case ELF::EF_AMDGPU_MACH_R600_REDWOOD:
    ElfIsaName += "redwood";
    break;
  case ELF::EF_AMDGPU_MACH_R600_SUMO:
    ElfIsaName += "sumo";
    break;
  case ELF::EF_AMDGPU_MACH_R600_BARTS:
    ElfIsaName += "barts";
    break;
  case ELF::EF_AMDGPU_MACH_R600_CAICOS:
    ElfIsaName += "caicos";
    break;
  case ELF::EF_AMDGPU_MACH_R600_CAYMAN:
    ElfIsaName += "cayman";
    break;
  case ELF::EF_AMDGPU_MACH_R600_TURKS:
    ElfIsaName += "turks";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX600:
    ElfIsaName += "gfx600";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX601:
    ElfIsaName += "gfx601";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX700:
    ElfIsaName += "gfx700";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX701:
    ElfIsaName += "gfx701";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX702:
    ElfIsaName += "gfx702";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX703:
    ElfIsaName += "gfx703";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX704:
    ElfIsaName += "gfx704";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX801:
    ElfIsaName += "gfx801";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX802:
    ElfIsaName += "gfx802";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX803:
    ElfIsaName += "gfx803";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX810:
    ElfIsaName += "gfx810";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX900:
    ElfIsaName += "gfx900";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX902:
    ElfIsaName += "gfx902";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX904:
    ElfIsaName += "gfx904";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX906:
    ElfIsaName += "gfx906";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX908:
    ElfIsaName += "gfx908";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX909:
    ElfIsaName += "gfx909";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1010:
    ElfIsaName += "gfx1010";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1011:
    ElfIsaName += "gfx1011";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1012:
    ElfIsaName += "gfx1012";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1030:
    ElfIsaName += "gfx1030";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1031:
    ElfIsaName += "gfx1031";
    break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1032:
    ElfIsaName += "gfx1032";
    break;
  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (EHdr.e_flags & ELF::EF_AMDGPU_XNACK)
    ElfIsaName += "+xnack";
  if (EHdr.e_flags & ELF::EF_AMDGPU_SRAM_ECC)
    ElfIsaName += "+sram-ecc";

  if (IsaName)
    memcpy(IsaName, ElfIsaName.c_str(), *Size);
  else
    *Size = ElfIsaName.size() + 1;

  return AMD_COMGR_STATUS_SUCCESS;
}

template <class ELFT>
static amd_comgr_status_t getElfIsaNameImpl(const ELFObjectFile<ELFT> *Obj,
                                            size_t *Size, char *IsaName) {
  auto EHdr = Obj->getELFFile()->getHeader();
  if (EHdr.e_machine == ELF::EM_AMDGPU)
    return getElfIsaNameV3(Obj, Size, IsaName);
  return getElfIsaNameV2(Obj, Size, IsaName);
}

amd_comgr_status_t getElfIsaName(DataObject *DataP, size_t *Size,
                                 char *IsaName) {
  auto ObjOrErr = getELFObjectFileBase(DataP);
  if (errorToBool(ObjOrErr.takeError()))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  auto Obj = ObjOrErr->get();

  if (auto ELF32LE = dyn_cast<ELF32LEObjectFile>(Obj))
    return getElfIsaNameImpl(ELF32LE, Size, IsaName);
  if (auto ELF64LE = dyn_cast<ELF64LEObjectFile>(Obj))
    return getElfIsaNameImpl(ELF64LE, Size, IsaName);
  if (auto ELF32BE = dyn_cast<ELF32BEObjectFile>(Obj))
    return getElfIsaNameImpl(ELF32BE, Size, IsaName);
  auto ELF64BE = dyn_cast<ELF64BEObjectFile>(Obj);
  return getElfIsaNameImpl(ELF64BE, Size, IsaName);
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

static_assert((sizeof(SupportedIsas) / sizeof(*SupportedIsas)) ==
                  (sizeof(IsaInfos) / sizeof(*IsaInfos)),
              "all SupportedIsas must have matching IsaInfos");

size_t getIsaCount() { return sizeof(SupportedIsas) / sizeof(*SupportedIsas); }

const char *getIsaName(size_t Index) { return SupportedIsas[Index]; }

amd_comgr_status_t getIsaMetadata(StringRef IsaName,
                                  llvm::msgpack::Document &Doc) {
  auto IsaIterator = find(SupportedIsas, IsaName);
  if (IsaIterator == std::end(SupportedIsas))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  auto IsaIndex = std::distance(std::begin(SupportedIsas), IsaIterator);

  TargetIdentifier Ident;
  if (auto Status = parseTargetIdentifier(IsaName, Ident))
    return Status;

  auto Root = Doc.getRoot().getMap(/*Convert=*/true);

  Root["Name"] = Doc.getNode(IsaName, /*Copy=*/true);
  Root["Architecture"] = Doc.getNode(Ident.Arch, /*Copy=*/true);
  Root["Vendor"] = Doc.getNode(Ident.Vendor, /*Copy=*/true);
  Root["OS"] = Doc.getNode(Ident.OS, /*Copy=*/true);
  Root["Environment"] = Doc.getNode(Ident.Environ, /*Copy=*/true);
  Root["Processor"] = Doc.getNode(Ident.Processor, /*Copy=*/true);
  auto XNACKEnabled = false;
  auto FeaturesNode = Doc.getArrayNode();
  for (size_t I = 0; I < Ident.Features.size(); ++I) {
    FeaturesNode.push_back(Doc.getNode(Ident.Features[I], /*Copy=*/true));
    if (Ident.Features[I] == "xnack")
      XNACKEnabled = true;
  }
  Root["Features"] = FeaturesNode;

  Root["XNACKEnabled"] =
      Doc.getNode(std::to_string(XNACKEnabled), /*Copy=*/true);

  auto Info = IsaInfos[IsaIndex];
  Root["TrapHandlerEnabled"] =
      Doc.getNode(std::to_string(Info.TrapHandlerEnabled), /*Copy=*/true);
  Root["LocalMemorySize"] =
      Doc.getNode(std::to_string(Info.LocalMemorySize), /*Copy=*/true);
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

bool isValidIsaName(StringRef IsaName) {
  auto IsaIterator = find(SupportedIsas, IsaName);
  if (IsaIterator == std::end(SupportedIsas))
    return false;

  TargetIdentifier Ident;
  return parseTargetIdentifier(IsaName, Ident) == AMD_COMGR_STATUS_SUCCESS;
}

} // namespace metadata
} // namespace COMGR
