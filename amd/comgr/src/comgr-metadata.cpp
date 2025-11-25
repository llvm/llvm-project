//===- comgr-metadata.cpp - Metadata query functions ----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains functions used to implement the Comgr metadata query
/// APIs, including:
///   amd_comgr_get_isa_count()
///   amd_comgr_get_isa_name()
///   amd_comgr_action_info_set_isa_name()
///   amd_comgr_get_isa_metadata()
///   amd_comgr_lookup_code_object()
///
//===----------------------------------------------------------------------===//

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

namespace {
Expected<std::unique_ptr<ELFObjectFileBase>>
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
bool mergeNoteRecords(llvm::msgpack::DocNode &From, llvm::msgpack::DocNode &To,
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
bool processNote(const Elf_Note<ELFT> &Note, DataMeta *MetaP,
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
amd_comgr_status_t getElfMetadataRoot(const ELFObjectFile<ELFT> *Obj,
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
} // namespace

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
  bool ImageSupport;
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
  // TODO: Update this to AvailableNumVGPRs to be more accurate
  unsigned AddressableNumVGPRs;
} IsaInfos[] = {
#define HANDLE_ISA(TARGET_TRIPLE, PROCESSOR, SRAMECC_SUPPORTED,                \
                   XNACK_SUPPORTED, ELF_MACHINE, TRAP_HANDLER_ENABLED,         \
                   IMAGE_SUPPORT, LDS_SIZE, LDS_BANK_COUNT, EUS_PER_CU,        \
                   MAX_WAVES_PER_CU, MAX_FLAT_WORK_GROUP_SIZE,                 \
                   SGPR_ALLOC_GRANULE, TOTAL_NUM_SGPRS, ADDRESSABLE_NUM_SGPRS, \
                   VGPR_ALLOC_GRANULE, TOTAL_NUM_VGPRS, ADDRESSABLE_NUM_VGPRS) \
  {TARGET_TRIPLE "-" PROCESSOR,                                                \
   PROCESSOR,                                                                  \
   SRAMECC_SUPPORTED,                                                          \
   XNACK_SUPPORTED,                                                            \
   ELF::ELF_MACHINE,                                                           \
   TRAP_HANDLER_ENABLED,                                                       \
   IMAGE_SUPPORT,                                                              \
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
namespace {
bool getMachInfo(unsigned Mach, std::string &Processor, bool &SrameccSupported,
                 bool &XnackSupported) {
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

template <class ELFT>
amd_comgr_status_t getElfIsaNameFromElfHeader(const ELFObjectFile<ELFT> *Obj,
                                              std::string &ElfIsaName) {
  auto ElfHeader = Obj->getELFFile().getHeader();

  if (ElfHeader.e_ident[ELF::EI_CLASS] == ELF::ELFCLASS64)
    ElfIsaName += "amdgcn";

  if (ElfHeader.e_machine != ELF::EM_AMDGPU) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ElfIsaName += "-amd-";

  if (ElfHeader.e_ident[ELF::EI_OSABI] == ELF::ELFOSABI_AMDGPU_HSA)
    ElfIsaName += "amdhsa";
  else
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  ElfIsaName += "--";

  std::string Processor;
  bool SrameccSupported, XnackSupported;
  if (!getMachInfo(ElfHeader.e_flags & ELF::EF_AMDGPU_MACH, Processor,
                   SrameccSupported, XnackSupported)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ElfIsaName += Processor;

  switch (ElfHeader.e_ident[ELF::EI_ABIVERSION]) {
  case ELF::ELFABIVERSION_AMDGPU_HSA_V4:
  case ELF::ELFABIVERSION_AMDGPU_HSA_V5:
  case ELF::ELFABIVERSION_AMDGPU_HSA_V6: {
    // Note for V6: generic version is not part of the ISA name so
    // we don't have to parse it.
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
} // namespace

amd_comgr_status_t getElfIsaName(DataObject *DataP, std::string &IsaName) {
  auto ObjOrErr = getELFObjectFileBase(DataP);
  if (errorToBool(ObjOrErr.takeError())) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  auto *Obj = ObjOrErr->get();

  if (auto *ELF64LE = dyn_cast<ELF64LEObjectFile>(Obj))
    return getElfIsaNameFromElfHeader(ELF64LE, IsaName);
  else
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
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
  Root["ImageSupport"] =
      Doc.getNode(std::to_string(Info.ImageSupport), /*Copy=*/true);
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

namespace {
size_t constexpr strLiteralLength(char const *Str) {
  size_t I = 0;
  while (Str[I]) {
    ++I;
  }
  return I;
}

constexpr const char *OffloadKindHip = "hip";
constexpr const char *OffloadKindHipV4 = "hipv4";
constexpr const char *OffloadKindHcc = "hcc";
constexpr const char *ClangOffloadBundlerMagic = "__CLANG_OFFLOAD_BUNDLE__";
constexpr size_t OffloadBundleMagicLen =
    strLiteralLength(ClangOffloadBundlerMagic);
} // namespace

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

  if (Magic != ClangOffloadBundlerMagic) {
    if (DataP->DataKind == AMD_COMGR_DATA_KIND_BYTES) {
      return lookUpCodeObjectInSharedObject(DataP, QueryList, QueryListSize);
    }
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  uint64_t NumOfCodeObjects = 0;
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
    uint64_t BundleEntryCodeObjectSize = 0;
    uint64_t BundleEntryCodeObjectOffset = 0;
    uint64_t BundleEntryIDSize = 0;
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
    if (OffloadAndTargetId.first != OffloadKindHip &&
        OffloadAndTargetId.first != OffloadKindHipV4 &&
        OffloadAndTargetId.first != OffloadKindHcc) {
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
    if (Seen == (int)QueryListSize) {
      break;
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace metadata
} // namespace COMGR
