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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include <cstddef>
#include <iostream>
#include <optional>

using namespace llvm;
using namespace llvm::object;

namespace COMGR {
namespace metadata {

template <typename ELFT> using Elf_Note = typename ELFT::Note;

namespace {
Expected<std::unique_ptr<ELFObjectFileBase>>
getELFObjectFileBase(MemoryBufferRef MB) {
  Expected<std::unique_ptr<ObjectFile>> ObjOrErr =
      ObjectFile::createELFObjectFile(MB);

  if (auto Err = ObjOrErr.takeError()) {
    return std::move(Err);
  }

  return unique_dyn_cast<ELFObjectFileBase>(std::move(*ObjOrErr));
}

Expected<std::unique_ptr<ELFObjectFileBase>>
getELFObjectFileBase(DataObject *DataP) {
  return getELFObjectFileBase(
      MemoryBufferRef(StringRef(DataP->Data, DataP->Size), ""));
}

// PAL currently produces MsgPack metadata in a note with this ID.
// FIXME: Unify with HSA note types?
#define PAL_METADATA_NOTE_TYPE 13

// Try to merge "amdhsa.kernels" from DocNode @p From to @p To.
// The merge is allowed only if
// 1. "amdhsa.version" exists and is same.
// 2. "amdhsa.kernels" exists in both nodes.
//
// "amdhsa.printf" is copied from @p From if @p To doesn't have it.
// If both have it, the merge is allowed only if they are identical
// (as expected from LTO partitions sharing the same module-level
// llvm.printf.fmts metadata).
//
// If merge is possible the function merges Kernel records
// to @p To and returns @c true.
// @p DestDoc is the document into which merged nodes are deep copied.
bool mergeNoteRecords(llvm::msgpack::DocNode &From, llvm::msgpack::DocNode &To,
                      const StringRef VersionStrKey,
                      const StringRef PrintfStrKey,
                      const StringRef KernelStrKey,
                      llvm::msgpack::Document &DestDoc) {
  if (!From.isMap()) {
    return false;
  }

  if (To.isEmpty()) {
    To = DestDoc.copyNode(From);
    return true;
  }

  assert(To.isMap());

  if (From.getMap().find(PrintfStrKey) != From.getMap().end()) {
    if (To.getMap().find(PrintfStrKey) != To.getMap().end()) {
      if (From.getMap()[PrintfStrKey] != To.getMap()[PrintfStrKey])
        return false;
    } else {
      To.getMap()[PrintfStrKey] = DestDoc.copyNode(From.getMap()[PrintfStrKey]);
    }
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
    ToKernelRecords.push_back(DestDoc.copyNode(Kernel));
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

    // Use a temporary document for parsing to avoid invalidating Root.
    // DocNode contains pointers to memory owned by its Document, so reusing
    // the same Document for parsing would invalidate nodes accumulated in Root.
    llvm::msgpack::Document TempDoc;
    if (!TempDoc.readFromBlob(MetaP->MetaDoc->RawDocumentList.back(), false)) {
      return false;
    }

    return mergeNoteRecords(TempDoc.getRoot(), Root, "amdhsa.version",
                            "amdhsa.printf", "amdhsa.kernels",
                            MetaP->MetaDoc->Document);
  }
  return false;
}

template <class ELFT>
llvm::Error getElfMetadataRoot(const ELFObjectFile<ELFT> *Obj,
                               DataMeta *MetaP) {
  bool Found = false;
  llvm::msgpack::DocNode Root;
  const ELFFile<ELFT> &ELFFile = Obj->getELFFile();

  auto ProgramHeadersOrError = ELFFile.program_headers();
  if (!ProgramHeadersOrError)
    return ProgramHeadersOrError.takeError();

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

    if (Err)
      return Err;
  }

  if (Found) {
    MetaP->MetaDoc->Document.getRoot() = Root;
    MetaP->DocNode = MetaP->MetaDoc->Document.getRoot();
    return Error::success();
  }

  auto SectionsOrError = ELFFile.sections();
  if (!SectionsOrError)
    return SectionsOrError.takeError();

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

    if (Err)
      return Err;
  }

  if (Found) {
    MetaP->MetaDoc->Document.getRoot() = Root;
    MetaP->DocNode = MetaP->MetaDoc->Document.getRoot();
    return Error::success();
  }

  return createStringError(inconvertibleErrorCode(),
                           "no AMDGPU metadata note found in code object");
}
} // namespace

namespace {
llvm::Error getMetadataRoot(ELFObjectFileBase *Obj, DataMeta *MetaP) {
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
} // namespace

amd_comgr_status_t getMetadataRoot(DataObject *DataP, DataMeta *MetaP) {
  auto ObjOrErr = getELFObjectFileBase(DataP);
  if (errorToBool(ObjOrErr.takeError())) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  // The internal walker preserves the underlying LLVM error; the DataObject
  // boundary keeps the historical status-only contract, so map any failure
  // (malformed input or an absent note) to a single invalid-argument status.
  if (errorToBool(getMetadataRoot(ObjOrErr->get(), MetaP)))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  return AMD_COMGR_STATUS_SUCCESS;
}

namespace {
struct IsaInfo {
  std::string IsaName;
  AMDGPU::GPUKind Kind;
};

ArrayRef<IsaInfo> getIsaInfos() {
  // Own the names for the lifetime of the library: getIsaName returns pointers
  // into these strings to callers of amd_comgr_get_isa_name.
  static const auto IsaInfos = [] {
    SmallVector<StringRef> Processors;
    AMDGPU::fillValidArchListAMDGCN(Processors);
    SmallVector<IsaInfo> Infos;
    for (StringRef Processor : Processors) {
      AMDGPU::GPUKind Kind = AMDGPU::parseArchAMDGCN(Processor);
      // TargetParser also lists aliases such as "tahiti". Enumerate only the
      // canonical names, including generic targets and target variants.
      if (Processor != AMDGPU::getArchNameAMDGCN(Kind))
        continue;
      Infos.push_back({("amdgcn-amd-amdhsa--" + Processor).str(), Kind});
    }
    return Infos;
  }();
  return IsaInfos;
}

// Every AMDGCN target Comgr supports ships a trap handler. Kept as a query so
// per-processor exceptions can be added without reintroducing a table column.
bool isTrapHandlerEnabled(AMDGPU::GPUKind Kind) {
  switch (Kind) {
  default:
    return true;
  }
}
} // namespace

size_t getIsaCount() { return getIsaInfos().size(); }

// NOLINTNEXTLINE(readability-identifier-naming)
typedef struct amdgpu_hsa_note_code_object_version_s {
  uint32_t major_version; // NOLINT(readability-identifier-naming)
  uint32_t minor_version; // NOLINT(readability-identifier-naming)
} amdgpu_hsa_note_code_object_version_t;

// NOLINTNEXTLINE(readability-identifier-naming)
namespace {
bool getMachInfo(unsigned Mach, std::string &Processor) {
  StringRef Name;
  switch (Mach) {
#define X(NUM, ENUM, NAME)                                                     \
  case ELF::ENUM:                                                              \
    Name = NAME;                                                               \
    break;
    AMDGPU_MACH_LIST(X)
#undef X
  default:
    return false;
  }

  // AMDGPU_MACH_LIST also covers the R600 machs, which are not AMDGCN
  // processors.
  if (AMDGPU::parseArchAMDGCN(Name) == AMDGPU::GK_NONE)
    return false;

  Processor = Name.str();
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
  if (!getMachInfo(ElfHeader.e_flags & ELF::EF_AMDGPU_MACH, Processor)) {
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

static Expected<std::string> getElfIsaName(MemoryBufferRef MB) {
  auto ObjOrErr = getELFObjectFileBase(MB);
  if (!ObjOrErr)
    return ObjOrErr.takeError();

  auto *ELF64LE = dyn_cast<ELF64LEObjectFile>(ObjOrErr->get());
  if (!ELF64LE)
    return createStringError(inconvertibleErrorCode(),
                             "code object is not a 64-bit little-endian ELF");

  std::string IsaName;
  if (getElfIsaNameFromElfHeader(ELF64LE, IsaName) != AMD_COMGR_STATUS_SUCCESS)
    return createStringError(inconvertibleErrorCode(),
                             "failed to derive ISA name from ELF header");
  return IsaName;
}

amd_comgr_status_t getElfIsaName(DataObject *DataP, std::string &IsaName) {
  auto IsaOrErr =
      getElfIsaName(MemoryBufferRef(StringRef(DataP->Data, DataP->Size), ""));
  if (!IsaOrErr) {
    consumeError(IsaOrErr.takeError());
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  IsaName = std::move(*IsaOrErr);
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t getIsaIndex(StringRef TargetIDString, size_t &Index,
                               StringRef *Processor) {
  // Treat bare amdgpu as its legacy alias even when the LLVM TargetID parser
  // requires the amdgpu spelling to include a subarch.
  // TODO: Remove this normalization once Comgr's LLVM baseline includes the
  // reland of llvm/llvm-project#213847 and accepts bare amdgpu directly.
  // Keep the alias and round-trip tests when removing the workaround.
  std::string LegacyTargetID;
  if (TargetIDString.consume_front("amdgpu-")) {
    LegacyTargetID = (Twine("amdgcn-") + TargetIDString).str();
    TargetIDString = LegacyTargetID;
  }

  // Validate the target ID and resolve the processor (derived from the subarch
  // when the name omits it).
  std::optional<AMDGPU::TargetID> TID =
      AMDGPU::TargetID::parseTargetIDString(TargetIDString);
  if (!TID || TID->getGPUKind() == AMDGPU::GK_NONE) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  StringRef CanonicalProcessor = AMDGPU::getArchNameAMDGCN(TID->getGPUKind());
  if (Processor) {
    *Processor = CanonicalProcessor;
  }

  // TargetID already validated the triple, so match only the resolved GPU kind.
  ArrayRef<IsaInfo> IsaInfos = getIsaInfos();
  auto IsaIterator = llvm::find_if(IsaInfos, [&](const IsaInfo &Info) {
    return TID->getGPUKind() == Info.Kind;
  });
  if (IsaIterator == IsaInfos.end()) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  Index = std::distance(IsaInfos.begin(), IsaIterator);

  return AMD_COMGR_STATUS_SUCCESS;
}

bool isSupportedFeature(size_t IsaIndex, StringRef Feature) {
  if (Feature.empty() ||
      (Feature.take_back() != "+" && Feature.take_back() != "-")) {
    return false;
  }

  const AMDGPU::AMDGPUFeatureBitset &Features =
      AMDGPU::getFeatureBitset(getIsaInfos()[IsaIndex].Kind);

  return (Feature.drop_back() == "xnack" &&
          Features.test(AMDGPU::FEAT_XNACK_ON_OFF_MODES)) ||
         (Feature.drop_back() == "sramecc" &&
          Features.test(AMDGPU::FEAT_SRAMECC_ON_OFF_MODES));
}

const char *getIsaName(size_t Index) {
  return getIsaInfos()[Index].IsaName.c_str();
}

// ISA metadata schema for amd_comgr_get_isa_metadata(). Keep this description
// in sync with amd/comgr/docs/ISAMetadata.md, which provides examples and LLVM
// reference links. These are target properties, not live device information or
// per-kernel resource usage. Code object metadata has a separate schema.
//
// The root is a map. All values are strings except Features, which is a map
// of strings. Numeric values use decimal notation; Boolean flags use "0"/"1".
//
// Target identity (preserving the spelling in the requested ISA name):
//   Name: Complete requested ISA name, including feature settings.
//   Architecture: Target triple's architecture component.
//   Vendor: Target triple's vendor component, normally "amd".
//   OS: Target triple's OS/runtime component, such as "amdhsa".
//   Environment: Target triple's environment component; may be empty.
//   Processor: Processor component of the requested name, such as "gfx90a".
//   Version: ISA metadata format version, currently "1.0.0"; separate from
//     the GPU ISA revision, Comgr library version, and code object version.
//
// Target feature settings:
//   Features: Map containing the selectable xnack and sramecc settings.
//     xnack: XNACK memory-fault replay setting.
//     sramecc: SRAM error-correcting code (ECC) setting.
//     Each value is "any" when unspecified, permitting either on or off.
//     An explicit '+' selects "on" (enabled required); '-' selects "off"
//     (disabled required). These requirements apply to the requested target ID.
//     Features without on/off modes are absent; requesting one explicitly is
//     an error. Hardwired-on features, such as gfx1250 xnack, are also absent.
//
// Capabilities:
//   TrapHandlerEnabled: Whether Comgr models trap handling as enabled;
//     currently "1" for every supported AMDGCN target. Does not report a
//     trap handler's register reservation or inspect an installed handler.
//   ImageSupport: Whether the target supports AMDGPU image instructions.
//
// Memory and execution limits:
//   LocalMemorySize: Total LDS bytes shared by workgroups in the full physical
//     block, from getLocalMemorySize(Kind, true). On RDNA and CDNA5 this is
//     a WGP capacity; it is not the allocation limit for one workgroup.
//   LDSBankCount: LDS banks in LLVM's target model, forwarded from
//     getLDSBankCount() without CU/WGP scaling. The bank set is
//     target-dependent.
//   EUsPerCU: SIMDs per physical CU: two with FEAT_GFX10_INSTS, otherwise four.
//   MaxWavesPerCU: Maximum resident waves per physical CU, computed
//     as getMaxWavesPerEU() * EUsPerCU, before kernel resource restrictions.
//   MaxFlatWorkGroupSize: Compiler-supported maximum work-items per
//     workgroup, currently 1024; limits the product of the dimensions.
//   These values do not select a kernel's CU/WGP mode. Account for that mode
//   and the different LDS and execution-count domains when computing occupancy.
//
// Scalar registers (one SGPR holds a 32-bit value shared by a wavefront):
//   SGPRAllocGranule: SGPRs per allocation unit in LLVM's target model.
//   TotalNumSGPRs: SGPR capacity per SIMD in LLVM's occupancy model only on
//     targets where SGPRs limit occupancy; not physical capacity on GFX10+.
//   AddressableNumSGPRs: Maximum SGPRs addressable by one wavefront, including
//     target-specific restrictions such as the SGPR initialization workaround.
//   On GFX10+, SGPRAllocGranule and AddressableNumSGPRs both report the fixed
//   allowance of 106 normal SGPRs per wave, not a physical allocation quantum.
//   TotalNumSGPRs still reports 800; LLVM does not use it to limit occupancy
//   on these targets. Kernel ABI and register reservations can further
//   constrain the register budget.
//
// Vector registers (one VGPR holds a 32-bit value for each wavefront lane):
//   VGPRAllocGranule: VGPRs per allocation unit for a wavefront.
//   TotalNumVGPRs: Wave-sized VGPR capacity per SIMD, shared by resident waves.
//   AddressableNumVGPRs: Maximum VGPRs addressable by one wavefront; includes
//     addressable AGPR capacity on targets with a unified register file.
//   All three use wave32 where FEAT_GFX10_INSTS is present, otherwise wave64.
//   The addressable count has target-specific wave-size rules; it cannot be
//   converted by universally halving it for wave64. The queries describe
//   static allocation without a kernel's dynamic VGPR block size.
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

  AMDGPU::GPUKind Kind = getIsaInfos()[IsaIndex].Kind;
  const AMDGPU::AMDGPUFeatureBitset &Features = AMDGPU::getFeatureBitset(Kind);

  auto FeaturesNode = Doc.getMapNode();
  if (Features.test(AMDGPU::FEAT_XNACK_ON_OFF_MODES)) {
    FeaturesNode["xnack"] = Doc.getNode("any", /*Copy=*/true);
  }
  if (Features.test(AMDGPU::FEAT_SRAMECC_ON_OFF_MODES)) {
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

  Root["TrapHandlerEnabled"] =
      Doc.getNode(std::to_string(isTrapHandlerEnabled(Kind)), /*Copy=*/true);
  Root["ImageSupport"] = Doc.getNode(
      std::to_string(Features.test(AMDGPU::FEAT_IMAGE_INSTS)), /*Copy=*/true);
  // Report total LDS capacity in full-SIMD mode.
  Root["LocalMemorySize"] = Doc.getNode(
      std::to_string(AMDGPU::getLocalMemorySize(Kind, /*FullSIMDMode=*/true)),
      /*Copy=*/true);
  // Report physical-CU counts independently of the workgroup execution mode.
  // GFX10+ has two SIMDs per CU; earlier targets have four.
  unsigned EUsPerCU = Features.test(AMDGPU::FEAT_GFX10_INSTS) ? 2 : 4;
  Root["EUsPerCU"] = Doc.getNode(std::to_string(EUsPerCU), /*Copy=*/true);
  Root["MaxWavesPerCU"] = Doc.getNode(
      std::to_string(AMDGPU::getMaxWavesPerEU(Kind) * EUsPerCU), /*Copy=*/true);
  Root["MaxFlatWorkGroupSize"] = Doc.getNode(
      std::to_string(AMDGPU::getMaxFlatWorkGroupSize()), /*Copy=*/true);
  Root["SGPRAllocGranule"] =
      Doc.getNode(std::to_string(AMDGPU::getSGPRAllocGranule(Kind)),
                  /*Copy=*/true);
  Root["TotalNumSGPRs"] = Doc.getNode(
      std::to_string(AMDGPU::getTotalNumSGPRs(Kind)), /*Copy=*/true);
  Root["AddressableNumSGPRs"] = Doc.getNode(
      std::to_string(AMDGPU::getAddressableNumSGPRs(Kind)), /*Copy=*/true);
  // Report wave32 VGPR counts and allocation granularity wherever wave32
  // exists.
  bool IsWave32 = Features.test(AMDGPU::FEAT_GFX10_INSTS);
  Root["VGPRAllocGranule"] =
      Doc.getNode(std::to_string(AMDGPU::getVGPRAllocGranule(Kind, IsWave32)),
                  /*Copy=*/true);
  Root["TotalNumVGPRs"] =
      Doc.getNode(std::to_string(AMDGPU::getTotalNumVGPRs(Kind, IsWave32)),
                  /*Copy=*/true);
  Root["AddressableNumVGPRs"] = Doc.getNode(
      std::to_string(AMDGPU::getAddressableNumVGPRs(Kind, IsWave32)),
      /*Copy=*/true);
  Root["LDSBankCount"] =
      Doc.getNode(std::to_string(AMDGPU::getLDSBankCount(Kind)), /*Copy=*/true);

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
    // The encoded size may include a null terminator; strip it.
    BundleEntryID = BundleEntryID.rtrim('\0');

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

amd_comgr_status_t parseTargetIdentifier(StringRef IdentStr,
                                         TargetIdentifier &Ident) {
  SmallVector<StringRef, 5> IsaNameComponents;
  IdentStr.split(IsaNameComponents, '-', 4);
  if (IsaNameComponents.size() != 5) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  Ident.Arch = IsaNameComponents[0];
  Ident.Vendor = IsaNameComponents[1];
  Ident.OS = IsaNameComponents[2];
  Ident.Environ = IsaNameComponents[3];

  Ident.Features.clear();
  IsaNameComponents[4].split(Ident.Features, ':');

  Ident.Processor = Ident.Features[0];
  Ident.Features.erase(Ident.Features.begin());

  if (IdentStr == "spirv64-amd-amdhsa--amdgcnspirv" ||
      IdentStr == "spirv64-amd-amdhsa-unknown-amdgcnspirv") {
    // Features not supported for SPIR-V
    if (!Ident.Features.empty())
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  size_t IsaIndex;
  amd_comgr_status_t Status = metadata::getIsaIndex(IdentStr, IsaIndex);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    return Status;
  }

  for (auto Feature : Ident.Features) {
    if (!metadata::isSupportedFeature(IsaIndex, Feature)) {
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace COMGR
