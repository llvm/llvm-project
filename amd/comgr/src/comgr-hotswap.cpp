//===- comgr-hotswap.cpp - HotSwap ISA rewriting: public API bridge -------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "comgr-hotswap-internal.h"
#include "comgr.h"

#include "llvm/TargetParser/AMDGPUTargetParser.h"

#include <optional>
#include <string>

using namespace COMGR;

namespace {

constexpr llvm::StringLiteral Gfx1250B0Feature = "gfx1250-b0-specific";
constexpr llvm::StringLiteral Gfx1250B0FeatureOn = "gfx1250-b0-specific+";
constexpr llvm::StringLiteral Gfx1250B0FeatureOff = "gfx1250-b0-specific-";

struct ParsedHotswapIsa {
  TargetIdentifier Ident;
  std::string CanonicalIsa;
  std::optional<bool> IsB0;
};

static bool parseGfx1250B0Feature(llvm::StringRef Feature,
                                  std::optional<bool> &IsB0) {
  if (Feature == Gfx1250B0FeatureOn) {
    IsB0 = true;
    return true;
  }
  if (Feature == Gfx1250B0FeatureOff) {
    IsB0 = false;
    return true;
  }
  return false;
}

static bool isGfx12_5Processor(llvm::StringRef Processor) {
  llvm::AMDGPU::IsaVersion Version = llvm::AMDGPU::getIsaVersion(Processor);
  return Version.Major == 12 && Version.Minor == 5;
}

static amd_comgr_status_t parseHotswapIsaName(const char *IsaName,
                                              ParsedHotswapIsa &Parsed) {
  Parsed = ParsedHotswapIsa{};

  llvm::SmallVector<llvm::StringRef, 8> Parts;
  llvm::StringRef OriginalIsa(IsaName);
  if (OriginalIsa.empty()) {
    hotswap::log() << "hotswap: error: parseHotswapIsaName: empty ISA name\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  OriginalIsa.split(Parts, ':');
  if (Parts.empty()) {
    hotswap::log() << "hotswap: error: parseHotswapIsaName: empty ISA name\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  llvm::SmallVector<llvm::StringRef, 8> CanonicalParts;
  for (llvm::StringRef Part : Parts) {
    std::optional<bool> IsB0;
    if (parseGfx1250B0Feature(Part, IsB0)) {
      if (Parsed.IsB0) {
        hotswap::log() << "hotswap: error: parseHotswapIsaName: duplicate "
                       << Gfx1250B0Feature << " feature in '" << OriginalIsa
                       << "'\n";
        return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
      }
      Parsed.IsB0 = IsB0;
      continue;
    }
    CanonicalParts.push_back(Part);
  }

  if (CanonicalParts.empty() || CanonicalParts[0].empty()) {
    hotswap::log()
        << "hotswap: error: parseHotswapIsaName: missing canonical ISA in '"
        << OriginalIsa << "'\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  Parsed.CanonicalIsa = CanonicalParts[0].str();
  for (size_t I = 1; I < CanonicalParts.size(); ++I) {
    Parsed.CanonicalIsa += ":";
    Parsed.CanonicalIsa += CanonicalParts[I].str();
  }

  if (parseTargetIdentifier(Parsed.CanonicalIsa, Parsed.Ident)) {
    hotswap::log()
        << "hotswap: error: parseHotswapIsaName: failed to parse ISA '"
        << Parsed.CanonicalIsa << "' from '" << OriginalIsa << "'\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (Parsed.IsB0 && Parsed.Ident.Processor != "gfx1250") {
    hotswap::log() << "hotswap: error: parseHotswapIsaName: "
                   << Gfx1250B0Feature << " is only valid for gfx1250, not '"
                   << Parsed.Ident.Processor << "'\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (Parsed.IsB0)
    Parsed.Ident.Features.push_back(*Parsed.IsB0 ? Gfx1250B0FeatureOn
                                                 : Gfx1250B0FeatureOff);

  return AMD_COMGR_STATUS_SUCCESS;
}

static bool shouldRunB0A0Patches(const ParsedHotswapIsa &Source,
                                 const ParsedHotswapIsa &Target) {
  // Legacy callers only pass gfx1250 today; preserve the existing B0-to-A0
  // rewrite behavior by defaulting an unspecified source to B0 and an
  // unspecified target to A0. If either side explicitly names a stepping, honor
  // that side instead of forcing the legacy path.
  const bool SourceIsB0 = Source.IsB0.value_or(true);
  const bool TargetIsB0 = Target.IsB0.value_or(false);
  return SourceIsB0 && !TargetIsB0;
}

static hotswap::MaskWorkaroundPolicy
getMaskWorkaroundPolicy(const ParsedHotswapIsa &Target, bool StrictMode,
                        bool RunB0A0Patches) {
  if (Target.Ident.Processor != "gfx1250")
    return hotswap::MaskWorkaroundPolicy::None;

  const bool TargetIsB0 = Target.IsB0.value_or(false);
  if (TargetIsB0)
    return StrictMode ? hotswap::MaskWorkaroundPolicy::B0
                      : hotswap::MaskWorkaroundPolicy::None;
  return RunB0A0Patches ? hotswap::MaskWorkaroundPolicy::A0
                        : hotswap::MaskWorkaroundPolicy::None;
}

static amd_comgr_status_t validateHotswapRewriteOptions(
    const amd_comgr_hotswap_rewrite_options_t *RewriteOptions,
    uint64_t &RewriteFlags) {
  if (!RewriteOptions) {
    hotswap::log()
        << "hotswap: error: amd_comgr_hotswap_rewrite_with_options: rewrite "
           "options are required\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (RewriteOptions->size < sizeof(amd_comgr_hotswap_rewrite_options_t)) {
    hotswap::log()
        << "hotswap: error: amd_comgr_hotswap_rewrite_with_options: rewrite "
           "options size "
        << RewriteOptions->size << " is smaller than "
        << sizeof(amd_comgr_hotswap_rewrite_options_t) << "\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  static constexpr uint64_t SupportedFlags =
      AMD_COMGR_HOTSWAP_REWRITE_FLAG_ENTRY_TRAMPOLINES |
      AMD_COMGR_HOTSWAP_REWRITE_FLAG_STRICT_MODE;
  if (RewriteOptions->flags & ~SupportedFlags) {
    hotswap::log() << "hotswap: error: amd_comgr_hotswap_rewrite_with_options: "
                      "unsupported rewrite option flags 0x";
    hotswap::log().write_hex(RewriteOptions->flags & ~SupportedFlags);
    hotswap::log() << "\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  RewriteFlags = RewriteOptions->flags;
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t
hotswapRewrite(amd_comgr_data_t input, const char *source_isa_name,
               const char *target_isa_name, uint64_t RewriteFlags,
               const char *ApiName, amd_comgr_data_t *output) {
  const bool RunEntryTrampolines =
      RewriteFlags & AMD_COMGR_HOTSWAP_REWRITE_FLAG_ENTRY_TRAMPOLINES;
  const bool StrictMode =
      RewriteFlags & AMD_COMGR_HOTSWAP_REWRITE_FLAG_STRICT_MODE;

  DataObject *InputP = DataObject::convert(input);
  if (!InputP) {
    hotswap::log() << "hotswap: error: " << ApiName
                   << ": invalid input data handle\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (!InputP->Data) {
    hotswap::log() << "hotswap: error: " << ApiName << ": input data is null\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (InputP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE) {
    hotswap::log() << "hotswap: error: " << ApiName
                   << ": input data kind must be executable\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (!source_isa_name || !target_isa_name || !output) {
    hotswap::log() << "hotswap: error: " << ApiName
                   << ": source ISA, target ISA, and output handle are "
                      "required\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ParsedHotswapIsa SourceIdent, TargetIdent;
  if (parseHotswapIsaName(source_isa_name, SourceIdent) ||
      parseHotswapIsaName(target_isa_name, TargetIdent))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (!isGfx12_5Processor(SourceIdent.Ident.Processor) ||
      !isGfx12_5Processor(TargetIdent.Ident.Processor)) {
    hotswap::log() << "hotswap: error: " << ApiName
                   << ": only gfx125x processors are supported, got source '"
                   << SourceIdent.Ident.Processor << "' and target '"
                   << TargetIdent.Ident.Processor << "'\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (SourceIdent.Ident.Processor != TargetIdent.Ident.Processor) {
    hotswap::log() << "hotswap: error: " << ApiName
                   << ": processor retargeting is not supported, got source '"
                   << SourceIdent.Ident.Processor << "' and target '"
                   << TargetIdent.Ident.Processor << "'\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hotswap::Gfx1250RewriteOptions Options;
  Options.RunB0A0Patches = SourceIdent.Ident.Processor == "gfx1250" &&
                           shouldRunB0A0Patches(SourceIdent, TargetIdent);
  Options.RunEntryTrampolines = RunEntryTrampolines;
  Options.MaskPolicy =
      getMaskWorkaroundPolicy(TargetIdent, StrictMode, Options.RunB0A0Patches);
  // Fast entry-trampoline path is B0->B0 only. Match the stepping defaults used
  // by shouldRunB0A0Patches: an unspecified source is treated as B0, but the
  // target must explicitly be B0 (unspecified defaults to A0), so the fast path
  // fails closed to the MC path when the target stepping is unknown.
  Options.UseB0B0EntryFastPath = TargetIdent.Ident.Processor == "gfx1250" &&
                                 SourceIdent.IsB0.value_or(true) &&
                                 TargetIdent.IsB0.value_or(false);

  std::unique_ptr<llvm::MemoryBuffer> OutBuffer;
  amd_comgr_status_t Status = hotswap::retargetCodeObject(
      InputP->Data, InputP->Size, TargetIdent.Ident, Options, OutBuffer);
  if (Status != AMD_COMGR_STATUS_SUCCESS)
    return Status;
  if (!OutBuffer) {
    hotswap::log() << "hotswap: error: " << ApiName
                   << ": rewrite returned no output buffer\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  DataObject *OutputP = DataObject::allocate(AMD_COMGR_DATA_KIND_EXECUTABLE);
  if (!OutputP) {
    hotswap::log() << "hotswap: error: " << ApiName
                   << ": output data allocation failed\n";
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (amd_comgr_status_t SetStatus = OutputP->setData(std::move(OutBuffer))) {
    hotswap::log() << "hotswap: error: " << ApiName
                   << ": output setData failed with status " << SetStatus
                   << "\n";
    OutputP->release();
    return SetStatus;
  }

  *output = DataObject::convert(OutputP);
  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_rewrite(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name, amd_comgr_data_t *output) {
  return hotswapRewrite(input, source_isa_name, target_isa_name,
                        AMD_COMGR_HOTSWAP_REWRITE_FLAG_NONE,
                        "amd_comgr_hotswap_rewrite", output);
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_rewrite_with_options(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name,
    const amd_comgr_hotswap_rewrite_options_t *rewrite_options,
    amd_comgr_data_t *output) {
  uint64_t RewriteFlags = AMD_COMGR_HOTSWAP_REWRITE_FLAG_NONE;
  if (amd_comgr_status_t Status =
          validateHotswapRewriteOptions(rewrite_options, RewriteFlags))
    return Status;

  return hotswapRewrite(input, source_isa_name, target_isa_name, RewriteFlags,
                        "amd_comgr_hotswap_rewrite_with_options", output);
}
