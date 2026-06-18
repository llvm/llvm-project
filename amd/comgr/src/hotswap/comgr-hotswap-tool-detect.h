//===- comgr-hotswap-tool-detect.h - gfx target + A0 gate helpers ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hotswap detection helpers (no HSA dep), split out so they can be unit-tested.
// HSA_TOOLS_LIB tool links symbol-hidden static comgr, no StringRef/Elf64_Ehdr

#ifndef COMGR_HOTSWAP_TOOL_DETECT_H
#define COMGR_HOTSWAP_TOOL_DETECT_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace COMGR::hotswap {

// Processor of an ISA name like amdgcn-amd-amdhsa--gfx1250[:feats] (the field
// after arch-vendor-os-environ; same model as comgr's parseTargetIdentifier).
inline std::string extractGfxTarget(const std::string &IsaName) {
  std::string Rest = IsaName;
  for (int I = 0; I < 4; ++I) {
    const std::string::size_type Dash = Rest.find('-');
    if (Dash == std::string::npos) {
      return std::string();
    }
    Rest = Rest.substr(Dash + 1);
  }
  return Rest.substr(0, Rest.find(':'));
}

// Arm only on gfx1250 at ASIC revision A0 (0). Callers must confirm the revision
// query succeeded before calling; a failed query is handled at the call site.
inline bool gateAllowsHotswap(const std::string &Gfx, uint32_t Revision) {
  return Gfx == "gfx1250" && Revision == 0;
}

// True for a 64-bit gfx1250 AMDGPU ELF (aligned-copy header read, e_machine
// checked). Raw ELF64 fields, no LLVM dependency (see file header).
inline bool isGfx1250CodeObject(const void *Data, size_t Size) {
  // Field names follow the ELF spec (cf. llvm::ELF::Elf64_Ehdr).
  // NOLINTBEGIN(readability-identifier-naming)
  struct Elf64Header {
    unsigned char e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
  };
  // NOLINTEND(readability-identifier-naming)
  static_assert(sizeof(Elf64Header) == 64,
                "Elf64Header must match the 64-byte ELF64 file header layout");
  static const unsigned char ElfMagic[4] = {0x7f, 'E', 'L', 'F'};
  constexpr int EiClass = 4;
  constexpr unsigned char ElfClass64 = 2;
  constexpr uint16_t EmAmdgpu = 224;
  constexpr uint32_t EfAmdgpuMach = 0x0ff;
  constexpr uint32_t EfAmdgpuMachGfx1250 = 0x49;

  if (!Data || Size < sizeof(Elf64Header)) {
    return false;
  }
  Elf64Header Header;
  std::memcpy(&Header, Data, sizeof(Header));
  return std::memcmp(Header.e_ident, ElfMagic, sizeof(ElfMagic)) == 0 &&
         Header.e_ident[EiClass] == ElfClass64 &&
         Header.e_machine == EmAmdgpu &&
         (Header.e_flags & EfAmdgpuMach) == EfAmdgpuMachGfx1250;
}

} // namespace COMGR::hotswap

#endif // COMGR_HOTSWAP_TOOL_DETECT_H
