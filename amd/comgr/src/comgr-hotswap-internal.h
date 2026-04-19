//===- comgr-hotswap-internal.h - HotSwap internal types and declarations -===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal header for the HotSwap ISA rewriting subsystem. Shared by all
/// comgr-hotswap-*.cpp compilation units. Not part of the public COMGR API.
///
/// Module structure:
///   comgr-hotswap-elf.cpp       ELF parsing, binary helpers, trampoline growth
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_INTERNAL_H
#define COMGR_HOTSWAP_INTERNAL_H

#include "amd_comgr.h"
#include "comgr-env.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace COMGR {
namespace hotswap {

// -- Logging ------------------------------------------------------------------
//
// Single output stream for all hotswap diagnostics (errors, warnings, and
// verbose traces). Returns llvm::errs() if AMD_COMGR_EMIT_VERBOSE_LOGS is set
// (via COMGR::env::shouldEmitVerboseLogs()) and llvm::nulls() otherwise, so
// hotswap output stays quiet in normal use but callers can opt in to the full
// diagnostic trail without relinking. Every function that returns a null /
// empty / failure result should emit here with a `"hotswap: error: ..."` or
// `"hotswap: ..."` prefix so the failure path is traceable.

inline llvm::raw_ostream &log() {
  return COMGR::env::shouldEmitVerboseLogs() ? llvm::errs() : llvm::nulls();
}

// -- Trampoline and NOP sled --------------------------------------------------

struct Trampoline {
  uint64_t OriginalOffset = 0;
  uint32_t OriginalSize = 0;
  llvm::SmallVector<uint8_t> Bytes;
};

struct NopSled {
  uint64_t Start = 0;
  uint64_t End = 0;
  uint64_t WritePos = 0;
};

// -- Rewrite rule -------------------------------------------------------------

struct RewriteRule {
  std::string ReplaceMnemonic;
  llvm::SmallVector<uint8_t> ReplaceBytes;
};

// -- Named constants ----------------------------------------------------------

// Minimum valid ELF64 size.
static constexpr uint64_t MinElfSize = sizeof(llvm::ELF::Elf64_Ehdr);

// Kernel descriptor size and RSRC1 offset from upstream
// AMDHSAKernelDescriptor.h.
static constexpr uint64_t KdSize = sizeof(llvm::amdhsa::kernel_descriptor_t);
static constexpr uint64_t KdRsrc1Offset = llvm::amdhsa::COMPUTE_PGM_RSRC1_OFFSET;

// Maximum distance (bytes) between an instruction and a NOP sled for the
// sled to be considered reachable by a single s_branch.
static constexpr int64_t MaxSledDistance = 131072;

// Minimum size (bytes) of a consecutive NOP run to be usable as a sled.
static constexpr uint64_t MinNopSledSize = 8;

// Minimum AMDGPU instruction size (one dword).
static constexpr uint32_t MinInstSize = 4;

// s_branch encoding: 16-bit signed dword offset field.
static constexpr int64_t BranchOffsetMin = -32768;
static constexpr int64_t BranchOffsetMax = 32767;
static constexpr uint32_t BranchOffsetMask = 0xFFFF;

// -- ElfView ------------------------------------------------------------------
//
// Thin wrapper around llvm::object::ELFFile<ELF64LE> that owns the structural
// view of a mutable code-object buffer. The caller retains ownership of the
// bytes; ElfView exposes LLVM's ELF iterators through member methods and
// caches the .text section lookup.

class ElfView {
public:
  using ELFT = llvm::object::ELF64LE;
  using ELFFileT = llvm::object::ELFFile<ELFT>;

  /// Parse \p Data / \p Size into an ElfView. Fails if the bytes are not a
  /// valid ELF64 or if no `.text` section is found.
  static llvm::Expected<ElfView> create(uint8_t *Data, size_t Size);

  ElfView(ElfView &&) = default;
  ElfView &operator=(ElfView &&) = default;
  ElfView(const ElfView &) = delete;
  ElfView &operator=(const ElfView &) = delete;

  const ELFFileT &file() const { return File; }
  size_t size() const { return File.getBufSize(); }

  /// Writable view of the underlying bytes. The caller that constructed this
  /// ElfView via `create(uint8_t *, size_t)` retains ownership of the buffer;
  /// ElfView just exposes a typed, mutable alias onto `ELFFile::base()`. Safe
  /// because the factory was handed a `uint8_t *` and the buffer outlives
  /// this ElfView.
  uint8_t *data() {
    return const_cast<uint8_t *>(File.base());
  }
  const uint8_t *data() const { return File.base(); }

  /// Section header range, cached at construction time. The underlying
  /// storage is the file buffer, which lives at least as long as this
  /// ElfView, so the range is always valid to iterate.
  ELFT::ShdrRange sections() const { return Sections; }

  /// Return the cached `.text` section header. Never null for a successfully
  /// constructed ElfView.
  const ELFT::Shdr *textSection() const { return TextSection; }

  uint64_t textOffset() const { return TextSection->sh_offset; }
  uint64_t textSize() const { return TextSection->sh_size; }
  uint64_t textAddr() const { return TextSection->sh_addr; }

  /// Index of the `.text` section in the section header table.
  unsigned textSectionIndex() const { return TextSectionIndex; }

  /// Pointer into the buffer for the first byte of `.text`.
  uint8_t *textData() { return data() + textOffset(); }
  const uint8_t *textData() const { return data() + textOffset(); }

  /// Find the kernel function symbol whose range includes \p TextOffset.
  /// Returns "" if no matching function symbol exists.
  std::string findKernelAtOffset(uint64_t TextOffset) const;

  /// Pointer to the kernel_descriptor for \p KernelName inside the buffer,
  /// or nullptr if not found.
  uint8_t *findKernelDescriptor(llvm::StringRef KernelName);

  /// Read the VGPR count from the kernel descriptor for \p KernelName.
  /// Returns std::nullopt if the descriptor is not found.
  std::optional<unsigned> getKernelVgprCount(llvm::StringRef KernelName,
                                             unsigned VgprGranuleSize) const;

  /// Update the RSRC1 VGPR/SGPR granule counts in the kernel descriptor for
  /// \p KernelName by adding \p ExtraVgprs / \p ExtraSgprs, using
  /// \p VgprGranuleSize / \p SgprGranuleSize so the call is ISA-agnostic.
  void updateKernelDescriptor(llvm::StringRef KernelName, unsigned ExtraVgprs,
                              unsigned ExtraSgprs, unsigned VgprGranuleSize,
                              unsigned SgprGranuleSize);

  /// Grow the ELF by inserting trampoline bytes after `.text` and adjusting
  /// all section and program headers. Returns a null unique_ptr on failure.
  ///
  /// Invariant: `.text` must be the last SHF_ALLOC section in its load
  /// segment. Any loaded section appearing past `.text` in the file causes
  /// the function to refuse (with a diagnostic on llvm::errs()) rather than
  /// silently emit stale virtual addresses.
  std::unique_ptr<llvm::WritableMemoryBuffer>
  growWithTrampolines(llvm::ArrayRef<Trampoline> Trampolines) const;

private:
  ElfView(ELFFileT File, ELFT::ShdrRange Sections,
          const ELFT::Shdr *TextSection, unsigned TextSectionIndex)
      : File(std::move(File)), Sections(Sections), TextSection(TextSection),
        TextSectionIndex(TextSectionIndex) {}

  ELFFileT File;
  ELFT::ShdrRange Sections;
  const ELFT::Shdr *TextSection;
  unsigned TextSectionIndex;
};

// -- Free-function ELF helpers (no ELF state required) ------------------------

/// Encode an s_branch from \p FromOffset to \p ToOffset using \p SBranchOpcode.
/// Writes MinInstSize bytes to \p OutBytes. Returns false if the delta is
/// unaligned or out of the 16-bit signed dword range.
[[nodiscard]] bool encodeSBranch(uint64_t FromOffset, uint64_t ToOffset,
                                 uint8_t OutBytes[MinInstSize],
                                 uint32_t SBranchOpcode);

/// Overwrite instruction bytes at \p InstOffset with \p Rule.ReplaceBytes,
/// padding remaining bytes with s_nop instructions. Returns false on bounds
/// violation.
[[nodiscard]] bool applyByteReplace(const RewriteRule &Rule,
                                    uint64_t InstOffset, uint32_t InstSize,
                                    uint8_t *Text, uint64_t TextSize,
                                    uint32_t SNopOpcode);

/// Find the nearest NOP sled to \p Offset with at least \p Needed bytes of
/// free space. Returns nullptr if none found within MaxSledDistance.
NopSled *findNearestSled(std::vector<NopSled> &Sleds, uint64_t Offset,
                         uint64_t Needed);

} // namespace hotswap
} // namespace COMGR

#endif // COMGR_HOTSWAP_INTERNAL_H
