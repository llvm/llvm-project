#ifndef HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_HPP
#define HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_HPP

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transpiler {

struct TextSection {
  std::vector<uint8_t> bytes;
  uint64_t offset = 0;
  uint64_t size = 0;
  bool valid = false;
};

struct KernelArgMeta {
  std::string name;
  int offset = 0;
  int size = 0;
  std::string valueKind;
  int addressSpace = -1;
};

// Per-kernel metadata extracted from the AMDGPU code object's MsgPack notes
// + kernel descriptor (`<name>.kd`).
struct KernelMeta {
  std::string name;
  int kernargSegmentSize = 0;
  int groupSegmentFixedSize = 0;
  int privateSegmentFixedSize = 0;
  int maxFlatWorkgroupSize = 256;
  std::vector<KernelArgMeta> args;

  // ---------------------------------------------------------------------
  // Kernel descriptor (KD) raw fields.
  //
  // Populated by extractKernelMeta from the 64-byte amd_kernel_code_t block
  // that lives at the symbol named `<kernelName>.kd` (always in the .rodata
  // section for amdhsa code objects). These fields are the entire
  // surface needed to derive the source-ISA SGPR ABI:
  //
  //   * privateSegmentFixedSize (KD bytes 4-7, mirrored from MsgPack): source
  //     private/scratch bytes per work-item. A non-zero value paired with
  //     `compute_pgm_rsrc2.ENABLE_PRIVATE_SEGMENT` is the launch-time ABI
  //     request that makes ROCR/SPI allocate scratch backing.
  //
  //   * kernelCodeProperties  (KD bytes 56-57): bit field selecting which
  //     `enable_sgpr_*` user SGPRs the loader / packet processor will pre-
  //     populate before kernel entry. See LLVM's AMDHSAKernelDescriptor.h
  //     KERNEL_CODE_PROPERTY_ENABLE_SGPR_* enum for the bit positions.
  //
  //   * kernargPreload        (KD bytes 58-59): packed
  //     {LENGTH[6:0], OFFSET[15:7]} per LLVM's KERNARG_PRELOAD_SPEC enum.
  //     LENGTH=N and OFFSET=K mean: the hardware copies N dwords of kernarg
  //     memory starting at byte (K*4) into user SGPRs immediately above the
  //     `enable_sgpr_*`-selected ones, before kernel entry. This is the
  //     gfx1250-specific "kernarg preload" mechanism that broke our
  //     hardcoded Phase-4 init.
  //
  //   * computePgmRsrc2       (KD bytes 52-55): contains
  //     ENABLE_SGPR_WORKGROUP_ID_{X,Y,Z} / WORKGROUP_INFO bits and the
  //     USER_SGPR_COUNT field (read for verification only — we recompute it
  //     from kernelCodeProperties + kernargPreload.length and assert
  //     equality).
  //
  //   * computePgmRsrc1       (KD bytes 48-51): not strictly required for
  //     the user-SGPR layout, but useful for diagnostics and for future
  //     wave-size-aware decisions. Captured for completeness.
  //
  // `hasKernelDescriptor` is true iff parsing succeeded. We do not silently
  // fall back to a hardcoded layout when it is false — the caller is
  // expected to refuse the lift instead.
  bool hasKernelDescriptor = false;
  uint32_t computePgmRsrc1 = 0;
  uint32_t computePgmRsrc2 = 0;
  uint16_t kernelCodeProperties = 0;
  uint16_t kernargPreload = 0;

  int implicitArgsBase() const {
    int maxEnd = 0;
    for (auto &a : args) {
      if (a.valueKind.rfind("hidden_", 0) == 0)
        continue;
      int end = a.offset + a.size;
      if (end > maxEnd) maxEnd = end;
    }
    return (maxEnd + 7) & ~7;
  }
};

std::vector<uint8_t> readFile(llvm::StringRef path);
TextSection extractTextSection(llvm::ArrayRef<uint8_t> elfData);
std::vector<std::string> listKernelNames(llvm::ArrayRef<uint8_t> elfData);
KernelMeta extractKernelMeta(llvm::ArrayRef<uint8_t> elfData,
                             llvm::StringRef kernelName);
uint64_t findKernelSymbolOffset(llvm::ArrayRef<uint8_t> elfData,
                                llvm::StringRef kernelName);

// Read the AMDGPU target ISA name (e.g. "gfx1250", "gfx942") encoded in
// the ELF e_flags MACH field per
// `EF_AMDGPU_MACH_AMDGCN_GFXLIST` in <llvm/BinaryFormat/ELF.h>. Returns
// the empty string when the ELF is malformed, when the MACH field is not
// an `EF_AMDGPU_MACH_AMDGCN_GFX*` value (R600 / NONE / vendor-extended
// codes), or when the file is not an AMDGPU ELF. Callers that have no
// other ISA source (e.g. raise_cli when the filename lacks `gfx*`)
// should treat an empty return as a hard failure rather than guessing.
std::string detectIsaFromElf(llvm::ArrayRef<uint8_t> elfData);

} // namespace transpiler

#endif
