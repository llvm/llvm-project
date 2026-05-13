//===- comgr-hotswap-patch-vop3px2-src2.cpp - VOP3PX2 SRC2 bit fix -------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// In-place bit-field patch for VOP3PX2 V_WMMA_SCALE* instructions.
/// The unused scale_src2 field at bits [58:50] is incorrectly decoded by
/// the SQ as an SGPR reference, causing a 3-cycle SALU stall after WMMA
/// co-execution. Setting this field to the VGPR0 encoding (0x100)
/// prevents the false dependency. Applies to both A0 and B0 steppings.
///
/// VGPR0 is chosen because any VGPR encoding (bit 8 set) eliminates the
/// false SGPR dependency, VGPR0 is always allocated, and it produces the
/// minimal bit-difference from the typical zeroed scale_src2 field.
///
/// VOP3PX2 encoding layout (128-bit / 16-byte instruction):
///   Source of truth: VOP3PX2e::Inst{58-50} in VOP3PInstructions.td
///   Bits [58:50] = scale_src2 (9-bit field, should be don't-care)
///   VGPR0 in a 9-bit SRC field = 0x100 (bit 8 set, bits 7:0 = 0)
///
///   Byte 6 bits [7:2] = scale_src2[5:0]  -> clear to 0
///   Byte 7 bit  [2]   = scale_src2[8]    -> set to 1
///   Byte 7 bits [1:0] = scale_src2[7:6]  -> clear to 0
///
/// This patch handles VOP3PX2 (WMMA) only. The same field layout exists
/// in VOP3PXe (MFMA V_MFMA_SCALE_* on gfx950, VOPInstructions.td:594)
/// but is out of scope here.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {
namespace {

// Bit constants derived from VOP3PX2e in VOP3PInstructions.td.
// See Inst{58-50} = ? (scale_src2, encoding-defined don't-care).
constexpr uint8_t Byte6ScaleSrc2Mask = 0xFC;
constexpr uint8_t Byte7ScaleSrc2LoMask = 0x03;
constexpr uint8_t Byte7ScaleSrc2HiBit = 0x04;

bool isVop3px2ScaleInst(StringRef Mnemonic) {
  return StringSwitch<bool>(Mnemonic)
      .Case("v_wmma_scale_f32_16x16x128_f8f6f4", true)
      .Case("v_wmma_scale16_f32_16x16x128_f8f6f4", true)
      .Case("v_wmma_scale_f32_32x16x128_f4", true)
      .Case("v_wmma_scale16_f32_32x16x128_f4", true)
      .Default(false);
}

} // anonymous namespace

/// Patch bits [58:50] (scale_src2) to VGPR0 encoding (0x100).
/// Returns true if the field was modified.
///
/// Raw byte manipulation is required here because scale_src2 is a
/// hardware encoding artifact not modeled as an MC operand. The MC
/// layer has no mechanism to read or set this field, so we patch the
/// encoding bytes directly using the bit layout documented above.
bool patchScaleSrc2(uint8_t *InstBytes) {
  uint8_t OldByte6 = InstBytes[6];
  uint8_t OldByte7 = InstBytes[7];

  uint8_t NewByte6 = OldByte6 & ~Byte6ScaleSrc2Mask;
  uint8_t NewByte7 = (OldByte7 & ~Byte7ScaleSrc2LoMask) | Byte7ScaleSrc2HiBit;

  if (NewByte6 == OldByte6 && NewByte7 == OldByte7)
    return false;

  InstBytes[6] = NewByte6;
  InstBytes[7] = NewByte7;
  return true;
}

// Must run before any pass that grows .text or invalidates
// Ctx.Decoded[i].Offset. Currently safe after applyWmmaHazardPatch
// because trampolines are deferred to the post-pass grow step.
//
// This only fires on the B0-to-A0 rewrite path (applyGfx1250B0toA0Rules).
// A0-native binaries are compiled with an A0-targeted Clang that sets the
// field correctly at codegen time, so they do not need hotswap rewriting.
static uint32_t applyVop3px2Src2FixImpl(PatchContext &Ctx) {
  uint32_t Patched = 0;
  unsigned Scanned = 0;

  for (InternalDecodedInst &DI : Ctx.Decoded) {
    if (!isVop3px2ScaleInst(DI.Mnemonic))
      continue;
    ++Scanned;

    if (patchScaleSrc2(Ctx.Text + DI.Offset)) {
      log() << "hotswap: VOP3PX2 SRC2 fix at 0x" << utohexstr(DI.Offset) << ": "
            << DI.Mnemonic << " scale_src2 -> VGPR0\n";
      ++Patched;
    }
  }

  if (Scanned > 0)
    log() << "hotswap: VOP3PX2 SRC2 scan: " << Scanned
          << " v_wmma_scale* found, " << Patched << " patched\n";
  return Patched;
}

void registerVop3px2Src2Patch(HotswapPatchVTable &VT) {
  VT.applyVop3px2Src2Fix = &applyVop3px2Src2FixImpl;
}

} // namespace hotswap
} // namespace COMGR
