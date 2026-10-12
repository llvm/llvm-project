//===- source-image.cpp - Transpiler --------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/source-image.h"

#include "transpiler/decoder/parsed-reg.h"
#include "transpiler/loader/code-object-utils.h"
#include "transpiler/raiser/handlers.h"
#include "transpiler/raiser/register-state.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Endian.h"

using namespace llvm;

namespace COMGR::transpiler {

namespace {

constexpr uint64_t DwordBytes = 4;

} // namespace

Expected<uint64_t> moveSourceImageAddress(RaiseContext &Ctx,
                                          const DecodedInst &Di,
                                          uint64_t SourceAddress,
                                          uint64_t ByteDisplacement) {
  // Unsigned arithmetic wraps, so compare against the address it started from
  // to catch a move off either end.
  uint64_t Moved = SourceAddress + ByteDisplacement;
  bool Backwards = ByteDisplacement >> 63;
  if (Backwards ? Moved > SourceAddress : Moved < SourceAddress)
    return unsupported(Ctx, Di,
                       "moves a source address out of the address space");
  return Moved;
}

std::optional<uint32_t> readSourceImageDword(const RaiseContext &Ctx,
                                             uint64_t Address) {
  for (const TextSection::ImageSection &Section : Ctx.sourceImageSections()) {
    if (Address < Section.Address)
      continue;
    uint64_t Offset = Address - Section.Address;
    uint64_t Size = Section.Bytes.size();
    if (Offset > Size || DwordBytes > Size - Offset)
      continue;
    return support::endian::read32le(Section.Bytes.data() + Offset);
  }
  return std::nullopt;
}

Expected<std::optional<uint64_t>> sourceImageSgprPairAddr(RaiseContext &Ctx,
                                                          const DecodedInst &Di,
                                                          unsigned BaseIdx) {
  if (std::optional<uint64_t> Address =
          Ctx.registers().lookupSourceImageSgprPairAddr(BaseIdx))
    return Address;
  // A source address the raise can no longer resolve must not reach a load or
  // a further displacement, both of which would then run against the memory
  // the raised kernel was handed.
  if (Ctx.registers().droppedSourceImageSgprPairAddr(BaseIdx))
    return unsupported(Ctx, Di,
                       "uses a source address that a block boundary or a write "
                       "to one half of the pair has dropped");
  return std::nullopt;
}

Expected<std::optional<uint64_t>> sourceImageOperandAddr(RaiseContext &Ctx,
                                                         const DecodedInst &Di,
                                                         unsigned Index) {
  // Only registers can name tracked source-image addresses.
  if (!Di.isReg(Index))
    return std::nullopt;
  Expected<ParsedReg> Reg = Ctx.registers().parseReg(Di, Index);
  if (!Reg)
    return Reg.takeError();
  if (Reg->RegKind != ParsedReg::SGPR || !Reg->BaseIdx)
    return std::nullopt;
  return sourceImageSgprPairAddr(Ctx, Di, *Reg->BaseIdx);
}

} // namespace COMGR::transpiler
