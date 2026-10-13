//===- setpc-analysis.h - Transpiler --------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_SETPC_ANALYSIS_H
#define TRANSPILER_SETPC_ANALYSIS_H

#include "transpiler/decoder/decoded-inst.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <set>
#include <variant>

namespace COMGR::transpiler {

struct MCState;

// Why the analysis cannot say where a register-indirect control transfer
// leads. The raiser turns each of these into the refusal it reports.
enum class SetPcRefusal {
  // The transfer reads its target from something other than a scalar register
  // pair.
  NotARegisterPair,
  // No path the decode recovered reaches the block that makes the transfer.
  BlockUnreachable,
  // Some path reaching the block leaves the pair without a source offset.
  PairWrittenWithoutOffset,
  // No path reaching the block writes a source offset into the pair.
  PairNeverHeldOffset,
  // More source offsets reach the pair than the analysis enumerates.
  TooManyTargets,
  // The pair names an offset that starts no decoded instruction.
  TargetNotAnInstruction,
};

// Where control goes, as the source offsets it reaches. The analysis computes
// each from a program-counter capture plus whatever displacement was added to
// it. One offset is a plain branch, several are a dispatch.
struct SetPcResolved {
  // Source offsets the transfer reaches, ascending and distinct. Never empty.
  llvm::SmallVector<uint64_t> Targets;
};

// Nothing the analysis models says where control goes.
struct SetPcUnresolvable {
  SetPcRefusal Why;
  // What the refusal is about: the low register of the source pair for the
  // refusals that name a pair, and the computed source offset for
  // TargetNotAnInstruction. NotARegisterPair names nothing and leaves it zero.
  uint64_t Subject;
};

// What one register-indirect control transfer was found to do. A site is one
// or the other, never a half-filled mix of both.
using SetPcSite = std::variant<SetPcResolved, SetPcUnresolvable>;

// Where the register-indirect control transfers of one decoded kernel lead.
struct SetPcAnalysis {
  // One entry per s_set_pc_i64 and s_swap_pc_i64, keyed by its source offset.
  llvm::DenseMap<uint64_t, SetPcSite> Sites;
  // Complete singleton source offsets at block entry, keyed by block and pair.
  llvm::DenseMap<uint64_t, llvm::DenseMap<unsigned, uint64_t>> EntryOffsets;
  // Block starts the analysis found on top of the ones the decode gave it.
  // Every one is the offset of a decoded instruction. The caller must start a
  // block at each, because the analysis split its own walk there and its
  // answers only hold for that shape of control flow.
  llvm::DenseSet<uint64_t> ExtraBlockStarts;
};

// Work out where every indirect jump in `Insts` goes. `Insts` must be in
// source order.
//
// Within a block the analysis follows the value a transfer reads: a
// program-counter capture makes the pair name an offset, a constant
// displacement added to it makes the pair name another, a call writes the
// offset it returns to, and any other write drops what the pair named.
//
// When a block does not write the pair its transfer reads, the transfer reads
// what the paths into the block left there. A forward dataflow over the
// recovered blocks collects those offsets. If any path leaves the pair holding
// something the analysis cannot name, it refuses the transfer instead of
// narrowing it to the paths that did name an offset.
//
// `BlockStarts` is the block-start set of the same decode. It is read, not
// written: the offsets contributed by the transfers are reported separately so
// that the caller can order the merge against the rest of its decode.
// `EntryOffset` is where control enters, which need not be the lowest offset in
// `Insts`: a callee followed into the decode may sit below its caller.
llvm::Expected<SetPcAnalysis>
analyzeSetPc(llvm::ArrayRef<DecodedInst> Insts,
             const std::set<uint64_t> &BlockStarts, uint64_t EntryOffset,
             const MCState &Mc);

} // namespace COMGR::transpiler

#endif
