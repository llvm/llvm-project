//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_ARCH_TARGETIMPL_H
#define LLD_ELF_ARCH_TARGETIMPL_H

#include "InputFiles.h"
#include "InputSection.h"
#include "Relocations.h"
#include "Symbols.h"
#include "llvm/BinaryFormat/ELF.h"

namespace lld::elf {

// getControlTransferAddend: If this relocation is used for control transfer
// instructions (e.g. branch, branch-link or call) or code references (e.g.
// virtual function pointers) and indicates an address-insignificant reference,
// return the effective addend for the relocation, otherwise return
// std::nullopt. The effective addend for a relocation is the addend that is
// used to determine its branch destination.
//
// getBranchInfoAtTarget: If a control transfer relocation referring to
// is+offset directly transfers control to a relocated branch instruction in the
// specified section, return the relocation for the branch target as well as its
// effective addend (see above). Otherwise return {nullptr, 0}.
//
// redirectControlTransferRelocations: Given r1, a relocation for which
// getControlTransferAddend() returned a value, and r2, a relocation returned by
// getBranchInfo(), modify r1 so that it branches directly to the target of r2.
template <typename GetControlTransferAddend, typename GetBranchInfoAtTarget,
          typename RedirectControlTransferRelocations>
inline void applyBranchToBranchOptImpl(
    Ctx &ctx, GetControlTransferAddend getControlTransferAddend,
    GetBranchInfoAtTarget getBranchInfoAtTarget,
    RedirectControlTransferRelocations redirectControlTransferRelocations) {
  // Needs to run serially because it writes to the relocations array as well as
  // reading relocations of other sections.
  for (ELFFileBase *f : ctx.objectFiles) {
    auto getRelocBranchInfo =
        [&getBranchInfoAtTarget](
            Relocation &r,
            uint64_t addend) -> std::pair<Relocation *, uint64_t> {
      auto *target = dyn_cast_or_null<Defined>(r.sym);
      // We don't allow preemptible symbols or ifuncs (may go somewhere else),
      // absolute symbols (runtime behavior unknown), non-executable or writable
      // memory (ditto) or non-regular sections (no section data).
      if (!target || target->isPreemptible || target->isGnuIFunc() ||
          !target->section ||
          !(target->section->flags & llvm::ELF::SHF_EXECINSTR) ||
          (target->section->flags & llvm::ELF::SHF_WRITE) ||
          target->section->kind() != SectionBase::Regular)
        return {nullptr, 0};
      return getBranchInfoAtTarget(*cast<InputSection>(target->section),
                                   target->value + addend);
    };
    for (InputSectionBase *sb : f->getSections()) {
      auto *s = dyn_cast_or_null<InputSection>(sb);
      if (!s)
        continue;
      for (Relocation &r : s->relocations) {
        std::optional<uint64_t> addend = getControlTransferAddend(*s, r);
        if (!addend)
          continue;
        std::pair<Relocation *, uint64_t> targetAndAddend =
            getRelocBranchInfo(r, *addend);
        if (!targetAndAddend.first)
          continue;
        // Avoid getting stuck in an infinite loop if we encounter a branch
        // that (possibly indirectly) branches to itself. It is unlikely
        // that more than 5 iterations will ever be needed in practice.
        size_t iterations = 5;
        while (iterations--) {
          std::pair<Relocation *, uint64_t> nextTargetAndAddend =
              getRelocBranchInfo(*targetAndAddend.first,
                                 targetAndAddend.second);
          if (!nextTargetAndAddend.first)
            break;
          targetAndAddend = nextTargetAndAddend;
        }
        redirectControlTransferRelocations(r, *targetAndAddend.first);
      }
    }
  }
}

} // namespace lld::elf

#endif
