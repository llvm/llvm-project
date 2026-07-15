//===--- TargetID.cpp - Utilities for parsing target ID -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetID.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

namespace clang {

llvm::StringRef getProcessorFromTargetID(const llvm::Triple &T,
                                         llvm::StringRef OffloadArch) {
  auto Split = OffloadArch.split(':');
  if (T.isAMDGPU())
    return llvm::AMDGPU::getCanonicalArchName(T, Split.first);
  return Split.first;
}

// For a specific processor, a feature either shows up in all target IDs, or
// does not show up in any target IDs. Otherwise the target ID combination is
// invalid.
std::optional<std::pair<llvm::StringRef, llvm::StringRef>>
getConflictTargetIDCombination(llvm::ArrayRef<TargetIDEntry> Entries) {
  struct Info {
    llvm::StringRef TargetID;
    bool HasXnack;
    bool HasSramEcc;
  };

  llvm::SmallDenseMap<llvm::AMDGPU::GPUKind, Info> Seen;
  for (const auto &[T, ID] : Entries) {
    std::optional<llvm::AMDGPU::TargetID> Parsed =
        llvm::AMDGPU::TargetID::parse(T, ID);
    if (!Parsed)
      continue;

    // A feature is present in a target ID only when an explicit '+'/'-'
    // modifier is given, not when it is left unspecified.
    Info Cur{ID, Parsed->isXnackOnOrOff(), Parsed->isSramEccOnOrOff()};
    auto [Loc, Inserted] = Seen.try_emplace(Parsed->getGPUKind(), Cur);
    if (Inserted)
      continue;

    const Info &Prev = Loc->second;
    if (Cur.HasXnack != Prev.HasXnack || Cur.HasSramEcc != Prev.HasSramEcc)
      return std::make_pair(Prev.TargetID, ID);
  }
  return std::nullopt;
}

std::string sanitizeTargetIDInFileName(llvm::StringRef TargetID) {
  std::string FileName = TargetID.str();
  if (llvm::sys::path::is_style_windows(llvm::sys::path::Style::native))
    llvm::replace(FileName, ':', '@');
  return FileName;
}

} // namespace clang
