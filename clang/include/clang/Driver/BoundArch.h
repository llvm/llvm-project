//===--- BoundArch.h - Bound Architecture struct ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_BOUNDARCH_H
#define LLVM_CLANG_DRIVER_BOUNDARCH_H

#include "clang/Basic/OffloadArch.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace driver {

/// Represents a bound architecture for offload / multiple architecture
/// compilation.
struct BoundArch {
  llvm::StringRef ArchName;

  /// The parsed offload architecture enum.
  /// Will be OffloadArch::Unknown if ArchName not recognized.
  OffloadArch Arch = OffloadArch::Unused;

  BoundArch() = default;
  explicit BoundArch(llvm::StringRef Name)
      : ArchName(Name),
        Arch(Name.empty() ? OffloadArch::Unknown : StringToOffloadArch(Name)) {}

  BoundArch(llvm::StringRef Name, OffloadArch A) : ArchName(Name), Arch(A) {}

  bool empty() const { return ArchName.empty(); }
  explicit operator bool() const { return Arch != OffloadArch::Unused; }

  bool operator==(const BoundArch &Other) const {
    return Arch == Other.Arch && ArchName == Other.ArchName;
  }

  bool operator<(const BoundArch &Other) const {
    return std::tie(Arch, ArchName) < std::tie(Other.Arch, Other.ArchName);
  }
};

} // namespace driver
} // namespace clang

#endif // LLVM_CLANG_DRIVER_BOUNDARCH_H
