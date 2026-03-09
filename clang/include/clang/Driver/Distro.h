//===--- Distro.h - Linux distribution detection support --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_DISTRO_H
#define LLVM_CLANG_DRIVER_DISTRO_H

#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace driver {

/// Distro - Helper class for detecting and classifying Linux distributions.
///
/// This class encapsulates the clang Linux distribution detection mechanism
/// as well as helper functions that match the specific (versioned) results
/// into wider distribution classes.
class Distro {
public:
  enum DistroType {
    // Special value means that no detection was performed yet.
    UninitializedDistro,
    // NB: Releases of a particular Linux distro should be kept together
    // in this enum, because some tests are done by integer comparison against
    // the first and last known member in the family, e.g. IsRedHat().
    AlpineLinux,
    ArchLinux,
    DebianJessie,
    DebianStretch,
    DebianBuster,
    DebianBullseye,
    DebianBookworm,
    DebianTrixie,
    DebianForky,
    DebianDuke,
    Exherbo,
    RHEL7,
    RHEL8,
    RHEL9,
    RHEL10,
    Fedora,
    Gentoo,
    OpenSUSE,
    UbuntuQuantal,
    UbuntuRaring,
    UbuntuSaucy,
    UbuntuTrusty,
    UbuntuUtopic,
    UbuntuVivid,
    UbuntuWily,
    UbuntuXenial,
    UbuntuYakkety,
    UbuntuZesty,
    UbuntuArtful,
    UbuntuBionic,
    UbuntuCosmic,
    UbuntuDisco,
    UbuntuEoan,
    UbuntuFocal,
    UbuntuGroovy,
    UbuntuHirsute,
    UbuntuImpish,
    UbuntuJammy,
    UbuntuKinetic,
    UbuntuLunar,
    UbuntuMantic,
    UbuntuNoble,
    UbuntuOracular,
    UbuntuPlucky,
    UbuntuQuesting,
    UbuntuResolute,
    UnknownDistro
  };

private:
  /// The distribution, possibly with specific version.
  DistroType DistroVal;

public:
  /// @name Constructors
  /// @{

  /// Default constructor leaves the distribution unknown.
  Distro() : DistroVal() {}

  /// Constructs a Distro type for specific distribution.
  Distro(DistroType D) : DistroVal(D) {}

  /// Detects the distribution using specified VFS.
  explicit Distro(llvm::vfs::FileSystem &VFS, const llvm::Triple &TargetOrHost);

  bool operator==(const Distro &Other) const {
    return DistroVal == Other.DistroVal;
  }

  bool operator!=(const Distro &Other) const {
    return DistroVal != Other.DistroVal;
  }

  bool operator>=(const Distro &Other) const {
    return DistroVal >= Other.DistroVal;
  }

  bool operator<=(const Distro &Other) const {
    return DistroVal <= Other.DistroVal;
  }

  /// @}
  /// @name Convenience Predicates
  /// @{

  bool IsRedhat() const {
    return DistroVal == Fedora || (DistroVal >= RHEL7 && DistroVal <= RHEL10);
  }

  bool IsOpenSUSE() const { return DistroVal == OpenSUSE; }

  bool IsDebian() const {
    return DistroVal >= DebianJessie && DistroVal <= DebianDuke;
  }

  bool IsUbuntu() const {
    return DistroVal >= UbuntuQuantal && DistroVal <= UbuntuResolute;
  }

  bool IsAlpineLinux() const { return DistroVal == AlpineLinux; }

  bool IsGentoo() const { return DistroVal == Gentoo; }

  /// @}
};

} // end namespace driver
} // end namespace clang

#endif
