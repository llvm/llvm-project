//===--- Distro.cpp - Linux distribution detection support ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Distro.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Threading.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

using namespace clang::driver;
using namespace clang;

static bool ParseVersion(StringRef Value, int &Major, int &Minor) {
  Value = Value.trim().trim('"');
  std::pair<StringRef, StringRef> Split = Value.split('.');
  if (Split.first.getAsInteger(10, Major))
    return false;
  Minor = 0;
  if (!Split.second.empty() && Split.second.getAsInteger(10, Minor))
    return false;
  return true;
}

static Distro::DistroType DetectOsRelease(llvm::vfs::FileSystem &VFS,
                                          int &Major, int &Minor) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File =
      VFS.getBufferForFile("/etc/os-release");
  if (!File)
    File = VFS.getBufferForFile("/usr/lib/os-release");
  if (!File)
    return Distro::UnknownDistro;

  SmallVector<StringRef, 16> Lines;
  File.get()->getBuffer().split(Lines, "\n");
  Distro::DistroType Version = Distro::UnknownDistro;

  // Obviously this can be improved a lot.
  for (StringRef Line : Lines)
    if (Line.starts_with("VERSION_ID=")) {
      int ParsedMajor, ParsedMinor;
      if (ParseVersion(Line.substr(11), ParsedMajor, ParsedMinor)) {
        Major = ParsedMajor;
        Minor = ParsedMinor;
      }
    } else if (Version == Distro::UnknownDistro && Line.starts_with("ID="))
      Version = llvm::StringSwitch<Distro::DistroType>(Line.substr(3))
                    .Case("alpine", Distro::AlpineLinux)
                    .Case("fedora", Distro::Fedora)
                    .Case("gentoo", Distro::Gentoo)
                    .Case("arch", Distro::ArchLinux)
                    // On SLES, /etc/os-release was introduced in SLES 11.
                    .Case("sles", Distro::OpenSUSE)
                    .Case("opensuse", Distro::OpenSUSE)
                    .Case("exherbo", Distro::Exherbo)
                    .Case("ubuntu", Distro::Ubuntu)
                    .Default(Distro::UnknownDistro);
  if (Version != Distro::Ubuntu)
    Major = Minor = 0;
  return Version;
}

static Distro::DistroType DetectLsbRelease(llvm::vfs::FileSystem &VFS,
                                           int &Major, int &Minor) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File =
      VFS.getBufferForFile("/etc/lsb-release");
  if (!File)
    return Distro::UnknownDistro;

  SmallVector<StringRef, 16> Lines;
  File.get()->getBuffer().split(Lines, "\n");
  bool IsUbuntuId = false;
  bool HasCodename = false;
  for (StringRef Line : Lines) {
    if (Line.starts_with("DISTRIB_ID="))
      IsUbuntuId = Line.substr(11).trim().equals_insensitive("ubuntu");
    else if (Line.starts_with("DISTRIB_CODENAME="))
      HasCodename = !Line.substr(17).trim().empty();
    else if (Line.starts_with("DISTRIB_RELEASE=")) {
      int ParsedMajor, ParsedMinor;
      if (ParseVersion(Line.substr(16), ParsedMajor, ParsedMinor)) {
        Major = ParsedMajor;
        Minor = ParsedMinor;
      }
    }
  }
  if (!IsUbuntuId || !HasCodename) {
    Major = Minor = 0;
    return Distro::UnknownDistro;
  }
  return Distro::Ubuntu;
}

static Distro::DistroType DetectDistro(llvm::vfs::FileSystem &VFS, int &Major,
                                       int &Minor) {
  Distro::DistroType Version = Distro::UnknownDistro;

  // Newer freedesktop.org's compilant systemd-based systems
  // should provide /etc/os-release or /usr/lib/os-release.
  Version = DetectOsRelease(VFS, Major, Minor);
  if (Version != Distro::UnknownDistro)
    return Version;

  // Older systems might provide /etc/lsb-release.
  Version = DetectLsbRelease(VFS, Major, Minor);
  if (Version != Distro::UnknownDistro)
    return Version;

  // Otherwise try some distro-specific quirks for Red Hat...
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File =
      VFS.getBufferForFile("/etc/redhat-release");

  if (File) {
    StringRef Data = File.get()->getBuffer();
    if (Data.starts_with("Fedora release"))
      return Distro::Fedora;
    if (Data.starts_with("Red Hat Enterprise Linux") ||
        Data.starts_with("CentOS") || Data.starts_with("AlmaLinux") ||
        Data.starts_with("Rocky Linux") ||
        Data.starts_with("Scientific Linux")) {
      if (Data.contains("release 10"))
        return Distro::RHEL10;
      if (Data.contains("release 9"))
        return Distro::RHEL9;
      if (Data.contains("release 8"))
        return Distro::RHEL8;
      if (Data.contains("release 7"))
        return Distro::RHEL7;
    }
    return Distro::UnknownDistro;
  }

  // ...for Debian
  File = VFS.getBufferForFile("/etc/debian_version");
  if (File) {
    StringRef Data = File.get()->getBuffer();
    // Contents: < major.minor > or < codename/sid >
    int MajorVersion;
    if (!Data.split('.').first.getAsInteger(10, MajorVersion)) {
      switch (MajorVersion) {
      case 8:
        return Distro::DebianJessie;
      case 9:
        return Distro::DebianStretch;
      case 10:
        return Distro::DebianBuster;
      case 11:
        return Distro::DebianBullseye;
      case 12:
        return Distro::DebianBookworm;
      case 13:
        return Distro::DebianTrixie;
      case 14:
        return Distro::DebianForky;
      case 15:
        return Distro::DebianDuke;
      default:
        return Distro::UnknownDistro;
      }
    }
    return llvm::StringSwitch<Distro::DistroType>(Data.split("\n").first)
        .Case("jessie/sid", Distro::DebianJessie)
        .Case("stretch/sid", Distro::DebianStretch)
        .Case("buster/sid", Distro::DebianBuster)
        .Case("bullseye/sid", Distro::DebianBullseye)
        .Case("bookworm/sid", Distro::DebianBookworm)
        .Case("trixie/sid", Distro::DebianTrixie)
        .Case("forky/sid", Distro::DebianForky)
        .Case("duke/sid", Distro::DebianDuke)
        .Default(Distro::UnknownDistro);
  }

  // ...for SUSE
  File = VFS.getBufferForFile("/etc/SuSE-release");
  if (File) {
    StringRef Data = File.get()->getBuffer();
    SmallVector<StringRef, 8> Lines;
    Data.split(Lines, "\n");
    for (const StringRef &Line : Lines) {
      if (!Line.trim().starts_with("VERSION"))
        continue;
      std::pair<StringRef, StringRef> SplitLine = Line.split('=');
      // Old versions have split VERSION and PATCHLEVEL
      // Newer versions use VERSION = x.y
      std::pair<StringRef, StringRef> SplitVer =
          SplitLine.second.trim().split('.');
      int Version;

      // OpenSUSE/SLES 10 and older are not supported and not compatible
      // with our rules, so just treat them as Distro::UnknownDistro.
      if (!SplitVer.first.getAsInteger(10, Version) && Version > 10)
        return Distro::OpenSUSE;
      return Distro::UnknownDistro;
    }
    return Distro::UnknownDistro;
  }

  // ...and others.
  if (VFS.exists("/etc/gentoo-release"))
    return Distro::Gentoo;

  return Distro::UnknownDistro;
}

static Distro::DistroType GetDistro(llvm::vfs::FileSystem &VFS,
                                    const llvm::Triple &TargetOrHost,
                                    int &Major, int &Minor) {
  Major = Minor = 0;

  // If we don't target Linux, no need to check the distro. This saves a few
  // OS calls.
  if (!TargetOrHost.isOSLinux())
    return Distro::UnknownDistro;

  // True if we're backed by a real file system.
  const bool onRealFS = (llvm::vfs::getRealFileSystem() == &VFS);

  // If the host is not running Linux, and we're backed by a real file
  // system, no need to check the distro. This is the case where someone
  // is cross-compiling from BSD or Windows to Linux, and it would be
  // meaningless to try to figure out the "distro" of the non-Linux host.
  llvm::Triple HostTriple(llvm::sys::getProcessTriple());
  if (!HostTriple.isOSLinux() && onRealFS)
    return Distro::UnknownDistro;

  if (onRealFS) {
    // If we're backed by a real file system, perform
    // the detection only once and save the result.
    static const Distro::DistroType LinuxDistro =
        DetectDistro(VFS, Major, Minor);
    static const int LinuxDistroMajor = Major;
    static const int LinuxDistroMinor = Minor;
    Major = LinuxDistroMajor;
    Minor = LinuxDistroMinor;
    return LinuxDistro;
  }
  // This is mostly for passing tests which uses llvm::vfs::InMemoryFileSystem,
  // which is not "real".
  return DetectDistro(VFS, Major, Minor);
}

Distro::Distro(llvm::vfs::FileSystem &VFS, const llvm::Triple &TargetOrHost)
    : DistroVal(), MajorVersionVal(), MinorVersionVal() {
  int Major, Minor;
  DistroVal = GetDistro(VFS, TargetOrHost, Major, Minor);
  MajorVersionVal = Major;
  MinorVersionVal = Minor;
}
