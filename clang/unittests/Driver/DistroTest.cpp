//===- unittests/Driver/DistroTest.cpp --- ToolChains tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Distro detection.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Distro.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::driver;

namespace {

// The tests include all release-related files for each distribution
// in the VFS, in order to make sure that earlier tests do not
// accidentally result in incorrect distribution guess.

TEST(DistroTest, DetectUbuntu) {
  llvm::vfs::InMemoryFileSystem UbuntuTrustyFileSystem;
  // Ubuntu uses Debian Sid version.
  UbuntuTrustyFileSystem.addFile("/etc/debian_version", 0,
      llvm::MemoryBuffer::getMemBuffer("jessie/sid\n"));
  UbuntuTrustyFileSystem.addFile("/etc/lsb-release", 0,
      llvm::MemoryBuffer::getMemBuffer("DISTRIB_ID=Ubuntu\n"
                                       "DISTRIB_RELEASE=14.04\n"
                                       "DISTRIB_CODENAME=trusty\n"
                                       "DISTRIB_DESCRIPTION=\"Ubuntu 14.04 LTS\"\n"));
  UbuntuTrustyFileSystem.addFile("/etc/os-release", 0,
      llvm::MemoryBuffer::getMemBuffer("NAME=\"Ubuntu\"\n"
                                       "VERSION=\"14.04, Trusty Tahr\"\n"
                                       "ID=ubuntu\n"
                                       "ID_LIKE=debian\n"
                                       "PRETTY_NAME=\"Ubuntu 14.04 LTS\"\n"
                                       "VERSION_ID=\"14.04\"\n"
                                       "HOME_URL=\"http://www.ubuntu.com/\"\n"
                                       "SUPPORT_URL=\"http://help.ubuntu.com/\"\n"
                                       "BUG_REPORT_URL=\"http://bugs.launchpad.net/ubuntu/\"\n"));

  Distro UbuntuTrusty{UbuntuTrustyFileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::UbuntuTrusty), UbuntuTrusty);
  ASSERT_TRUE(UbuntuTrusty.IsUbuntu());
  ASSERT_FALSE(UbuntuTrusty.IsRedhat());
  ASSERT_FALSE(UbuntuTrusty.IsOpenSUSE());
  ASSERT_FALSE(UbuntuTrusty.IsDebian());
  ASSERT_FALSE(UbuntuTrusty.IsGentoo());

  Distro UbuntuTrusty2{UbuntuTrustyFileSystem, llvm::Triple("unknown-pc-windows")};
  ASSERT_EQ(Distro(Distro::UnknownDistro), UbuntuTrusty2);

  llvm::vfs::InMemoryFileSystem UbuntuYakketyFileSystem;
  UbuntuYakketyFileSystem.addFile("/etc/debian_version", 0,
      llvm::MemoryBuffer::getMemBuffer("stretch/sid\n"));
  UbuntuYakketyFileSystem.addFile("/etc/lsb-release", 0,
      llvm::MemoryBuffer::getMemBuffer("DISTRIB_ID=Ubuntu\n"
                                       "DISTRIB_RELEASE=16.10\n"
                                       "DISTRIB_CODENAME=yakkety\n"
                                       "DISTRIB_DESCRIPTION=\"Ubuntu 16.10\"\n"));
  UbuntuYakketyFileSystem.addFile("/etc/os-release", 0,
      llvm::MemoryBuffer::getMemBuffer("NAME=\"Ubuntu\"\n"
                                       "VERSION=\"16.10 (Yakkety Yak)\"\n"
                                       "ID=ubuntu\n"
                                       "ID_LIKE=debian\n"
                                       "PRETTY_NAME=\"Ubuntu 16.10\"\n"
                                       "VERSION_ID=\"16.10\"\n"
                                       "HOME_URL=\"http://www.ubuntu.com/\"\n"
                                       "SUPPORT_URL=\"http://help.ubuntu.com/\"\n"
                                       "BUG_REPORT_URL=\"http://bugs.launchpad.net/ubuntu/\"\n"
                                       "PRIVACY_POLICY_URL=\"http://www.ubuntu.com/legal/terms-and-policies/privacy-policy\"\n"
                                       "VERSION_CODENAME=yakkety\n"
                                       "UBUNTU_CODENAME=yakkety\n"));

  Distro UbuntuYakkety{UbuntuYakketyFileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::UbuntuYakkety), UbuntuYakkety);
  ASSERT_TRUE(UbuntuYakkety.IsUbuntu());
  ASSERT_FALSE(UbuntuYakkety.IsRedhat());
  ASSERT_FALSE(UbuntuYakkety.IsOpenSUSE());
  ASSERT_FALSE(UbuntuYakkety.IsDebian());
  ASSERT_FALSE(UbuntuYakkety.IsGentoo());
}

TEST(DistroTest, DetectRedhat) {
  llvm::vfs::InMemoryFileSystem Fedora25FileSystem;
  Fedora25FileSystem.addFile("/etc/system-release-cpe", 0,
      llvm::MemoryBuffer::getMemBuffer("cpe:/o:fedoraproject:fedora:25\n"));
  // Both files are symlinks to fedora-release.
  Fedora25FileSystem.addFile("/etc/system-release", 0,
      llvm::MemoryBuffer::getMemBuffer("Fedora release 25 (Twenty Five)\n"));
  Fedora25FileSystem.addFile("/etc/redhat-release", 0,
      llvm::MemoryBuffer::getMemBuffer("Fedora release 25 (Twenty Five)\n"));
  Fedora25FileSystem.addFile("/etc/fedora-release", 0,
      llvm::MemoryBuffer::getMemBuffer("Fedora release 25 (Twenty Five)\n"));
  Fedora25FileSystem.addFile("/etc/os-release", 0,
      llvm::MemoryBuffer::getMemBuffer("NAME=Fedora\n"
                                       "VERSION=\"25 (Twenty Five)\"\n"
                                       "ID=fedora\n"
                                       "VERSION_ID=25\n"
                                       "PRETTY_NAME=\"Fedora 25 (Twenty Five)\"\n"
                                       "ANSI_COLOR=\"0;34\"\n"
                                       "CPE_NAME=\"cpe:/o:fedoraproject:fedora:25\"\n"
                                       "HOME_URL=\"https://fedoraproject.org/\"\n"
                                       "BUG_REPORT_URL=\"https://bugzilla.redhat.com/\"\n"
                                       "REDHAT_BUGZILLA_PRODUCT=\"Fedora\"\n"
                                       "REDHAT_BUGZILLA_PRODUCT_VERSION=25\n"
                                       "REDHAT_SUPPORT_PRODUCT=\"Fedora\"\n"
                                       "REDHAT_SUPPORT_PRODUCT_VERSION=25\n"
                                       "PRIVACY_POLICY_URL=https://fedoraproject.org/wiki/Legal:PrivacyPolicy\n"));
  Distro Fedora25{Fedora25FileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::Fedora), Fedora25);
  ASSERT_FALSE(Fedora25.IsUbuntu());
  ASSERT_TRUE(Fedora25.IsRedhat());
  ASSERT_FALSE(Fedora25.IsOpenSUSE());
  ASSERT_FALSE(Fedora25.IsDebian());
  ASSERT_FALSE(Fedora25.IsGentoo());

  llvm::vfs::InMemoryFileSystem CentOS7FileSystem;
  CentOS7FileSystem.addFile("/etc/system-release-cpe", 0,
      llvm::MemoryBuffer::getMemBuffer("cpe:/o:centos:centos:7\n"));
  // Both files are symlinks to centos-release.
  CentOS7FileSystem.addFile("/etc/system-release", 0,
      llvm::MemoryBuffer::getMemBuffer("CentOS Linux release 7.2.1511 (Core) \n"));
  CentOS7FileSystem.addFile("/etc/redhat-release", 0,
      llvm::MemoryBuffer::getMemBuffer("CentOS Linux release 7.2.1511 (Core) \n"));
  CentOS7FileSystem.addFile("/etc/centos-release", 0,
      llvm::MemoryBuffer::getMemBuffer("CentOS Linux release 7.2.1511 (Core) \n"));
  CentOS7FileSystem.addFile("/etc/centos-release-upstream", 0,
      llvm::MemoryBuffer::getMemBuffer("Derived from Red Hat Enterprise Linux 7.2 (Source)\n"));
  CentOS7FileSystem.addFile("/etc/os-release", 0,
      llvm::MemoryBuffer::getMemBuffer("NAME=\"CentOS Linux\"\n"
                                       "VERSION=\"7 (Core)\"\n"
                                       "ID=\"centos\"\n"
                                       "ID_LIKE=\"rhel fedora\"\n"
                                       "VERSION_ID=\"7\"\n"
                                       "PRETTY_NAME=\"CentOS Linux 7 (Core)\"\n"
                                       "ANSI_COLOR=\"0;31\"\n"
                                       "CPE_NAME=\"cpe:/o:centos:centos:7\"\n"
                                       "HOME_URL=\"https://www.centos.org/\"\n"
                                       "BUG_REPORT_URL=\"https://bugs.centos.org/\"\n"
                                       "\n"
                                       "CENTOS_MANTISBT_PROJECT=\"CentOS-7\"\n"
                                       "CENTOS_MANTISBT_PROJECT_VERSION=\"7\"\n"
                                       "REDHAT_SUPPORT_PRODUCT=\"centos\"\n"
                                       "REDHAT_SUPPORT_PRODUCT_VERSION=\"7\"\n"));

  Distro CentOS7{CentOS7FileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::RHEL7), CentOS7);
  ASSERT_FALSE(CentOS7.IsUbuntu());
  ASSERT_TRUE(CentOS7.IsRedhat());
  ASSERT_FALSE(CentOS7.IsOpenSUSE());
  ASSERT_FALSE(CentOS7.IsDebian());
  ASSERT_FALSE(CentOS7.IsGentoo());
}

TEST(DistroTest, DetectOpenSUSE) {
  llvm::vfs::InMemoryFileSystem OpenSUSELeap421FileSystem;
  OpenSUSELeap421FileSystem.addFile("/etc/SuSE-release", 0,
      llvm::MemoryBuffer::getMemBuffer("openSUSE 42.1 (x86_64)\n"
                                       "VERSION = 42.1\n"
                                       "CODENAME = Malachite\n"
                                       "# /etc/SuSE-release is deprecated and will be removed in the future, use /etc/os-release instead\n"));
  OpenSUSELeap421FileSystem.addFile("/etc/os-release", 0,
      llvm::MemoryBuffer::getMemBuffer("NAME=\"openSUSE Leap\"\n"
                                       "VERSION=\"42.1\"\n"
                                       "VERSION_ID=\"42.1\"\n"
                                       "PRETTY_NAME=\"openSUSE Leap 42.1 (x86_64)\"\n"
                                       "ID=opensuse\n"
                                       "ANSI_COLOR=\"0;32\"\n"
                                       "CPE_NAME=\"cpe:/o:opensuse:opensuse:42.1\"\n"
                                       "BUG_REPORT_URL=\"https://bugs.opensuse.org\"\n"
                                       "HOME_URL=\"https://opensuse.org/\"\n"
                                       "ID_LIKE=\"suse\"\n"));

  Distro OpenSUSELeap421{OpenSUSELeap421FileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::OpenSUSE), OpenSUSELeap421);
  ASSERT_FALSE(OpenSUSELeap421.IsUbuntu());
  ASSERT_FALSE(OpenSUSELeap421.IsRedhat());
  ASSERT_TRUE(OpenSUSELeap421.IsOpenSUSE());
  ASSERT_FALSE(OpenSUSELeap421.IsDebian());
  ASSERT_FALSE(OpenSUSELeap421.IsGentoo());

  llvm::vfs::InMemoryFileSystem OpenSUSE132FileSystem;
  OpenSUSE132FileSystem.addFile("/etc/SuSE-release", 0,
      llvm::MemoryBuffer::getMemBuffer("openSUSE 13.2 (x86_64)\n"
                                       "VERSION = 13.2\n"
                                       "CODENAME = Harlequin\n"
                                       "# /etc/SuSE-release is deprecated and will be removed in the future, use /etc/os-release instead\n"));
  OpenSUSE132FileSystem.addFile("/etc/os-release", 0,
      llvm::MemoryBuffer::getMemBuffer("NAME=openSUSE\n"
                                       "VERSION=\"13.2 (Harlequin)\"\n"
                                       "VERSION_ID=\"13.2\"\n"
                                       "PRETTY_NAME=\"openSUSE 13.2 (Harlequin) (x86_64)\"\n"
                                       "ID=opensuse\n"
                                       "ANSI_COLOR=\"0;32\"\n"
                                       "CPE_NAME=\"cpe:/o:opensuse:opensuse:13.2\"\n"
                                       "BUG_REPORT_URL=\"https://bugs.opensuse.org\"\n"
                                       "HOME_URL=\"https://opensuse.org/\"\n"
                                       "ID_LIKE=\"suse\"\n"));

  Distro OpenSUSE132{OpenSUSE132FileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::OpenSUSE), OpenSUSE132);
  ASSERT_FALSE(OpenSUSE132.IsUbuntu());
  ASSERT_FALSE(OpenSUSE132.IsRedhat());
  ASSERT_TRUE(OpenSUSE132.IsOpenSUSE());
  ASSERT_FALSE(OpenSUSE132.IsDebian());
  ASSERT_FALSE(OpenSUSE132.IsGentoo());

  llvm::vfs::InMemoryFileSystem SLES10FileSystem;
  SLES10FileSystem.addFile("/etc/SuSE-release", 0,
      llvm::MemoryBuffer::getMemBuffer("SUSE Linux Enterprise Server 10 (x86_64)\n"
                                       "VERSION = 10\n"
                                       "PATCHLEVEL = 4\n"));
  SLES10FileSystem.addFile("/etc/lsb_release", 0,
      llvm::MemoryBuffer::getMemBuffer("LSB_VERSION=\"core-2.0-noarch:core-3.0-noarch:core-2.0-x86_64:core-3.0-x86_64\"\n"));

  // SLES10 is unsupported and therefore evaluates to unknown
  Distro SLES10{SLES10FileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::UnknownDistro), SLES10);
  ASSERT_FALSE(SLES10.IsUbuntu());
  ASSERT_FALSE(SLES10.IsRedhat());
  ASSERT_FALSE(SLES10.IsOpenSUSE());
  ASSERT_FALSE(SLES10.IsDebian());
  ASSERT_FALSE(SLES10.IsGentoo());
}

TEST(DistroTest, DetectDebian) {
  llvm::vfs::InMemoryFileSystem DebianJessieFileSystem;
  DebianJessieFileSystem.addFile("/etc/debian_version", 0,
                                 llvm::MemoryBuffer::getMemBuffer("8.6\n"));
  DebianJessieFileSystem.addFile("/etc/os-release", 0,
      llvm::MemoryBuffer::getMemBuffer("PRETTY_NAME=\"Debian GNU/Linux 8 (jessie)\"\n"
                                       "NAME=\"Debian GNU/Linux\"\n"
                                       "VERSION_ID=\"8\"\n"
                                       "VERSION=\"8 (jessie)\"\n"
                                       "ID=debian\n"
                                       "HOME_URL=\"http://www.debian.org/\"\n"
                                       "SUPPORT_URL=\"http://www.debian.org/support\"\n"
                                       "BUG_REPORT_URL=\"https://bugs.debian.org/\"\n"));

  Distro DebianJessie{DebianJessieFileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::DebianJessie), DebianJessie);
  ASSERT_FALSE(DebianJessie.IsUbuntu());
  ASSERT_FALSE(DebianJessie.IsRedhat());
  ASSERT_FALSE(DebianJessie.IsOpenSUSE());
  ASSERT_TRUE(DebianJessie.IsDebian());
  ASSERT_FALSE(DebianJessie.IsGentoo());

  llvm::vfs::InMemoryFileSystem DebianStretchSidFileSystem;
  DebianStretchSidFileSystem.addFile("/etc/debian_version", 0,
                                 llvm::MemoryBuffer::getMemBuffer("stretch/sid\n"));
  DebianStretchSidFileSystem.addFile("/etc/os-release", 0,
      llvm::MemoryBuffer::getMemBuffer("PRETTY_NAME=\"Debian GNU/Linux stretch/sid\"\n"
                                       "NAME=\"Debian GNU/Linux\"\n"
                                       "ID=debian\n"
                                       "HOME_URL=\"http://www.debian.org/\"\n"
                                       "SUPPORT_URL=\"http://www.debian.org/support\"\n"
                                       "BUG_REPORT_URL=\"https://bugs.debian.org/\"\n"));

  Distro DebianStretchSid{DebianStretchSidFileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::DebianStretch), DebianStretchSid);
  ASSERT_FALSE(DebianStretchSid.IsUbuntu());
  ASSERT_FALSE(DebianStretchSid.IsRedhat());
  ASSERT_FALSE(DebianStretchSid.IsOpenSUSE());
  ASSERT_TRUE(DebianStretchSid.IsDebian());
  ASSERT_FALSE(DebianStretchSid.IsGentoo());
}

TEST(DistroTest, DetectExherbo) {
  llvm::vfs::InMemoryFileSystem ExherboFileSystem;
  ExherboFileSystem.addFile("/etc/os-release", 0,
      llvm::MemoryBuffer::getMemBuffer("NAME=\"Exherbo\"\n"
                                       "PRETTY_NAME=\"Exherbo Linux\"\n"
                                       "ID=exherbo\n"
                                       "ANSI_COLOR=\"0;32\"\n"
                                       "HOME_URL=\"https://www.exherbo.org/\"\n"
                                       "SUPPORT_URL=\"irc://irc.freenode.net/#exherbo\"\n"
                                       "BUG_REPORT_URL=\"https://bugs.exherbo.org/\"\n"));

  Distro Exherbo{ExherboFileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::Exherbo), Exherbo);
  ASSERT_FALSE(Exherbo.IsUbuntu());
  ASSERT_FALSE(Exherbo.IsRedhat());
  ASSERT_FALSE(Exherbo.IsOpenSUSE());
  ASSERT_FALSE(Exherbo.IsDebian());
  ASSERT_FALSE(Exherbo.IsGentoo());
}

TEST(DistroTest, DetectArchLinux) {
  llvm::vfs::InMemoryFileSystem ArchLinuxFileSystem;
  ArchLinuxFileSystem.addFile("/etc/arch-release", 0, // (empty)
                                 llvm::MemoryBuffer::getMemBuffer(""));
  ArchLinuxFileSystem.addFile("/etc/os-release", 0,
      llvm::MemoryBuffer::getMemBuffer("NAME=\"Arch Linux\"\n"
                                       "ID=arch\n"
                                       "PRETTY_NAME=\"Arch Linux\"\n"
                                       "ANSI_COLOR=\"0;36\"\n"
                                       "HOME_URL=\"https://www.archlinux.org/\"\n"
                                       "SUPPORT_URL=\"https://bbs.archlinux.org/\"\n"
                                       "BUG_REPORT_URL=\"https://bugs.archlinux.org/\"\n"));

  Distro ArchLinux{ArchLinuxFileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::ArchLinux), ArchLinux);
  ASSERT_FALSE(ArchLinux.IsUbuntu());
  ASSERT_FALSE(ArchLinux.IsRedhat());
  ASSERT_FALSE(ArchLinux.IsOpenSUSE());
  ASSERT_FALSE(ArchLinux.IsDebian());
  ASSERT_FALSE(ArchLinux.IsGentoo());
}

TEST(DistroTest, DetectGentoo) {
  llvm::vfs::InMemoryFileSystem GentooFileSystem;
  GentooFileSystem.addFile(
      "/etc/gentoo-release", 0,
      llvm::MemoryBuffer::getMemBuffer("Gentoo Base System release 2.6"));
  GentooFileSystem.addFile(
      "/etc/os-release", 0,
      llvm::MemoryBuffer::getMemBuffer(
          "NAME=Gentoo\n"
          "ID=gentoo\n"
          "PRETTY_NAME=\"Gentoo/Linux\"\n"
          "ANSI_COLOR=\"1;32\"\n"
          "HOME_URL=\"https://www.gentoo.org/\"\n"
          "SUPPORT_URL=\"https://www.gentoo.org/support/\"\n"
          "BUG_REPORT_URL=\"https://bugs.gentoo.org/\"\n"));

  Distro Gentoo{GentooFileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::Gentoo), Gentoo);
  ASSERT_FALSE(Gentoo.IsUbuntu());
  ASSERT_FALSE(Gentoo.IsRedhat());
  ASSERT_FALSE(Gentoo.IsOpenSUSE());
  ASSERT_FALSE(Gentoo.IsDebian());
  ASSERT_TRUE(Gentoo.IsGentoo());
}

TEST(DistroTest, DetectWindowsAndCrossCompile) {

  class CountingFileSystem : public llvm::vfs::ProxyFileSystem {
  public:
    CountingFileSystem() : ProxyFileSystem(llvm::vfs::getRealFileSystem()) {}

    llvm::ErrorOr<llvm::vfs::Status> status(const llvm::Twine &Path) override {
      ++Count;
      return llvm::vfs::ProxyFileSystem::status(Path);
    }

    llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
    openFileForRead(const llvm::Twine &Path) override {
      ++Count;
      return llvm::vfs::ProxyFileSystem::openFileForRead(Path);
    }

    unsigned Count{};
  };

  llvm::Triple Host(llvm::sys::getProcessTriple());
  if (!Host.isOSWindows())
    GTEST_SKIP();

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> RFS =
      llvm::vfs::getRealFileSystem();

  CountingFileSystem CFileSystem;
  Distro LinuxDistro{CFileSystem, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::UnknownDistro), LinuxDistro);
  ASSERT_GT(CFileSystem.Count, 0U);

  Distro WinDistro{CFileSystem, llvm::Triple("unknown-pc-windows")};
  ASSERT_EQ(Distro(Distro::UnknownDistro), WinDistro);
  ASSERT_GT(CFileSystem.Count, 0U);

  // When running on Windows along with a real file system, ensure that no
  // distro is returned if targeting Linux
  Distro LinuxRealDistro{*RFS, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(Distro(Distro::UnknownDistro), LinuxRealDistro);

  Distro WinRealDistro{*RFS, llvm::Triple("unknown-pc-windows")};
  ASSERT_EQ(Distro(Distro::UnknownDistro), WinRealDistro);
}

TEST(DistroTest, DetectLinux) {
  llvm::Triple Host(llvm::sys::getProcessTriple());
  if (!Host.isOSLinux())
    GTEST_SKIP();

  // When running on Linux, check if the distro is the same as the host when
  // targeting Linux
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> RFS =
      llvm::vfs::getRealFileSystem();
  Distro HostDistro{*RFS, Host};
  Distro LinuxRealDistro{*RFS, llvm::Triple("unknown-pc-linux")};
  ASSERT_EQ(HostDistro, LinuxRealDistro);
}

} // end anonymous namespace
