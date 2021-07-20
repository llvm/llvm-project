//===--- DarwinSDKInfo.h - SDK Information parser for darwin ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_DARWIN_SDK_INFO_H
#define LLVM_CLANG_BASIC_DARWIN_SDK_INFO_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace clang {

/// The information about the darwin SDK that was used during this compilation.
class DarwinSDKInfo {
public:
  /// Represents the mapping between macOS and IosMac versions.
  class IOSMacVersionMapping {
  public:
    VersionTuple getiOSMacVersionForMacOSVersion();

    llvm::StringMap<VersionTuple> MacOS2iOSMacMapping;
    llvm::StringMap<VersionTuple> IosMac2macOSMapping;
  };

  DarwinSDKInfo(llvm::VersionTuple Version) : Version(Version) {}

  const llvm::VersionTuple &getVersion() const { return Version; }

  IOSMacVersionMapping &getVersionMap() { return VersionMap; }

  const IOSMacVersionMapping &getVersionMap() const { return VersionMap; }

private:
  llvm::VersionTuple Version;
  IOSMacVersionMapping VersionMap;
};

/// Parse the SDK information from the SDKSettings.json file.
///
/// \returns an error if the SDKSettings.json file is invalid, None if the
/// SDK has no SDKSettings.json, or a valid \c DarwinSDKInfo otherwise.
Expected<Optional<DarwinSDKInfo>> parseDarwinSDKInfo(llvm::vfs::FileSystem &VFS,
                                                     StringRef SDKRootPath);

} // end namespace clang

#endif // LLVM_CLANG_BASIC_DARWIN_SDK_INFO_H
