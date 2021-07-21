//===--- DarwinSDKInfo.cpp - SDK Information parser for darwin - ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DarwinSDKInfo.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace clang;

static Optional<DarwinSDKInfo>
parseDarwinSDKSettingsJSON(const llvm::json::Object *Obj) {
  auto VersionString = Obj->getString("Version");
  if (!VersionString)
    return None;
  VersionTuple Version;
  if (Version.tryParse(*VersionString))
    return None;
  DarwinSDKInfo SDKInfo(Version);
  if (const auto *VM = Obj->getObject("VersionMap")) {
    auto parseVersionMap = [](const llvm::json::Object &Obj,
                              llvm::StringMap<VersionTuple> &Mapping) -> bool {
      for (const auto &KV : Obj) {
        if (auto Val = KV.getSecond().getAsString()) {
          llvm::VersionTuple Version;
          if (Version.tryParse(*Val))
            return true;
          Mapping[KV.getFirst()] = Version;
        }
      }
      return false;
    };
    if (const auto *Mapping = VM->getObject("macOS_iOSMac")) {
      if (parseVersionMap(*Mapping,
                          SDKInfo.getVersionMap().MacOS2iOSMacMapping))
        return None;
    }
    if (const auto *Mapping = VM->getObject("iOSMac_macOS")) {
      if (parseVersionMap(*Mapping,
                          SDKInfo.getVersionMap().IosMac2macOSMapping))
        return None;
    }
  }
  return std::move(SDKInfo);
}

Expected<Optional<DarwinSDKInfo>>
clang::parseDarwinSDKInfo(llvm::vfs::FileSystem &VFS, StringRef SDKRootPath) {
  llvm::SmallString<256> Filepath = SDKRootPath;
  llvm::sys::path::append(Filepath, "SDKSettings.json");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File =
      VFS.getBufferForFile(Filepath);
  if (!File) {
    // If the file couldn't be read, assume it just doesn't exist.
    return None;
  }
  Expected<llvm::json::Value> Result =
      llvm::json::parse(File.get()->getBuffer());
  if (!Result)
    return Result.takeError();

  if (const auto *Obj = Result->getAsObject()) {
    if (auto SDKInfo = parseDarwinSDKSettingsJSON(Obj))
      return std::move(SDKInfo);
  }
  return llvm::make_error<llvm::StringError>("invalid SDKSettings.json",
                                             llvm::inconvertibleErrorCode());
}
