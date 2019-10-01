//===--- DarwinSDKInfo.cpp - SDK Information parser for darwin - ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/DarwinSDKInfo.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang;

Expected<Optional<DarwinSDKInfo>>
driver::parseDarwinSDKInfo(llvm::vfs::FileSystem &VFS, StringRef SDKRootPath) {
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
    auto VersionString = Obj->getString("Version");
    if (VersionString) {
      VersionTuple Version;
      if (!Version.tryParse(*VersionString))
        return DarwinSDKInfo(Version);
    }
  }
  return llvm::make_error<llvm::StringError>("invalid SDKSettings.json",
                                             llvm::inconvertibleErrorCode());
}
