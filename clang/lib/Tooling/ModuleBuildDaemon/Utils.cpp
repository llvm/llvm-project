//===------------------------------ Utils.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/Utils.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

#include <string>

using namespace llvm;

namespace clang::tooling::cc1modbuildd {

void writeError(llvm::Error Err, std::string Msg) {
  handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
    errs() << Msg << EIB.message() << '\n';
  });
}

std::string getFullErrorMsg(llvm::Error Err, std::string Msg) {
  std::string ErrMessage;
  handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
    ErrMessage = Msg + EIB.message();
  });
  return ErrMessage;
}

llvm::Error makeStringError(llvm::Error Err, std::string Msg) {
  std::string ErrMsg = getFullErrorMsg(std::move(Err), Msg);
  return llvm::make_error<StringError>(ErrMsg, inconvertibleErrorCode());
}

std::string getBasePath() {
  llvm::BLAKE3 Hash;
  Hash.update(clang::getClangFullVersion());
  auto HashResult = Hash.final<sizeof(uint64_t)>();
  uint64_t HashValue =
      llvm::support::endian::read<uint64_t, llvm::endianness::native>(
          HashResult.data());
  std::string Key = toString(llvm::APInt(64, HashValue), 36, /*Signed*/ false);

  // Set paths
  llvm::SmallString<128> BasePath;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot*/ true, BasePath);
  llvm::sys::path::append(BasePath, "clang-" + Key);
  return BasePath.c_str();
}

bool validBasePathLength(llvm::StringRef Address) {
  // Confirm that the user provided BasePath is short enough to allow the socket
  // address to fit within the space alloted to sockaddr_un::sun_path
  if (Address.str().length() > BasePathMaxLength) {
    return false;
  }
  return true;
}

} // namespace clang::tooling::cc1modbuildd
