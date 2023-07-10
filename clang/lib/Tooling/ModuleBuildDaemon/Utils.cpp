//===------------------------------ Utils.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clang/Tooling/ModuleBuildDaemon/Utils.h>
#include <llvm/Support/Error.h>
#include <string>

using namespace llvm;

void cc1modbuildd::writeError(llvm::Error Err, std::string Msg) {
  handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
    errs() << Msg << EIB.message() << '\n';
  });
}

std::string cc1modbuildd::getFullErrorMsg(llvm::Error Err, std::string Msg) {
  std::string ErrMessage;
  handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
    ErrMessage = Msg + EIB.message();
  });
  return ErrMessage;
}

llvm::Error cc1modbuildd::makeStringError(llvm::Error Err, std::string Msg) {
  std::string ErrMsg = getFullErrorMsg(std::move(Err), Msg);
  return llvm::make_error<StringError>(ErrMsg, inconvertibleErrorCode());
}