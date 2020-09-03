//===--- Config.cpp - User configuration of clangd behavior ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "support/Context.h"

namespace clang {
namespace clangd {

Key<Config> Config::Key;

const Config &Config::current() {
  if (const Config *C = Context::current().get(Key))
    return *C;
  static Config Default;
  return Default;
}

} // namespace clangd
} // namespace clang
