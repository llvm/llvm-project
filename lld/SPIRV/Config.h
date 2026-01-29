//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_SPIRV_CONFIG_H
#define LLD_SPIRV_CONFIG_H

#include <string>

namespace lld::spirv {

struct Config {
  std::string targetTriple = "spirv64";
  unsigned ltoOptLevel = 2;
};

} // namespace lld::spirv

#endif
