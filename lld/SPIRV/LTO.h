//===- LTO.h ----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_SPIRV_LTO_H
#define LLD_SPIRV_LTO_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/LTO/LTO.h"
#include <memory>
#include <string>
#include <vector>

namespace lld::spirv {

struct Config;

class BitcodeCompiler {
public:
  BitcodeCompiler(Config &config);

  void add(MemoryBufferRef mb);
  std::vector<char> compile();

private:
  Config &config;
  std::unique_ptr<llvm::lto::LTO> ltoObj;
  std::vector<SmallString<0>> buf;
  std::vector<std::string> tempFiles;
};

} // namespace lld::spirv

#endif
