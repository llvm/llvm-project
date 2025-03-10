//===------------ bolt/Rewrite/MetadataRewriter.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/MetadataRewriter.h"
#include "bolt/Rewrite/RewriteInstance.h"

using namespace llvm;
using namespace bolt;

MetadataRewriter::MetadataRewriter(StringRef Name, RewriteInstance &RI)
    : Name(Name), RI(RI), BC(*RI.BC) {}

std::optional<uint64_t> MetadataRewriter::lookupSymbol(const StringRef Name) {
  return RI.Linker->lookupSymbol(Name);
}
