//===- ParsedAttrInfo.cpp - Registry for attribute plugins ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Registry of attributes added by plugins which
// derive the ParsedAttrInfo class.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/ParsedAttrInfo.h"
#include "llvm/Support/ManagedStatic.h"
#include <list>
#include <memory>

using namespace clang;

LLVM_INSTANTIATE_REGISTRY(ParsedAttrInfoRegistry)

static std::list<std::unique_ptr<ParsedAttrInfo>> instantiateEntries() {
  std::list<std::unique_ptr<ParsedAttrInfo>> Instances;
  for (const auto &It : ParsedAttrInfoRegistry::entries())
    Instances.emplace_back(It.instantiate());
  return Instances;
}

const std::list<std::unique_ptr<ParsedAttrInfo>> &
clang::getAttributePluginInstances() {
  static std::list<std::unique_ptr<ParsedAttrInfo>> Instances =
      instantiateEntries();
  return Instances;
}
