//===-- lib/Support/PluginDirectives.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/PluginDirectives.h"
#include <list>

namespace Fortran::common {

// A list, so that the specs never move once registered.
static std::list<PluginDirectiveSpec> &getPluginDirectives() {
  static std::list<PluginDirectiveSpec> specs;
  return specs;
}

void registerPluginDirective(PluginDirectiveSpec spec) {
  getPluginDirectives().push_back(std::move(spec));
}

bool isPluginDirectivePrefix(std::string_view prefix) {
  for (const PluginDirectiveSpec &spec : getPluginDirectives()) {
    if (spec.prefix == prefix) {
      return true;
    }
  }
  return false;
}

const PluginDirectiveSpec *lookupPluginDirective(
    std::string_view prefix, std::string_view keyword) {
  for (const PluginDirectiveSpec &spec : getPluginDirectives()) {
    if (spec.prefix == prefix && spec.keyword == keyword) {
      return &spec;
    }
  }
  return nullptr;
}

static std::vector<std::string> &getSentinels() {
  static std::vector<std::string> sentinels;
  return sentinels;
}

void registerPluginDirectiveSentinel(std::string_view prefix) {
  std::string sentinel{"$"};
  sentinel += prefix;
  for (const std::string &s : getSentinels()) {
    if (s == sentinel) {
      return;
    }
  }
  getSentinels().push_back(std::move(sentinel));
}

const std::vector<std::string> &getPluginDirectiveSentinels() {
  return getSentinels();
}

} // namespace Fortran::common
