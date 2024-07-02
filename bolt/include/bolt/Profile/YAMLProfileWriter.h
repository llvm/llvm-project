//===- bolt/Profile/YAMLProfileWriter.h - Write profile in YAML -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PROFILE_YAML_PROFILE_WRITER_H
#define BOLT_PROFILE_YAML_PROFILE_WRITER_H

#include "bolt/Profile/ProfileYAMLMapping.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

namespace llvm {
namespace bolt {
class BoltAddressTranslation;
class RewriteInstance;

class YAMLProfileWriter {
  YAMLProfileWriter() = delete;

  std::string Filename;

  std::unique_ptr<raw_fd_ostream> OS;

public:
  explicit YAMLProfileWriter(const std::string &Filename)
      : Filename(Filename) {}

  /// Save execution profile for that instance.
  std::error_code writeProfile(const RewriteInstance &RI);

  static yaml::bolt::BinaryFunctionProfile
  convert(const BinaryFunction &BF, bool UseDFS,
          const BoltAddressTranslation *BAT = nullptr);

  /// Set CallSiteInfo destination fields from \p Symbol and return a target
  /// BinaryFunction for that symbol.
  static const BinaryFunction *
  setCSIDestination(const BinaryContext &BC, yaml::bolt::CallSiteInfo &CSI,
                    const MCSymbol *Symbol, const BoltAddressTranslation *BAT,
                    uint32_t Offset = 0);
};

} // namespace bolt
} // namespace llvm

#endif
