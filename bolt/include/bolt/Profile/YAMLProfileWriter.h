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

  /// Callback to determine if a function is covered by BAT.
  using IsBATCallbackTy = std::optional<function_ref<bool(uint64_t Address)>>;
  /// Callback to get secondary entry point id for a given function and offset.
  using GetBATSecondaryEntryPointIdCallbackTy =
      std::optional<function_ref<unsigned(uint64_t Address, uint32_t Offset)>>;

  static yaml::bolt::BinaryFunctionProfile
  convert(const BinaryFunction &BF, bool UseDFS,
          IsBATCallbackTy IsBATFunction = std::nullopt,
          GetBATSecondaryEntryPointIdCallbackTy GetBATSecondaryEntryPointId =
              std::nullopt);
};

} // namespace bolt
} // namespace llvm

#endif
