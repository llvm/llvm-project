//===- bolt/Profile/YAMLProfileWriter.h - Write profile in YAML -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PROFILE_YAML_PROFILE_WRITER_H
#define BOLT_PROFILE_YAML_PROFILE_WRITER_H

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
};

} // namespace bolt
} // namespace llvm

#endif
