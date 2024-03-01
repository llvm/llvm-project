//===-- include/flang/Semantics/module-dependences.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_MODULE_DEPENDENCES_H_
#define FORTRAN_SEMANTICS_MODULE_DEPENDENCES_H_

#include <cinttypes>
#include <map>
#include <optional>
#include <string>

namespace Fortran::semantics {

using ModuleCheckSumType = std::uint64_t;

class ModuleDependences {
public:
  void AddDependence(
      std::string &&name, bool intrinsic, ModuleCheckSumType hash) {
    if (intrinsic) {
      intrinsicMap_.emplace(std::move(name), hash);
    } else {
      nonIntrinsicMap_.emplace(std::move(name), hash);
    }
  }
  std::optional<ModuleCheckSumType> GetRequiredHash(
      const std::string &name, bool intrinsic) {
    if (intrinsic) {
      if (auto iter{intrinsicMap_.find(name)}; iter != intrinsicMap_.end()) {
        return iter->second;
      }
    } else {
      if (auto iter{nonIntrinsicMap_.find(name)};
          iter != nonIntrinsicMap_.end()) {
        return iter->second;
      }
    }
    return std::nullopt;
  }

private:
  std::map<std::string, ModuleCheckSumType> intrinsicMap_, nonIntrinsicMap_;
};

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_MODULE_DEPENDENCES_H_
