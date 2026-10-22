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
#include <set>
#include <string>
#include <utility>

namespace Fortran::semantics {

using ModuleCheckSumType = std::uint64_t;

class ModuleDependences {
public:
  // 'requiredBy' is the name of the module whose module file
  // records the dependence.
  void AddDependence(std::string &&name, bool intrinsic,
      ModuleCheckSumType hash, std::string &&requiredBy) {
    Dependence dependence{hash, std::move(requiredBy)};
    if (intrinsic) {
      intrinsicMap_.insert_or_assign(std::move(name), std::move(dependence));
    } else {
      nonIntrinsicMap_.insert_or_assign(std::move(name), std::move(dependence));
    }
  }
  std::optional<ModuleCheckSumType> GetRequiredHash(
      const std::string &name, bool intrinsic) const {
    if (const Dependence *dependence{Find(name, intrinsic)}) {
      return dependence->hash;
    }
    return std::nullopt;
  }
  // Returns the name of the module that most recently required 'name'
  const std::string *GetRequiredBy(
      const std::string &name, bool intrinsic) const {
    if (const Dependence *dependence{Find(name, intrinsic)}) {
      return &dependence->requiredBy;
    }
    return nullptr;
  }

  // Returns true the first time it is called for a module name and required
  // hash, so that the module search path is scanned for homonyms only once.
  bool MarkHomonymSearchDone(const std::string &name, ModuleCheckSumType hash) {
    return homonymSearchDone_.emplace(name, hash).second;
  }

private:
  struct Dependence {
    ModuleCheckSumType hash;
    std::string requiredBy;
  };
  const Dependence *Find(const std::string &name, bool intrinsic) const {
    const auto &map{intrinsic ? intrinsicMap_ : nonIntrinsicMap_};
    if (auto iter{map.find(name)}; iter != map.end()) {
      return &iter->second;
    }
    return nullptr;
  }
  std::map<std::string, Dependence> intrinsicMap_, nonIntrinsicMap_;
  std::set<std::pair<std::string, ModuleCheckSumType>> homonymSearchDone_;
};

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_MODULE_DEPENDENCES_H_
