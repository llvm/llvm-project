//===--- PathRemapper.h - Remap filepath prefixes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a data structure that stores a string-to-string
//  mapping used to transform file paths based on a prefix mapping. It
//  is optimized for the common case of having 0, 1, or 2 mappings.
//
//  Remappings are stored such that they are applied in the order they
//  are passed on the command line, with the first one winning - a path will
//  only be remapped by a single mapping, if any, not multiple. The ordering
//  would only matter if one source mapping was a prefix of another.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_PATHREMAPPER_H
#define LLVM_CLANG_BASIC_PATHREMAPPER_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"

#include <string>
#include <utility>

namespace clang {

class PathRemapper {
  SmallVector<std::pair<std::string, std::string>, 2> PathMappings;
public:
  /// Adds a mapping such that any paths starting with `FromPrefix` have that
  /// portion replaced with `ToPrefix`.
  void addMapping(StringRef FromPrefix, StringRef ToPrefix) {
    PathMappings.emplace_back(FromPrefix.str(), ToPrefix.str());
  }

  /// Returns a remapped `Path` if it starts with a prefix in the remapper;
  /// otherwise the original path is returned.
  std::string remapPath(StringRef Path) const {
    for (const auto &Mapping : PathMappings)
      if (Path.startswith(Mapping.first))
        return (Twine(Mapping.second) +
                Path.substr(Mapping.first.size())).str();
    return Path.str();
  }

  /// Remaps `PathBuf` if it starts with a prefix in the remapper. Avoids any
  /// heap allocations if the path is not remapped.
  void remapPath(SmallVectorImpl<char> &PathBuf) const {
    for (const auto &E : PathMappings)
      if (llvm::sys::path::replace_path_prefix(PathBuf, E.first, E.second))
        break;
  }

  /// Return true if there are no path remappings (meaning remapPath will always
  /// return the path given).
  bool empty() const {
    return PathMappings.empty();
  }

  ArrayRef<std::pair<std::string, std::string>> getMappings() const {
    return PathMappings;
  }
};

} // end namespace clang

#endif
