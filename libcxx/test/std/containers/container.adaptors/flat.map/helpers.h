//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_FLAT_MAP_HELPERS_H
#define SUPPORT_FLAT_MAP_HELPERS_H

#include <cassert>
#include <string>
#include <vector>

struct StartsWith {
  explicit StartsWith(char ch) : lower_(1, ch), upper_(1, ch + 1) {}
  StartsWith(const StartsWith&)     = delete;
  void operator=(const StartsWith&) = delete;
  struct Less {
    using is_transparent = void;
    bool operator()(const std::string& a, const std::string& b) const { return a < b; }
    bool operator()(const StartsWith& a, const std::string& b) const { return a.upper_ <= b; }
    bool operator()(const std::string& a, const StartsWith& b) const { return a < b.lower_; }
    bool operator()(const StartsWith&, const StartsWith&) const {
      assert(false); // should not be called
      return false;
    }
  };

private:
  std::string lower_;
  std::string upper_;
};

template <class T>
struct CopyOnlyVector : std::vector<T> {
  using std::vector<T>::vector;

  CopyOnlyVector(const CopyOnlyVector&) = default;
  CopyOnlyVector(CopyOnlyVector&& other) : CopyOnlyVector(other){}
  CopyOnlyVector(CopyOnlyVector&& other, std::vector<T>::allocator_type alloc) : CopyOnlyVector(other, alloc){}

  CopyOnlyVector& operator=(const CopyOnlyVector&) = default;
  CopyOnlyVector& operator=(CopyOnlyVector& other) {
    return this->operator=(other);
  }
};

#endif // SUPPORT_FLAT_MAP_HELPERS_H
