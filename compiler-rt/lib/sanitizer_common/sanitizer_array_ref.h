//===-- sanitizer_array_ref.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_ARRAY_REF_H
#define SANITIZER_ARRAY_REF_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

/// ArrayRef - Represent a constant reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the ArrayRef. For this reason, it is not in general
/// safe to store an ArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
template <typename T>
class ArrayRef {
 public:
  ArrayRef() {}
  ArrayRef(const T *begin, const T *end) : begin_(begin), end_(end) {}

  template <typename C>
  ArrayRef(const C &src) : ArrayRef(src.data(), src.data() + src.size()) {}

  const T *begin() const { return begin_; }
  const T *end() const { return end_; }

  bool empty() const { return begin_ == end_; }

  uptr size() const { return end_ - begin_; }

 private:
  const T *begin_ = nullptr;
  const T *end_ = nullptr;
};

}  // namespace __sanitizer

#endif  // SANITIZER_ARRAY_REF_H
