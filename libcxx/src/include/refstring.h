//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_REFSTRING_H
#define _LIBCPP_REFSTRING_H

#include "atomic_support.h"
#include <__config>
#include <cstddef>
#include <cstring>
#include <stdexcept>

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wc99-extensions")

struct __libcpp_refstring::__rep {
  ptrdiff_t refcount;
  char data[];
};

inline __libcpp_refstring::__libcpp_refstring(const char* msg) {
  std::size_t len = strlen(msg);
  auto* rep       = static_cast<__rep*>(::operator new(sizeof(__rep) + len + 1));
  rep->refcount      = 0;
  std::memcpy(rep->data, msg, len + 1);
  __imp_ = rep;
}

inline __libcpp_refstring::__libcpp_refstring(const __libcpp_refstring& s) noexcept : __imp_(s.__imp_) {
  __libcpp_atomic_add(&__imp_->refcount, 1);
}

inline __libcpp_refstring& __libcpp_refstring::operator=(__libcpp_refstring const& s) noexcept {
  __rep* old_rep = __imp_;
  __imp_         = s.__imp_;
  __libcpp_atomic_add(&__imp_->refcount, 1);

  if (__libcpp_atomic_add(&old_rep->refcount, ptrdiff_t(-1)) < 0)
    ::operator delete(old_rep);
  return *this;
}

inline __libcpp_refstring::~__libcpp_refstring() {
  if (__libcpp_atomic_add(&__imp_->refcount, ptrdiff_t(-1)) < 0)
    ::operator delete(__imp_);
}

inline const char* __libcpp_refstring::c_str() const noexcept { return __imp_->data; }

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_REFSTRING_H
