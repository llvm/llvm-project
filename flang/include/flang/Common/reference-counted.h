//===-- include/flang/Common/reference-counted.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_REFERENCE_COUNTED_H_
#define FORTRAN_COMMON_REFERENCE_COUNTED_H_

// A class template of smart pointers to objects with their own
// reference counting object lifetimes that's lighter weight
// than std::shared_ptr<>.  Not thread-safe.

namespace Fortran::common {

// A base class for reference-counted objects.  Must be public.
template <typename A> class ReferenceCounted {
public:
  ReferenceCounted() {}
  int references() const { return references_; }
  void TakeReference() { ++references_; }
  void DropReference() {
    if (--references_ == 0) {
      delete static_cast<A *>(this);
    }
  }

private:
  int references_{0};
};

// A reference to a reference-counted object.
template <typename A> class CountedReference {
public:
  using type = A;
  CountedReference() {}
  CountedReference(type *m) : p_{m} { Take(); }
  CountedReference(const CountedReference &c) : p_{c.p_} { Take(); }
  CountedReference(CountedReference &&c) : p_{c.p_} { c.p_ = nullptr; }
  CountedReference &operator=(const CountedReference &c) {
    c.Take();
    Drop();
    p_ = c.p_;
    return *this;
  }
  CountedReference &operator=(CountedReference &&c) {
    A *p{c.p_};
    c.p_ = nullptr;
    Drop();
    p_ = p;
    return *this;
  }
  ~CountedReference() { Drop(); }
  operator bool() const { return p_ != nullptr; }
  type *get() const { return p_; }
  type &operator*() const { return *p_; }
  type *operator->() const { return p_; }

private:
  void Take() const {
    if (p_) {
      p_->TakeReference();
    }
  }
  void Drop() {
    if (p_) {
      p_->DropReference();
      p_ = nullptr;
    }
  }

  type *p_{nullptr};
};
} // namespace Fortran::common
#endif // FORTRAN_COMMON_REFERENCE_COUNTED_H_
