//===-- sanitizer_vector.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between sanitizers run-time libraries.
//
//===----------------------------------------------------------------------===//

// Low-fat STL-like vector container.

#pragma once

#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "rsan_defs.hpp"

namespace Robustness {

template<typename T>
class Vector {
 public:
  Vector() : begin_(), end_(), last_() {}

  ~Vector() {
    if (begin_)
      InternalFree(begin_);
  }

  void clear() {
    if (begin_)
      InternalFree(begin_);
    begin_ = 0;
    end_ = 0;
    last_ = 0;
  }

  uptr size() const {
    return end_ - begin_;
  }

  bool empty() const {
    return end_ == begin_;
  }

  T &operator[](uptr i) {
    DCHECK_LT(i, end_ - begin_);
    return begin_[i];
  }

  const T &operator[](uptr i) const {
    DCHECK_LT(i, end_ - begin_);
    return begin_[i];
  }

  T *push_back() {
    EnsureSize(size() + 1);
    T *p = &end_[-1];
    internal_memset(p, 0, sizeof(*p));
    return p;
  }

  T *push_back(const T& v) {
    EnsureSize(size() + 1);
    T *p = &end_[-1];
    internal_memcpy(p, &v, sizeof(*p));
    return p;
  }

  T *insert(u64 i, const T& v) {
    DCHECK_LE(i, end_ - begin_);
    EnsureSize(size() + 1);
	auto start = begin_ + i;
	internal_memmove(start+1, start, ((end_-1) - start) * sizeof(T));
    T *p = &begin_[i];
    internal_memcpy(p, &v, sizeof(*p));
    return p;
  }

  void pop_back() {
    DCHECK_GT(end_, begin_);
    end_--;
  }

  void resize(uptr size_) {
    uptr old_size = size();
    if (size_ <= old_size) {
      end_ = begin_ + size_;
      return;
    }
    EnsureSize(size_);
	if (size_ > old_size)
		internal_memset(&begin_[old_size], 0,
				sizeof(T) * (size_ - old_size));
  }

  void ensureSize(uptr size_){
	  auto oldSize = size();
	  EnsureSize(size_);
	  if (size_ > oldSize)
		  internal_memset(&begin_[oldSize], 0,
				  sizeof(T) * (size_ - oldSize));
  }

  Vector& operator=(const Vector &w){
	  resize(w.size());
	  internal_memcpy(begin_, w.begin_, w.size()* sizeof(T));
	  return *this;
  }

  T* begin() const{
    return begin_;
  }
  T* end() const{
    return end_;
  }
  const T* cbegin() const{
    return begin_;
  }
  const T* cend() const{
    return end_;
  }

  void reserve(uptr size_){
    if (size_ <= (uptr)(last_ - begin_)) {
      return;
    }
	uptr oldSize = end_ - begin_;
    uptr cap0 = last_ - begin_;
    uptr cap = cap0 * 5 / 4;  // 25% growth
    if (cap == 0)
      cap = 16;
    if (cap < size_)
      cap = size_;
    T *p = (T*)InternalAlloc(cap * sizeof(T));
    if (cap0) {
      internal_memcpy(p, begin_, oldSize * sizeof(T));
      InternalFree(begin_);
    }
    begin_ = p;
    end_ = begin_ + oldSize;
    last_ = begin_ + cap;
  }

 private:
  T *begin_;
  T *end_;
  T *last_;

  void EnsureSize(uptr size_) {
    if (size_ <= size())
      return;
    if (size_ <= (uptr)(last_ - begin_)) {
      end_ = begin_ + size_;
      return;
    }
    uptr cap0 = last_ - begin_;
    uptr cap = cap0 * 5 / 4;  // 25% growth
    if (cap == 0)
      cap = 16;
    if (cap < size_)
      cap = size_;
    T *p = (T*)InternalAlloc(cap * sizeof(T));
    if (cap0) {
      internal_memcpy(p, begin_, cap0 * sizeof(T));
      InternalFree(begin_);
    }
    begin_ = p;
    end_ = begin_ + size_;
    last_ = begin_ + cap;
  }

  //Vector(const Vector&);
};
}  // namespace Robustness
