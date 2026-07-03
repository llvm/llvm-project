//===-- Iterable.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_ITERABLE_H
#define LLDB_UTILITY_ITERABLE_H

#include <utility>

#include <llvm/ADT/iterator.h>

namespace lldb_private {

template <typename WrappedIteratorT,
          typename T = typename std::iterator_traits<
              WrappedIteratorT>::value_type::second_type>
struct ValueMapIterator
    : llvm::iterator_adaptor_base<
          ValueMapIterator<WrappedIteratorT, T>, WrappedIteratorT,
          typename std::iterator_traits<WrappedIteratorT>::iterator_category,
          T> {
  ValueMapIterator() = default;
  explicit ValueMapIterator(WrappedIteratorT u)
      : ValueMapIterator::iterator_adaptor_base(std::move(u)) {}

  const T &operator*() { return (*this->I).second; }
  const T &operator*() const { return (*this->I).second; }
};

template <typename MutexType, typename C,
          typename IteratorT = typename C::const_iterator>
class LockingAdaptedIterable : public llvm::iterator_range<IteratorT> {
public:
  LockingAdaptedIterable(const C &container, MutexType &mutex)
      : llvm::iterator_range<IteratorT>(container), m_mutex(&mutex) {
    m_mutex->lock();
  }

  LockingAdaptedIterable(LockingAdaptedIterable &&rhs)
      : llvm::iterator_range<IteratorT>(rhs), m_mutex(rhs.m_mutex) {
    rhs.m_mutex = nullptr;
  }

  ~LockingAdaptedIterable() {
    if (m_mutex)
      m_mutex->unlock();
  }

private:
  MutexType *m_mutex = nullptr;

  LockingAdaptedIterable(const LockingAdaptedIterable &) = delete;
  LockingAdaptedIterable &operator=(const LockingAdaptedIterable &) = delete;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_ITERABLE_H
