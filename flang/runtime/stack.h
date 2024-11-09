//===-- runtime/stack.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Trivial implementation of stack that can be used on all targets.
// It is a list based stack with dynamic allocation/deallocation
// of the list nodes.

#ifndef FORTRAN_RUNTIME_STACK_H
#define FORTRAN_RUNTIME_STACK_H

#include "terminator.h"
#include "flang/Runtime/memory.h"

namespace Fortran::runtime {
// Storage for the Stack elements of type T.
template <typename T, unsigned N> struct StackStorage {
  RT_API_ATTRS void *getElement(unsigned i) {
    if (i < N) {
      return storage[i];
    } else {
      return nullptr;
    }
  }
  RT_API_ATTRS const void *getElement(unsigned i) const {
    if (i < N) {
      return storage[i];
    } else {
      return nullptr;
    }
  }

private:
  // Storage to hold N elements of type T.
  // It is declared as an array of bytes to avoid
  // default construction (if any is implied by type T).
  alignas(T) char storage[N][sizeof(T)];
};

// 0-size specialization that provides no storage.
template <typename T> struct alignas(T) StackStorage<T, 0> {
  RT_API_ATTRS void *getElement(unsigned) { return nullptr; }
  RT_API_ATTRS const void *getElement(unsigned) const { return nullptr; }
};

template <typename T, unsigned N = 0> class Stack : public StackStorage<T, N> {
public:
  Stack() = delete;
  Stack(const Stack &) = delete;
  Stack(Stack &&) = delete;
  RT_API_ATTRS Stack(Terminator &terminator) : terminator_{terminator} {}
  RT_API_ATTRS ~Stack() {
    while (!empty()) {
      pop();
    }
  }
  RT_API_ATTRS void push(const T &object) {
    if (void *ptr{this->getElement(size_)}) {
      new (ptr) T{object};
    } else {
      top_ = New<List>{terminator_}(top_, object).release();
    }
    ++size_;
  }
  RT_API_ATTRS void push(T &&object) {
    if (void *ptr{this->getElement(size_)}) {
      new (ptr) T{std::move(object)};
    } else {
      top_ = New<List>{terminator_}(top_, std::move(object)).release();
    }
    ++size_;
  }
  template <typename... Args> RT_API_ATTRS void emplace(Args &&...args) {
    if (void *ptr{this->getElement(size_)}) {
      new (ptr) T{std::forward<Args>(args)...};
    } else {
      top_ =
          New<List>{terminator_}(top_, std::forward<Args>(args)...).release();
    }
    ++size_;
  }
  RT_API_ATTRS T &top() {
    RUNTIME_CHECK(terminator_, size_ > 0);
    if (void *ptr{this->getElement(size_ - 1)}) {
      return *reinterpret_cast<T *>(ptr);
    } else {
      RUNTIME_CHECK(terminator_, top_);
      return top_->object_;
    }
  }
  RT_API_ATTRS const T &top() const {
    RUNTIME_CHECK(terminator_, size_ > 0);
    if (void *ptr{this->getElement(size_ - 1)}) {
      return *reinterpret_cast<const T *>(ptr);
    } else {
      RUNTIME_CHECK(terminator_, top_);
      return top_->object_;
    }
  }
  RT_API_ATTRS void pop() {
    RUNTIME_CHECK(terminator_, size_ > 0);
    if (void *ptr{this->getElement(size_ - 1)}) {
      reinterpret_cast<T *>(ptr)->~T();
    } else {
      RUNTIME_CHECK(terminator_, top_);
      List *next{top_->next_};
      top_->~List();
      FreeMemory(top_);
      top_ = next;
    }
    --size_;
  }
  RT_API_ATTRS bool empty() const { return size_ == 0; }

private:
  struct List {
    template <typename... Args>
    RT_API_ATTRS List(List *next, Args &&...args)
        : next_(next), object_(std::forward<Args>(args)...) {}
    RT_API_ATTRS List(List *next, const T &object)
        : next_(next), object_(object) {}
    RT_API_ATTRS List(List *next, T &&object)
        : next_(next), object_(std::move(object)) {}
    List *next_{nullptr};
    T object_;
  };
  List *top_{nullptr};
  std::size_t size_{0};
  Terminator &terminator_;
};
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_STACK_H
