//===-- Either.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Either_h_
#define liblldb_Either_h_

#include "llvm/ADT/Optional.h"

#include <functional>

namespace lldb_utility {
template <typename T1, typename T2> class Either {
private:
  enum class Selected { One, Two };

  Selected m_selected;
  union {
    T1 m_t1;
    T2 m_t2;
  };

public:
  Either(const T1 &t1) {
    m_t1 = t1;
    m_selected = Selected::One;
  }

  Either(const T2 &t2) {
    m_t2 = t2;
    m_selected = Selected::Two;
  }

  Either(const Either<T1, T2> &rhs) {
    switch (rhs.m_selected) {
    case Selected::One:
      m_t1 = rhs.GetAs<T1>().getValue();
      m_selected = Selected::One;
      break;
    case Selected::Two:
      m_t2 = rhs.GetAs<T2>().getValue();
      m_selected = Selected::Two;
      break;
    }
  }

  template <class X, typename std::enable_if<std::is_same<T1, X>::value>::type
                         * = nullptr>
  llvm::Optional<T1> GetAs() const {
    switch (m_selected) {
    case Selected::One:
      return m_t1;
    default:
      return llvm::Optional<T1>();
    }
  }

  template <class X, typename std::enable_if<std::is_same<T2, X>::value>::type
                         * = nullptr>
  llvm::Optional<T2> GetAs() const {
    switch (m_selected) {
    case Selected::Two:
      return m_t2;
    default:
      return llvm::Optional<T2>();
    }
  }

  template <class ResultType>
  ResultType Apply(std::function<ResultType(T1)> if_T1,
                   std::function<ResultType(T2)> if_T2) const {
    switch (m_selected) {
    case Selected::One:
      return if_T1(m_t1);
    case Selected::Two:
      return if_T2(m_t2);
    }

    llvm_unreachable("Unhandled Selected in switch.");
  }

  bool operator==(const Either<T1, T2> &rhs) {
    return (GetAs<T1>() == rhs.GetAs<T1>()) && (GetAs<T2>() == rhs.GetAs<T2>());
  }

  explicit operator bool() {
    switch (m_selected) {
    case Selected::One:
      return (bool)m_t1;
    case Selected::Two:
      return (bool)m_t2;
    }

    llvm_unreachable("Unhandled Selected in switch.");
  }

  Either<T1, T2> &operator=(const Either<T1, T2> &rhs) {
    switch (rhs.m_selected) {
    case Selected::One:
      m_t1 = rhs.GetAs<T1>().getValue();
      m_selected = Selected::One;
      break;
    case Selected::Two:
      m_t2 = rhs.GetAs<T2>().getValue();
      m_selected = Selected::Two;
      break;
    }
    return *this;
  }

  ~Either() {
    switch (m_selected) {
    case Selected::One:
      m_t1.T1::~T1();
      break;
    case Selected::Two:
      m_t2.T2::~T2();
      break;
    }
  }
};

} // namespace lldb_utility

#endif // #ifndef liblldb_Either_h_
