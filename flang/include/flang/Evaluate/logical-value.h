//===-- include/flang/Evaluate/logical-value.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_LOGICAL_VALUE_H_
#define FORTRAN_EVALUATE_LOGICAL_VALUE_H_

#include "integer-value.h"
#include "llvm/Support/Compiler.h"
#include <utility>

namespace Fortran::evaluate::value {
using common::KindsEnum;

/// A Fortran LOGICAL value.
///
/// The kind is dynamic, but only a predefined set of Fortran kinds are
/// allowed. It is also kind-aware, i.e. knows which LOGICAL kind it currently
/// represents.
///
/// It is implemented as a wrapper around IntegerValue.
class LogicalValue {
public:
  using Word = IntegerValue;

  LogicalValue() {}
  LogicalValue(const LogicalValue &) = default;
  LogicalValue(LogicalValue &&) = default;
  LogicalValue &operator=(const LogicalValue &) = default;
  LogicalValue &operator=(LogicalValue &&) = default;

  LogicalValue(KindsEnum kind, const LogicalValue &v) : LogicalValue{v} {
    CHECK(kind == v.kind());
  }

  LogicalValue(KindsEnum kind, LogicalValue &&v) : LogicalValue{std::move(v)} {
    CHECK(kind == v.kind());
  }

  LogicalValue(KindsEnum kind, bool truth) : word_(Represent(kind, truth)) {}

  LogicalValue(KindsEnum kind, const Word &w) : word_(kind, w) {}

  /// Creates a logical with value 'false' of a given kind. This is in contrast
  /// to the default-ctor which creates a "monostate" that represents 'false' of
  /// a not-yet-known kind.
  static LogicalValue Zero(KindsEnum kind) { return LogicalValue{kind, false}; }

  void print(llvm::raw_ostream &os) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// Whether this object represents a default-initialized value ('false') of
  /// unknown kind.
  bool IsMonostate() const { return word_.IsMonostate(); }

  /// The kind of the value currently stored.
  KindsEnum kind() const { return word_.kind(); }

  int bits() const { return bits(kind()); }
  static constexpr int bits(KindsEnum kind) { return Word::bits(kind); }

  /// Number of bytes accessed by FromRawBytes/StoreRawBytes
  std::size_t bytesStored() const { return bytesStored(kind()); }
  static constexpr std::size_t bytesStored(KindsEnum kind) {
    return Word::bytesStored(kind);
  }

  Word word() const { return word_; }

  bool IsCanonical() const {
    const KindsEnum kind{this->kind()};
    return word_ == canonicalFalse(kind) || word_ == canonicalTrue(kind);
  }

  /// Fortran actually has only .EQV. & .NEQV. relational operations
  /// for LOGICAL, but this class supports more so that it can be used
  /// with the STL for sorting and as a key type for std::set<> & std::map<>.
  bool operator<(const LogicalValue &that) const {
    return !IsTrue() && that.IsTrue();
  }
  bool operator<=(const LogicalValue &that) const { return !IsTrue(); }
  bool operator==(const LogicalValue &that) const {
    return IsTrue() == that.IsTrue();
  }
  bool operator!=(const LogicalValue &that) const {
    return IsTrue() != that.IsTrue();
  }

  bool operator>=(const LogicalValue &that) const { return IsTrue(); }

  bool operator>(const LogicalValue &that) const {
    return IsTrue() && !that.IsTrue();
  }

  bool IsTrue() const { return !word_.IsZero(); }

  LogicalValue NOT() const {
    return FromWord(word_.IEOR(canonicalTrue(kind())));
  }

  LogicalValue AND(const LogicalValue &that) const {
    return FromWord(word_.IAND(that.word()));
  }

  LogicalValue OR(const LogicalValue &that) const {
    return FromWord(word_.IOR(that.word()));
  }

  LogicalValue EQV(const LogicalValue &that) const { return NEQV(that).NOT(); }

  LogicalValue NEQV(const LogicalValue &that) const {
    return FromWord(word_.IEOR(that.word()));
  }

  static LogicalValue FromRawBytes(
      KindsEnum kind, const void *raw, std::size_t expectedSize) {
    Word w{Word::FromRawBytes(kind, raw, expectedSize)};
    return LogicalValue{w.kind(), w};
  }

  void StoreRawBytes(void *dst, size_t size, bool *changed = nullptr) const {
    word_.StoreRawBytes(dst, size, changed);
  }

private:
  static Word canonicalTrue(KindsEnum kind) { return Word{kind, 1}; }

  static Word canonicalFalse(KindsEnum kind) { return Word{kind, 0}; }

  static Word Represent(KindsEnum kind, bool x) {
    return x ? canonicalTrue(kind) : canonicalFalse(kind);
  }

  static LogicalValue FromWord(const Word &w) {
    LogicalValue v;
    v.word_ = w;
    return v;
  }

  static LogicalValue FromWord(Word &&w) {
    LogicalValue v;
    v.word_ = std::move(w);
    return v;
  }

  Word word_;
};

} // namespace Fortran::evaluate::value

namespace llvm {
/// For pretty printing in GTest
inline raw_ostream &operator<<(
    raw_ostream &os, const Fortran::evaluate::value::LogicalValue &v) {
  v.print(os);
  return os;
}
} // namespace llvm

#endif // FORTRAN_EVALUATE_LOGICAL_VALUE_H_
