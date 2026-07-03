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
#include <utility>

namespace Fortran::evaluate::value {

// ----------------------------------------------------------------------------
// LogicalValue: runtime-kind logical.
// ----------------------------------------------------------------------------
class LogicalValue {
public:
  // Raw bit pattern of the value, held as the runtime-kind integer facade.
  using Word = IntegerValue;
  // The value is held as a single IntegerValue whose bit width is the logical
  // kind's width (8/16/32/64 for kinds 1/2/4/8), preserving the kind's raw bit
  // representation for word()/TRANSFER.  Its 1-bit monostate (inherited from
  // IntegerValue) models the former monostate.  All supported logical kinds use
  // C's representation (.TRUE. == 1, .FALSE. == 0), so any non-zero word is
  // .TRUE.
  using Storage = IntegerValue;

  LogicalValue() : storage_(0, 4) {} // kind-4 .FALSE.
  LogicalValue(const LogicalValue &) = default;
  LogicalValue(LogicalValue &&) = default;
  LogicalValue &operator=(const LogicalValue &) = default;
  LogicalValue &operator=(LogicalValue &&) = default;

  // The default kind (4) is used when no runtime kind is supplied.
  LogicalValue(bool truth, int kind = 4) : storage_(truth ? 1 : 0, kind) {}
  // Interpret w as the raw bit pattern of a logical of the given runtime kind.
  LogicalValue(const Word &w, int kind);

  // Comparison operators
  bool operator==(const LogicalValue &y) const {
    return IsTrue() == y.IsTrue();
  }
  bool operator!=(const LogicalValue &y) const {
    return IsTrue() != y.IsTrue();
  }
  bool operator<(const LogicalValue &y) const {
    return !IsTrue() && y.IsTrue();
  }
  bool operator>(const LogicalValue &y) const { return y < *this; }
  bool operator<=(const LogicalValue &y) const { return !(y < *this); }
  bool operator>=(const LogicalValue &y) const { return !(*this < y); }

  // Runtime kind / width accessors
  int kind() const;
  int bits() const;
  // IntegerValue's 1-bit monostate marks the default-initialized state.
  bool IsMonostate() const { return storage_.IsMonostate(); }
  bool IsZero() const;
  bool IsTrue() const;
  bool IsCanonical() const;
  bool StoreRawBytes(void *to, std::size_t bytes) const;
  static LogicalValue FromRawBytes(const void *raw, std::size_t bytes);
  static LogicalValue Zero(int kind);

  Word word() const;

  // Logical operations
  LogicalValue NOT() const;
  LogicalValue AND(const LogicalValue &y) const;
  LogicalValue OR(const LogicalValue &y) const;
  LogicalValue EQV(const LogicalValue &y) const;
  LogicalValue NEQV(const LogicalValue &y) const;
  LogicalValue ConvertToKind(int kind) const { return {IsTrue(), kind}; }

private:
  // Wraps a raw integer word result back into the facade.
  static LogicalValue Wrap(Word w) {
    LogicalValue v;
    v.storage_ = std::move(w);
    return v;
  }

  // The canonical .TRUE. word (1) at this value's width.
  Word One() const { return Word(1, kind()); }

  // Maps a Fortran logical kind (1/2/4/8) to its bit width (8 * kind).
  static int BitsForKind(int kind);

  Storage storage_;
};

} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_LOGICAL_VALUE_H_
