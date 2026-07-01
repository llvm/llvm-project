//===-- lib/Evaluate/logical-value.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/logical-value.h"

namespace Fortran::evaluate::value {

// ============================================================================
// LogicalValue out-of-line definitions.
// ============================================================================

int LogicalValue::BitsForKind(int kind) {
  switch (kind) {
  case 1:
    return 8;
  case 2:
    return 16;
  case 4:
    return 32;
  case 8:
    return 64;
  default:
    llvm_unreachable("arbitrary kinds not supported");
    return 32;
  }
}

int LogicalValue::kind() const { return storage_.kind(); }

int LogicalValue::bits() const { return storage_.bits(); }

LogicalValue LogicalValue::Zero(int kind) { return LogicalValue{false, kind}; }

bool LogicalValue::IsZero() const { return IsMonostate() || !IsTrue(); }

bool LogicalValue::IsTrue() const { return !storage_.IsZero(); }

bool LogicalValue::IsCanonical() const {
  if (IsMonostate()) {
    return true;
  }
  return storage_.ToUInt64() <= 1;
}

LogicalValue LogicalValue::FromRawBytes(const void *raw, std::size_t bytes) {
  Word w{Word::FromRawBytes(raw, static_cast<int>(bytes))};
  return LogicalValue{w, w.kind()};
}

typename LogicalValue::Word LogicalValue::word() const { return storage_; }

bool LogicalValue::StoreRawBytes(void *to, std::size_t bytes) const {
  return storage_.StoreRawBytes(to, static_cast<int>(bytes));
}

LogicalValue LogicalValue::NOT() const {
  if (IsMonostate()) {
    return LogicalValue{};
  }
  return Wrap(storage_.IEOR(One()));
}

LogicalValue LogicalValue::AND(const LogicalValue &y) const {
  if (IsMonostate()) {
    return LogicalValue{};
  }
  return Wrap(storage_.IAND(y.word()));
}

LogicalValue LogicalValue::OR(const LogicalValue &y) const {
  if (IsMonostate()) {
    return LogicalValue{};
  }
  return Wrap(storage_.IOR(y.word()));
}

LogicalValue LogicalValue::EQV(const LogicalValue &y) const {
  if (IsMonostate()) {
    return LogicalValue{};
  }
  return Wrap(storage_.IEOR(y.word()).IEOR(One()));
}

LogicalValue LogicalValue::NEQV(const LogicalValue &y) const {
  if (IsMonostate()) {
    return LogicalValue{};
  }
  return Wrap(storage_.IEOR(y.word()));
}

LogicalValue::LogicalValue(const Word &w, int kind) {
  int width{BitsForKind(kind)};
  storage_ =
      w.IsMonostate() ? Word(0, kind) : Word::ConvertUnsigned(w, width).value;
}

} // namespace Fortran::evaluate::value
