//===-- include/flang/Evaluate/common.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_COMMON_H_
#define FORTRAN_EVALUATE_COMMON_H_

#include "flang/Common/Fortran.h"
#include "flang/Common/default-kinds.h"
#include "flang/Common/enum-set.h"
#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Common/restorer.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/message.h"
#include <cinttypes>
#include <map>
#include <string>

namespace Fortran::semantics {
class DerivedTypeSpec;
}

namespace Fortran::evaluate {
class IntrinsicProcTable;

using common::ConstantSubscript;
using common::RelationalOperator;

// Integers are always ordered; reals may not be.
ENUM_CLASS(Ordering, Less, Equal, Greater)
ENUM_CLASS(Relation, Less, Equal, Greater, Unordered)

template <typename A>
static constexpr Ordering Compare(const A &x, const A &y) {
  if (x < y) {
    return Ordering::Less;
  } else if (x > y) {
    return Ordering::Greater;
  } else {
    return Ordering::Equal;
  }
}

template <typename CH>
static constexpr Ordering Compare(
    const std::basic_string<CH> &x, const std::basic_string<CH> &y) {
  std::size_t xLen{x.size()}, yLen{y.size()};
  using String = std::basic_string<CH>;
  // Fortran CHARACTER comparison is defined with blank padding
  // to extend a shorter operand.
  if (xLen < yLen) {
    return Compare(String{x}.append(yLen - xLen, CH{' '}), y);
  } else if (xLen > yLen) {
    return Compare(x, String{y}.append(xLen - yLen, CH{' '}));
  } else if (x < y) {
    return Ordering::Less;
  } else if (x > y) {
    return Ordering::Greater;
  } else {
    return Ordering::Equal;
  }
}

static constexpr Ordering Reverse(Ordering ordering) {
  if (ordering == Ordering::Less) {
    return Ordering::Greater;
  } else if (ordering == Ordering::Greater) {
    return Ordering::Less;
  } else {
    return Ordering::Equal;
  }
}

static constexpr Relation RelationFromOrdering(Ordering ordering) {
  if (ordering == Ordering::Less) {
    return Relation::Less;
  } else if (ordering == Ordering::Greater) {
    return Relation::Greater;
  } else {
    return Relation::Equal;
  }
}

static constexpr Relation Reverse(Relation relation) {
  if (relation == Relation::Less) {
    return Relation::Greater;
  } else if (relation == Relation::Greater) {
    return Relation::Less;
  } else {
    return relation;
  }
}

static constexpr bool Satisfies(RelationalOperator op, Ordering order) {
  switch (order) {
  case Ordering::Less:
    return op == RelationalOperator::LT || op == RelationalOperator::LE ||
        op == RelationalOperator::NE;
  case Ordering::Equal:
    return op == RelationalOperator::LE || op == RelationalOperator::EQ ||
        op == RelationalOperator::GE;
  case Ordering::Greater:
    return op == RelationalOperator::NE || op == RelationalOperator::GE ||
        op == RelationalOperator::GT;
  }
  return false; // silence g++ warning
}

static constexpr bool Satisfies(RelationalOperator op, Relation relation) {
  switch (relation) {
  case Relation::Less:
    return Satisfies(op, Ordering::Less);
  case Relation::Equal:
    return Satisfies(op, Ordering::Equal);
  case Relation::Greater:
    return Satisfies(op, Ordering::Greater);
  case Relation::Unordered:
    return false;
  }
  return false; // silence g++ warning
}

ENUM_CLASS(
    RealFlag, Overflow, DivideByZero, InvalidArgument, Underflow, Inexact)

using RealFlags = common::EnumSet<RealFlag, RealFlag_enumSize>;

template <typename A> struct ValueWithRealFlags {
  A AccumulateFlags(RealFlags &f) {
    f |= flags;
    return value;
  }
  A value;
  RealFlags flags{};
};

struct Rounding {
  common::RoundingMode mode{common::RoundingMode::TiesToEven};
  // When set, emulate status flag behavior peculiar to x86
  // (viz., fail to set the Underflow flag when an inexact product of a
  // multiplication is rounded up to a normal number from a subnormal
  // in some rounding modes)
#if __x86_64__
  bool x86CompatibleBehavior{true};
#else
  bool x86CompatibleBehavior{false};
#endif
};

static constexpr Rounding defaultRounding;

#if FLANG_BIG_ENDIAN
constexpr bool isHostLittleEndian{false};
#elif FLANG_LITTLE_ENDIAN
constexpr bool isHostLittleEndian{true};
#else
#error host endianness is not known
#endif

// HostUnsignedInt<BITS> finds the smallest native unsigned integer type
// whose size is >= BITS.
template <bool LE8, bool LE16, bool LE32, bool LE64> struct SmallestUInt {};
template <> struct SmallestUInt<true, true, true, true> {
  using type = std::uint8_t;
};
template <> struct SmallestUInt<false, true, true, true> {
  using type = std::uint16_t;
};
template <> struct SmallestUInt<false, false, true, true> {
  using type = std::uint32_t;
};
template <> struct SmallestUInt<false, false, false, true> {
  using type = std::uint64_t;
};
template <int BITS>
using HostUnsignedInt =
    typename SmallestUInt<BITS <= 8, BITS <= 16, BITS <= 32, BITS <= 64>::type;

// Many classes in this library follow a common paradigm.
// - There is no default constructor (Class() {}), usually to prevent the
//   need for std::monostate as a default constituent in a std::variant<>.
// - There are full copy and move semantics for construction and assignment.
// - Discriminated unions have a std::variant<> member "u" and support
//   explicit copy and move constructors as well as comparison for equality.
#define DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(t) \
  t(const t &); \
  t(t &&); \
  t &operator=(const t &); \
  t &operator=(t &&);
#define DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(t) \
  t(const t &) = default; \
  t(t &&) = default; \
  t &operator=(const t &) = default; \
  t &operator=(t &&) = default;
#define DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(t) \
  t::t(const t &) = default; \
  t::t(t &&) = default; \
  t &t::operator=(const t &) = default; \
  t &t::operator=(t &&) = default;
#define CONSTEXPR_CONSTRUCTORS_AND_ASSIGNMENTS(t) \
  constexpr t(const t &) = default; \
  constexpr t(t &&) = default; \
  constexpr t &operator=(const t &) = default; \
  constexpr t &operator=(t &&) = default;

#define CLASS_BOILERPLATE(t) \
  t() = delete; \
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(t)

#define UNION_CONSTRUCTORS(t) \
  template <typename _A> explicit t(const _A &x) : u{x} {} \
  template <typename _A, typename = common::NoLvalue<_A>> \
  explicit t(_A &&x) : u(std::move(x)) {}

#define EVALUATE_UNION_CLASS_BOILERPLATE(t) \
  CLASS_BOILERPLATE(t) \
  UNION_CONSTRUCTORS(t) \
  bool operator==(const t &) const;

// Forward definition of Expr<> so that it can be indirectly used in its own
// definition
template <typename A> class Expr;

class FoldingContext {
public:
  FoldingContext(
      const common::IntrinsicTypeDefaultKinds &d, const IntrinsicProcTable &t)
      : defaults_{d}, intrinsics_{t} {}
  FoldingContext(const parser::ContextualMessages &m,
      const common::IntrinsicTypeDefaultKinds &d, const IntrinsicProcTable &t,
      Rounding round = defaultRounding, bool flush = false)
      : messages_{m}, defaults_{d}, intrinsics_{t}, rounding_{round},
        flushSubnormalsToZero_{flush} {}
  FoldingContext(const FoldingContext &that)
      : messages_{that.messages_}, defaults_{that.defaults_},
        intrinsics_{that.intrinsics_}, rounding_{that.rounding_},
        flushSubnormalsToZero_{that.flushSubnormalsToZero_},
        pdtInstance_{that.pdtInstance_}, impliedDos_{that.impliedDos_} {}
  FoldingContext(
      const FoldingContext &that, const parser::ContextualMessages &m)
      : messages_{m}, defaults_{that.defaults_},
        intrinsics_{that.intrinsics_}, rounding_{that.rounding_},
        flushSubnormalsToZero_{that.flushSubnormalsToZero_},
        pdtInstance_{that.pdtInstance_}, impliedDos_{that.impliedDos_} {}

  parser::ContextualMessages &messages() { return messages_; }
  const parser::ContextualMessages &messages() const { return messages_; }
  const common::IntrinsicTypeDefaultKinds &defaults() const {
    return defaults_;
  }
  Rounding rounding() const { return rounding_; }
  bool flushSubnormalsToZero() const { return flushSubnormalsToZero_; }
  bool bigEndian() const { return bigEndian_; }
  std::size_t maxAlignment() const { return maxAlignment_; }
  const semantics::DerivedTypeSpec *pdtInstance() const { return pdtInstance_; }
  const IntrinsicProcTable &intrinsics() const { return intrinsics_; }

  ConstantSubscript &StartImpliedDo(parser::CharBlock, ConstantSubscript = 1);
  std::optional<ConstantSubscript> GetImpliedDo(parser::CharBlock) const;
  void EndImpliedDo(parser::CharBlock);

  std::map<parser::CharBlock, ConstantSubscript> &impliedDos() {
    return impliedDos_;
  }

  common::Restorer<const semantics::DerivedTypeSpec *> WithPDTInstance(
      const semantics::DerivedTypeSpec &spec) {
    return common::ScopedSet(pdtInstance_, &spec);
  }

private:
  parser::ContextualMessages messages_;
  const common::IntrinsicTypeDefaultKinds &defaults_;
  const IntrinsicProcTable &intrinsics_;
  Rounding rounding_{defaultRounding};
  bool flushSubnormalsToZero_{false};
  static constexpr bool bigEndian_{false}; // TODO: configure for target
  static constexpr std::size_t maxAlignment_{8}; // TODO: configure for target
  const semantics::DerivedTypeSpec *pdtInstance_{nullptr};
  std::map<parser::CharBlock, ConstantSubscript> impliedDos_;
};

void RealFlagWarnings(FoldingContext &, const RealFlags &, const char *op);
} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_COMMON_H_
