//===-- include/flang/Evaluate/constant.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_CONSTANT_H_
#define FORTRAN_EVALUATE_CONSTANT_H_

#include "formatting.h"
#include "type.h"
#include "flang/Common/reference.h"
#include "flang/Common/type-kinds.h"
#include "flang/Support/default-kinds.h"
#include <map>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace Fortran::semantics {
class Symbol;
}

namespace Fortran::evaluate {

using semantics::Symbol;
using SymbolRef = common::Reference<const Symbol>;

// Wraps a constant value in a class templated by its resolved type.
// This Constant<> template class should be instantiated only for
// concrete intrinsic types and SomeDerived.  There is no instance
// Constant<SomeType> since there is no way to constrain each
// element of its array to hold the same type.  To represent a generic
// constant, use a generic expression like Expr<SomeInteger> or
// Expr<SomeType>) to wrap the appropriate instantiation of Constant<>.

template <typename> class Constant;

// When describing shapes of constants or specifying 1-based subscript
// values as indices into constants, use a vector of integers.
using ConstantSubscripts = std::vector<ConstantSubscript>;
inline int GetRank(const ConstantSubscripts &s) {
  return static_cast<int>(s.size());
}

// Returns the number of elements of shape, if no overflow occurs.
std::optional<uint64_t> TotalElementCount(const ConstantSubscripts &shape);

// Validate dimension re-ordering like ORDER in RESHAPE.
// On success, return a vector that can be used as dimOrder in
// ConstantBounds::IncrementSubscripts().
std::optional<std::vector<int>> ValidateDimensionOrder(
    int rank, const std::vector<int> &order);

bool HasNegativeExtent(const ConstantSubscripts &);

class ConstantBounds {
public:
  ConstantBounds() = default;
  explicit ConstantBounds(const ConstantSubscripts &shape);
  explicit ConstantBounds(ConstantSubscripts &&shape);
  ~ConstantBounds();
  const ConstantSubscripts &shape() const { return shape_; }
  int Rank() const { return GetRank(shape_); }
  static constexpr int Corank() { return 0; }
  Constant<SubscriptInteger> SHAPE() const;

  // It is possible in this representation for a constant array to have
  // lower bounds other than 1, which is of course not expressible in
  // Fortran.  This case arises only from definitions of named constant
  // arrays with such bounds, as in:
  //   REAL, PARAMETER :: NAMED(0:1) = [1.,2.]
  // Bundling the lower bounds of the named constant with its
  // constant value allows folding of subscripted array element
  // references, LBOUND, and UBOUND without having to thread the named
  // constant or its bounds throughout folding.
  const ConstantSubscripts &lbounds() const { return lbounds_; }
  ConstantSubscripts ComputeUbounds(std::optional<int> dim) const;
  void set_lbounds(ConstantSubscripts &&);
  void SetLowerBoundsToOne();
  bool HasNonDefaultLowerBound() const;

  // If no optional dimension order argument is passed, increments a vector of
  // subscripts in Fortran array order (first dimension varying most quickly).
  // Otherwise, increments the vector of subscripts according to the given
  // dimension order (dimension dimOrder[0] varying most quickly; dimension
  // indexing is zero based here). Returns false when last element was visited.
  bool IncrementSubscripts(
      ConstantSubscripts &, const std::vector<int> *dimOrder = nullptr) const;

protected:
  ConstantSubscript SubscriptsToOffset(const ConstantSubscripts &) const;

private:
  ConstantSubscripts shape_;
  ConstantSubscripts lbounds_;
};

// Constant<> is specialized for Character kinds and SomeDerived.
// The non-Character intrinsic types, and SomeDerived, share enough
// common behavior that they use this common base class.
template <typename RESULT, typename ELEMENT = Scalar<RESULT>>
class ConstantBase : public ConstantBounds {
  static_assert(RESULT::category != TypeCategory::Character);

public:
  using Result = RESULT;
  using Element = ELEMENT;

  constexpr KindsEnum kind() const { return kind_; }

  // Constructor for creating ConstantBase from an actual value (i.e.
  // literals, etc.)
  template <typename A>
  ConstantBase(KindsEnum kind, const A &x, Result res)
      : kind_{kind}, result_{res}, values_{A{kind, x}} {
    CHECK_KIND(kind, RESULT);
  }
  ConstantBase(KindsEnum kind, ELEMENT &&x)
      : kind_{kind}, result_{Result{kind}}, values_{std::move(x)} {
    CHECK_KIND(kind, RESULT);
  }

  template <TypeCategory CAT>
  ConstantBase(KindsEnum kind, const SomeKind<CAT> &x)
      : kind_{kind}, result_{Result{kind}}, values_{x} {
    CHECK_KIND(kind, RESULT);
  }
  template <TypeCategory CAT>
  ConstantBase(KindsEnum kind, SomeKind<CAT> &&x)
      : kind_{kind}, result_{Result{kind}}, values_{std::move(x)} {
    CHECK_KIND(kind, RESULT);
  }

  template <typename A>
  ConstantBase(KindsEnum kind, const A &x)
      : kind_{kind}, result_{Result{kind}}, values_{A{kind, x}} {
    CHECK_KIND(kind, RESULT);
  }
  ConstantBase(KindsEnum kind, ELEMENT &&x, Result res)
      : kind_{kind}, result_{res}, values_{std::move(x)} {
    CHECK_KIND(kind, RESULT);
  }

  ConstantBase(
      KindsEnum kind, std::vector<Element> &&x, ConstantSubscripts &&sh)
      : ConstantBase{kind, std::move(x), std::move(sh), Result{kind}} {}
  ConstantBase(
      KindsEnum kind, std::vector<Element> &&, ConstantSubscripts &&, Result);
  template <typename A, typename B, typename C>
  ConstantBase(
      KindsEnum kind, const std::map<A, B, C> &x, Result res = Result{})
      : kind_{kind}, result_{res}, values_{x} {}

  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(ConstantBase)
  ~ConstantBase();

  bool operator==(const ConstantBase &) const;
  bool empty() const { return values_.empty(); }
  std::size_t size() const { return values_.size(); }
  const std::vector<Element> &values() const { return values_; }
  Result &result() { return result_; }
  const Result &result() const { return result_; }

  constexpr DynamicType GetType() const { return result_.GetType(); }
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;
  std::string AsFortran() const;

protected:
  std::vector<Element> Reshape(const ConstantSubscripts &) const;
  std::size_t CopyFrom(const ConstantBase &source, std::size_t count,
      ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder);

  KindsEnum kind_;
  Result result_; // usually empty except for Real & Complex
  std::vector<Element> values_;
};

template <typename T> class Constant : public ConstantBase<T> {
public:
  using Result = T;
  using Base = ConstantBase<T>;
  using Element = Scalar<T>;

  using Base::Base;
  CLASS_BOILERPLATE(Constant)

  std::optional<Scalar<T>> GetScalarValue() const {
    if (ConstantBounds::Rank() == 0) {
      return Base::values_.at(0);
    } else {
      return std::nullopt;
    }
  }

  // Apply subscripts.  Excess subscripts are ignored, including the
  // case of a scalar.
  Element At(const ConstantSubscripts &) const;

  Constant Reshape(ConstantSubscripts &&) const;
  std::size_t CopyFrom(const Constant &source, std::size_t count,
      ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder);
};

template <>
class Constant<Type<TypeCategory::Character>> : public ConstantBounds {
public:
  using Result = Type<TypeCategory::Character>;
  using Element = Scalar<Result>;

  constexpr KindsEnum kind() const { return kind_; }

  CLASS_BOILERPLATE(Constant)
  explicit Constant(KindsEnum kind, const Scalar<Result> &);
  explicit Constant(KindsEnum kind, Scalar<Result> &&);
  Constant(KindsEnum kind, ConstantSubscript length, std::vector<Element> &&,
      ConstantSubscripts &&);
  ~Constant();

  bool operator==(const Constant &that) const {
    return kind() == that.kind() && LEN() == that.LEN() &&
        shape() == that.shape() && values_ == that.values_;
  }
  bool empty() const;
  std::size_t size() const;

  const Scalar<Result> &values() const { return values_; }
  ConstantSubscript LEN() const { return length_; }
  bool wasHollerith() const { return wasHollerith_; }
  void set_wasHollerith(bool yes = true) { wasHollerith_ = yes; }

  std::optional<Scalar<Result>> GetScalarValue() const {
    if (Rank() == 0) {
      return values_;
    } else {
      return std::nullopt;
    }
  }

  // Apply subscripts, if any.
  Scalar<Result> At(const ConstantSubscripts &) const;

  // Extract substring(s); returns nullopt for errors.
  std::optional<Constant> Substring(ConstantSubscript, ConstantSubscript) const;

  Constant Reshape(ConstantSubscripts &&) const;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;
  std::string AsFortran() const;
  DynamicType GetType() const { return {kind_, length_}; }
  std::size_t CopyFrom(const Constant &source, std::size_t count,
      ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder);

private:
  KindsEnum kind_;
  Scalar<Result> values_; // one contiguous string
  ConstantSubscript length_;
  bool wasHollerith_{false};
};

class StructureConstructor;
struct ComponentCompare {
  bool operator()(SymbolRef x, SymbolRef y) const;
};
using StructureConstructorValues = std::map<SymbolRef,
    common::CopyableIndirection<Expr<SomeType>>, ComponentCompare>;

template <>
class Constant<SomeDerived>
    : public ConstantBase<SomeDerived, StructureConstructorValues> {
public:
  using Result = SomeDerived;
  using Element = StructureConstructorValues;
  using Base = ConstantBase<SomeDerived, StructureConstructorValues>;

  Constant(const StructureConstructor &);
  Constant(StructureConstructor &&);
  Constant(KindsEnum kind, const StructureConstructor &v) : Constant(v) {
    CHECK(kind == NoKind);
  }
  Constant(KindsEnum kind, StructureConstructor &&v) : Constant(std::move(v)) {
    CHECK(kind == NoKind);
  }
  Constant(const semantics::DerivedTypeSpec &,
      std::vector<StructureConstructorValues> &&, ConstantSubscripts &&);
  Constant(const semantics::DerivedTypeSpec &,
      std::vector<StructureConstructor> &&, ConstantSubscripts &&);
  CLASS_BOILERPLATE(Constant)

  std::optional<StructureConstructor> GetScalarValue() const;
  StructureConstructor At(const ConstantSubscripts &) const;

  bool operator==(const Constant &) const;
  Constant Reshape(ConstantSubscripts &&) const;
  std::size_t CopyFrom(const Constant &source, std::size_t count,
      ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder);
};

inline Constant<SubscriptInteger> MakeSubscriptIntConstant(int64_t v) {
  return Constant<SubscriptInteger>{
      SubscriptIntegerKind, Scalar<SubscriptInteger>{SubscriptIntegerKind, v}};
}

inline Constant<CInteger> MakeCIntegerConstant(int32_t v) {
  return Constant<CInteger>{CIntegerKind, Scalar<CInteger>{CIntegerKind, v}};
}

inline Constant<LogicalResult> MakeLogicalResultConstant(bool v) {
  return Constant<LogicalResult>{
      LogicalResultKind, Scalar<LogicalResult>{LogicalResultKind, v}};
}

template <typename T, typename CharT,
    typename =
        std::enable_if_t<std::is_same_v<T, Type<TypeCategory::Character>>>>
inline Constant<T> MakeConstant(
    KindsEnum kind, const std::basic_string<CharT> &v) {
  return Constant<T>{kind, value::CharacterValue{kind, v}};
}

template <typename T, typename CharT,
    typename =
        std::enable_if_t<std::is_same_v<T, Type<TypeCategory::Character>>>>
inline Constant<T> MakeConstant(KindsEnum kind, std::basic_string<CharT> &&v) {
  return Constant<T>{kind, value::CharacterValue{kind, std::move(v)}};
}

template <typename T,
    typename = std::enable_if_t<std::is_same_v<T, Type<TypeCategory::Integer>>>>
inline Constant<T> MakeConstant(KindsEnum kind, int64_t v) {
  return Constant<T>{kind, value::IntegerValue{kind, v}};
}

FOR_EACH_LENGTHLESS_INTRINSIC_KIND(extern template class ConstantBase, )
extern template class ConstantBase<SomeDerived, StructureConstructorValues>;
FOR_EACH_INTRINSIC_KIND(extern template class Constant, )

#define INSTANTIATE_CONSTANT_TEMPLATES \
  FOR_EACH_LENGTHLESS_INTRINSIC_KIND(template class ConstantBase, ) \
  template class ConstantBase<SomeDerived, StructureConstructorValues>; \
  FOR_EACH_INTRINSIC_KIND(template class Constant, )
} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_CONSTANT_H_
