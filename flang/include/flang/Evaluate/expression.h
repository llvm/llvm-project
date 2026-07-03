//===-- include/flang/Evaluate/expression.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_EXPRESSION_H_
#define FORTRAN_EVALUATE_EXPRESSION_H_

// Represent Fortran expressions in a type-safe manner.
// Expressions are the sole owners of their constituents; i.e., there is no
// context-independent hash table or sharing of common subexpressions, and
// thus these are trees, not DAGs.  Both deep copy and move semantics are
// supported for expression construction.  Expressions may be compared
// for equality.

#include "common.h"
#include "constant.h"
#include "formatting.h"
#include "type.h"
#include "variable.h"
#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Common/template.h"
#include "flang/Parser/char-block.h"
#include "flang/Support/Fortran.h"
#include <algorithm>
#include <list>
#include <tuple>
#include <type_traits>
#include <variant>

namespace llvm {
class raw_ostream;
}

namespace Fortran::evaluate {

using common::LogicalOperator;
using common::RelationalOperator;

// Expressions are represented by specializations of the class template Expr.
// Each of these specializations wraps a single data member "u" that
// is a std::variant<> discriminated union over all of the representational
// types for the constants, variables, operations, and other entities that
// can be valid expressions in that context:
// - Expr<Type<CATEGORY, KIND>> represents an expression whose result is of a
//   specific intrinsic type category and kind, e.g. Type<TypeCategory::Real, 4>
// - Expr<SomeDerived> wraps data and procedure references that result in an
//   instance of a derived type (or CLASS(*) unlimited polymorphic)
// - Expr<SomeKind<CATEGORY>> is a union of Expr<Type<CATEGORY, K>> for each
//   kind type parameter value K in that intrinsic type category.  It represents
//   an expression with known category and any kind.
// - Expr<SomeType> is a union of Expr<SomeKind<CATEGORY>> over the five
//   intrinsic type categories of Fortran.  It represents any valid expression.
//
// Everything that can appear in, or as, a valid Fortran expression must be
// represented with an instance of some class containing a Result typedef that
// maps to some instantiation of Type<CATEGORY, KIND>, SomeKind<CATEGORY>,
// or SomeType.  (Exception: BOZ literal constants in generic Expr<SomeType>.)
template <typename A> using ResultType = typename std::decay_t<A>::Result;

// Common Expr<> behaviors: every Expr<T> derives from ExpressionBase<T>.
template <typename RESULT> class ExpressionBase {
public:
  using Result = RESULT;

private:
  using Derived = Expr<Result>;
#if defined(__APPLE__) && defined(__GNUC__)
  Derived &derived();
  const Derived &derived() const;
#else
  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<const Derived *>(this); }
#endif

public:
  template <typename A> Derived &operator=(const A &x) {
    Derived &d{derived()};
    d.u = x;
    return d;
  }

  template <typename A> common::IfNoLvalue<Derived &, A> operator=(A &&x) {
    Derived &d{derived()};
    d.u = std::move(x);
    return d;
  }

  std::optional<DynamicType> GetType() const;
  int Rank() const;
  int Corank() const;
  std::string AsFortran() const;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;
  static Derived Rewrite(FoldingContext &, Derived &&);
};

// Operations always have specific Fortran result types (i.e., with known
// intrinsic type category and kind parameter value).  The classes that
// represent the operations all inherit from this Operation<> base class
// template.  Note that Operation has as its first type parameter (DERIVED) a
// "curiously reoccurring template pattern (CRTP)" reference to the specific
// operation class being derived from Operation; e.g., Add is defined with
// struct Add : public Operation<Add, ...>.  Uses of instances of Operation<>,
// including its own member functions, can access each specific class derived
// from it via its derived() member function with compile-time type safety.
template <typename DERIVED, typename RESULT, typename... OPERANDS>
class Operation {
  // The extra final member is a dummy that allows a safe unused reference
  // to element 1 to arise indirectly in the definition of "right()" below
  // when the operation has but a single operand.
  using OperandTypes = std::tuple<OPERANDS..., std::monostate>;

public:
  using Derived = DERIVED;
  using Result = RESULT;
  static constexpr std::size_t operands{sizeof...(OPERANDS)};
  // Allow specific intrinsic types and Parentheses<SomeDerived>
  static_assert(IsSpecificIntrinsicType<Result> ||
      (operands == 1 && std::is_same_v<Result, SomeDerived>));
  template <int J> using Operand = std::tuple_element_t<J, OperandTypes>;

  // Unary operations wrap a single Expr with a CopyableIndirection.
  // Binary operations wrap a tuple of CopyableIndirections to Exprs.
private:
  using Container = std::conditional_t<operands == 1,
      common::CopyableIndirection<Expr<Operand<0>>>,
      std::tuple<common::CopyableIndirection<Expr<OPERANDS>>...>>;

public:
  CLASS_BOILERPLATE(Operation)
  explicit Operation(const Expr<OPERANDS> &...x) : operand_{x...} {}
  explicit Operation(Expr<OPERANDS> &&...x) : operand_{std::move(x)...} {}

  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<const Derived *>(this); }

  // References to operand expressions from member functions of derived
  // classes for specific operators can be made by index, e.g. operand<0>(),
  // which must be spelled like "this->template operand<0>()" when
  // inherited in a derived class template.  There are convenience aliases
  // left() and right() that are not templates.
  template <int J> Expr<Operand<J>> &operand() {
    if constexpr (operands == 1) {
      static_assert(J == 0);
      return operand_.value();
    } else {
      return std::get<J>(operand_).value();
    }
  }
  template <int J> const Expr<Operand<J>> &operand() const {
    if constexpr (operands == 1) {
      static_assert(J == 0);
      return operand_.value();
    } else {
      return std::get<J>(operand_).value();
    }
  }

  Expr<Operand<0>> &left() { return operand<0>(); }
  const Expr<Operand<0>> &left() const { return operand<0>(); }

  std::conditional_t<(operands > 1), Expr<Operand<1>> &, void> right() {
    if constexpr (operands > 1) {
      return operand<1>();
    }
  }
  std::conditional_t<(operands > 1), const Expr<Operand<1>> &, void>
  right() const {
    if constexpr (operands > 1) {
      return operand<1>();
    }
  }

  std::conditional_t<Result::category != TypeCategory::Derived,
      std::optional<DynamicType>, void>
  GetType() const {
    if constexpr (Result::category != TypeCategory::Derived) {
      // The result kind of most operations equals their first operand's kind;
      // derive it at runtime rather than from the compile-time Result::kind.
      // Operations whose result kind differs (Convert, Relational) override
      // this.  An operand may be typeless on an error path, so propagate that
      // rather than aborting.
      if (auto operandType{this->left().GetType()}) {
        return DynamicType{Result::category, operandType->kind()};
      }
      return std::nullopt;
    }
  }
  int Rank() const {
    int rank{left().Rank()};
    if constexpr (operands > 1) {
      return std::max(rank, right().Rank());
    } else {
      return rank;
    }
  }
  static constexpr int Corank() { return 0; }

  bool operator==(const Operation &that) const {
    return operand_ == that.operand_;
  }

  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;

private:
  Container operand_;
};

// Unary operations

// Conversions to specific types from expressions of known category and
// dynamic kind.
template <typename TO, TypeCategory FROMCAT = TO::category>
struct Convert : public Operation<Convert<TO, FROMCAT>, TO, SomeKind<FROMCAT>> {
  // Fortran doesn't have conversions between kinds of CHARACTER apart from
  // assignments, and in those the data must be convertible to/from 7-bit ASCII.
  static_assert(
      ((TO::category == TypeCategory::Integer ||
           TO::category == TypeCategory::Real ||
           TO::category == TypeCategory::Unsigned) &&
          (FROMCAT == TypeCategory::Integer || FROMCAT == TypeCategory::Real ||
              FROMCAT == TypeCategory::Unsigned)) ||
      TO::category == FROMCAT);
  using Result = TO;
  using Operand = SomeKind<FROMCAT>;
  using Base = Operation<Convert, Result, Operand>;
  using Base::Base;
  // A conversion's result kind is the target kind, not the operand kind, so it
  // is stored at runtime (defaulting to the compile-time target kind while it
  // remains a template parameter) rather than derived from the operand.
  std::optional<DynamicType> GetType() const {
    return DynamicType{Result::category, resultKind_};
  }
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;
  // TODO(kind-flip): the conversion target kind must be supplied at
  // construction (it was formerly the compile-time Result::kind, now removed).
  int resultKind_{0};
};

template <typename A>
struct Parentheses : public Operation<Parentheses<A>, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Parentheses, A, A>;
  using Base::Base;
};

template <>
struct Parentheses<SomeDerived>
    : public Operation<Parentheses<SomeDerived>, SomeDerived, SomeDerived> {
public:
  using Result = SomeDerived;
  using Operand = SomeDerived;
  using Base = Operation<Parentheses, SomeDerived, SomeDerived>;
  using Base::Base;
  DynamicType GetType() const;
};

template <typename A> struct Negate : public Operation<Negate<A>, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Negate, A, A>;
  using Base::Base;
};

struct ComplexComponent
    : public Operation<ComplexComponent, Type<TypeCategory::Real>,
          Type<TypeCategory::Complex>> {
  using Result = Type<TypeCategory::Real>;
  using Operand = Type<TypeCategory::Complex>;
  using Base = Operation<ComplexComponent, Result, Operand>;
  CLASS_BOILERPLATE(ComplexComponent)
  ComplexComponent(bool isImaginary, const Expr<Operand> &x)
      : Base{x}, isImaginaryPart{isImaginary} {}
  ComplexComponent(bool isImaginary, Expr<Operand> &&x)
      : Base{std::move(x)}, isImaginaryPart{isImaginary} {}

  bool isImaginaryPart{true};
};

struct Not : public Operation<Not, Type<TypeCategory::Logical>,
                 Type<TypeCategory::Logical>> {
  using Result = Type<TypeCategory::Logical>;
  using Operand = Result;
  using Base = Operation<Not, Result, Operand>;
  using Base::Base;
};

// CharacterValue lengths are determined by context in Fortran and do not
// have explicit syntax for changing them.  Expressions represent
// changes of length (e.g., for assignments and structure constructors)
// with this operation.
struct SetLength : public Operation<SetLength, Type<TypeCategory::Character>,
                       Type<TypeCategory::Character>, SubscriptInteger> {
  using Result = Type<TypeCategory::Character>;
  using CharacterOperand = Result;
  using LengthOperand = SubscriptInteger;
  using Base = Operation<SetLength, Result, CharacterOperand, LengthOperand>;
  using Base::Base;
};

// Binary operations

template <typename A> struct Add : public Operation<Add<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Add, A, A, A>;
  using Base::Base;
};

template <typename A> struct Subtract : public Operation<Subtract<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Subtract, A, A, A>;
  using Base::Base;
};

template <typename A> struct Multiply : public Operation<Multiply<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Multiply, A, A, A>;
  using Base::Base;
};

template <typename A> struct Divide : public Operation<Divide<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Divide, A, A, A>;
  using Base::Base;
};

template <typename A> struct Power : public Operation<Power<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Power, A, A, A>;
  using Base::Base;
};

template <typename A>
struct RealToIntPower : public Operation<RealToIntPower<A>, A, A, SomeInteger> {
  using Base = Operation<RealToIntPower, A, A, SomeInteger>;
  using Result = A;
  using BaseOperand = A;
  using ExponentOperand = SomeInteger;
  using Base::Base;
};

template <typename A> struct Extremum : public Operation<Extremum<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Extremum, A, A, A>;
  CLASS_BOILERPLATE(Extremum)
  Extremum(Ordering ord, const Expr<Operand> &x, const Expr<Operand> &y)
      : Base{x, y}, ordering{ord} {}
  Extremum(Ordering ord, Expr<Operand> &&x, Expr<Operand> &&y)
      : Base{std::move(x), std::move(y)}, ordering{ord} {}
  bool operator==(const Extremum &) const;
  Ordering ordering{Ordering::Greater};
};

struct ComplexConstructor
    : public Operation<ComplexConstructor, Type<TypeCategory::Complex>,
          Type<TypeCategory::Real>, Type<TypeCategory::Real>> {
  using Result = Type<TypeCategory::Complex>;
  using Operand = Type<TypeCategory::Real>;
  using Base = Operation<ComplexConstructor, Result, Operand, Operand>;
  using Base::Base;
};

struct Concat
    : public Operation<Concat, Type<TypeCategory::Character>,
          Type<TypeCategory::Character>, Type<TypeCategory::Character>> {
  using Result = Type<TypeCategory::Character>;
  using Operand = Result;
  using Base = Operation<Concat, Result, Operand, Operand>;
  using Base::Base;
};

struct LogicalOperation
    : public Operation<LogicalOperation, Type<TypeCategory::Logical>,
          Type<TypeCategory::Logical>, Type<TypeCategory::Logical>> {
  using Result = Type<TypeCategory::Logical>;
  using Operand = Result;
  using Base = Operation<LogicalOperation, Result, Operand, Operand>;
  CLASS_BOILERPLATE(LogicalOperation)
  LogicalOperation(
      LogicalOperator opr, const Expr<Operand> &x, const Expr<Operand> &y)
      : Base{x, y}, logicalOperator{opr} {}
  LogicalOperation(LogicalOperator opr, Expr<Operand> &&x, Expr<Operand> &&y)
      : Base{std::move(x), std::move(y)}, logicalOperator{opr} {}
  bool operator==(const LogicalOperation &) const;
  LogicalOperator logicalOperator;
};

// Fortran 2023 conditional expression: (cond ? val : cond ? val : ... : else)
// All branches have the same type and rank (verified during semantic analysis).
template <typename T> class ConditionalExpr {
public:
  using Result = T;
  CLASS_BOILERPLATE(ConditionalExpr)
  ConditionalExpr(Expr<LogicalResult> &&cond, Expr<Result> &&thenVal,
      Expr<Result> &&elseVal)
      : condition_{std::move(cond)}, thenValue_{std::move(thenVal)},
        elseValue_{std::move(elseVal)} {}
  bool operator==(const ConditionalExpr &) const;
  Expr<LogicalResult> &condition() { return condition_.value(); }
  const Expr<LogicalResult> &condition() const { return condition_.value(); }
  Expr<Result> &thenValue() { return thenValue_.value(); }
  const Expr<Result> &thenValue() const { return thenValue_.value(); }
  Expr<Result> &elseValue() { return elseValue_.value(); }
  const Expr<Result> &elseValue() const { return elseValue_.value(); }
  int Rank() const { return thenValue().Rank(); }
  std::optional<DynamicType> GetType() const {
    const auto thenType{thenValue().GetType()};
    if constexpr (T::category == TypeCategory::Derived) {
      // F2023 10.1.4(7) A conditional-expr is polymorphic if any branch is
      if (thenType && !thenType->IsPolymorphic()) {
        if (const auto elseType{elseValue().GetType()}) {
          if (elseType->IsPolymorphic()) {
            return elseType;
          }
        }
      }
    }
    return thenType;
  }
  static constexpr int Corank() { return 0; }
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;

private:
  common::CopyableIndirection<Expr<LogicalResult>> condition_;
  common::CopyableIndirection<Expr<Result>> thenValue_;
  common::CopyableIndirection<Expr<Result>> elseValue_;
};

// Array constructors
template <typename RESULT> class ArrayConstructorValues;

struct ImpliedDoIndex {
  using Result = SubscriptInteger;
  bool operator==(const ImpliedDoIndex &) const;
  static constexpr DynamicType GetType() {
    return DynamicType{TypeCategory::Integer, subscriptIntegerKind};
  }
  static constexpr int Rank() { return 0; }
  static constexpr int Corank() { return 0; }
  parser::CharBlock name; // nested implied DOs must use distinct names
};

template <typename RESULT> class ImpliedDo {
public:
  using Result = RESULT;
  using Index = ResultType<ImpliedDoIndex>;
  ImpliedDo(parser::CharBlock name, Expr<Index> &&lower, Expr<Index> &&upper,
      Expr<Index> &&stride, ArrayConstructorValues<Result> &&values)
      : name_{name}, lower_{std::move(lower)}, upper_{std::move(upper)},
        stride_{std::move(stride)}, values_{std::move(values)} {}
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(ImpliedDo)
  bool operator==(const ImpliedDo &) const;
  parser::CharBlock name() const { return name_; }
  Expr<Index> &lower() { return lower_.value(); }
  const Expr<Index> &lower() const { return lower_.value(); }
  Expr<Index> &upper() { return upper_.value(); }
  const Expr<Index> &upper() const { return upper_.value(); }
  Expr<Index> &stride() { return stride_.value(); }
  const Expr<Index> &stride() const { return stride_.value(); }
  ArrayConstructorValues<Result> &values() { return values_.value(); }
  const ArrayConstructorValues<Result> &values() const {
    return values_.value();
  }

private:
  parser::CharBlock name_;
  common::CopyableIndirection<Expr<Index>> lower_, upper_, stride_;
  common::CopyableIndirection<ArrayConstructorValues<Result>> values_;
};

template <typename RESULT> struct ArrayConstructorValue {
  using Result = RESULT;
  EVALUATE_UNION_CLASS_BOILERPLATE(ArrayConstructorValue)
  std::variant<Expr<Result>, ImpliedDo<Result>> u;
};

template <typename RESULT> class ArrayConstructorValues {
public:
  using Result = RESULT;
  using Values = std::vector<ArrayConstructorValue<Result>>;
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(ArrayConstructorValues)
  ArrayConstructorValues() {}

  bool operator==(const ArrayConstructorValues &) const;
  static constexpr int Rank() { return 1; }
  static constexpr int Corank() { return 0; }
  template <typename A> common::NoLvalue<A> Push(A &&x) {
    values_.emplace_back(std::move(x));
  }

  typename Values::iterator begin() { return values_.begin(); }
  typename Values::const_iterator begin() const { return values_.begin(); }
  typename Values::iterator end() { return values_.end(); }
  typename Values::const_iterator end() const { return values_.end(); }

protected:
  Values values_;
};

// Determine the runtime result kind of an array constructor by inspecting
// its element values (recursing into implied DOs).  Returns 0 if none of the
// elements have a known type.
template <typename RESULT>
int GleanArrayConstructorResultKind(
    const ArrayConstructorValues<RESULT> &values) {
  for (const auto &v : values) {
    int kind{common::visit(
        [](const auto &x) -> int {
          if constexpr (std::is_same_v<std::decay_t<decltype(x)>,
                            Expr<RESULT>>) {
            if (auto type{x.GetType()}) {
              return type->kind();
            }
            return 0;
          } else {
            return GleanArrayConstructorResultKind<RESULT>(x.values());
          }
        },
        v.u)};
    if (kind != 0) {
      return kind;
    }
  }
  return 0;
}

// Note that there are specializations of ArrayConstructor for character
// and derived types, since they must carry additional type information,
// but that an empty ArrayConstructor can be constructed for any type
// given an expression from which such type information may be gleaned.
template <typename RESULT>
class ArrayConstructor : public ArrayConstructorValues<RESULT> {
public:
  using Result = RESULT;
  using Base = ArrayConstructorValues<Result>;
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(ArrayConstructor)
  explicit ArrayConstructor(Base &&values)
      : Base{std::move(values)},
        resultKind_{GleanArrayConstructorResultKind<Result>(*this)} {}
  template <typename T> explicit ArrayConstructor(const Expr<T> &x) {
    if (auto type{x.GetType()}) {
      resultKind_ = type->kind();
    }
  }
  static constexpr Result result() { return Result{}; }
  // The element kind is stored at runtime (defaulting to the compile-time kind
  // while it remains a template parameter) rather than taken from Result.
  DynamicType GetType() const {
    return DynamicType{Result::category, resultKind_};
  }
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;
  // The element kind is supplied at construction (formerly the compile-time
  // Result::kind, now removed).
  int resultKind_{0};
};

template <>
class ArrayConstructor<Type<TypeCategory::Character>>
    : public ArrayConstructorValues<Type<TypeCategory::Character>> {
public:
  using Result = Type<TypeCategory::Character>;
  using Base = ArrayConstructorValues<Result>;
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(ArrayConstructor)
  explicit ArrayConstructor(Base &&values) : Base{std::move(values)} {}
  template <typename T> explicit ArrayConstructor(const Expr<T> &x) {
    if (auto type{x.GetType()}) {
      resultKind_ = type->kind();
    }
  }
  ArrayConstructor &set_LEN(Expr<SubscriptInteger> &&);
  bool operator==(const ArrayConstructor &) const;
  static constexpr Result result() { return Result{}; }
  DynamicType GetType() const {
    return DynamicType{Result::category, resultKind_};
  }
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;
  const Expr<SubscriptInteger> *LEN() const {
    return length_ ? &length_->value() : nullptr;
  }
  // The character kind is supplied at construction (formerly the compile-time
  // Result::kind, now removed).
  int resultKind_{0};

private:
  std::optional<common::CopyableIndirection<Expr<SubscriptInteger>>> length_;
};

template <>
class ArrayConstructor<SomeDerived>
    : public ArrayConstructorValues<SomeDerived> {
public:
  using Result = SomeDerived;
  using Base = ArrayConstructorValues<Result>;
  CLASS_BOILERPLATE(ArrayConstructor)

  ArrayConstructor(const semantics::DerivedTypeSpec &spec, Base &&v)
      : Base{std::move(v)}, result_{spec} {}
  template <typename A>
  explicit ArrayConstructor(const A &prototype)
      : result_{prototype.GetType().value().GetDerivedTypeSpec()} {}

  bool operator==(const ArrayConstructor &) const;
  constexpr Result result() const { return result_; }
  constexpr DynamicType GetType() const { return result_.GetType(); }
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;

private:
  Result result_;
};

// Expression representations for each type category.

template <>
class Expr<Type<TypeCategory::Integer>>
    : public ExpressionBase<Type<TypeCategory::Integer>> {
public:
  using Result = Type<TypeCategory::Integer>;

  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  // Compiler-internal integer literals (subscripts, bounds, shapes) default to
  // the subscript integer kind, which is the widest commonly-used kind and so
  // is range-safe; the kind is carried at runtime in the resulting Constant.
  explicit Expr(int n) : Expr(static_cast<long long>(n)) {}
  explicit Expr(unsigned n) : Expr(static_cast<unsigned long long>(n)) {}
  explicit Expr(long n) : Expr(static_cast<long long>(n)) {}
  explicit Expr(unsigned long n) : Expr(static_cast<unsigned long long>(n)) {}
  explicit Expr(long long n)
      : u{Constant<Result>{typename Result::Scalar{n, subscriptIntegerKind},
            Result{subscriptIntegerKind}}} {}
  explicit Expr(unsigned long long n)
      : u{Constant<Result>{typename Result::Scalar{n, subscriptIntegerKind},
            Result{subscriptIntegerKind}}} {}

private:
  using Conversions = std::tuple<Convert<Result, TypeCategory::Integer>,
      Convert<Result, TypeCategory::Real>,
      Convert<Result, TypeCategory::Unsigned>>;
  using Operations = std::tuple<Parentheses<Result>, Negate<Result>,
      Add<Result>, Subtract<Result>, Multiply<Result>, Divide<Result>,
      Power<Result>, Extremum<Result>, ConditionalExpr<Result>>;
  using Indices = std::tuple<ImpliedDoIndex>;
  using TypeParamInquiries = std::tuple<TypeParamInquiry>;
  using DescriptorInquiries = std::tuple<DescriptorInquiry>;
  using Others = std::tuple<Constant<Result>, ArrayConstructor<Result>,
      Designator<Result>, FunctionRef<Result>>;

public:
  common::TupleToVariant<common::CombineTuples<Operations, Conversions, Indices,
      TypeParamInquiries, DescriptorInquiries, Others>>
      u;
};

template <>
class Expr<Type<TypeCategory::Unsigned>>
    : public ExpressionBase<Type<TypeCategory::Unsigned>> {
public:
  using Result = Type<TypeCategory::Unsigned>;

  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  explicit Expr(int n) : Expr(static_cast<long long>(n)) {}
  explicit Expr(unsigned n) : Expr(static_cast<unsigned long long>(n)) {}
  explicit Expr(long n) : Expr(static_cast<long long>(n)) {}
  explicit Expr(unsigned long n) : Expr(static_cast<unsigned long long>(n)) {}
  explicit Expr(long long n)
      : u{Constant<Result>{typename Result::Scalar{n, subscriptIntegerKind},
            Result{subscriptIntegerKind}}} {}
  explicit Expr(unsigned long long n)
      : u{Constant<Result>{typename Result::Scalar{n, subscriptIntegerKind},
            Result{subscriptIntegerKind}}} {}

private:
  using Conversions = std::tuple<Convert<Result, TypeCategory::Integer>,
      Convert<Result, TypeCategory::Real>,
      Convert<Result, TypeCategory::Unsigned>>;
  using Operations = std::tuple<Parentheses<Result>, Negate<Result>,
      Add<Result>, Subtract<Result>, Multiply<Result>, Divide<Result>,
      Power<Result>, Extremum<Result>, ConditionalExpr<Result>>;
  using Others = std::tuple<Constant<Result>, ArrayConstructor<Result>,
      Designator<Result>, FunctionRef<Result>>;

public:
  common::TupleToVariant<common::CombineTuples<Operations, Conversions, Others>>
      u;
};

template <>
class Expr<Type<TypeCategory::Real>>
    : public ExpressionBase<Type<TypeCategory::Real>> {
public:
  using Result = Type<TypeCategory::Real>;

  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  explicit Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}

private:
  // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
  // and part access operations (resp.).
  using Conversions = std::variant<Convert<Result, TypeCategory::Integer>,
      Convert<Result, TypeCategory::Real>,
      Convert<Result, TypeCategory::Unsigned>>;
  using Operations = std::variant<ComplexComponent, Parentheses<Result>,
      Negate<Result>, Add<Result>, Subtract<Result>, Multiply<Result>,
      Divide<Result>, Power<Result>, RealToIntPower<Result>, Extremum<Result>,
      ConditionalExpr<Result>>;
  using Others = std::variant<Constant<Result>, ArrayConstructor<Result>,
      Designator<Result>, FunctionRef<Result>>;

public:
  common::CombineVariants<Operations, Conversions, Others> u;
};

template <>
class Expr<Type<TypeCategory::Complex>>
    : public ExpressionBase<Type<TypeCategory::Complex>> {
public:
  using Result = Type<TypeCategory::Complex>;
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  explicit Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}
  using Operations = std::variant<Parentheses<Result>, Negate<Result>,
      Convert<Result, TypeCategory::Complex>, Add<Result>, Subtract<Result>,
      Multiply<Result>, Divide<Result>, Power<Result>, RealToIntPower<Result>,
      ComplexConstructor, ConditionalExpr<Result>>;
  using Others = std::variant<Constant<Result>, ArrayConstructor<Result>,
      Designator<Result>, FunctionRef<Result>>;

public:
  common::CombineVariants<Operations, Others> u;
};

FOR_EACH_INTEGER_KIND(extern template class Expr, )
FOR_EACH_UNSIGNED_KIND(extern template class Expr, )
FOR_EACH_REAL_KIND(extern template class Expr, )
FOR_EACH_COMPLEX_KIND(extern template class Expr, )

template <>
class Expr<Type<TypeCategory::Character>>
    : public ExpressionBase<Type<TypeCategory::Character>> {
public:
  using Result = Type<TypeCategory::Character>;
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  explicit Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}
  explicit Expr(Scalar<Result> &&x) : u{Constant<Result>{std::move(x)}} {}

  std::optional<Expr<SubscriptInteger>> LEN() const;

  std::variant<Constant<Result>, ArrayConstructor<Result>, Designator<Result>,
      FunctionRef<Result>, Parentheses<Result>, Convert<Result>, Concat,
      Extremum<Result>, SetLength, ConditionalExpr<Result>>
      u;
};

FOR_EACH_CHARACTER_KIND(extern template class Expr, )

// The Relational class template is a helper for constructing logical
// expressions with polymorphism over the cross product of the possible
// categories and kinds of comparable operands.
// Fortran defines a numeric relation with distinct types or kinds as
// first undergoing the same operand conversions that occur with the intrinsic
// addition operator.  CharacterValue relations must have the same kind.
// There are no relations between LOGICAL values.

template <typename T>
class Relational : public Operation<Relational<T>, LogicalResult, T, T> {
public:
  using Result = LogicalResult;
  using Base = Operation<Relational, LogicalResult, T, T>;
  using Operand = typename Base::template Operand<0>;
  static_assert(Operand::category == TypeCategory::Integer ||
      Operand::category == TypeCategory::Real ||
      Operand::category == TypeCategory::Complex ||
      Operand::category == TypeCategory::Character ||
      Operand::category == TypeCategory::Unsigned);
  CLASS_BOILERPLATE(Relational)
  Relational(
      RelationalOperator r, const Expr<Operand> &a, const Expr<Operand> &b)
      : Base{a, b}, opr{r} {}
  Relational(RelationalOperator r, Expr<Operand> &&a, Expr<Operand> &&b)
      : Base{std::move(a), std::move(b)}, opr{r} {}
  // A relation's result is always LogicalResult, independent of operand kind.
  static constexpr std::optional<DynamicType> GetType() {
    return DynamicType{TypeCategory::Logical, logicalResultKind};
  }
  bool operator==(const Relational &) const;
  RelationalOperator opr;
};

template <> class Relational<SomeType> {
  using DirectlyComparableTypes = common::CombineTuples<IntegerTypes, RealTypes,
      ComplexTypes, CharacterTypes, UnsignedTypes>;

public:
  using Result = LogicalResult;
  EVALUATE_UNION_CLASS_BOILERPLATE(Relational)
  static constexpr DynamicType GetType() {
    return DynamicType{TypeCategory::Logical, logicalResultKind};
  }
  int Rank() const {
    return common::visit([](const auto &x) { return x.Rank(); }, u);
  }
  static constexpr int Corank() { return 0; }
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &o) const;
  common::MapTemplate<Relational, DirectlyComparableTypes> u;
};

FOR_EACH_INTEGER_KIND(extern template class Relational, )
FOR_EACH_UNSIGNED_KIND(extern template class Relational, )
FOR_EACH_REAL_KIND(extern template class Relational, )
FOR_EACH_CHARACTER_KIND(extern template class Relational, )
extern template class Relational<SomeType>;

// Logical expressions of a kind bigger than LogicalResult
// do not include Relational<> operations as possibilities,
// since the results of Relationals are always LogicalResult
// (kind=4).
template <>
class Expr<Type<TypeCategory::Logical>>
    : public ExpressionBase<Type<TypeCategory::Logical>> {
public:
  using Result = Type<TypeCategory::Logical>;
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  explicit Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}
  explicit Expr(bool x) : u{Constant<Result>{x, Result{logicalResultKind}}} {}

private:
  using Operations = std::tuple<Convert<Result>, Parentheses<Result>, Not,
      LogicalOperation, ConditionalExpr<Result>>;
  using Relations = std::tuple<Relational<SomeType>>;
  using Others = std::tuple<Constant<Result>, ArrayConstructor<Result>,
      Designator<Result>, FunctionRef<Result>>;

public:
  common::TupleToVariant<common::CombineTuples<Operations, Relations, Others>>
      u;
};

FOR_EACH_LOGICAL_KIND(extern template class Expr, )

// StructureConstructor pairs a StructureConstructorValues instance
// (a map associating symbols with expressions) with a derived type
// specification.  There are two other similar classes:
//  - ArrayConstructor<SomeDerived> comprises a derived type spec &
//    zero or more instances of Expr<SomeDerived>; it has rank 1
//    but not (in the most general case) a known shape.
//  - Constant<SomeDerived> comprises a derived type spec, zero or more
//    homogeneous instances of StructureConstructorValues whose type
//    parameters and component expressions are all constant, and a
//    known shape (possibly scalar).
// StructureConstructor represents a scalar value of derived type that
// is not necessarily a constant.  It is used only as an Expr<SomeDerived>
// alternative and as the type Scalar<SomeDerived> (with an assumption
// of constant component value expressions).
class StructureConstructor {
public:
  using Result = SomeDerived;

  explicit StructureConstructor(const semantics::DerivedTypeSpec &spec)
      : result_{spec} {}
  StructureConstructor(
      const semantics::DerivedTypeSpec &, const StructureConstructorValues &);
  StructureConstructor(
      const semantics::DerivedTypeSpec &, StructureConstructorValues &&);
  CLASS_BOILERPLATE(StructureConstructor)

  constexpr Result result() const { return result_; }
  const semantics::DerivedTypeSpec &derivedTypeSpec() const {
    return result_.derivedTypeSpec();
  }
  StructureConstructorValues &values() { return values_; }
  const StructureConstructorValues &values() const { return values_; }

  bool operator==(const StructureConstructor &) const;

  StructureConstructorValues::iterator begin() { return values_.begin(); }
  StructureConstructorValues::const_iterator begin() const {
    return values_.begin();
  }
  StructureConstructorValues::iterator end() { return values_.end(); }
  StructureConstructorValues::const_iterator end() const {
    return values_.end();
  }

  // can return nullopt
  std::optional<Expr<SomeType>> Find(const Symbol &) const;

  StructureConstructor &Add(const semantics::Symbol &, Expr<SomeType> &&);
  static constexpr int Rank() { return 0; }
  static constexpr int Corank() { return 0; }
  DynamicType GetType() const;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;

private:
  std::optional<Expr<SomeType>> CreateParentComponent(const Symbol &) const;
  Result result_;
  StructureConstructorValues values_;
};

// An expression whose result has a derived type.
template <> class Expr<SomeDerived> : public ExpressionBase<SomeDerived> {
public:
  using Result = SomeDerived;
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  std::variant<Constant<Result>, ArrayConstructor<Result>, StructureConstructor,
      Designator<Result>, FunctionRef<Result>, Parentheses<Result>,
      ConditionalExpr<Result>>
      u;
};

// A polymorphic expression of known intrinsic type category, but dynamic
// kind, represented as a discriminated union over Expr<Type<CAT, K>>
// for each supported kind K in the category.
template <TypeCategory CAT>
class Expr<SomeKind<CAT>> : public ExpressionBase<SomeKind<CAT>> {
public:
  using Result = SomeKind<CAT>;
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  int GetKind() const;
  common::MapTemplate<evaluate::Expr, CategoryTypes<CAT>> u;
};

template <> class Expr<SomeCharacter> : public ExpressionBase<SomeCharacter> {
public:
  using Result = SomeCharacter;
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  int GetKind() const;
  std::optional<Expr<SubscriptInteger>> LEN() const;
  common::MapTemplate<Expr, CategoryTypes<TypeCategory::Character>> u;
};

// A variant comprising the Expr<> instantiations over SomeDerived and
// SomeKind<CATEGORY>.
using CategoryExpression = common::MapTemplate<Expr, SomeCategory>;

// BOZ literal "typeless" constants must be wide enough to hold a numeric
// value of any supported kind of INTEGER or REAL.  They must also be
// distinguishable from other integer constants, since they are permitted
// to be used in only a few situations.
using BOZLiteralConstant = typename LargestReal::Scalar::Word;

// Null pointers without MOLD= arguments are typed by context.
struct NullPointer {
  constexpr bool operator==(const NullPointer &) const { return true; }
  static constexpr int Rank() { return 0; }
  static constexpr int Corank() { return 0; }
};

// Procedure pointer targets are treated as if they were typeless.
// They are either procedure designators or values returned from
// references to functions that return procedure (not object) pointers.
using TypelessExpression = std::variant<BOZLiteralConstant, NullPointer,
    ProcedureDesignator, ProcedureRef>;

// A completely generic expression, polymorphic across all of the intrinsic type
// categories and each of their kinds.
template <> class Expr<SomeType> : public ExpressionBase<SomeType> {
public:
  using Result = SomeType;
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)

  // Owning references to these generic expressions can appear in other
  // compiler data structures (viz., the parse tree and symbol table), so
  // its destructor is externalized to reduce redundant default instances.
  ~Expr();

  template <TypeCategory CAT>
  explicit Expr(const Expr<Type<CAT>> &x) : u{Expr<SomeKind<CAT>>{x}} {}

  template <TypeCategory CAT>
  explicit Expr(Expr<Type<CAT>> &&x) : u{Expr<SomeKind<CAT>>{std::move(x)}} {}

  template <TypeCategory CAT> Expr &operator=(const Expr<Type<CAT>> &x) {
    u = Expr<SomeKind<CAT>>{x};
    return *this;
  }

  template <TypeCategory CAT> Expr &operator=(Expr<Type<CAT>> &&x) {
    u = Expr<SomeKind<CAT>>{std::move(x)};
    return *this;
  }

public:
  common::CombineVariants<TypelessExpression, CategoryExpression> u;
};

// An assignment is either intrinsic, user-defined (with a ProcedureRef to
// specify the procedure to call), or pointer assignment (with possibly empty
// BoundsSpec or non-empty BoundsRemapping). In all cases there are Exprs
// representing the LHS and RHS of the assignment.
class Assignment {
public:
  Assignment(Expr<SomeType> &&lhs, Expr<SomeType> &&rhs)
      : lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  struct Intrinsic {};
  using BoundsSpec = std::vector<Expr<SubscriptInteger>>;
  using BoundsRemapping =
      std::vector<std::pair<Expr<SubscriptInteger>, Expr<SubscriptInteger>>>;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;

  Expr<SomeType> lhs;
  Expr<SomeType> rhs;
  std::variant<Intrinsic, ProcedureRef, BoundsSpec, BoundsRemapping> u;
};

// This wrapper class is used, by means of a forward reference with
// an owning pointer, to cache analyzed expressions in parse tree nodes.
struct GenericExprWrapper {
  GenericExprWrapper() {}
  explicit GenericExprWrapper(std::optional<Expr<SomeType>> &&x)
      : v{std::move(x)} {}
  ~GenericExprWrapper();
  static void Deleter(GenericExprWrapper *);
  std::optional<Expr<SomeType>> v; // vacant if error
};

// Like GenericExprWrapper but for analyzed assignments
struct GenericAssignmentWrapper {
  GenericAssignmentWrapper() {}
  explicit GenericAssignmentWrapper(Assignment &&x) : v{std::move(x)} {}
  explicit GenericAssignmentWrapper(std::optional<Assignment> &&x)
      : v{std::move(x)} {}
  ~GenericAssignmentWrapper();
  static void Deleter(GenericAssignmentWrapper *);
  std::optional<Assignment> v; // vacant if error
};

FOR_EACH_CATEGORY_TYPE(extern template class Expr, )
FOR_EACH_TYPE_AND_KIND(extern template class ExpressionBase, )
FOR_EACH_INTRINSIC_KIND(extern template class ArrayConstructorValues, )
FOR_EACH_INTRINSIC_KIND(extern template class ArrayConstructor, )

// Template instantiations to resolve these "extern template" declarations.
#define INSTANTIATE_EXPRESSION_TEMPLATES \
  FOR_EACH_INTRINSIC_KIND(template class Expr, ) \
  FOR_EACH_CATEGORY_TYPE(template class Expr, ) \
  FOR_EACH_INTEGER_KIND(template class Relational, ) \
  FOR_EACH_UNSIGNED_KIND(template class Relational, ) \
  FOR_EACH_REAL_KIND(template class Relational, ) \
  FOR_EACH_CHARACTER_KIND(template class Relational, ) \
  template class Relational<SomeType>; \
  FOR_EACH_TYPE_AND_KIND(template class ExpressionBase, ) \
  FOR_EACH_INTRINSIC_KIND(template class ArrayConstructorValues, ) \
  FOR_EACH_INTRINSIC_KIND(template class ArrayConstructor, ) \
  FOR_EACH_INTRINSIC_KIND(template class ConditionalExpr, )
} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_EXPRESSION_H_
