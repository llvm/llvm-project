//===-- lib/Semantics/expression.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/expression.h"
#include "check-call.h"
#include "pointer-assignment.h"
#include "resolve-names-utils.h"
#include "resolve-names.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <optional>
#include <set>
#include <vector>

// Typedef for optional generic expressions (ubiquitous in this file)
using MaybeExpr =
    std::optional<Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>;

// Much of the code that implements semantic analysis of expressions is
// tightly coupled with their typed representations in lib/Evaluate,
// and appears here in namespace Fortran::evaluate for convenience.
namespace Fortran::evaluate {

using common::LanguageFeature;
using common::NumericOperator;
using common::TypeCategory;

static inline std::string ToUpperCase(std::string_view str) {
  return parser::ToUpperCaseLetters(str);
}

struct DynamicTypeWithLength : public DynamicType {
  explicit DynamicTypeWithLength(const DynamicType &t) : DynamicType{t} {}
  std::optional<Expr<SubscriptInteger>> LEN() const;
  std::optional<Expr<SubscriptInteger>> length;
};

std::optional<Expr<SubscriptInteger>> DynamicTypeWithLength::LEN() const {
  if (length) {
    return length;
  } else {
    return GetCharLength();
  }
}

static std::optional<DynamicTypeWithLength> AnalyzeTypeSpec(
    const std::optional<parser::TypeSpec> &spec) {
  if (spec) {
    if (const semantics::DeclTypeSpec *typeSpec{spec->declTypeSpec}) {
      // Name resolution sets TypeSpec::declTypeSpec only when it's valid
      // (viz., an intrinsic type with valid known kind or a non-polymorphic
      // & non-ABSTRACT derived type).
      if (const semantics::IntrinsicTypeSpec *intrinsic{
              typeSpec->AsIntrinsic()}) {
        TypeCategory category{intrinsic->category()};
        if (auto optKind{ToInt64(intrinsic->kind())}) {
          int kind{static_cast<int>(*optKind)};
          if (category == TypeCategory::Character) {
            const semantics::CharacterTypeSpec &cts{
                typeSpec->characterTypeSpec()};
            const semantics::ParamValue &len{cts.length()};
            // N.B. CHARACTER(LEN=*) is allowed in type-specs in ALLOCATE() &
            // type guards, but not in array constructors.
            return DynamicTypeWithLength{DynamicType{kind, len}};
          } else {
            return DynamicTypeWithLength{DynamicType{category, kind}};
          }
        }
      } else if (const semantics::DerivedTypeSpec *derived{
                     typeSpec->AsDerived()}) {
        return DynamicTypeWithLength{DynamicType{*derived}};
      }
    }
  }
  return std::nullopt;
}

// Utilities to set a source location, if we have one, on an actual argument,
// when it is statically present.
static void SetArgSourceLocation(ActualArgument &x, parser::CharBlock at) {
  x.set_sourceLocation(at);
}
static void SetArgSourceLocation(
    std::optional<ActualArgument> &x, parser::CharBlock at) {
  if (x) {
    x->set_sourceLocation(at);
  }
}
static void SetArgSourceLocation(
    std::optional<ActualArgument> &x, std::optional<parser::CharBlock> at) {
  if (x && at) {
    x->set_sourceLocation(*at);
  }
}

class ArgumentAnalyzer {
public:
  explicit ArgumentAnalyzer(ExpressionAnalyzer &context)
      : context_{context}, source_{context.GetContextualMessages().at()},
        isProcedureCall_{false} {}
  ArgumentAnalyzer(ExpressionAnalyzer &context, parser::CharBlock source,
      bool isProcedureCall = false)
      : context_{context}, source_{source}, isProcedureCall_{isProcedureCall} {}
  bool fatalErrors() const { return fatalErrors_; }
  ActualArguments &&GetActuals() {
    CHECK(!fatalErrors_);
    return std::move(actuals_);
  }
  const Expr<SomeType> &GetExpr(std::size_t i) const {
    return DEREF(actuals_.at(i).value().UnwrapExpr());
  }
  Expr<SomeType> &&MoveExpr(std::size_t i) {
    return std::move(DEREF(actuals_.at(i).value().UnwrapExpr()));
  }
  void Analyze(const common::Indirection<parser::Expr> &x) {
    Analyze(x.value());
  }
  void Analyze(const parser::Expr &x) {
    actuals_.emplace_back(AnalyzeExpr(x));
    SetArgSourceLocation(actuals_.back(), x.source);
    fatalErrors_ |= !actuals_.back();
  }
  void Analyze(const parser::Variable &);
  void Analyze(const parser::ActualArgSpec &, bool isSubroutine);
  void ConvertBOZ(std::optional<DynamicType> &thisType, std::size_t i,
      std::optional<DynamicType> otherType);

  bool IsIntrinsicRelational(
      RelationalOperator, const DynamicType &, const DynamicType &) const;
  bool IsIntrinsicLogical() const;
  bool IsIntrinsicNumeric(NumericOperator) const;
  bool IsIntrinsicConcat() const;

  bool CheckConformance();
  bool CheckAssignmentConformance();
  bool CheckForNullPointer(const char *where = "as an operand here");

  // Find and return a user-defined operator or report an error.
  // The provided message is used if there is no such operator.
  // If a definedOpSymbolPtr is provided, the caller must check
  // for its accessibility.
  MaybeExpr TryDefinedOp(
      const char *, parser::MessageFixedText, bool isUserOp = false);
  template <typename E>
  MaybeExpr TryDefinedOp(E opr, parser::MessageFixedText msg) {
    return TryDefinedOp(
        context_.context().languageFeatures().GetNames(opr), msg);
  }
  // Find and return a user-defined assignment
  std::optional<ProcedureRef> TryDefinedAssignment();
  std::optional<ProcedureRef> GetDefinedAssignmentProc();
  std::optional<DynamicType> GetType(std::size_t) const;
  void Dump(llvm::raw_ostream &);

private:
  MaybeExpr TryDefinedOp(std::vector<const char *>, parser::MessageFixedText);
  MaybeExpr TryBoundOp(const Symbol &, int passIndex);
  std::optional<ActualArgument> AnalyzeExpr(const parser::Expr &);
  MaybeExpr AnalyzeExprOrWholeAssumedSizeArray(const parser::Expr &);
  bool AreConformable() const;
  const Symbol *FindBoundOp(parser::CharBlock, int passIndex,
      const Symbol *&generic, bool isSubroutine);
  void AddAssignmentConversion(
      const DynamicType &lhsType, const DynamicType &rhsType);
  bool OkLogicalIntegerAssignment(TypeCategory lhs, TypeCategory rhs);
  int GetRank(std::size_t) const;
  bool IsBOZLiteral(std::size_t i) const {
    return evaluate::IsBOZLiteral(GetExpr(i));
  }
  void SayNoMatch(const std::string &, bool isAssignment = false);
  std::string TypeAsFortran(std::size_t);
  bool AnyUntypedOrMissingOperand();

  ExpressionAnalyzer &context_;
  ActualArguments actuals_;
  parser::CharBlock source_;
  bool fatalErrors_{false};
  const bool isProcedureCall_; // false for user-defined op or assignment
};

// Wraps a data reference in a typed Designator<>, and a procedure
// or procedure pointer reference in a ProcedureDesignator.
MaybeExpr ExpressionAnalyzer::Designate(DataRef &&ref) {
  const Symbol &last{ref.GetLastSymbol()};
  const Symbol &symbol{BypassGeneric(last).GetUltimate()};
  if (semantics::IsProcedure(symbol)) {
    if (symbol.attrs().test(semantics::Attr::ABSTRACT)) {
      Say("Abstract procedure interface '%s' may not be used as a designator"_err_en_US,
          last.name());
    }
    if (auto *component{std::get_if<Component>(&ref.u)}) {
      if (!CheckDataRef(ref)) {
        return std::nullopt;
      }
      return Expr<SomeType>{ProcedureDesignator{std::move(*component)}};
    } else if (!std::holds_alternative<SymbolRef>(ref.u)) {
      DIE("unexpected alternative in DataRef");
    } else if (!symbol.attrs().test(semantics::Attr::INTRINSIC)) {
      if (symbol.has<semantics::GenericDetails>()) {
        Say("'%s' is not a specific procedure"_err_en_US, last.name());
      } else {
        return Expr<SomeType>{ProcedureDesignator{symbol}};
      }
    } else if (auto interface{context_.intrinsics().IsSpecificIntrinsicFunction(
                   symbol.name().ToString())};
               interface && !interface->isRestrictedSpecific) {
      SpecificIntrinsic intrinsic{
          symbol.name().ToString(), std::move(*interface)};
      intrinsic.isRestrictedSpecific = interface->isRestrictedSpecific;
      return Expr<SomeType>{ProcedureDesignator{std::move(intrinsic)}};
    } else {
      Say("'%s' is not an unrestricted specific intrinsic procedure"_err_en_US,
          last.name());
    }
    return std::nullopt;
  } else if (MaybeExpr result{AsGenericExpr(std::move(ref))}) {
    return result;
  } else {
    if (!context_.HasError(last) && !context_.HasError(symbol)) {
      AttachDeclaration(
          Say("'%s' is not an object that can appear in an expression"_err_en_US,
              last.name()),
          symbol);
      context_.SetError(last);
    }
    return std::nullopt;
  }
}

// Some subscript semantic checks must be deferred until all of the
// subscripts are in hand.
MaybeExpr ExpressionAnalyzer::CompleteSubscripts(ArrayRef &&ref) {
  const Symbol &symbol{ref.GetLastSymbol().GetUltimate()};
  int symbolRank{symbol.Rank()};
  int subscripts{static_cast<int>(ref.size())};
  if (subscripts == 0) {
    return std::nullopt; // error recovery
  } else if (subscripts != symbolRank) {
    if (symbolRank != 0) {
      Say("Reference to rank-%d object '%s' has %d subscripts"_err_en_US,
          symbolRank, symbol.name(), subscripts);
    }
    return std::nullopt;
  } else if (const auto *object{
                 symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    // C928 & C1002
    if (Triplet *last{std::get_if<Triplet>(&ref.subscript().back().u)}) {
      if (!last->upper() && object->IsAssumedSize()) {
        Say("Assumed-size array '%s' must have explicit final "
            "subscript upper bound value"_err_en_US,
            symbol.name());
        return std::nullopt;
      }
    }
  } else {
    // Shouldn't get here from Analyze(ArrayElement) without a valid base,
    // which, if not an object, must be a construct entity from
    // SELECT TYPE/RANK or ASSOCIATE.
    CHECK(symbol.has<semantics::AssocEntityDetails>());
  }
  if (!semantics::IsNamedConstant(symbol) && !inDataStmtObject_) {
    // Subscripts of named constants are checked in folding.
    // Subscripts of DATA statement objects are checked in data statement
    // conversion to initializers.
    CheckConstantSubscripts(ref);
  }
  return Designate(DataRef{std::move(ref)});
}

// Applies subscripts to a data reference.
MaybeExpr ExpressionAnalyzer::ApplySubscripts(
    DataRef &&dataRef, std::vector<Subscript> &&subscripts) {
  if (subscripts.empty()) {
    return std::nullopt; // error recovery
  }
  return common::visit(
      common::visitors{
          [&](SymbolRef &&symbol) {
            return CompleteSubscripts(ArrayRef{symbol, std::move(subscripts)});
          },
          [&](Component &&c) {
            return CompleteSubscripts(
                ArrayRef{std::move(c), std::move(subscripts)});
          },
          [&](auto &&) -> MaybeExpr {
            DIE("bad base for ArrayRef");
            return std::nullopt;
          },
      },
      std::move(dataRef.u));
}

void ExpressionAnalyzer::CheckConstantSubscripts(ArrayRef &ref) {
  // Fold subscript expressions and check for an empty triplet.
  const Symbol &arraySymbol{ref.base().GetLastSymbol()};
  Shape lb{GetLBOUNDs(foldingContext_, NamedEntity{arraySymbol})};
  CHECK(lb.size() >= ref.subscript().size());
  Shape ub{GetUBOUNDs(foldingContext_, NamedEntity{arraySymbol})};
  CHECK(ub.size() >= ref.subscript().size());
  bool anyPossiblyEmptyDim{false};
  int dim{0};
  for (Subscript &ss : ref.subscript()) {
    if (Triplet * triplet{std::get_if<Triplet>(&ss.u)}) {
      auto expr{Fold(triplet->stride())};
      auto stride{ToInt64(expr)};
      triplet->set_stride(std::move(expr));
      std::optional<ConstantSubscript> lower, upper;
      if (auto expr{triplet->lower()}) {
        *expr = Fold(std::move(*expr));
        lower = ToInt64(*expr);
        triplet->set_lower(std::move(*expr));
      } else {
        lower = ToInt64(lb[dim]);
      }
      if (auto expr{triplet->upper()}) {
        *expr = Fold(std::move(*expr));
        upper = ToInt64(*expr);
        triplet->set_upper(std::move(*expr));
      } else {
        upper = ToInt64(ub[dim]);
      }
      if (stride) {
        if (*stride == 0) {
          Say("Stride of triplet must not be zero"_err_en_US);
          return;
        }
        if (lower && upper) {
          if (*stride > 0) {
            anyPossiblyEmptyDim |= *lower > *upper;
          } else {
            anyPossiblyEmptyDim |= *lower < *upper;
          }
        } else {
          anyPossiblyEmptyDim = true;
        }
      } else { // non-constant stride
        if (lower && upper && *lower == *upper) {
          // stride is not relevant
        } else {
          anyPossiblyEmptyDim = true;
        }
      }
    } else { // not triplet
      auto &expr{std::get<IndirectSubscriptIntegerExpr>(ss.u).value()};
      expr = Fold(std::move(expr));
      anyPossiblyEmptyDim |= expr.Rank() > 0; // vector subscript
    }
    ++dim;
  }
  if (anyPossiblyEmptyDim) {
    return;
  }
  dim = 0;
  for (Subscript &ss : ref.subscript()) {
    auto dimLB{ToInt64(lb[dim])};
    auto dimUB{ToInt64(ub[dim])};
    std::optional<ConstantSubscript> val[2];
    int vals{0};
    if (auto *triplet{std::get_if<Triplet>(&ss.u)}) {
      auto stride{ToInt64(triplet->stride())};
      std::optional<ConstantSubscript> lower, upper;
      if (const auto *lowerExpr{triplet->GetLower()}) {
        lower = ToInt64(*lowerExpr);
      } else if (lb[dim]) {
        lower = ToInt64(*lb[dim]);
      }
      if (const auto *upperExpr{triplet->GetUpper()}) {
        upper = ToInt64(*upperExpr);
      } else if (ub[dim]) {
        upper = ToInt64(*ub[dim]);
      }
      if (lower) {
        val[vals++] = *lower;
        if (upper && *upper != lower && (stride && *stride != 0)) {
          // Normalize upper bound for non-unit stride
          // 1:10:2 -> 1:9:2, 10:1:-2 -> 10:2:-2
          val[vals++] = *lower + *stride * ((*upper - *lower) / *stride);
        }
      }
    } else {
      val[vals++] =
          ToInt64(std::get<IndirectSubscriptIntegerExpr>(ss.u).value());
    }
    for (int j{0}; j < vals; ++j) {
      if (val[j]) {
        if (dimLB && *val[j] < *dimLB) {
          AttachDeclaration(
              Say("Subscript %jd is less than lower bound %jd for dimension %d of array"_err_en_US,
                  static_cast<std::intmax_t>(*val[j]),
                  static_cast<std::intmax_t>(*dimLB), dim + 1),
              ref.base().GetLastSymbol());
        }
        if (dimUB && *val[j] > *dimUB) {
          AttachDeclaration(
              Say("Subscript %jd is greater than upper bound %jd for dimension %d of array"_err_en_US,
                  static_cast<std::intmax_t>(*val[j]),
                  static_cast<std::intmax_t>(*dimUB), dim + 1),
              ref.base().GetLastSymbol());
        }
      }
    }
    ++dim;
  }
}

// C919a - only one part-ref of a data-ref may have rank > 0
bool ExpressionAnalyzer::CheckRanks(const DataRef &dataRef) {
  return common::visit(
      common::visitors{
          [this](const Component &component) {
            const Symbol &symbol{component.GetLastSymbol()};
            if (int componentRank{symbol.Rank()}; componentRank > 0) {
              if (int baseRank{component.base().Rank()}; baseRank > 0) {
                Say("Reference to whole rank-%d component '%s' of rank-%d array of derived type is not allowed"_err_en_US,
                    componentRank, symbol.name(), baseRank);
                return false;
              }
            } else {
              return CheckRanks(component.base());
            }
            return true;
          },
          [this](const ArrayRef &arrayRef) {
            if (const auto *component{arrayRef.base().UnwrapComponent()}) {
              int subscriptRank{0};
              for (const Subscript &subscript : arrayRef.subscript()) {
                subscriptRank += subscript.Rank();
              }
              if (subscriptRank > 0) {
                if (int componentBaseRank{component->base().Rank()};
                    componentBaseRank > 0) {
                  Say("Subscripts of component '%s' of rank-%d derived type array have rank %d but must all be scalar"_err_en_US,
                      component->GetLastSymbol().name(), componentBaseRank,
                      subscriptRank);
                  return false;
                }
              } else {
                return CheckRanks(component->base());
              }
            }
            return true;
          },
          [](const SymbolRef &) { return true; },
          [](const CoarrayRef &) { return true; },
      },
      dataRef.u);
}

// C911 - if the last name in a data-ref has an abstract derived type,
// it must also be polymorphic.
bool ExpressionAnalyzer::CheckPolymorphic(const DataRef &dataRef) {
  if (auto type{DynamicType::From(dataRef.GetLastSymbol())}) {
    if (type->category() == TypeCategory::Derived && !type->IsPolymorphic()) {
      const Symbol &typeSymbol{
          type->GetDerivedTypeSpec().typeSymbol().GetUltimate()};
      if (typeSymbol.attrs().test(semantics::Attr::ABSTRACT)) {
        AttachDeclaration(
            Say("Reference to object with abstract derived type '%s' must be polymorphic"_err_en_US,
                typeSymbol.name()),
            typeSymbol);
        return false;
      }
    }
  }
  return true;
}

bool ExpressionAnalyzer::CheckDataRef(const DataRef &dataRef) {
  // Always check both, don't short-circuit
  bool ranksOk{CheckRanks(dataRef)};
  bool polyOk{CheckPolymorphic(dataRef)};
  return ranksOk && polyOk;
}

// Parse tree correction after a substring S(j:k) was misparsed as an
// array section.  Fortran substrings must have a range, not a
// single index.
static std::optional<parser::Substring> FixMisparsedSubstringDataRef(
    parser::DataRef &dataRef) {
  if (auto *ae{
          std::get_if<common::Indirection<parser::ArrayElement>>(&dataRef.u)}) {
    // ...%a(j:k) and "a" is a character scalar
    parser::ArrayElement &arrElement{ae->value()};
    if (arrElement.subscripts.size() == 1) {
      if (auto *triplet{std::get_if<parser::SubscriptTriplet>(
              &arrElement.subscripts.front().u)}) {
        if (!std::get<2 /*stride*/>(triplet->t).has_value()) {
          if (const Symbol *symbol{
                  parser::GetLastName(arrElement.base).symbol}) {
            const Symbol &ultimate{symbol->GetUltimate()};
            if (const semantics::DeclTypeSpec *type{ultimate.GetType()}) {
              if (!ultimate.IsObjectArray() &&
                  type->category() == semantics::DeclTypeSpec::Character) {
                // The ambiguous S(j:k) was parsed as an array section
                // reference, but it's now clear that it's a substring.
                // Fix the parse tree in situ.
                return arrElement.ConvertToSubstring();
              }
            }
          }
        }
      }
    }
  }
  return std::nullopt;
}

// When a designator is a misparsed type-param-inquiry of a misparsed
// substring -- it looks like a structure component reference of an array
// slice -- fix the substring and then convert to an intrinsic function
// call to KIND() or LEN().  And when the designator is a misparsed
// substring, convert it into a substring reference in place.
MaybeExpr ExpressionAnalyzer::FixMisparsedSubstring(
    const parser::Designator &d) {
  auto &mutate{const_cast<parser::Designator &>(d)};
  if (auto *dataRef{std::get_if<parser::DataRef>(&mutate.u)}) {
    if (auto *sc{std::get_if<common::Indirection<parser::StructureComponent>>(
            &dataRef->u)}) {
      parser::StructureComponent &structComponent{sc->value()};
      parser::CharBlock which{structComponent.component.source};
      if (which == "kind" || which == "len") {
        if (auto substring{
                FixMisparsedSubstringDataRef(structComponent.base)}) {
          // ...%a(j:k)%kind or %len and "a" is a character scalar
          mutate.u = std::move(*substring);
          if (MaybeExpr substringExpr{Analyze(d)}) {
            return MakeFunctionRef(which,
                ActualArguments{ActualArgument{std::move(*substringExpr)}});
          }
        }
      }
    } else if (auto substring{FixMisparsedSubstringDataRef(*dataRef)}) {
      mutate.u = std::move(*substring);
    }
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Designator &d) {
  auto restorer{GetContextualMessages().SetLocation(d.source)};
  if (auto substringInquiry{FixMisparsedSubstring(d)}) {
    return substringInquiry;
  }
  // These checks have to be deferred to these "top level" data-refs where
  // we can be sure that there are no following subscripts (yet).
  MaybeExpr result{Analyze(d.u)};
  if (result) {
    std::optional<DataRef> dataRef{ExtractDataRef(std::move(result))};
    if (!dataRef) {
      dataRef = ExtractDataRef(std::move(result), /*intoSubstring=*/true);
    }
    if (!dataRef) {
      dataRef = ExtractDataRef(std::move(result),
          /*intoSubstring=*/false, /*intoComplexPart=*/true);
    }
    if (dataRef && !CheckDataRef(*dataRef)) {
      result.reset();
    }
  }
  return result;
}

// A utility subroutine to repackage optional expressions of various levels
// of type specificity as fully general MaybeExpr values.
template <typename A> common::IfNoLvalue<MaybeExpr, A> AsMaybeExpr(A &&x) {
  return AsGenericExpr(std::move(x));
}
template <typename A> MaybeExpr AsMaybeExpr(std::optional<A> &&x) {
  if (x) {
    return AsMaybeExpr(std::move(*x));
  }
  return std::nullopt;
}

// Type kind parameter values for literal constants.
int ExpressionAnalyzer::AnalyzeKindParam(
    const std::optional<parser::KindParam> &kindParam, int defaultKind) {
  if (!kindParam) {
    return defaultKind;
  }
  std::int64_t kind{common::visit(
      common::visitors{
          [](std::uint64_t k) { return static_cast<std::int64_t>(k); },
          [&](const parser::Scalar<
              parser::Integer<parser::Constant<parser::Name>>> &n) {
            if (MaybeExpr ie{Analyze(n)}) {
              return ToInt64(*ie).value_or(defaultKind);
            }
            return static_cast<std::int64_t>(defaultKind);
          },
      },
      kindParam->u)};
  if (kind != static_cast<int>(kind)) {
    Say("Unsupported type kind value (%jd)"_err_en_US,
        static_cast<std::intmax_t>(kind));
    kind = defaultKind;
  }
  return static_cast<int>(kind);
}

// Common handling of parser::IntLiteralConstant and SignedIntLiteralConstant
struct IntTypeVisitor {
  using Result = MaybeExpr;
  using Types = IntegerTypes;
  template <typename T> Result Test() {
    if (T::kind >= kind) {
      const char *p{digits.begin()};
      using Int = typename T::Scalar;
      typename Int::ValueWithOverflow num{0, false};
      if (isNegated) {
        auto unsignedNum{Int::Read(p, 10, false /*unsigned*/)};
        num.value = unsignedNum.value.Negate().value;
        num.overflow = unsignedNum.overflow || num.value > Int{0};
        if (!num.overflow && num.value.Negate().overflow &&
            !analyzer.context().IsInModuleFile(digits)) {
          analyzer.Say(digits,
              "negated maximum INTEGER(KIND=%d) literal"_port_en_US, T::kind);
        }
      } else {
        num = Int::Read(p, 10, true /*signed*/);
      }
      if (!num.overflow) {
        if (T::kind > kind) {
          if (!isDefaultKind ||
              !analyzer.context().IsEnabled(LanguageFeature::BigIntLiterals)) {
            return std::nullopt;
          } else if (analyzer.context().ShouldWarn(
                         LanguageFeature::BigIntLiterals)) {
            analyzer.Say(digits,
                "Integer literal is too large for default INTEGER(KIND=%d); "
                "assuming INTEGER(KIND=%d)"_port_en_US,
                kind, T::kind);
          }
        }
        return Expr<SomeType>{
            Expr<SomeInteger>{Expr<T>{Constant<T>{std::move(num.value)}}}};
      }
    }
    return std::nullopt;
  }
  ExpressionAnalyzer &analyzer;
  parser::CharBlock digits;
  std::int64_t kind;
  bool isDefaultKind;
  bool isNegated;
};

template <typename PARSED>
MaybeExpr ExpressionAnalyzer::IntLiteralConstant(
    const PARSED &x, bool isNegated) {
  const auto &kindParam{std::get<std::optional<parser::KindParam>>(x.t)};
  bool isDefaultKind{!kindParam};
  int kind{AnalyzeKindParam(kindParam, GetDefaultKind(TypeCategory::Integer))};
  if (CheckIntrinsicKind(TypeCategory::Integer, kind)) {
    auto digits{std::get<parser::CharBlock>(x.t)};
    if (MaybeExpr result{common::SearchTypes(
            IntTypeVisitor{*this, digits, kind, isDefaultKind, isNegated})}) {
      return result;
    } else if (isDefaultKind) {
      Say(digits,
          "Integer literal is too large for any allowable "
          "kind of INTEGER"_err_en_US);
    } else {
      Say(digits, "Integer literal is too large for INTEGER(KIND=%d)"_err_en_US,
          kind);
    }
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::IntLiteralConstant &x, bool isNegated) {
  auto restorer{
      GetContextualMessages().SetLocation(std::get<parser::CharBlock>(x.t))};
  return IntLiteralConstant(x, isNegated);
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::SignedIntLiteralConstant &x) {
  auto restorer{GetContextualMessages().SetLocation(x.source)};
  return IntLiteralConstant(x);
}

template <typename TYPE>
Constant<TYPE> ReadRealLiteral(
    parser::CharBlock source, FoldingContext &context) {
  const char *p{source.begin()};
  auto valWithFlags{
      Scalar<TYPE>::Read(p, context.targetCharacteristics().roundingMode())};
  CHECK(p == source.end());
  RealFlagWarnings(context, valWithFlags.flags, "conversion of REAL literal");
  auto value{valWithFlags.value};
  if (context.targetCharacteristics().areSubnormalsFlushedToZero()) {
    value = value.FlushSubnormalToZero();
  }
  return {value};
}

struct RealTypeVisitor {
  using Result = std::optional<Expr<SomeReal>>;
  using Types = RealTypes;

  RealTypeVisitor(int k, parser::CharBlock lit, FoldingContext &ctx)
      : kind{k}, literal{lit}, context{ctx} {}

  template <typename T> Result Test() {
    if (kind == T::kind) {
      return {AsCategoryExpr(ReadRealLiteral<T>(literal, context))};
    }
    return std::nullopt;
  }

  int kind;
  parser::CharBlock literal;
  FoldingContext &context;
};

// Reads a real literal constant and encodes it with the right kind.
MaybeExpr ExpressionAnalyzer::Analyze(const parser::RealLiteralConstant &x) {
  // Use a local message context around the real literal for better
  // provenance on any messages.
  auto restorer{GetContextualMessages().SetLocation(x.real.source)};
  // If a kind parameter appears, it defines the kind of the literal and the
  // letter used in an exponent part must be 'E' (e.g., the 'E' in
  // "6.02214E+23").  In the absence of an explicit kind parameter, any
  // exponent letter determines the kind.  Otherwise, defaults apply.
  auto &defaults{context_.defaultKinds()};
  int defaultKind{defaults.GetDefaultKind(TypeCategory::Real)};
  const char *end{x.real.source.end()};
  char expoLetter{' '};
  std::optional<int> letterKind;
  for (const char *p{x.real.source.begin()}; p < end; ++p) {
    if (parser::IsLetter(*p)) {
      expoLetter = *p;
      switch (expoLetter) {
      case 'e':
        letterKind = defaults.GetDefaultKind(TypeCategory::Real);
        break;
      case 'd':
        letterKind = defaults.doublePrecisionKind();
        break;
      case 'q':
        letterKind = defaults.quadPrecisionKind();
        break;
      default:
        Say("Unknown exponent letter '%c'"_err_en_US, expoLetter);
      }
      break;
    }
  }
  if (letterKind) {
    defaultKind = *letterKind;
  }
  // C716 requires 'E' as an exponent.
  // Extension: allow exponent-letter matching the kind-param.
  auto kind{AnalyzeKindParam(x.kind, defaultKind)};
  if (letterKind && expoLetter != 'e') {
    if (kind != *letterKind) {
      Say("Explicit kind parameter on real constant disagrees with "
          "exponent letter '%c'"_warn_en_US,
          expoLetter);
    } else if (x.kind &&
        context_.ShouldWarn(
            common::LanguageFeature::ExponentMatchingKindParam)) {
      Say("Explicit kind parameter together with non-'E' exponent letter "
          "is not standard"_port_en_US);
    }
  }
  auto result{common::SearchTypes(
      RealTypeVisitor{kind, x.real.source, GetFoldingContext()})};
  if (!result) { // C717
    Say("Unsupported REAL(KIND=%d)"_err_en_US, kind);
  }
  return AsMaybeExpr(std::move(result));
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::SignedRealLiteralConstant &x) {
  if (auto result{Analyze(std::get<parser::RealLiteralConstant>(x.t))}) {
    auto &realExpr{std::get<Expr<SomeReal>>(result->u)};
    if (auto sign{std::get<std::optional<parser::Sign>>(x.t)}) {
      if (sign == parser::Sign::Negative) {
        return AsGenericExpr(-std::move(realExpr));
      }
    }
    return result;
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::SignedComplexLiteralConstant &x) {
  auto result{Analyze(std::get<parser::ComplexLiteralConstant>(x.t))};
  if (!result) {
    return std::nullopt;
  } else if (std::get<parser::Sign>(x.t) == parser::Sign::Negative) {
    return AsGenericExpr(-std::move(std::get<Expr<SomeComplex>>(result->u)));
  } else {
    return result;
  }
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::ComplexPart &x) {
  return Analyze(x.u);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::ComplexLiteralConstant &z) {
  return AnalyzeComplex(Analyze(std::get<0>(z.t)), Analyze(std::get<1>(z.t)),
      "complex literal constant");
}

// CHARACTER literal processing.
MaybeExpr ExpressionAnalyzer::AnalyzeString(std::string &&string, int kind) {
  if (!CheckIntrinsicKind(TypeCategory::Character, kind)) {
    return std::nullopt;
  }
  switch (kind) {
  case 1:
    return AsGenericExpr(Constant<Type<TypeCategory::Character, 1>>{
        parser::DecodeString<std::string, parser::Encoding::LATIN_1>(
            string, true)});
  case 2:
    return AsGenericExpr(Constant<Type<TypeCategory::Character, 2>>{
        parser::DecodeString<std::u16string, parser::Encoding::UTF_8>(
            string, true)});
  case 4:
    return AsGenericExpr(Constant<Type<TypeCategory::Character, 4>>{
        parser::DecodeString<std::u32string, parser::Encoding::UTF_8>(
            string, true)});
  default:
    CRASH_NO_CASE;
  }
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::CharLiteralConstant &x) {
  int kind{
      AnalyzeKindParam(std::get<std::optional<parser::KindParam>>(x.t), 1)};
  auto value{std::get<std::string>(x.t)};
  return AnalyzeString(std::move(value), kind);
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::HollerithLiteralConstant &x) {
  int kind{GetDefaultKind(TypeCategory::Character)};
  auto value{x.v};
  return AnalyzeString(std::move(value), kind);
}

// .TRUE. and .FALSE. of various kinds
MaybeExpr ExpressionAnalyzer::Analyze(const parser::LogicalLiteralConstant &x) {
  auto kind{AnalyzeKindParam(std::get<std::optional<parser::KindParam>>(x.t),
      GetDefaultKind(TypeCategory::Logical))};
  bool value{std::get<bool>(x.t)};
  auto result{common::SearchTypes(
      TypeKindVisitor<TypeCategory::Logical, Constant, bool>{
          kind, std::move(value)})};
  if (!result) {
    Say("unsupported LOGICAL(KIND=%d)"_err_en_US, kind); // C728
  }
  return result;
}

// BOZ typeless literals
MaybeExpr ExpressionAnalyzer::Analyze(const parser::BOZLiteralConstant &x) {
  const char *p{x.v.c_str()};
  std::uint64_t base{16};
  switch (*p++) {
  case 'b':
    base = 2;
    break;
  case 'o':
    base = 8;
    break;
  case 'z':
    break;
  case 'x':
    break;
  default:
    CRASH_NO_CASE;
  }
  CHECK(*p == '"');
  ++p;
  auto value{BOZLiteralConstant::Read(p, base, false /*unsigned*/)};
  if (*p != '"') {
    Say("Invalid digit ('%c') in BOZ literal '%s'"_err_en_US, *p,
        x.v); // C7107, C7108
    return std::nullopt;
  }
  if (value.overflow) {
    Say("BOZ literal '%s' too large"_err_en_US, x.v);
    return std::nullopt;
  }
  return AsGenericExpr(std::move(value.value));
}

// Names and named constants
MaybeExpr ExpressionAnalyzer::Analyze(const parser::Name &n) {
  auto restorer{GetContextualMessages().SetLocation(n.source)};
  if (std::optional<int> kind{IsImpliedDo(n.source)}) {
    return AsMaybeExpr(ConvertToKind<TypeCategory::Integer>(
        *kind, AsExpr(ImpliedDoIndex{n.source})));
  }
  if (context_.HasError(n.symbol)) { // includes case of no symbol
    return std::nullopt;
  } else {
    const Symbol &ultimate{n.symbol->GetUltimate()};
    if (ultimate.has<semantics::TypeParamDetails>()) {
      // A bare reference to a derived type parameter within a parameterized
      // derived type definition.
      auto dyType{DynamicType::From(ultimate)};
      if (!dyType) {
        // When the integer kind of this type parameter is not known now,
        // it's either an error or because it depends on earlier-declared kind
        // type parameters.  So assume that it's a subscript integer for now
        // while processing other specification expressions in the PDT
        // definition; the right kind value will be used later in each of its
        // instantiations.
        int kind{SubscriptInteger::kind};
        if (const auto *typeSpec{ultimate.GetType()}) {
          if (const semantics::IntrinsicTypeSpec *
              intrinType{typeSpec->AsIntrinsic()}) {
            if (auto k{ToInt64(Fold(semantics::KindExpr{intrinType->kind()}))};
                k && IsValidKindOfIntrinsicType(TypeCategory::Integer, *k)) {
              kind = *k;
            }
          }
        }
        dyType = DynamicType{TypeCategory::Integer, kind};
      }
      return Fold(ConvertToType(
          *dyType, AsGenericExpr(TypeParamInquiry{std::nullopt, ultimate})));
    } else {
      if (n.symbol->attrs().test(semantics::Attr::VOLATILE)) {
        if (const semantics::Scope *pure{semantics::FindPureProcedureContaining(
                context_.FindScope(n.source))}) {
          SayAt(n,
              "VOLATILE variable '%s' may not be referenced in pure subprogram '%s'"_err_en_US,
              n.source, DEREF(pure->symbol()).name());
          n.symbol->attrs().reset(semantics::Attr::VOLATILE);
        }
      }
      if (!isWholeAssumedSizeArrayOk_ &&
          semantics::IsAssumedSizeArray(*n.symbol)) { // C1002, C1014, C1231
        AttachDeclaration(
            SayAt(n,
                "Whole assumed-size array '%s' may not appear here without subscripts"_err_en_US,
                n.source),
            *n.symbol);
      }
      return Designate(DataRef{*n.symbol});
    }
  }
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::NamedConstant &n) {
  auto restorer{GetContextualMessages().SetLocation(n.v.source)};
  if (MaybeExpr value{Analyze(n.v)}) {
    Expr<SomeType> folded{Fold(std::move(*value))};
    if (IsConstantExpr(folded)) {
      return folded;
    }
    Say(n.v.source, "must be a constant"_err_en_US); // C718
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::NullInit &n) {
  auto restorer{AllowNullPointer()};
  if (MaybeExpr value{Analyze(n.v.value())}) {
    // Subtle: when the NullInit is a DataStmtConstant, it might
    // be a misparse of a structure constructor without parameters
    // or components (e.g., T()).  Checking the result to ensure
    // that a "=>" data entity initializer actually resolved to
    // a null pointer has to be done by the caller.
    return Fold(std::move(*value));
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::StmtFunctionStmt &stmtFunc) {
  inStmtFunctionDefinition_ = true;
  return Analyze(std::get<parser::Scalar<parser::Expr>>(stmtFunc.t));
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::InitialDataTarget &x) {
  return Analyze(x.value());
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::DataStmtValue &x) {
  if (const auto &repeat{
          std::get<std::optional<parser::DataStmtRepeat>>(x.t)}) {
    x.repetitions = -1;
    if (MaybeExpr expr{Analyze(repeat->u)}) {
      Expr<SomeType> folded{Fold(std::move(*expr))};
      if (auto value{ToInt64(folded)}) {
        if (*value >= 0) { // C882
          x.repetitions = *value;
        } else {
          Say(FindSourceLocation(repeat),
              "Repeat count (%jd) for data value must not be negative"_err_en_US,
              *value);
        }
      }
    }
  }
  return Analyze(std::get<parser::DataStmtConstant>(x.t));
}

// Substring references
std::optional<Expr<SubscriptInteger>> ExpressionAnalyzer::GetSubstringBound(
    const std::optional<parser::ScalarIntExpr> &bound) {
  if (bound) {
    if (MaybeExpr expr{Analyze(*bound)}) {
      if (expr->Rank() > 1) {
        Say("substring bound expression has rank %d"_err_en_US, expr->Rank());
      }
      if (auto *intExpr{std::get_if<Expr<SomeInteger>>(&expr->u)}) {
        if (auto *ssIntExpr{std::get_if<Expr<SubscriptInteger>>(&intExpr->u)}) {
          return {std::move(*ssIntExpr)};
        }
        return {Expr<SubscriptInteger>{
            Convert<SubscriptInteger, TypeCategory::Integer>{
                std::move(*intExpr)}}};
      } else {
        Say("substring bound expression is not INTEGER"_err_en_US);
      }
    }
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Substring &ss) {
  if (MaybeExpr baseExpr{Analyze(std::get<parser::DataRef>(ss.t))}) {
    if (std::optional<DataRef> dataRef{ExtractDataRef(std::move(*baseExpr))}) {
      if (MaybeExpr newBaseExpr{Designate(std::move(*dataRef))}) {
        if (std::optional<DataRef> checked{
                ExtractDataRef(std::move(*newBaseExpr))}) {
          const parser::SubstringRange &range{
              std::get<parser::SubstringRange>(ss.t)};
          std::optional<Expr<SubscriptInteger>> first{
              GetSubstringBound(std::get<0>(range.t))};
          std::optional<Expr<SubscriptInteger>> last{
              GetSubstringBound(std::get<1>(range.t))};
          const Symbol &symbol{checked->GetLastSymbol()};
          if (std::optional<DynamicType> dynamicType{
                  DynamicType::From(symbol)}) {
            if (dynamicType->category() == TypeCategory::Character) {
              return WrapperHelper<TypeCategory::Character, Designator,
                  Substring>(dynamicType->kind(),
                  Substring{std::move(checked.value()), std::move(first),
                      std::move(last)});
            }
          }
          Say("substring may apply only to CHARACTER"_err_en_US);
        }
      }
    }
  }
  return std::nullopt;
}

// CHARACTER literal substrings
MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::CharLiteralConstantSubstring &x) {
  const parser::SubstringRange &range{std::get<parser::SubstringRange>(x.t)};
  std::optional<Expr<SubscriptInteger>> lower{
      GetSubstringBound(std::get<0>(range.t))};
  std::optional<Expr<SubscriptInteger>> upper{
      GetSubstringBound(std::get<1>(range.t))};
  if (MaybeExpr string{Analyze(std::get<parser::CharLiteralConstant>(x.t))}) {
    if (auto *charExpr{std::get_if<Expr<SomeCharacter>>(&string->u)}) {
      Expr<SubscriptInteger> length{
          common::visit([](const auto &ckExpr) { return ckExpr.LEN().value(); },
              charExpr->u)};
      if (!lower) {
        lower = Expr<SubscriptInteger>{1};
      }
      if (!upper) {
        upper = Expr<SubscriptInteger>{
            static_cast<std::int64_t>(ToInt64(length).value())};
      }
      return common::visit(
          [&](auto &&ckExpr) -> MaybeExpr {
            using Result = ResultType<decltype(ckExpr)>;
            auto *cp{std::get_if<Constant<Result>>(&ckExpr.u)};
            CHECK(DEREF(cp).size() == 1);
            StaticDataObject::Pointer staticData{StaticDataObject::Create()};
            staticData->set_alignment(Result::kind)
                .set_itemBytes(Result::kind)
                .Push(cp->GetScalarValue().value(),
                    foldingContext_.targetCharacteristics().isBigEndian());
            Substring substring{std::move(staticData), std::move(lower.value()),
                std::move(upper.value())};
            return AsGenericExpr(
                Expr<Result>{Designator<Result>{std::move(substring)}});
          },
          std::move(charExpr->u));
    }
  }
  return std::nullopt;
}

// substring%KIND/LEN
MaybeExpr ExpressionAnalyzer::Analyze(const parser::SubstringInquiry &x) {
  if (MaybeExpr substring{Analyze(x.v)}) {
    CHECK(x.source.size() >= 8);
    int nameLen{x.source.end()[-1] == 'n' ? 3 /*LEN*/ : 4 /*KIND*/};
    parser::CharBlock name{
        x.source.end() - nameLen, static_cast<std::size_t>(nameLen)};
    CHECK(name == "len" || name == "kind");
    return MakeFunctionRef(
        name, ActualArguments{ActualArgument{std::move(*substring)}});
  } else {
    return std::nullopt;
  }
}

// Subscripted array references
std::optional<Expr<SubscriptInteger>> ExpressionAnalyzer::AsSubscript(
    MaybeExpr &&expr) {
  if (expr) {
    if (expr->Rank() > 1) {
      Say("Subscript expression has rank %d greater than 1"_err_en_US,
          expr->Rank());
    }
    if (auto *intExpr{std::get_if<Expr<SomeInteger>>(&expr->u)}) {
      if (auto *ssIntExpr{std::get_if<Expr<SubscriptInteger>>(&intExpr->u)}) {
        return std::move(*ssIntExpr);
      } else {
        return Expr<SubscriptInteger>{
            Convert<SubscriptInteger, TypeCategory::Integer>{
                std::move(*intExpr)}};
      }
    } else {
      Say("Subscript expression is not INTEGER"_err_en_US);
    }
  }
  return std::nullopt;
}

std::optional<Expr<SubscriptInteger>> ExpressionAnalyzer::TripletPart(
    const std::optional<parser::Subscript> &s) {
  if (s) {
    return AsSubscript(Analyze(*s));
  } else {
    return std::nullopt;
  }
}

std::optional<Subscript> ExpressionAnalyzer::AnalyzeSectionSubscript(
    const parser::SectionSubscript &ss) {
  return common::visit(
      common::visitors{
          [&](const parser::SubscriptTriplet &t) -> std::optional<Subscript> {
            const auto &lower{std::get<0>(t.t)};
            const auto &upper{std::get<1>(t.t)};
            const auto &stride{std::get<2>(t.t)};
            auto result{Triplet{
                TripletPart(lower), TripletPart(upper), TripletPart(stride)}};
            if ((lower && !result.lower()) || (upper && !result.upper())) {
              return std::nullopt;
            } else {
              return std::make_optional<Subscript>(result);
            }
          },
          [&](const auto &s) -> std::optional<Subscript> {
            if (auto subscriptExpr{AsSubscript(Analyze(s))}) {
              return Subscript{std::move(*subscriptExpr)};
            } else {
              return std::nullopt;
            }
          },
      },
      ss.u);
}

// Empty result means an error occurred
std::vector<Subscript> ExpressionAnalyzer::AnalyzeSectionSubscripts(
    const std::list<parser::SectionSubscript> &sss) {
  bool error{false};
  std::vector<Subscript> subscripts;
  for (const auto &s : sss) {
    if (auto subscript{AnalyzeSectionSubscript(s)}) {
      subscripts.emplace_back(std::move(*subscript));
    } else {
      error = true;
    }
  }
  return !error ? subscripts : std::vector<Subscript>{};
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::ArrayElement &ae) {
  MaybeExpr baseExpr;
  {
    auto restorer{AllowWholeAssumedSizeArray()};
    baseExpr = Analyze(ae.base);
  }
  if (baseExpr) {
    if (ae.subscripts.empty()) {
      // will be converted to function call later or error reported
    } else if (baseExpr->Rank() == 0) {
      if (const Symbol *symbol{GetLastSymbol(*baseExpr)}) {
        if (!context_.HasError(symbol)) {
          if (inDataStmtConstant_) {
            // Better error for NULL(X) with a MOLD= argument
            Say("'%s' must be an array or structure constructor if used with non-empty parentheses as a DATA statement constant"_err_en_US,
                symbol->name());
          } else {
            Say("'%s' is not an array"_err_en_US, symbol->name());
          }
          context_.SetError(*symbol);
        }
      }
    } else if (std::optional<DataRef> dataRef{
                   ExtractDataRef(std::move(*baseExpr))}) {
      return ApplySubscripts(
          std::move(*dataRef), AnalyzeSectionSubscripts(ae.subscripts));
    } else {
      Say("Subscripts may be applied only to an object, component, or array constant"_err_en_US);
    }
  }
  // error was reported: analyze subscripts without reporting more errors
  auto restorer{GetContextualMessages().DiscardMessages()};
  AnalyzeSectionSubscripts(ae.subscripts);
  return std::nullopt;
}

// Type parameter inquiries apply to data references, but don't depend
// on any trailing (co)subscripts.
static NamedEntity IgnoreAnySubscripts(Designator<SomeDerived> &&designator) {
  return common::visit(
      common::visitors{
          [](SymbolRef &&symbol) { return NamedEntity{symbol}; },
          [](Component &&component) {
            return NamedEntity{std::move(component)};
          },
          [](ArrayRef &&arrayRef) { return std::move(arrayRef.base()); },
          [](CoarrayRef &&coarrayRef) {
            return NamedEntity{coarrayRef.GetLastSymbol()};
          },
      },
      std::move(designator.u));
}

// Components of parent derived types are explicitly represented as such.
std::optional<Component> ExpressionAnalyzer::CreateComponent(
    DataRef &&base, const Symbol &component, const semantics::Scope &scope) {
  if (IsAllocatableOrPointer(component) && base.Rank() > 0) { // C919b
    Say("An allocatable or pointer component reference must be applied to a scalar base"_err_en_US);
  }
  if (&component.owner() == &scope) {
    return Component{std::move(base), component};
  }
  if (const Symbol *typeSymbol{scope.GetSymbol()}) {
    if (const Symbol *parentComponent{typeSymbol->GetParentComponent(&scope)}) {
      if (const auto *object{
              parentComponent->detailsIf<semantics::ObjectEntityDetails>()}) {
        if (const auto *parentType{object->type()}) {
          if (const semantics::Scope *parentScope{
                  parentType->derivedTypeSpec().scope()}) {
            return CreateComponent(
                DataRef{Component{std::move(base), *parentComponent}},
                component, *parentScope);
          }
        }
      }
    }
  }
  return std::nullopt;
}

// Derived type component references and type parameter inquiries
MaybeExpr ExpressionAnalyzer::Analyze(const parser::StructureComponent &sc) {
  MaybeExpr base{Analyze(sc.base)};
  Symbol *sym{sc.component.symbol};
  if (!base || !sym || context_.HasError(sym)) {
    return std::nullopt;
  }
  const auto &name{sc.component.source};
  if (auto *dtExpr{UnwrapExpr<Expr<SomeDerived>>(*base)}) {
    const auto *dtSpec{GetDerivedTypeSpec(dtExpr->GetType())};
    if (sym->detailsIf<semantics::TypeParamDetails>()) {
      if (auto *designator{UnwrapExpr<Designator<SomeDerived>>(*dtExpr)}) {
        if (std::optional<DynamicType> dyType{DynamicType::From(*sym)}) {
          if (dyType->category() == TypeCategory::Integer) {
            auto restorer{GetContextualMessages().SetLocation(name)};
            return Fold(ConvertToType(*dyType,
                AsGenericExpr(TypeParamInquiry{
                    IgnoreAnySubscripts(std::move(*designator)), *sym})));
          }
        }
        Say(name, "Type parameter is not INTEGER"_err_en_US);
      } else {
        Say(name,
            "A type parameter inquiry must be applied to "
            "a designator"_err_en_US);
      }
    } else if (!dtSpec || !dtSpec->scope()) {
      CHECK(context_.AnyFatalError() || !foldingContext_.messages().empty());
      return std::nullopt;
    } else if (std::optional<DataRef> dataRef{
                   ExtractDataRef(std::move(*dtExpr))}) {
      auto restorer{GetContextualMessages().SetLocation(name)};
      if (auto component{
              CreateComponent(std::move(*dataRef), *sym, *dtSpec->scope())}) {
        return Designate(DataRef{std::move(*component)});
      } else {
        Say(name, "Component is not in scope of derived TYPE(%s)"_err_en_US,
            dtSpec->typeSymbol().name());
      }
    } else {
      Say(name,
          "Base of component reference must be a data reference"_err_en_US);
    }
  } else if (auto *details{sym->detailsIf<semantics::MiscDetails>()}) {
    // special part-ref: %re, %im, %kind, %len
    // Type errors on the base of %re/%im/%len are detected and
    // reported in name resolution.
    using MiscKind = semantics::MiscDetails::Kind;
    MiscKind kind{details->kind()};
    if (kind == MiscKind::ComplexPartRe || kind == MiscKind::ComplexPartIm) {
      if (auto *zExpr{std::get_if<Expr<SomeComplex>>(&base->u)}) {
        if (std::optional<DataRef> dataRef{ExtractDataRef(*zExpr)}) {
          // Represent %RE/%IM as a designator
          Expr<SomeReal> realExpr{common::visit(
              [&](const auto &z) {
                using PartType = typename ResultType<decltype(z)>::Part;
                auto part{kind == MiscKind::ComplexPartRe
                        ? ComplexPart::Part::RE
                        : ComplexPart::Part::IM};
                return AsCategoryExpr(Designator<PartType>{
                    ComplexPart{std::move(*dataRef), part}});
              },
              zExpr->u)};
          return AsGenericExpr(std::move(realExpr));
        }
      }
    } else if (kind == MiscKind::KindParamInquiry ||
        kind == MiscKind::LenParamInquiry) {
      ActualArgument arg{std::move(*base)};
      SetArgSourceLocation(arg, name);
      return MakeFunctionRef(name, ActualArguments{std::move(arg)});
    } else {
      DIE("unexpected MiscDetails::Kind");
    }
  } else {
    Say(name, "derived type required before component reference"_err_en_US);
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::CoindexedNamedObject &x) {
  if (auto maybeDataRef{ExtractDataRef(Analyze(x.base))}) {
    DataRef *dataRef{&*maybeDataRef};
    std::vector<Subscript> subscripts;
    SymbolVector reversed;
    if (auto *aRef{std::get_if<ArrayRef>(&dataRef->u)}) {
      subscripts = std::move(aRef->subscript());
      reversed.push_back(aRef->GetLastSymbol());
      if (Component *component{aRef->base().UnwrapComponent()}) {
        dataRef = &component->base();
      } else {
        dataRef = nullptr;
      }
    }
    if (dataRef) {
      while (auto *component{std::get_if<Component>(&dataRef->u)}) {
        reversed.push_back(component->GetLastSymbol());
        dataRef = &component->base();
      }
      if (auto *baseSym{std::get_if<SymbolRef>(&dataRef->u)}) {
        reversed.push_back(*baseSym);
      } else {
        Say("Base of coindexed named object has subscripts or cosubscripts"_err_en_US);
      }
    }
    std::vector<Expr<SubscriptInteger>> cosubscripts;
    bool cosubsOk{true};
    for (const auto &cosub :
        std::get<std::list<parser::Cosubscript>>(x.imageSelector.t)) {
      MaybeExpr coex{Analyze(cosub)};
      if (auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(coex)}) {
        cosubscripts.push_back(
            ConvertToType<SubscriptInteger>(std::move(*intExpr)));
      } else {
        cosubsOk = false;
      }
    }
    if (cosubsOk && !reversed.empty()) {
      int numCosubscripts{static_cast<int>(cosubscripts.size())};
      const Symbol &symbol{reversed.front()};
      if (numCosubscripts != symbol.Corank()) {
        Say("'%s' has corank %d, but coindexed reference has %d cosubscripts"_err_en_US,
            symbol.name(), symbol.Corank(), numCosubscripts);
      }
    }
    for (const auto &imageSelSpec :
        std::get<std::list<parser::ImageSelectorSpec>>(x.imageSelector.t)) {
      common::visit(
          common::visitors{
              [&](const auto &x) { Analyze(x.v); },
          },
          imageSelSpec.u);
    }
    // Reverse the chain of symbols so that the base is first and coarray
    // ultimate component is last.
    if (cosubsOk) {
      return Designate(
          DataRef{CoarrayRef{SymbolVector{reversed.crbegin(), reversed.crend()},
              std::move(subscripts), std::move(cosubscripts)}});
    }
  }
  return std::nullopt;
}

int ExpressionAnalyzer::IntegerTypeSpecKind(
    const parser::IntegerTypeSpec &spec) {
  Expr<SubscriptInteger> value{
      AnalyzeKindSelector(TypeCategory::Integer, spec.v)};
  if (auto kind{ToInt64(value)}) {
    return static_cast<int>(*kind);
  }
  SayAt(spec, "Constant INTEGER kind value required here"_err_en_US);
  return GetDefaultKind(TypeCategory::Integer);
}

// Array constructors

// Inverts a collection of generic ArrayConstructorValues<SomeType> that
// all happen to have the same actual type T into one ArrayConstructor<T>.
template <typename T>
ArrayConstructorValues<T> MakeSpecific(
    ArrayConstructorValues<SomeType> &&from) {
  ArrayConstructorValues<T> to;
  for (ArrayConstructorValue<SomeType> &x : from) {
    common::visit(
        common::visitors{
            [&](common::CopyableIndirection<Expr<SomeType>> &&expr) {
              auto *typed{UnwrapExpr<Expr<T>>(expr.value())};
              to.Push(std::move(DEREF(typed)));
            },
            [&](ImpliedDo<SomeType> &&impliedDo) {
              to.Push(ImpliedDo<T>{impliedDo.name(),
                  std::move(impliedDo.lower()), std::move(impliedDo.upper()),
                  std::move(impliedDo.stride()),
                  MakeSpecific<T>(std::move(impliedDo.values()))});
            },
        },
        std::move(x.u));
  }
  return to;
}

class ArrayConstructorContext {
public:
  ArrayConstructorContext(
      ExpressionAnalyzer &c, std::optional<DynamicTypeWithLength> &&t)
      : exprAnalyzer_{c}, type_{std::move(t)} {}

  void Add(const parser::AcValue &);
  MaybeExpr ToExpr();

  // These interfaces allow *this to be used as a type visitor argument to
  // common::SearchTypes() to convert the array constructor to a typed
  // expression in ToExpr().
  using Result = MaybeExpr;
  using Types = AllTypes;
  template <typename T> Result Test() {
    if (type_ && type_->category() == T::category) {
      if constexpr (T::category == TypeCategory::Derived) {
        if (!type_->IsUnlimitedPolymorphic()) {
          return AsMaybeExpr(ArrayConstructor<T>{type_->GetDerivedTypeSpec(),
              MakeSpecific<T>(std::move(values_))});
        }
      } else if (type_->kind() == T::kind) {
        ArrayConstructor<T> result{MakeSpecific<T>(std::move(values_))};
        if constexpr (T::category == TypeCategory::Character) {
          if (auto len{LengthIfGood()}) {
            // The ac-do-variables may be treated as constant expressions,
            // if some conditions on ac-implied-do-control hold (10.1.12 (12)).
            // At the same time, they may be treated as constant expressions
            // only in the context of the ac-implied-do, but setting
            // the character length here may result in complete elimination
            // of the ac-implied-do. For example:
            //   character(10) :: c
            //   ... len([(c(i:i), integer(8)::i = 1,4)])
            // would be evaulated into:
            //   ... int(max(0_8,i-i+1_8),kind=4)
            // with a dangling reference to the ac-do-variable.
            // Prevent this by checking for the ac-do-variable references
            // in the 'len' expression.
            result.set_LEN(std::move(*len));
          }
        }
        return AsMaybeExpr(std::move(result));
      }
    }
    return std::nullopt;
  }

private:
  using ImpliedDoIntType = ResultType<ImpliedDoIndex>;

  std::optional<Expr<SubscriptInteger>> LengthIfGood() const {
    if (type_) {
      auto len{type_->LEN()};
      if (len && IsConstantExpr(*len) && !ContainsAnyImpliedDoIndex(*len)) {
        return len;
      }
    }
    return std::nullopt;
  }
  bool NeedLength() const {
    return type_ && type_->category() == TypeCategory::Character &&
        !LengthIfGood();
  }
  void Push(MaybeExpr &&);
  void Add(const parser::AcValue::Triplet &);
  void Add(const parser::Expr &);
  void Add(const parser::AcImpliedDo &);
  void UnrollConstantImpliedDo(const parser::AcImpliedDo &,
      parser::CharBlock name, std::int64_t lower, std::int64_t upper,
      std::int64_t stride);

  template <int KIND, typename A>
  std::optional<Expr<Type<TypeCategory::Integer, KIND>>> GetSpecificIntExpr(
      const A &x) {
    if (MaybeExpr y{exprAnalyzer_.Analyze(x)}) {
      Expr<SomeInteger> *intExpr{UnwrapExpr<Expr<SomeInteger>>(*y)};
      return Fold(exprAnalyzer_.GetFoldingContext(),
          ConvertToType<Type<TypeCategory::Integer, KIND>>(
              std::move(DEREF(intExpr))));
    }
    return std::nullopt;
  }

  // Nested array constructors all reference the same ExpressionAnalyzer,
  // which represents the nest of active implied DO loop indices.
  ExpressionAnalyzer &exprAnalyzer_;
  std::optional<DynamicTypeWithLength> type_;
  bool explicitType_{type_.has_value()};
  std::optional<std::int64_t> constantLength_;
  ArrayConstructorValues<SomeType> values_;
  std::uint64_t messageDisplayedSet_{0};
};

void ArrayConstructorContext::Push(MaybeExpr &&x) {
  if (!x) {
    return;
  }
  if (!type_) {
    if (auto *boz{std::get_if<BOZLiteralConstant>(&x->u)}) {
      // Treat an array constructor of BOZ as if default integer.
      if (exprAnalyzer_.context().ShouldWarn(
              common::LanguageFeature::BOZAsDefaultInteger)) {
        exprAnalyzer_.Say(
            "BOZ literal in array constructor without explicit type is assumed to be default INTEGER"_port_en_US);
      }
      x = AsGenericExpr(ConvertToKind<TypeCategory::Integer>(
          exprAnalyzer_.GetDefaultKind(TypeCategory::Integer),
          std::move(*boz)));
    }
  }
  std::optional<DynamicType> dyType{x->GetType()};
  if (!dyType) {
    if (auto *boz{std::get_if<BOZLiteralConstant>(&x->u)}) {
      if (!type_) {
        // Treat an array constructor of BOZ as if default integer.
        if (exprAnalyzer_.context().ShouldWarn(
                common::LanguageFeature::BOZAsDefaultInteger)) {
          exprAnalyzer_.Say(
              "BOZ literal in array constructor without explicit type is assumed to be default INTEGER"_port_en_US);
        }
        x = AsGenericExpr(ConvertToKind<TypeCategory::Integer>(
            exprAnalyzer_.GetDefaultKind(TypeCategory::Integer),
            std::move(*boz)));
        dyType = x.value().GetType();
      } else if (auto cast{ConvertToType(*type_, std::move(*x))}) {
        x = std::move(cast);
        dyType = *type_;
      } else {
        if (!(messageDisplayedSet_ & 0x80)) {
          exprAnalyzer_.Say(
              "BOZ literal is not suitable for use in this array constructor"_err_en_US);
          messageDisplayedSet_ |= 0x80;
        }
        return;
      }
    } else { // procedure name, &c.
      if (!(messageDisplayedSet_ & 0x40)) {
        exprAnalyzer_.Say(
            "Item is not suitable for use in an array constructor"_err_en_US);
        messageDisplayedSet_ |= 0x40;
      }
      return;
    }
  } else if (dyType->IsUnlimitedPolymorphic()) {
    if (!(messageDisplayedSet_ & 8)) {
      exprAnalyzer_.Say("Cannot have an unlimited polymorphic value in an "
                        "array constructor"_err_en_US); // C7113
      messageDisplayedSet_ |= 8;
    }
    return;
  } else if (dyType->category() == TypeCategory::Derived &&
      dyType->GetDerivedTypeSpec().typeSymbol().attrs().test(
          semantics::Attr::ABSTRACT)) { // F'2023 C7125
    if (!(messageDisplayedSet_ & 0x200)) {
      exprAnalyzer_.Say(
          "An item whose declared type is ABSTRACT may not appear in an array constructor"_err_en_US);
      messageDisplayedSet_ |= 0x200;
    }
  }
  DynamicTypeWithLength xType{dyType.value()};
  if (Expr<SomeCharacter> * charExpr{UnwrapExpr<Expr<SomeCharacter>>(*x)}) {
    CHECK(xType.category() == TypeCategory::Character);
    xType.length =
        common::visit([](const auto &kc) { return kc.LEN(); }, charExpr->u);
  }
  if (!type_) {
    // If there is no explicit type-spec in an array constructor, the type
    // of the array is the declared type of all of the elements, which must
    // be well-defined and all match.
    // TODO: Possible language extension: use the most general type of
    // the values as the type of a numeric constructed array, convert all
    // of the other values to that type.  Alternative: let the first value
    // determine the type, and convert the others to that type.
    CHECK(!explicitType_);
    type_ = std::move(xType);
    constantLength_ = ToInt64(type_->length);
    values_.Push(std::move(*x));
  } else if (!explicitType_) {
    if (type_->IsTkCompatibleWith(xType) && xType.IsTkCompatibleWith(*type_)) {
      values_.Push(std::move(*x));
      auto xLen{xType.LEN()};
      if (auto thisLen{ToInt64(xLen)}) {
        if (constantLength_) {
          if (exprAnalyzer_.context().ShouldWarn(
                  common::LanguageFeature::DistinctArrayConstructorLengths) &&
              *thisLen != *constantLength_) {
            if (!(messageDisplayedSet_ & 1)) {
              exprAnalyzer_.Say(
                  "Character literal in array constructor without explicit "
                  "type has different length than earlier elements"_port_en_US);
              messageDisplayedSet_ |= 1;
            }
          }
          if (*thisLen > *constantLength_) {
            // Language extension: use the longest literal to determine the
            // length of the array constructor's character elements, not the
            // first, when there is no explicit type.
            *constantLength_ = *thisLen;
            type_->length = std::move(xLen);
          }
        } else {
          constantLength_ = *thisLen;
          type_->length = std::move(xLen);
        }
      } else if (xLen && NeedLength()) {
        type_->length = std::move(xLen);
      }
    } else {
      if (!(messageDisplayedSet_ & 2)) {
        exprAnalyzer_.Say(
            "Values in array constructor must have the same declared type "
            "when no explicit type appears"_err_en_US); // C7110
        messageDisplayedSet_ |= 2;
      }
    }
  } else {
    if (auto cast{ConvertToType(*type_, std::move(*x))}) {
      values_.Push(std::move(*cast));
    } else if (!(messageDisplayedSet_ & 4)) {
      exprAnalyzer_.Say("Value in array constructor of type '%s' could not "
                        "be converted to the type of the array '%s'"_err_en_US,
          x->GetType()->AsFortran(), type_->AsFortran()); // C7111, C7112
      messageDisplayedSet_ |= 4;
    }
  }
}

void ArrayConstructorContext::Add(const parser::AcValue &x) {
  common::visit(
      common::visitors{
          [&](const parser::AcValue::Triplet &triplet) { Add(triplet); },
          [&](const common::Indirection<parser::Expr> &expr) {
            Add(expr.value());
          },
          [&](const common::Indirection<parser::AcImpliedDo> &impliedDo) {
            Add(impliedDo.value());
          },
      },
      x.u);
}

// Transforms l:u(:s) into (_,_=l,u(,s)) with an anonymous index '_'
void ArrayConstructorContext::Add(const parser::AcValue::Triplet &triplet) {
  std::optional<Expr<ImpliedDoIntType>> lower{
      GetSpecificIntExpr<ImpliedDoIntType::kind>(std::get<0>(triplet.t))};
  std::optional<Expr<ImpliedDoIntType>> upper{
      GetSpecificIntExpr<ImpliedDoIntType::kind>(std::get<1>(triplet.t))};
  std::optional<Expr<ImpliedDoIntType>> stride{
      GetSpecificIntExpr<ImpliedDoIntType::kind>(std::get<2>(triplet.t))};
  if (lower && upper) {
    if (!stride) {
      stride = Expr<ImpliedDoIntType>{1};
    }
    if (!type_) {
      type_ = DynamicTypeWithLength{ImpliedDoIntType::GetType()};
    }
    auto v{std::move(values_)};
    parser::CharBlock anonymous;
    Push(Expr<SomeType>{
        Expr<SomeInteger>{Expr<ImpliedDoIntType>{ImpliedDoIndex{anonymous}}}});
    std::swap(v, values_);
    values_.Push(ImpliedDo<SomeType>{anonymous, std::move(*lower),
        std::move(*upper), std::move(*stride), std::move(v)});
  }
}

void ArrayConstructorContext::Add(const parser::Expr &expr) {
  auto restorer{exprAnalyzer_.GetContextualMessages().SetLocation(expr.source)};
  Push(exprAnalyzer_.Analyze(expr));
}

void ArrayConstructorContext::Add(const parser::AcImpliedDo &impliedDo) {
  const auto &control{std::get<parser::AcImpliedDoControl>(impliedDo.t)};
  const auto &bounds{std::get<parser::AcImpliedDoControl::Bounds>(control.t)};
  exprAnalyzer_.Analyze(bounds.name);
  parser::CharBlock name{bounds.name.thing.thing.source};
  const Symbol *symbol{bounds.name.thing.thing.symbol};
  int kind{ImpliedDoIntType::kind};
  if (const auto dynamicType{DynamicType::From(symbol)}) {
    kind = dynamicType->kind();
  }
  std::optional<Expr<ImpliedDoIntType>> lower{
      GetSpecificIntExpr<ImpliedDoIntType::kind>(bounds.lower)};
  std::optional<Expr<ImpliedDoIntType>> upper{
      GetSpecificIntExpr<ImpliedDoIntType::kind>(bounds.upper)};
  if (lower && upper) {
    std::optional<Expr<ImpliedDoIntType>> stride{
        GetSpecificIntExpr<ImpliedDoIntType::kind>(bounds.step)};
    if (!stride) {
      stride = Expr<ImpliedDoIntType>{1};
    }
    if (exprAnalyzer_.AddImpliedDo(name, kind)) {
      // Check for constant bounds; the loop may require complete unrolling
      // of the parse tree if all bounds are constant in order to allow the
      // implied DO loop index to qualify as a constant expression.
      auto cLower{ToInt64(lower)};
      auto cUpper{ToInt64(upper)};
      auto cStride{ToInt64(stride)};
      if (!(messageDisplayedSet_ & 0x10) && cStride && *cStride == 0) {
        exprAnalyzer_.SayAt(bounds.step.value().thing.thing.value().source,
            "The stride of an implied DO loop must not be zero"_err_en_US);
        messageDisplayedSet_ |= 0x10;
      }
      bool isConstant{cLower && cUpper && cStride && *cStride != 0};
      bool isNonemptyConstant{isConstant &&
          ((*cStride > 0 && *cLower <= *cUpper) ||
              (*cStride < 0 && *cLower >= *cUpper))};
      bool isEmpty{isConstant && !isNonemptyConstant};
      bool unrollConstantLoop{false};
      parser::Messages buffer;
      auto saveMessagesDisplayed{messageDisplayedSet_};
      {
        auto messageRestorer{
            exprAnalyzer_.GetContextualMessages().SetMessages(buffer)};
        auto v{std::move(values_)};
        for (const auto &value :
            std::get<std::list<parser::AcValue>>(impliedDo.t)) {
          Add(value);
        }
        std::swap(v, values_);
        if (isNonemptyConstant && buffer.AnyFatalError()) {
          unrollConstantLoop = true;
        } else {
          values_.Push(ImpliedDo<SomeType>{name, std::move(*lower),
              std::move(*upper), std::move(*stride), std::move(v)});
        }
      }
      // F'2023 7.8 p5
      if (!(messageDisplayedSet_ & 0x100) && isEmpty && NeedLength()) {
        exprAnalyzer_.SayAt(name,
            "Array constructor implied DO loop has no iterations and indeterminate character length"_err_en_US);
        messageDisplayedSet_ |= 0x100;
      }
      if (unrollConstantLoop) {
        messageDisplayedSet_ = saveMessagesDisplayed;
        UnrollConstantImpliedDo(impliedDo, name, *cLower, *cUpper, *cStride);
      } else if (auto *messages{
                     exprAnalyzer_.GetContextualMessages().messages()}) {
        messages->Annex(std::move(buffer));
      }
      exprAnalyzer_.RemoveImpliedDo(name);
    } else if (!(messageDisplayedSet_ & 0x20)) {
      exprAnalyzer_.SayAt(name,
          "Implied DO index '%s' is active in a surrounding implied DO loop "
          "and may not have the same name"_err_en_US,
          name); // C7115
      messageDisplayedSet_ |= 0x20;
    }
  }
}

// Fortran considers an implied DO index of an array constructor to be
// a constant expression if the bounds of the implied DO loop are constant.
// Usually this doesn't matter, but if we emitted spurious messages as a
// result of not using constant values for the index while analyzing the
// items, we need to do it again the "hard" way with multiple iterations over
// the parse tree.
void ArrayConstructorContext::UnrollConstantImpliedDo(
    const parser::AcImpliedDo &impliedDo, parser::CharBlock name,
    std::int64_t lower, std::int64_t upper, std::int64_t stride) {
  auto &foldingContext{exprAnalyzer_.GetFoldingContext()};
  auto restorer{exprAnalyzer_.DoNotUseSavedTypedExprs()};
  for (auto &at{foldingContext.StartImpliedDo(name, lower)};
       (stride > 0 && at <= upper) || (stride < 0 && at >= upper);
       at += stride) {
    for (const auto &value :
        std::get<std::list<parser::AcValue>>(impliedDo.t)) {
      Add(value);
    }
  }
  foldingContext.EndImpliedDo(name);
}

MaybeExpr ArrayConstructorContext::ToExpr() {
  return common::SearchTypes(std::move(*this));
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::ArrayConstructor &array) {
  const parser::AcSpec &acSpec{array.v};
  ArrayConstructorContext acContext{*this, AnalyzeTypeSpec(acSpec.type)};
  for (const parser::AcValue &value : acSpec.values) {
    acContext.Add(value);
  }
  return acContext.ToExpr();
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::StructureConstructor &structure) {
  auto &parsedType{std::get<parser::DerivedTypeSpec>(structure.t)};
  parser::Name structureType{std::get<parser::Name>(parsedType.t)};
  parser::CharBlock &typeName{structureType.source};
  if (semantics::Symbol *typeSymbol{structureType.symbol}) {
    if (typeSymbol->has<semantics::DerivedTypeDetails>()) {
      semantics::DerivedTypeSpec dtSpec{typeName, typeSymbol->GetUltimate()};
      if (!CheckIsValidForwardReference(dtSpec)) {
        return std::nullopt;
      }
    }
  }
  if (!parsedType.derivedTypeSpec) {
    return std::nullopt;
  }
  const auto &spec{*parsedType.derivedTypeSpec};
  const Symbol &typeSymbol{spec.typeSymbol()};
  if (!spec.scope() || !typeSymbol.has<semantics::DerivedTypeDetails>()) {
    return std::nullopt; // error recovery
  }
  const semantics::Scope &scope{context_.FindScope(typeName)};
  const semantics::Scope *pureContext{FindPureProcedureContaining(scope)};
  const auto &typeDetails{typeSymbol.get<semantics::DerivedTypeDetails>()};
  const Symbol *parentComponent{typeDetails.GetParentComponent(*spec.scope())};

  if (typeSymbol.attrs().test(semantics::Attr::ABSTRACT)) { // C796
    AttachDeclaration(Say(typeName,
                          "ABSTRACT derived type '%s' may not be used in a "
                          "structure constructor"_err_en_US,
                          typeName),
        typeSymbol); // C7114
  }

  // This iterator traverses all of the components in the derived type and its
  // parents.  The symbols for whole parent components appear after their
  // own components and before the components of the types that extend them.
  // E.g., TYPE :: A; REAL X; END TYPE
  //       TYPE, EXTENDS(A) :: B; REAL Y; END TYPE
  // produces the component list X, A, Y.
  // The order is important below because a structure constructor can
  // initialize X or A by name, but not both.
  auto components{semantics::OrderedComponentIterator{spec}};
  auto nextAnonymous{components.begin()};

  std::set<parser::CharBlock> unavailable;
  bool anyKeyword{false};
  StructureConstructor result{spec};
  bool checkConflicts{true}; // until we hit one
  auto &messages{GetContextualMessages()};

  // NULL() can be a valid component
  auto restorer{AllowNullPointer()};

  for (const auto &component :
      std::get<std::list<parser::ComponentSpec>>(structure.t)) {
    const parser::Expr &expr{
        std::get<parser::ComponentDataSource>(component.t).v.value()};
    parser::CharBlock source{expr.source};
    auto restorer{messages.SetLocation(source)};
    const Symbol *symbol{nullptr};
    MaybeExpr value{Analyze(expr)};
    std::optional<DynamicType> valueType{DynamicType::From(value)};
    if (const auto &kw{std::get<std::optional<parser::Keyword>>(component.t)}) {
      anyKeyword = true;
      source = kw->v.source;
      symbol = kw->v.symbol;
      if (!symbol) {
        // Skip overridden inaccessible parent components in favor of
        // their later overrides.
        for (const Symbol &sym : components) {
          if (sym.name() == source) {
            symbol = &sym;
          }
        }
      }
      if (!symbol) { // C7101
        Say(source,
            "Keyword '%s=' does not name a component of derived type '%s'"_err_en_US,
            source, typeName);
      }
    } else {
      if (anyKeyword) { // C7100
        Say(source,
            "Value in structure constructor lacks a component name"_err_en_US);
        checkConflicts = false; // stem cascade
      }
      // Here's a regrettably common extension of the standard: anonymous
      // initialization of parent components, e.g., T(PT(1)) rather than
      // T(1) or T(PT=PT(1)).
      if (nextAnonymous == components.begin() && parentComponent &&
          valueType == DynamicType::From(*parentComponent) &&
          context().IsEnabled(LanguageFeature::AnonymousParents)) {
        auto iter{
            std::find(components.begin(), components.end(), *parentComponent)};
        if (iter != components.end()) {
          symbol = parentComponent;
          nextAnonymous = ++iter;
          if (context().ShouldWarn(LanguageFeature::AnonymousParents)) {
            Say(source,
                "Whole parent component '%s' in structure "
                "constructor should not be anonymous"_port_en_US,
                symbol->name());
          }
        }
      }
      while (!symbol && nextAnonymous != components.end()) {
        const Symbol &next{*nextAnonymous};
        ++nextAnonymous;
        if (!next.test(Symbol::Flag::ParentComp)) {
          symbol = &next;
        }
      }
      if (!symbol) {
        Say(source, "Unexpected value in structure constructor"_err_en_US);
      }
    }
    if (symbol) {
      const semantics::Scope &innermost{context_.FindScope(expr.source)};
      if (auto msg{CheckAccessibleSymbol(innermost, *symbol)}) {
        Say(expr.source, std::move(*msg));
      }
      if (checkConflicts) {
        auto componentIter{
            std::find(components.begin(), components.end(), *symbol)};
        if (unavailable.find(symbol->name()) != unavailable.cend()) {
          // C797, C798
          Say(source,
              "Component '%s' conflicts with another component earlier in "
              "this structure constructor"_err_en_US,
              symbol->name());
        } else if (symbol->test(Symbol::Flag::ParentComp)) {
          // Make earlier components unavailable once a whole parent appears.
          for (auto it{components.begin()}; it != componentIter; ++it) {
            unavailable.insert(it->name());
          }
        } else {
          // Make whole parent components unavailable after any of their
          // constituents appear.
          for (auto it{componentIter}; it != components.end(); ++it) {
            if (it->test(Symbol::Flag::ParentComp)) {
              unavailable.insert(it->name());
            }
          }
        }
      }
      unavailable.insert(symbol->name());
      if (value) {
        if (symbol->has<semantics::TypeParamDetails>()) {
          Say(expr.source,
              "Type parameter '%s' may not appear as a component of a structure constructor"_err_en_US,
              symbol->name());
        }
        if (!(symbol->has<semantics::ProcEntityDetails>() ||
                symbol->has<semantics::ObjectEntityDetails>())) {
          continue; // recovery
        }
        if (IsPointer(*symbol)) { // C7104, C7105, C1594(4)
          semantics::CheckStructConstructorPointerComponent(
              context_, *symbol, *value, innermost);
          result.Add(*symbol, Fold(std::move(*value)));
          continue;
        }
        if (IsNullPointer(*value)) {
          if (IsAllocatable(*symbol)) {
            if (IsBareNullPointer(&*value)) {
              // NULL() with no arguments allowed by 7.5.10 para 6 for
              // ALLOCATABLE.
              result.Add(*symbol, Expr<SomeType>{NullPointer{}});
              continue;
            }
            if (IsNullObjectPointer(*value)) {
              AttachDeclaration(
                  Say(expr.source,
                      "NULL() with arguments is not standard conforming as the value for allocatable component '%s'"_port_en_US,
                      symbol->name()),
                  *symbol);
              // proceed to check type & shape
            } else {
              AttachDeclaration(
                  Say(expr.source,
                      "A NULL procedure pointer may not be used as the value for component '%s'"_err_en_US,
                      symbol->name()),
                  *symbol);
              continue;
            }
          } else {
            AttachDeclaration(
                Say(expr.source,
                    "A NULL pointer may not be used as the value for component '%s'"_err_en_US,
                    symbol->name()),
                *symbol);
            continue;
          }
        } else if (const Symbol * pointer{FindPointerComponent(*symbol)};
                   pointer && pureContext) { // C1594(4)
          if (const Symbol *
              visible{semantics::FindExternallyVisibleObject(
                  *value, *pureContext)}) {
            Say(expr.source,
                "The externally visible object '%s' may not be used in a pure procedure as the value for component '%s' which has the pointer component '%s'"_err_en_US,
                visible->name(), symbol->name(), pointer->name());
          }
        }
        if (MaybeExpr converted{ConvertToType(*symbol, std::move(*value))}) {
          if (auto componentShape{GetShape(GetFoldingContext(), *symbol)}) {
            if (auto valueShape{GetShape(GetFoldingContext(), *converted)}) {
              if (GetRank(*componentShape) == 0 && GetRank(*valueShape) > 0) {
                AttachDeclaration(
                    Say(expr.source,
                        "Rank-%d array value is not compatible with scalar component '%s'"_err_en_US,
                        GetRank(*valueShape), symbol->name()),
                    *symbol);
              } else {
                auto checked{
                    CheckConformance(messages, *componentShape, *valueShape,
                        CheckConformanceFlags::RightIsExpandableDeferred,
                        "component", "value")};
                if (checked && *checked && GetRank(*componentShape) > 0 &&
                    GetRank(*valueShape) == 0 &&
                    (IsDeferredShape(*symbol) ||
                        !IsExpandableScalar(*converted, GetFoldingContext(),
                            *componentShape, true /*admit PURE call*/))) {
                  AttachDeclaration(
                      Say(expr.source,
                          "Scalar value cannot be expanded to shape of array component '%s'"_err_en_US,
                          symbol->name()),
                      *symbol);
                }
                if (checked.value_or(true)) {
                  result.Add(*symbol, std::move(*converted));
                }
              }
            } else {
              Say(expr.source, "Shape of value cannot be determined"_err_en_US);
            }
          } else {
            AttachDeclaration(
                Say(expr.source,
                    "Shape of component '%s' cannot be determined"_err_en_US,
                    symbol->name()),
                *symbol);
          }
        } else if (auto symType{DynamicType::From(symbol)}) {
          if (IsAllocatable(*symbol) && symType->IsUnlimitedPolymorphic() &&
              valueType) {
            // ok
          } else if (valueType) {
            AttachDeclaration(
                Say(expr.source,
                    "Value in structure constructor of type '%s' is "
                    "incompatible with component '%s' of type '%s'"_err_en_US,
                    valueType->AsFortran(), symbol->name(),
                    symType->AsFortran()),
                *symbol);
          } else {
            AttachDeclaration(
                Say(expr.source,
                    "Value in structure constructor is incompatible with "
                    "component '%s' of type %s"_err_en_US,
                    symbol->name(), symType->AsFortran()),
                *symbol);
          }
        }
      }
    }
  }

  // Ensure that unmentioned component objects have default initializers.
  for (const Symbol &symbol : components) {
    if (!symbol.test(Symbol::Flag::ParentComp) &&
        unavailable.find(symbol.name()) == unavailable.cend()) {
      if (IsAllocatable(symbol)) {
        // Set all remaining allocatables to explicit NULL().
        result.Add(symbol, Expr<SomeType>{NullPointer{}});
      } else {
        const auto *object{symbol.detailsIf<semantics::ObjectEntityDetails>()};
        if (object && object->init()) {
          result.Add(symbol, common::Clone(*object->init()));
        } else if (IsPointer(symbol)) {
          result.Add(symbol, Expr<SomeType>{NullPointer{}});
        } else if (object) { // C799
          AttachDeclaration(Say(typeName,
                                "Structure constructor lacks a value for "
                                "component '%s'"_err_en_US,
                                symbol.name()),
              symbol);
        }
      }
    }
  }

  return AsMaybeExpr(Expr<SomeDerived>{std::move(result)});
}

static std::optional<parser::CharBlock> GetPassName(
    const semantics::Symbol &proc) {
  return common::visit(
      [](const auto &details) {
        if constexpr (std::is_base_of_v<semantics::WithPassArg,
                          std::decay_t<decltype(details)>>) {
          return details.passName();
        } else {
          return std::optional<parser::CharBlock>{};
        }
      },
      proc.details());
}

static std::optional<int> GetPassIndex(const Symbol &proc) {
  CHECK(!proc.attrs().test(semantics::Attr::NOPASS));
  std::optional<parser::CharBlock> passName{GetPassName(proc)};
  const auto *interface {
    semantics::FindInterface(proc)
  };
  if (!passName || !interface) {
    return 0; // first argument is passed-object
  }
  const auto &subp{interface->get<semantics::SubprogramDetails>()};
  int index{0};
  for (const auto *arg : subp.dummyArgs()) {
    if (arg && arg->name() == passName) {
      return index;
    }
    ++index;
  }
  return std::nullopt;
}

// Injects an expression into an actual argument list as the "passed object"
// for a type-bound procedure reference that is not NOPASS.  Adds an
// argument keyword if possible, but not when the passed object goes
// before a positional argument.
// e.g., obj%tbp(x) -> tbp(obj,x).
static void AddPassArg(ActualArguments &actuals, const Expr<SomeDerived> &expr,
    const Symbol &component, bool isPassedObject = true) {
  if (component.attrs().test(semantics::Attr::NOPASS)) {
    return;
  }
  std::optional<int> passIndex{GetPassIndex(component)};
  if (!passIndex) {
    return; // error recovery
  }
  auto iter{actuals.begin()};
  int at{0};
  while (iter < actuals.end() && at < *passIndex) {
    if (*iter && (*iter)->keyword()) {
      iter = actuals.end();
      break;
    }
    ++iter;
    ++at;
  }
  ActualArgument passed{AsGenericExpr(common::Clone(expr))};
  passed.set_isPassedObject(isPassedObject);
  if (iter == actuals.end()) {
    if (auto passName{GetPassName(component)}) {
      passed.set_keyword(*passName);
    }
  }
  actuals.emplace(iter, std::move(passed));
}

// Return the compile-time resolution of a procedure binding, if possible.
static const Symbol *GetBindingResolution(
    const std::optional<DynamicType> &baseType, const Symbol &component) {
  const auto *binding{component.detailsIf<semantics::ProcBindingDetails>()};
  if (!binding) {
    return nullptr;
  }
  if (!component.attrs().test(semantics::Attr::NON_OVERRIDABLE) &&
      (!baseType || baseType->IsPolymorphic())) {
    return nullptr;
  }
  return &binding->symbol();
}

auto ExpressionAnalyzer::AnalyzeProcedureComponentRef(
    const parser::ProcComponentRef &pcr, ActualArguments &&arguments,
    bool isSubroutine) -> std::optional<CalleeAndArguments> {
  const parser::StructureComponent &sc{pcr.v.thing};
  if (MaybeExpr base{Analyze(sc.base)}) {
    if (const Symbol *sym{sc.component.symbol}) {
      if (context_.HasError(sym)) {
        return std::nullopt;
      }
      if (!IsProcedure(*sym)) {
        AttachDeclaration(
            Say(sc.component.source, "'%s' is not a procedure"_err_en_US,
                sc.component.source),
            *sym);
        return std::nullopt;
      }
      if (auto *dtExpr{UnwrapExpr<Expr<SomeDerived>>(*base)}) {
        if (sym->has<semantics::GenericDetails>()) {
          const Symbol &generic{*sym};
          auto dyType{dtExpr->GetType()};
          AdjustActuals adjustment{
              [&](const Symbol &proc, ActualArguments &actuals) {
                if (!proc.attrs().test(semantics::Attr::NOPASS)) {
                  AddPassArg(actuals, std::move(*dtExpr), proc);
                }
                return true;
              }};
          auto pair{
              ResolveGeneric(generic, arguments, adjustment, isSubroutine)};
          sym = pair.first;
          if (!sym) {
            EmitGenericResolutionError(generic, pair.second, isSubroutine);
            return std::nullopt;
          }
          // re-resolve the name to the specific binding
          CHECK(sym->has<semantics::ProcBindingDetails>());
          // Use the most recent override of a binding, respecting
          // the rule that inaccessible bindings may not be overridden
          // outside their module.  Fortran doesn't allow a PUBLIC
          // binding to be overridden by a PRIVATE one.
          CHECK(dyType && dyType->category() == TypeCategory::Derived &&
              !dyType->IsUnlimitedPolymorphic());
          if (const Symbol *
              latest{DEREF(dyType->GetDerivedTypeSpec().typeSymbol().scope())
                         .FindComponent(sym->name())}) {
            if (sym->attrs().test(semantics::Attr::PRIVATE)) {
              const auto *bindingModule{FindModuleContaining(generic.owner())};
              const Symbol *s{latest};
              while (s && FindModuleContaining(s->owner()) != bindingModule) {
                if (const auto *parent{s->owner().GetDerivedTypeParent()}) {
                  s = parent->FindComponent(sym->name());
                } else {
                  s = nullptr;
                }
              }
              if (s && !s->attrs().test(semantics::Attr::PRIVATE)) {
                // The latest override in the same module as the binding
                // is public, so it can be overridden.
              } else {
                latest = s;
              }
            }
            if (latest) {
              sym = latest;
            }
          }
          sc.component.symbol = const_cast<Symbol *>(sym);
        }
        std::optional<DataRef> dataRef{ExtractDataRef(std::move(*dtExpr))};
        if (dataRef && !CheckDataRef(*dataRef)) {
          return std::nullopt;
        }
        if (dataRef && dataRef->Rank() > 0) {
          if (sym->has<semantics::ProcBindingDetails>() &&
              sym->attrs().test(semantics::Attr::NOPASS)) {
            // C1529 seems unnecessary and most compilers don't enforce it.
            AttachDeclaration(
                Say(sc.component.source,
                    "Base of NOPASS type-bound procedure reference should be scalar"_port_en_US),
                *sym);
          } else if (IsProcedurePointer(*sym)) { // C919
            Say(sc.component.source,
                "Base of procedure component reference must be scalar"_err_en_US);
          }
        }
        if (const Symbol *resolution{
                GetBindingResolution(dtExpr->GetType(), *sym)}) {
          AddPassArg(arguments, std::move(*dtExpr), *sym, false);
          return CalleeAndArguments{
              ProcedureDesignator{*resolution}, std::move(arguments)};
        } else if (dataRef.has_value()) {
          if (sym->attrs().test(semantics::Attr::NOPASS)) {
            return CalleeAndArguments{
                ProcedureDesignator{Component{std::move(*dataRef), *sym}},
                std::move(arguments)};
          } else {
            AddPassArg(arguments,
                Expr<SomeDerived>{Designator<SomeDerived>{std::move(*dataRef)}},
                *sym);
            return CalleeAndArguments{
                ProcedureDesignator{*sym}, std::move(arguments)};
          }
        }
      }
      Say(sc.component.source,
          "Base of procedure component reference is not a derived-type object"_err_en_US);
    }
  }
  CHECK(context_.AnyFatalError());
  return std::nullopt;
}

// Can actual be argument associated with dummy?
static bool CheckCompatibleArgument(bool isElemental,
    const ActualArgument &actual, const characteristics::DummyArgument &dummy) {
  const auto *expr{actual.UnwrapExpr()};
  return common::visit(
      common::visitors{
          [&](const characteristics::DummyDataObject &x) {
            if (x.attrs.test(characteristics::DummyDataObject::Attr::Pointer) &&
                IsBareNullPointer(expr)) {
              // NULL() without MOLD= is compatible with any dummy data pointer
              // but cannot be allowed to lead to ambiguity.
              return true;
            } else if (!isElemental && actual.Rank() != x.type.Rank() &&
                !x.type.attrs().test(
                    characteristics::TypeAndShape::Attr::AssumedRank) &&
                !x.ignoreTKR.test(common::IgnoreTKR::Rank)) {
              return false;
            } else if (auto actualType{actual.GetType()}) {
              return x.type.type().IsTkCompatibleWith(*actualType, x.ignoreTKR);
            }
            return false;
          },
          [&](const characteristics::DummyProcedure &) {
            return expr && IsProcedurePointerTarget(*expr);
          },
          [&](const characteristics::AlternateReturn &) {
            return actual.isAlternateReturn();
          },
      },
      dummy.u);
}

// Are the actual arguments compatible with the dummy arguments of procedure?
static bool CheckCompatibleArguments(
    const characteristics::Procedure &procedure,
    const ActualArguments &actuals) {
  bool isElemental{procedure.IsElemental()};
  const auto &dummies{procedure.dummyArguments};
  CHECK(dummies.size() == actuals.size());
  for (std::size_t i{0}; i < dummies.size(); ++i) {
    const characteristics::DummyArgument &dummy{dummies[i]};
    const std::optional<ActualArgument> &actual{actuals[i]};
    if (actual && !CheckCompatibleArgument(isElemental, *actual, dummy)) {
      return false;
    }
  }
  return true;
}

// Handles a forward reference to a module function from what must
// be a specification expression.  Return false if the symbol is
// an invalid forward reference.
bool ExpressionAnalyzer::ResolveForward(const Symbol &symbol) {
  if (context_.HasError(symbol)) {
    return false;
  }
  if (const auto *details{
          symbol.detailsIf<semantics::SubprogramNameDetails>()}) {
    if (details->kind() == semantics::SubprogramKind::Module) {
      // If this symbol is still a SubprogramNameDetails, we must be
      // checking a specification expression in a sibling module
      // procedure.  Resolve its names now so that its interface
      // is known.
      semantics::ResolveSpecificationParts(context_, symbol);
      if (symbol.has<semantics::SubprogramNameDetails>()) {
        // When the symbol hasn't had its details updated, we must have
        // already been in the process of resolving the function's
        // specification part; but recursive function calls are not
        // allowed in specification parts (10.1.11 para 5).
        Say("The module function '%s' may not be referenced recursively in a specification expression"_err_en_US,
            symbol.name());
        context_.SetError(symbol);
        return false;
      }
    } else if (inStmtFunctionDefinition_) {
      semantics::ResolveSpecificationParts(context_, symbol);
      CHECK(symbol.has<semantics::SubprogramDetails>());
    } else { // 10.1.11 para 4
      Say("The internal function '%s' may not be referenced in a specification expression"_err_en_US,
          symbol.name());
      context_.SetError(symbol);
      return false;
    }
  }
  return true;
}

// Resolve a call to a generic procedure with given actual arguments.
// adjustActuals is called on procedure bindings to handle pass arg.
std::pair<const Symbol *, bool> ExpressionAnalyzer::ResolveGeneric(
    const Symbol &symbol, const ActualArguments &actuals,
    const AdjustActuals &adjustActuals, bool isSubroutine,
    bool mightBeStructureConstructor) {
  const Symbol *elemental{nullptr}; // matching elemental specific proc
  const Symbol *nonElemental{nullptr}; // matching non-elemental specific
  const Symbol &ultimate{symbol.GetUltimate()};
  // Check for a match with an explicit INTRINSIC
  if (ultimate.attrs().test(semantics::Attr::INTRINSIC)) {
    parser::Messages buffer;
    auto restorer{foldingContext_.messages().SetMessages(buffer)};
    ActualArguments localActuals{actuals};
    if (context_.intrinsics().Probe(
            CallCharacteristics{ultimate.name().ToString(), isSubroutine},
            localActuals, foldingContext_) &&
        !buffer.AnyFatalError()) {
      return {&ultimate, false};
    }
  }
  if (const auto *details{ultimate.detailsIf<semantics::GenericDetails>()}) {
    for (const Symbol &specific0 : details->specificProcs()) {
      const Symbol &specific{BypassGeneric(specific0)};
      if (isSubroutine != !IsFunction(specific)) {
        continue;
      }
      if (!ResolveForward(specific)) {
        continue;
      }
      if (std::optional<characteristics::Procedure> procedure{
              characteristics::Procedure::Characterize(
                  ProcedureDesignator{specific}, context_.foldingContext())}) {
        ActualArguments localActuals{actuals};
        if (specific.has<semantics::ProcBindingDetails>()) {
          if (!adjustActuals.value()(specific, localActuals)) {
            continue;
          }
        }
        if (semantics::CheckInterfaceForGeneric(*procedure, localActuals,
                context_, false /* no integer conversions */) &&
            CheckCompatibleArguments(*procedure, localActuals)) {
          if ((procedure->IsElemental() && elemental) ||
              (!procedure->IsElemental() && nonElemental)) {
            // 16.9.144(6): a bare NULL() is not allowed as an actual
            // argument to a generic procedure if the specific procedure
            // cannot be unambiguously distinguished
            // Underspecified external procedure actual arguments can
            // also lead to ambiguity.
            return {nullptr, true /* due to ambiguity */};
          }
          if (!procedure->IsElemental()) {
            // takes priority over elemental match
            nonElemental = &specific;
          } else {
            elemental = &specific;
          }
        }
      }
    }
    if (nonElemental) {
      return {&AccessSpecific(symbol, *nonElemental), false};
    } else if (elemental) {
      return {&AccessSpecific(symbol, *elemental), false};
    }
    // Check parent derived type
    if (const auto *parentScope{symbol.owner().GetDerivedTypeParent()}) {
      if (const Symbol *extended{parentScope->FindComponent(symbol.name())}) {
        auto pair{ResolveGeneric(
            *extended, actuals, adjustActuals, isSubroutine, false)};
        if (pair.first) {
          return pair;
        }
      }
    }
    if (mightBeStructureConstructor && details->derivedType()) {
      return {details->derivedType(), false};
    }
  }
  // Check for generic or explicit INTRINSIC of the same name in outer scopes.
  // See 15.5.5.2 for details.
  if (!symbol.owner().IsGlobal() && !symbol.owner().IsDerivedType()) {
    for (const std::string &n : GetAllNames(context_, symbol.name())) {
      if (const Symbol *outer{symbol.owner().parent().FindSymbol(n)}) {
        auto pair{ResolveGeneric(*outer, actuals, adjustActuals, isSubroutine,
            mightBeStructureConstructor)};
        if (pair.first) {
          return pair;
        }
      }
    }
  }
  return {nullptr, false};
}

const Symbol &ExpressionAnalyzer::AccessSpecific(
    const Symbol &originalGeneric, const Symbol &specific) {
  if (const auto *hosted{
          originalGeneric.detailsIf<semantics::HostAssocDetails>()}) {
    return AccessSpecific(hosted->symbol(), specific);
  } else if (const auto *used{
                 originalGeneric.detailsIf<semantics::UseDetails>()}) {
    const auto &scope{originalGeneric.owner()};
    if (auto iter{scope.find(specific.name())}; iter != scope.end()) {
      if (const auto *useDetails{
              iter->second->detailsIf<semantics::UseDetails>()}) {
        const Symbol &usedSymbol{useDetails->symbol()};
        const auto *usedGeneric{
            usedSymbol.detailsIf<semantics::GenericDetails>()};
        if (&usedSymbol == &specific ||
            (usedGeneric && usedGeneric->specific() == &specific)) {
          return specific;
        }
      }
    }
    // Create a renaming USE of the specific procedure.
    auto rename{context_.SaveTempName(
        used->symbol().owner().GetName().value().ToString() + "$" +
        specific.owner().GetName().value().ToString() + "$" +
        specific.name().ToString())};
    return *const_cast<semantics::Scope &>(scope)
                .try_emplace(rename, specific.attrs(),
                    semantics::UseDetails{rename, specific})
                .first->second;
  } else {
    return specific;
  }
}

void ExpressionAnalyzer::EmitGenericResolutionError(
    const Symbol &symbol, bool dueToAmbiguity, bool isSubroutine) {
  Say(dueToAmbiguity
          ? "One or more actual arguments to the generic procedure '%s' matched multiple specific procedures, perhaps due to use of NULL() without MOLD= or an actual procedure with an implicit interface"_err_en_US
          : semantics::IsGenericDefinedOp(symbol)
          ? "No specific procedure of generic operator '%s' matches the actual arguments"_err_en_US
          : isSubroutine
          ? "No specific subroutine of generic '%s' matches the actual arguments"_err_en_US
          : "No specific function of generic '%s' matches the actual arguments"_err_en_US,
      symbol.name());
}

auto ExpressionAnalyzer::GetCalleeAndArguments(
    const parser::ProcedureDesignator &pd, ActualArguments &&arguments,
    bool isSubroutine, bool mightBeStructureConstructor)
    -> std::optional<CalleeAndArguments> {
  return common::visit(common::visitors{
                           [&](const parser::Name &name) {
                             return GetCalleeAndArguments(name,
                                 std::move(arguments), isSubroutine,
                                 mightBeStructureConstructor);
                           },
                           [&](const parser::ProcComponentRef &pcr) {
                             return AnalyzeProcedureComponentRef(
                                 pcr, std::move(arguments), isSubroutine);
                           },
                       },
      pd.u);
}

auto ExpressionAnalyzer::GetCalleeAndArguments(const parser::Name &name,
    ActualArguments &&arguments, bool isSubroutine,
    bool mightBeStructureConstructor) -> std::optional<CalleeAndArguments> {
  const Symbol *symbol{name.symbol};
  if (context_.HasError(symbol)) {
    return std::nullopt; // also handles null symbol
  }
  const Symbol &ultimate{DEREF(symbol).GetUltimate()};
  CheckForBadRecursion(name.source, ultimate);
  bool dueToAmbiguity{false};
  bool isGenericInterface{ultimate.has<semantics::GenericDetails>()};
  bool isExplicitIntrinsic{ultimate.attrs().test(semantics::Attr::INTRINSIC)};
  const Symbol *resolution{nullptr};
  if (isGenericInterface || isExplicitIntrinsic) {
    ExpressionAnalyzer::AdjustActuals noAdjustment;
    auto pair{ResolveGeneric(*symbol, arguments, noAdjustment, isSubroutine,
        mightBeStructureConstructor)};
    resolution = pair.first;
    dueToAmbiguity = pair.second;
    if (resolution) {
      if (context_.GetPPCBuiltinsScope() &&
          resolution->name().ToString().rfind("__ppc_", 0) == 0) {
        semantics::CheckPPCIntrinsic(
            *symbol, *resolution, arguments, GetFoldingContext());
      }
      // re-resolve name to the specific procedure
      name.symbol = const_cast<Symbol *>(resolution);
    }
  } else if (IsProcedure(ultimate) &&
      ultimate.attrs().test(semantics::Attr::ABSTRACT)) {
    Say("Abstract procedure interface '%s' may not be referenced"_err_en_US,
        name.source);
  } else {
    resolution = symbol;
  }
  if (!resolution || resolution->attrs().test(semantics::Attr::INTRINSIC)) {
    auto name{resolution ? resolution->name() : ultimate.name()};
    if (std::optional<SpecificCall> specificCall{context_.intrinsics().Probe(
            CallCharacteristics{name.ToString(), isSubroutine}, arguments,
            GetFoldingContext())}) {
      CheckBadExplicitType(*specificCall, *symbol);
      return CalleeAndArguments{
          ProcedureDesignator{std::move(specificCall->specificIntrinsic)},
          std::move(specificCall->arguments)};
    } else {
      if (isGenericInterface) {
        EmitGenericResolutionError(*symbol, dueToAmbiguity, isSubroutine);
      }
      return std::nullopt;
    }
  }
  if (resolution->GetUltimate().has<semantics::DerivedTypeDetails>()) {
    if (mightBeStructureConstructor) {
      return CalleeAndArguments{
          semantics::SymbolRef{*resolution}, std::move(arguments)};
    }
  } else if (IsProcedure(*resolution)) {
    return CalleeAndArguments{
        ProcedureDesignator{*resolution}, std::move(arguments)};
  }
  if (!context_.HasError(*resolution)) {
    AttachDeclaration(
        Say(name.source, "'%s' is not a callable procedure"_err_en_US,
            name.source),
        *resolution);
  }
  return std::nullopt;
}

// Fortran 2018 expressly states (8.2 p3) that any declared type for a
// generic intrinsic function "has no effect" on the result type of a
// call to that intrinsic.  So one can declare "character*8 cos" and
// still get a real result from "cos(1.)".  This is a dangerous feature,
// especially since implementations are free to extend their sets of
// intrinsics, and in doing so might clash with a name in a program.
// So we emit a warning in this situation, and perhaps it should be an
// error -- any correctly working program can silence the message by
// simply deleting the pointless type declaration.
void ExpressionAnalyzer::CheckBadExplicitType(
    const SpecificCall &call, const Symbol &intrinsic) {
  if (intrinsic.GetUltimate().GetType()) {
    const auto &procedure{call.specificIntrinsic.characteristics.value()};
    if (const auto &result{procedure.functionResult}) {
      if (const auto *typeAndShape{result->GetTypeAndShape()}) {
        if (auto declared{
                typeAndShape->Characterize(intrinsic, GetFoldingContext())}) {
          if (!declared->type().IsTkCompatibleWith(typeAndShape->type())) {
            if (auto *msg{Say(
                    "The result type '%s' of the intrinsic function '%s' is not the explicit declared type '%s'"_warn_en_US,
                    typeAndShape->AsFortran(), intrinsic.name(),
                    declared->AsFortran())}) {
              msg->Attach(intrinsic.name(),
                  "Ignored declaration of intrinsic function '%s'"_en_US,
                  intrinsic.name());
            }
          }
        }
      }
    }
  }
}

void ExpressionAnalyzer::CheckForBadRecursion(
    parser::CharBlock callSite, const semantics::Symbol &proc) {
  if (const auto *scope{proc.scope()}) {
    if (scope->sourceRange().Contains(callSite)) {
      parser::Message *msg{nullptr};
      if (proc.attrs().test(semantics::Attr::NON_RECURSIVE)) { // 15.6.2.1(3)
        msg = Say("NON_RECURSIVE procedure '%s' cannot call itself"_err_en_US,
            callSite);
      } else if (IsAssumedLengthCharacter(proc) && IsExternal(proc)) {
        // TODO: Also catch assumed PDT type parameters
        msg = Say( // 15.6.2.1(3)
            "Assumed-length CHARACTER(*) function '%s' cannot call itself"_err_en_US,
            callSite);
      } else if (FindCUDADeviceContext(scope)) {
        msg = Say(
            "Device subprogram '%s' cannot call itself"_err_en_US, callSite);
      }
      AttachDeclaration(msg, proc);
    }
  }
}

template <typename A> static const Symbol *AssumedTypeDummy(const A &x) {
  if (const auto *designator{
          std::get_if<common::Indirection<parser::Designator>>(&x.u)}) {
    if (const auto *dataRef{
            std::get_if<parser::DataRef>(&designator->value().u)}) {
      if (const auto *name{std::get_if<parser::Name>(&dataRef->u)}) {
        return AssumedTypeDummy(*name);
      }
    }
  }
  return nullptr;
}
template <>
const Symbol *AssumedTypeDummy<parser::Name>(const parser::Name &name) {
  if (const Symbol *symbol{name.symbol}) {
    if (const auto *type{symbol->GetType()}) {
      if (type->category() == semantics::DeclTypeSpec::TypeStar) {
        return symbol;
      }
    }
  }
  return nullptr;
}
template <typename A>
static const Symbol *AssumedTypePointerOrAllocatableDummy(const A &object) {
  // It is illegal for allocatable of pointer objects to be TYPE(*), but at that
  // point it is is not guaranteed that it has been checked the object has
  // POINTER or ALLOCATABLE attribute, so do not assume nullptr can be directly
  // returned.
  return common::visit(
      common::visitors{
          [&](const parser::StructureComponent &x) {
            return AssumedTypeDummy(x.component);
          },
          [&](const parser::Name &x) { return AssumedTypeDummy(x); },
      },
      object.u);
}
template <>
const Symbol *AssumedTypeDummy<parser::AllocateObject>(
    const parser::AllocateObject &x) {
  return AssumedTypePointerOrAllocatableDummy(x);
}
template <>
const Symbol *AssumedTypeDummy<parser::PointerObject>(
    const parser::PointerObject &x) {
  return AssumedTypePointerOrAllocatableDummy(x);
}

bool ExpressionAnalyzer::CheckIsValidForwardReference(
    const semantics::DerivedTypeSpec &dtSpec) {
  if (dtSpec.IsForwardReferenced()) {
    Say("Cannot construct value for derived type '%s' "
        "before it is defined"_err_en_US,
        dtSpec.name());
    return false;
  }
  return true;
}

std::optional<Chevrons> ExpressionAnalyzer::AnalyzeChevrons(
    const parser::CallStmt &call) {
  Chevrons result;
  auto checkLaunchArg{[&](const Expr<SomeType> &expr, const char *which) {
    if (auto dyType{expr.GetType()}) {
      if (dyType->category() == TypeCategory::Integer) {
        return true;
      }
      if (dyType->category() == TypeCategory::Derived &&
          !dyType->IsPolymorphic() &&
          IsBuiltinDerivedType(&dyType->GetDerivedTypeSpec(), "dim3")) {
        return true;
      }
    }
    Say("Kernel launch %s parameter must be either integer or TYPE(dim3)"_err_en_US,
        which);
    return false;
  }};
  if (const auto &chevrons{call.chevrons}) {
    if (auto expr{Analyze(std::get<0>(chevrons->t))};
        expr && checkLaunchArg(*expr, "grid")) {
      result.emplace_back(*expr);
    } else {
      return std::nullopt;
    }
    if (auto expr{Analyze(std::get<1>(chevrons->t))};
        expr && checkLaunchArg(*expr, "block")) {
      result.emplace_back(*expr);
    } else {
      return std::nullopt;
    }
    if (const auto &maybeExpr{std::get<2>(chevrons->t)}) {
      if (auto expr{Analyze(*maybeExpr)}) {
        result.emplace_back(*expr);
      } else {
        return std::nullopt;
      }
    }
    if (const auto &maybeExpr{std::get<3>(chevrons->t)}) {
      if (auto expr{Analyze(*maybeExpr)}) {
        result.emplace_back(*expr);
      } else {
        return std::nullopt;
      }
    }
  }
  return std::move(result);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::FunctionReference &funcRef,
    std::optional<parser::StructureConstructor> *structureConstructor) {
  const parser::Call &call{funcRef.v};
  auto restorer{GetContextualMessages().SetLocation(funcRef.source)};
  ArgumentAnalyzer analyzer{*this, funcRef.source, true /* isProcedureCall */};
  for (const auto &arg : std::get<std::list<parser::ActualArgSpec>>(call.t)) {
    analyzer.Analyze(arg, false /* not subroutine call */);
  }
  if (analyzer.fatalErrors()) {
    return std::nullopt;
  }
  bool mightBeStructureConstructor{structureConstructor != nullptr};
  if (std::optional<CalleeAndArguments> callee{GetCalleeAndArguments(
          std::get<parser::ProcedureDesignator>(call.t), analyzer.GetActuals(),
          false /* not subroutine */, mightBeStructureConstructor)}) {
    if (auto *proc{std::get_if<ProcedureDesignator>(&callee->u)}) {
      return MakeFunctionRef(
          funcRef.source, std::move(*proc), std::move(callee->arguments));
    }
    CHECK(std::holds_alternative<semantics::SymbolRef>(callee->u));
    const Symbol &symbol{*std::get<semantics::SymbolRef>(callee->u)};
    if (mightBeStructureConstructor) {
      // Structure constructor misparsed as function reference?
      const auto &designator{std::get<parser::ProcedureDesignator>(call.t)};
      if (const auto *name{std::get_if<parser::Name>(&designator.u)}) {
        semantics::Scope &scope{context_.FindScope(name->source)};
        semantics::DerivedTypeSpec dtSpec{name->source, symbol.GetUltimate()};
        if (!CheckIsValidForwardReference(dtSpec)) {
          return std::nullopt;
        }
        const semantics::DeclTypeSpec &type{
            semantics::FindOrInstantiateDerivedType(scope, std::move(dtSpec))};
        auto &mutableRef{const_cast<parser::FunctionReference &>(funcRef)};
        *structureConstructor =
            mutableRef.ConvertToStructureConstructor(type.derivedTypeSpec());
        return Analyze(structureConstructor->value());
      }
    }
    if (!context_.HasError(symbol)) {
      AttachDeclaration(
          Say("'%s' is called like a function but is not a procedure"_err_en_US,
              symbol.name()),
          symbol);
      context_.SetError(symbol);
    }
  }
  return std::nullopt;
}

static bool HasAlternateReturns(const evaluate::ActualArguments &args) {
  for (const auto &arg : args) {
    if (arg && arg->isAlternateReturn()) {
      return true;
    }
  }
  return false;
}

void ExpressionAnalyzer::Analyze(const parser::CallStmt &callStmt) {
  const parser::Call &call{callStmt.call};
  auto restorer{GetContextualMessages().SetLocation(callStmt.source)};
  ArgumentAnalyzer analyzer{*this, callStmt.source, true /* isProcedureCall */};
  const auto &actualArgList{std::get<std::list<parser::ActualArgSpec>>(call.t)};
  for (const auto &arg : actualArgList) {
    analyzer.Analyze(arg, true /* is subroutine call */);
  }
  auto chevrons{AnalyzeChevrons(callStmt)};
  if (!analyzer.fatalErrors() && chevrons) {
    if (std::optional<CalleeAndArguments> callee{
            GetCalleeAndArguments(std::get<parser::ProcedureDesignator>(call.t),
                analyzer.GetActuals(), true /* subroutine */)}) {
      ProcedureDesignator *proc{std::get_if<ProcedureDesignator>(&callee->u)};
      CHECK(proc);
      bool isKernel{false};
      if (const Symbol * procSym{proc->GetSymbol()}) {
        const Symbol &ultimate{procSym->GetUltimate()};
        if (const auto *subpDetails{
                ultimate.detailsIf<semantics::SubprogramDetails>()}) {
          if (auto attrs{subpDetails->cudaSubprogramAttrs()}) {
            isKernel = *attrs == common::CUDASubprogramAttrs::Global ||
                *attrs == common::CUDASubprogramAttrs::Grid_Global;
          }
        } else if (const auto *procDetails{
                       ultimate.detailsIf<semantics::ProcEntityDetails>()}) {
          isKernel = procDetails->isCUDAKernel();
        }
        if (isKernel && chevrons->empty()) {
          Say("'%s' is a kernel subroutine and must be called with kernel launch parameters in chevrons"_err_en_US,
              procSym->name());
        }
      }
      if (!isKernel && !chevrons->empty()) {
        Say("Kernel launch parameters in chevrons may not be used unless calling a kernel subroutine"_err_en_US);
      }
      if (CheckCall(callStmt.source, *proc, callee->arguments)) {
        callStmt.typedCall.Reset(
            new ProcedureRef{std::move(*proc), std::move(callee->arguments),
                HasAlternateReturns(callee->arguments)},
            ProcedureRef::Deleter);
        DEREF(callStmt.typedCall.get()).set_chevrons(std::move(*chevrons));
        return;
      }
    }
    if (!context_.AnyFatalError()) {
      std::string buf;
      llvm::raw_string_ostream dump{buf};
      parser::DumpTree(dump, callStmt);
      Say("Internal error: Expression analysis failed on CALL statement: %s"_err_en_US,
          dump.str());
    }
  }
}

const Assignment *ExpressionAnalyzer::Analyze(const parser::AssignmentStmt &x) {
  if (!x.typedAssignment) {
    ArgumentAnalyzer analyzer{*this};
    const auto &variable{std::get<parser::Variable>(x.t)};
    analyzer.Analyze(variable);
    analyzer.Analyze(std::get<parser::Expr>(x.t));
    std::optional<Assignment> assignment;
    if (!analyzer.fatalErrors()) {
      auto restorer{GetContextualMessages().SetLocation(variable.GetSource())};
      std::optional<ProcedureRef> procRef{analyzer.TryDefinedAssignment()};
      if (!procRef) {
        analyzer.CheckForNullPointer(
            "in a non-pointer intrinsic assignment statement");
        const Expr<SomeType> &lhs{analyzer.GetExpr(0)};
        if (auto dyType{lhs.GetType()};
            dyType && dyType->IsPolymorphic()) { // 10.2.1.2p1(1)
          const Symbol *lastWhole0{UnwrapWholeSymbolOrComponentDataRef(lhs)};
          const Symbol *lastWhole{
              lastWhole0 ? &lastWhole0->GetUltimate() : nullptr};
          if (!lastWhole || !IsAllocatable(*lastWhole)) {
            Say("Left-hand side of assignment may not be polymorphic unless assignment is to an entire allocatable"_err_en_US);
          } else if (evaluate::IsCoarray(*lastWhole)) {
            Say("Left-hand side of assignment may not be polymorphic if it is a coarray"_err_en_US);
          }
        }
      }
      assignment.emplace(analyzer.MoveExpr(0), analyzer.MoveExpr(1));
      if (procRef) {
        assignment->u = std::move(*procRef);
      }
    }
    x.typedAssignment.Reset(new GenericAssignmentWrapper{std::move(assignment)},
        GenericAssignmentWrapper::Deleter);
  }
  return common::GetPtrFromOptional(x.typedAssignment->v);
}

const Assignment *ExpressionAnalyzer::Analyze(
    const parser::PointerAssignmentStmt &x) {
  if (!x.typedAssignment) {
    MaybeExpr lhs{Analyze(std::get<parser::DataRef>(x.t))};
    MaybeExpr rhs;
    {
      auto restorer{AllowNullPointer()};
      rhs = Analyze(std::get<parser::Expr>(x.t));
    }
    if (!lhs || !rhs) {
      x.typedAssignment.Reset(
          new GenericAssignmentWrapper{}, GenericAssignmentWrapper::Deleter);
    } else {
      Assignment assignment{std::move(*lhs), std::move(*rhs)};
      common::visit(
          common::visitors{
              [&](const std::list<parser::BoundsRemapping> &list) {
                Assignment::BoundsRemapping bounds;
                for (const auto &elem : list) {
                  auto lower{AsSubscript(Analyze(std::get<0>(elem.t)))};
                  auto upper{AsSubscript(Analyze(std::get<1>(elem.t)))};
                  if (lower && upper) {
                    bounds.emplace_back(
                        Fold(std::move(*lower)), Fold(std::move(*upper)));
                  }
                }
                assignment.u = std::move(bounds);
              },
              [&](const std::list<parser::BoundsSpec> &list) {
                Assignment::BoundsSpec bounds;
                for (const auto &bound : list) {
                  if (auto lower{AsSubscript(Analyze(bound.v))}) {
                    bounds.emplace_back(Fold(std::move(*lower)));
                  }
                }
                assignment.u = std::move(bounds);
              },
          },
          std::get<parser::PointerAssignmentStmt::Bounds>(x.t).u);
      x.typedAssignment.Reset(
          new GenericAssignmentWrapper{std::move(assignment)},
          GenericAssignmentWrapper::Deleter);
    }
  }
  return common::GetPtrFromOptional(x.typedAssignment->v);
}

static bool IsExternalCalledImplicitly(
    parser::CharBlock callSite, const ProcedureDesignator &proc) {
  if (const auto *symbol{proc.GetSymbol()}) {
    return symbol->has<semantics::SubprogramDetails>() &&
        symbol->owner().IsGlobal() &&
        (!symbol->scope() /*ENTRY*/ ||
            !symbol->scope()->sourceRange().Contains(callSite));
  } else {
    return false;
  }
}

std::optional<characteristics::Procedure> ExpressionAnalyzer::CheckCall(
    parser::CharBlock callSite, const ProcedureDesignator &proc,
    ActualArguments &arguments) {
  bool treatExternalAsImplicit{IsExternalCalledImplicitly(callSite, proc)};
  const Symbol *procSymbol{proc.GetSymbol()};
  std::optional<characteristics::Procedure> chars;
  if (procSymbol && procSymbol->has<semantics::ProcEntityDetails>() &&
      procSymbol->owner().IsGlobal()) {
    // Unknown global external, implicit interface; assume
    // characteristics from the actual arguments, and check
    // for consistency with other references.
    chars = characteristics::Procedure::FromActuals(
        proc, arguments, context_.foldingContext());
    if (chars && procSymbol) {
      // Ensure calls over implicit interfaces are consistent
      auto name{procSymbol->name()};
      if (auto iter{implicitInterfaces_.find(name)};
          iter != implicitInterfaces_.end()) {
        std::string whyNot;
        if (!chars->IsCompatibleWith(iter->second.second, &whyNot)) {
          if (auto *msg{Say(callSite,
                  "Reference to the procedure '%s' has an implicit interface that is distinct from another reference: %s"_warn_en_US,
                  name, whyNot)}) {
            msg->Attach(
                iter->second.first, "previous reference to '%s'"_en_US, name);
          }
        }
      } else {
        implicitInterfaces_.insert(
            std::make_pair(name, std::make_pair(callSite, *chars)));
      }
    }
  }
  if (!chars) {
    chars = characteristics::Procedure::Characterize(
        proc, context_.foldingContext());
  }
  bool ok{true};
  if (chars) {
    if (treatExternalAsImplicit && !chars->CanBeCalledViaImplicitInterface()) {
      Say(callSite,
          "References to the procedure '%s' require an explicit interface"_err_en_US,
          DEREF(procSymbol).name());
    }
    const SpecificIntrinsic *specificIntrinsic{proc.GetSpecificIntrinsic()};
    bool procIsDummy{procSymbol && IsDummy(*procSymbol)};
    if (chars->functionResult &&
        chars->functionResult->IsAssumedLengthCharacter() &&
        !specificIntrinsic && !procIsDummy) {
      Say(callSite,
          "Assumed-length character function must be defined with a length to be called"_err_en_US);
    }
    ok &= semantics::CheckArguments(*chars, arguments, context_,
        context_.FindScope(callSite), treatExternalAsImplicit,
        specificIntrinsic);
  }
  if (procSymbol && !IsPureProcedure(*procSymbol)) {
    if (const semantics::Scope *
        pure{semantics::FindPureProcedureContaining(
            context_.FindScope(callSite))}) {
      Say(callSite,
          "Procedure '%s' referenced in pure subprogram '%s' must be pure too"_err_en_US,
          procSymbol->name(), DEREF(pure->symbol()).name());
    }
  }
  if (ok && !treatExternalAsImplicit && procSymbol &&
      !(chars && chars->HasExplicitInterface())) {
    if (const Symbol *global{FindGlobal(*procSymbol)};
        global && global != procSymbol && IsProcedure(*global)) {
      // Check a known global definition behind a local interface
      if (auto globalChars{characteristics::Procedure::Characterize(
              *global, context_.foldingContext())}) {
        semantics::CheckArguments(*globalChars, arguments, context_,
            context_.FindScope(callSite), true,
            nullptr /*not specific intrinsic*/);
      }
    }
  }
  return chars;
}

// Unary operations

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Parentheses &x) {
  if (MaybeExpr operand{Analyze(x.v.value())}) {
    if (const semantics::Symbol *symbol{GetLastSymbol(*operand)}) {
      if (const semantics::Symbol *result{FindFunctionResult(*symbol)}) {
        if (semantics::IsProcedurePointer(*result)) {
          Say("A function reference that returns a procedure "
              "pointer may not be parenthesized"_err_en_US); // C1003
        }
      }
    }
    return Parenthesize(std::move(*operand));
  }
  return std::nullopt;
}

static MaybeExpr NumericUnaryHelper(ExpressionAnalyzer &context,
    NumericOperator opr, const parser::Expr::IntrinsicUnary &x) {
  ArgumentAnalyzer analyzer{context};
  analyzer.Analyze(x.v);
  if (!analyzer.fatalErrors()) {
    if (analyzer.IsIntrinsicNumeric(opr)) {
      analyzer.CheckForNullPointer();
      if (opr == NumericOperator::Add) {
        return analyzer.MoveExpr(0);
      } else {
        return Negation(context.GetContextualMessages(), analyzer.MoveExpr(0));
      }
    } else {
      return analyzer.TryDefinedOp(AsFortran(opr),
          "Operand of unary %s must be numeric; have %s"_err_en_US);
    }
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::UnaryPlus &x) {
  return NumericUnaryHelper(*this, NumericOperator::Add, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Negate &x) {
  if (const auto *litConst{
          std::get_if<parser::LiteralConstant>(&x.v.value().u)}) {
    if (const auto *intConst{
            std::get_if<parser::IntLiteralConstant>(&litConst->u)}) {
      return Analyze(*intConst, true);
    }
  }
  return NumericUnaryHelper(*this, NumericOperator::Subtract, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::NOT &x) {
  ArgumentAnalyzer analyzer{*this};
  analyzer.Analyze(x.v);
  if (!analyzer.fatalErrors()) {
    if (analyzer.IsIntrinsicLogical()) {
      analyzer.CheckForNullPointer();
      return AsGenericExpr(
          LogicalNegation(std::get<Expr<SomeLogical>>(analyzer.MoveExpr(0).u)));
    } else {
      return analyzer.TryDefinedOp(LogicalOperator::Not,
          "Operand of %s must be LOGICAL; have %s"_err_en_US);
    }
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::PercentLoc &x) {
  // Represent %LOC() exactly as if it had been a call to the LOC() extension
  // intrinsic function.
  // Use the actual source for the name of the call for error reporting.
  std::optional<ActualArgument> arg;
  if (const Symbol *assumedTypeDummy{AssumedTypeDummy(x.v.value())}) {
    arg = ActualArgument{ActualArgument::AssumedType{*assumedTypeDummy}};
  } else if (MaybeExpr argExpr{Analyze(x.v.value())}) {
    arg = ActualArgument{std::move(*argExpr)};
  } else {
    return std::nullopt;
  }
  parser::CharBlock at{GetContextualMessages().at()};
  CHECK(at.size() >= 4);
  parser::CharBlock loc{at.begin() + 1, 3};
  CHECK(loc == "loc");
  return MakeFunctionRef(loc, ActualArguments{std::move(*arg)});
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::DefinedUnary &x) {
  const auto &name{std::get<parser::DefinedOpName>(x.t).v};
  ArgumentAnalyzer analyzer{*this, name.source};
  analyzer.Analyze(std::get<1>(x.t));
  return analyzer.TryDefinedOp(name.source.ToString().c_str(),
      "No operator %s defined for %s"_err_en_US, true);
}

// Binary (dyadic) operations

template <template <typename> class OPR>
MaybeExpr NumericBinaryHelper(ExpressionAnalyzer &context, NumericOperator opr,
    const parser::Expr::IntrinsicBinary &x) {
  ArgumentAnalyzer analyzer{context};
  analyzer.Analyze(std::get<0>(x.t));
  analyzer.Analyze(std::get<1>(x.t));
  if (!analyzer.fatalErrors()) {
    if (analyzer.IsIntrinsicNumeric(opr)) {
      analyzer.CheckForNullPointer();
      analyzer.CheckConformance();
      return NumericOperation<OPR>(context.GetContextualMessages(),
          analyzer.MoveExpr(0), analyzer.MoveExpr(1),
          context.GetDefaultKind(TypeCategory::Real));
    } else {
      return analyzer.TryDefinedOp(AsFortran(opr),
          "Operands of %s must be numeric; have %s and %s"_err_en_US);
    }
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Power &x) {
  return NumericBinaryHelper<Power>(*this, NumericOperator::Power, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Multiply &x) {
  return NumericBinaryHelper<Multiply>(*this, NumericOperator::Multiply, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Divide &x) {
  return NumericBinaryHelper<Divide>(*this, NumericOperator::Divide, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Add &x) {
  return NumericBinaryHelper<Add>(*this, NumericOperator::Add, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Subtract &x) {
  return NumericBinaryHelper<Subtract>(*this, NumericOperator::Subtract, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::Expr::ComplexConstructor &z) {
  return AnalyzeComplex(Analyze(std::get<0>(z.t).value()),
      Analyze(std::get<1>(z.t).value()), "complex constructor");
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Concat &x) {
  ArgumentAnalyzer analyzer{*this};
  analyzer.Analyze(std::get<0>(x.t));
  analyzer.Analyze(std::get<1>(x.t));
  if (!analyzer.fatalErrors()) {
    if (analyzer.IsIntrinsicConcat()) {
      analyzer.CheckForNullPointer();
      return common::visit(
          [&](auto &&x, auto &&y) -> MaybeExpr {
            using T = ResultType<decltype(x)>;
            if constexpr (std::is_same_v<T, ResultType<decltype(y)>>) {
              return AsGenericExpr(Concat<T::kind>{std::move(x), std::move(y)});
            } else {
              DIE("different types for intrinsic concat");
            }
          },
          std::move(std::get<Expr<SomeCharacter>>(analyzer.MoveExpr(0).u).u),
          std::move(std::get<Expr<SomeCharacter>>(analyzer.MoveExpr(1).u).u));
    } else {
      return analyzer.TryDefinedOp("//",
          "Operands of %s must be CHARACTER with the same kind; have %s and %s"_err_en_US);
    }
  }
  return std::nullopt;
}

// The Name represents a user-defined intrinsic operator.
// If the actuals match one of the specific procedures, return a function ref.
// Otherwise report the error in messages.
MaybeExpr ExpressionAnalyzer::AnalyzeDefinedOp(
    const parser::Name &name, ActualArguments &&actuals) {
  if (auto callee{GetCalleeAndArguments(name, std::move(actuals))}) {
    CHECK(std::holds_alternative<ProcedureDesignator>(callee->u));
    return MakeFunctionRef(name.source,
        std::move(std::get<ProcedureDesignator>(callee->u)),
        std::move(callee->arguments));
  } else {
    return std::nullopt;
  }
}

MaybeExpr RelationHelper(ExpressionAnalyzer &context, RelationalOperator opr,
    const parser::Expr::IntrinsicBinary &x) {
  ArgumentAnalyzer analyzer{context};
  analyzer.Analyze(std::get<0>(x.t));
  analyzer.Analyze(std::get<1>(x.t));
  if (!analyzer.fatalErrors()) {
    std::optional<DynamicType> leftType{analyzer.GetType(0)};
    std::optional<DynamicType> rightType{analyzer.GetType(1)};
    analyzer.ConvertBOZ(leftType, 0, rightType);
    analyzer.ConvertBOZ(rightType, 1, leftType);
    if (leftType && rightType &&
        analyzer.IsIntrinsicRelational(opr, *leftType, *rightType)) {
      analyzer.CheckForNullPointer("as a relational operand");
      return AsMaybeExpr(Relate(context.GetContextualMessages(), opr,
          analyzer.MoveExpr(0), analyzer.MoveExpr(1)));
    } else {
      return analyzer.TryDefinedOp(opr,
          leftType && leftType->category() == TypeCategory::Logical &&
                  rightType && rightType->category() == TypeCategory::Logical
              ? "LOGICAL operands must be compared using .EQV. or .NEQV."_err_en_US
              : "Operands of %s must have comparable types; have %s and %s"_err_en_US);
    }
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::LT &x) {
  return RelationHelper(*this, RelationalOperator::LT, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::LE &x) {
  return RelationHelper(*this, RelationalOperator::LE, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::EQ &x) {
  return RelationHelper(*this, RelationalOperator::EQ, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::NE &x) {
  return RelationHelper(*this, RelationalOperator::NE, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::GE &x) {
  return RelationHelper(*this, RelationalOperator::GE, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::GT &x) {
  return RelationHelper(*this, RelationalOperator::GT, x);
}

MaybeExpr LogicalBinaryHelper(ExpressionAnalyzer &context, LogicalOperator opr,
    const parser::Expr::IntrinsicBinary &x) {
  ArgumentAnalyzer analyzer{context};
  analyzer.Analyze(std::get<0>(x.t));
  analyzer.Analyze(std::get<1>(x.t));
  if (!analyzer.fatalErrors()) {
    if (analyzer.IsIntrinsicLogical()) {
      analyzer.CheckForNullPointer("as a logical operand");
      return AsGenericExpr(BinaryLogicalOperation(opr,
          std::get<Expr<SomeLogical>>(analyzer.MoveExpr(0).u),
          std::get<Expr<SomeLogical>>(analyzer.MoveExpr(1).u)));
    } else {
      return analyzer.TryDefinedOp(
          opr, "Operands of %s must be LOGICAL; have %s and %s"_err_en_US);
    }
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::AND &x) {
  return LogicalBinaryHelper(*this, LogicalOperator::And, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::OR &x) {
  return LogicalBinaryHelper(*this, LogicalOperator::Or, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::EQV &x) {
  return LogicalBinaryHelper(*this, LogicalOperator::Eqv, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::NEQV &x) {
  return LogicalBinaryHelper(*this, LogicalOperator::Neqv, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::DefinedBinary &x) {
  const auto &name{std::get<parser::DefinedOpName>(x.t).v};
  ArgumentAnalyzer analyzer{*this, name.source};
  analyzer.Analyze(std::get<1>(x.t));
  analyzer.Analyze(std::get<2>(x.t));
  return analyzer.TryDefinedOp(name.source.ToString().c_str(),
      "No operator %s defined for %s and %s"_err_en_US, true);
}

// Returns true if a parsed function reference should be converted
// into an array element reference.
static bool CheckFuncRefToArrayElement(semantics::SemanticsContext &context,
    const parser::FunctionReference &funcRef) {
  // Emit message if the function reference fix will end up an array element
  // reference with no subscripts, or subscripts on a scalar, because it will
  // not be possible to later distinguish in expressions between an empty
  // subscript list due to bad subscripts error recovery or because the
  // user did not put any.
  auto &proc{std::get<parser::ProcedureDesignator>(funcRef.v.t)};
  const auto *name{std::get_if<parser::Name>(&proc.u)};
  if (!name) {
    name = &std::get<parser::ProcComponentRef>(proc.u).v.thing.component;
  }
  if (!name->symbol) {
    return false;
  } else if (name->symbol->Rank() == 0) {
    if (const Symbol *function{
            semantics::IsFunctionResultWithSameNameAsFunction(*name->symbol)}) {
      auto &msg{context.Say(funcRef.source,
          function->flags().test(Symbol::Flag::StmtFunction)
              ? "Recursive call to statement function '%s' is not allowed"_err_en_US
              : "Recursive call to '%s' requires a distinct RESULT in its declaration"_err_en_US,
          name->source)};
      AttachDeclaration(&msg, *function);
      name->symbol = const_cast<Symbol *>(function);
    }
    return false;
  } else {
    if (std::get<std::list<parser::ActualArgSpec>>(funcRef.v.t).empty()) {
      auto &msg{context.Say(funcRef.source,
          "Reference to array '%s' with empty subscript list"_err_en_US,
          name->source)};
      if (name->symbol) {
        AttachDeclaration(&msg, *name->symbol);
      }
    }
    return true;
  }
}

// Converts, if appropriate, an original misparse of ambiguous syntax like
// A(1) as a function reference into an array reference.
// Misparsed structure constructors are detected elsewhere after generic
// function call resolution fails.
template <typename... A>
static void FixMisparsedFunctionReference(
    semantics::SemanticsContext &context, const std::variant<A...> &constU) {
  // The parse tree is updated in situ when resolving an ambiguous parse.
  using uType = std::decay_t<decltype(constU)>;
  auto &u{const_cast<uType &>(constU)};
  if (auto *func{
          std::get_if<common::Indirection<parser::FunctionReference>>(&u)}) {
    parser::FunctionReference &funcRef{func->value()};
    // Ensure that there are no argument keywords
    for (const auto &arg :
        std::get<std::list<parser::ActualArgSpec>>(funcRef.v.t)) {
      if (std::get<std::optional<parser::Keyword>>(arg.t)) {
        return;
      }
    }
    auto &proc{std::get<parser::ProcedureDesignator>(funcRef.v.t)};
    if (Symbol *origSymbol{
            common::visit(common::visitors{
                              [&](parser::Name &name) { return name.symbol; },
                              [&](parser::ProcComponentRef &pcr) {
                                return pcr.v.thing.component.symbol;
                              },
                          },
                proc.u)}) {
      Symbol &symbol{origSymbol->GetUltimate()};
      if (symbol.has<semantics::ObjectEntityDetails>() ||
          symbol.has<semantics::AssocEntityDetails>()) {
        // Note that expression in AssocEntityDetails cannot be a procedure
        // pointer as per C1105 so this cannot be a function reference.
        if constexpr (common::HasMember<common::Indirection<parser::Designator>,
                          uType>) {
          if (CheckFuncRefToArrayElement(context, funcRef)) {
            u = common::Indirection{funcRef.ConvertToArrayElementRef()};
          }
        } else {
          DIE("can't fix misparsed function as array reference");
        }
      }
    }
  }
}

// Common handling of parse tree node types that retain the
// representation of the analyzed expression.
template <typename PARSED>
MaybeExpr ExpressionAnalyzer::ExprOrVariable(
    const PARSED &x, parser::CharBlock source) {
  auto restorer{GetContextualMessages().SetLocation(source)};
  if constexpr (std::is_same_v<PARSED, parser::Expr> ||
      std::is_same_v<PARSED, parser::Variable>) {
    FixMisparsedFunctionReference(context_, x.u);
  }
  if (AssumedTypeDummy(x)) { // C710
    Say("TYPE(*) dummy argument may only be used as an actual argument"_err_en_US);
    ResetExpr(x);
    return std::nullopt;
  }
  MaybeExpr result;
  if constexpr (common::HasMember<parser::StructureConstructor,
                    std::decay_t<decltype(x.u)>> &&
      common::HasMember<common::Indirection<parser::FunctionReference>,
          std::decay_t<decltype(x.u)>>) {
    if (const auto *funcRef{
            std::get_if<common::Indirection<parser::FunctionReference>>(
                &x.u)}) {
      // Function references in Exprs might turn out to be misparsed structure
      // constructors; we have to try generic procedure resolution
      // first to be sure.
      std::optional<parser::StructureConstructor> ctor;
      result = Analyze(funcRef->value(), &ctor);
      if (result && ctor) {
        // A misparsed function reference is really a structure
        // constructor.  Repair the parse tree in situ.
        const_cast<PARSED &>(x).u = std::move(*ctor);
      }
    } else {
      result = Analyze(x.u);
    }
  } else {
    result = Analyze(x.u);
  }
  if (result) {
    if constexpr (std::is_same_v<PARSED, parser::Expr>) {
      if (!isNullPointerOk_ && IsNullPointer(*result)) {
        Say(source,
            "NULL() may not be used as an expression in this context"_err_en_US);
      }
    }
    SetExpr(x, Fold(std::move(*result)));
    return x.typedExpr->v;
  } else {
    ResetExpr(x);
    if (!context_.AnyFatalError()) {
      std::string buf;
      llvm::raw_string_ostream dump{buf};
      parser::DumpTree(dump, x);
      Say("Internal error: Expression analysis failed on: %s"_err_en_US,
          dump.str());
    }
    return std::nullopt;
  }
}

// This is an optional preliminary pass over parser::Expr subtrees.
// Given an expression tree, iteratively traverse it in a bottom-up order
// to analyze all of its subexpressions.  A later normal top-down analysis
// will then be able to use the results that will have been saved in the
// parse tree without having to recurse deeply.  This technique keeps
// absurdly deep expression parse trees from causing the analyzer to overflow
// its stack.
MaybeExpr ExpressionAnalyzer::IterativelyAnalyzeSubexpressions(
    const parser::Expr &top) {
  std::vector<const parser::Expr *> queue, finish;
  queue.push_back(&top);
  do {
    const parser::Expr &expr{*queue.back()};
    queue.pop_back();
    if (!expr.typedExpr) {
      const parser::Expr::IntrinsicUnary *unary{nullptr};
      const parser::Expr::IntrinsicBinary *binary{nullptr};
      common::visit(
          [&unary, &binary](auto &y) {
            if constexpr (std::is_convertible_v<decltype(&y),
                              decltype(unary)>) {
              // Don't evaluate a constant operand to Negate
              if (!std::holds_alternative<parser::LiteralConstant>(
                      y.v.value().u)) {
                unary = &y;
              }
            } else if constexpr (std::is_convertible_v<decltype(&y),
                                     decltype(binary)>) {
              binary = &y;
            }
          },
          expr.u);
      if (unary) {
        queue.push_back(&unary->v.value());
      } else if (binary) {
        queue.push_back(&std::get<0>(binary->t).value());
        queue.push_back(&std::get<1>(binary->t).value());
      }
      finish.push_back(&expr);
    }
  } while (!queue.empty());
  // Analyze the collected subexpressions in bottom-up order.
  // On an error, bail out and leave partial results in place.
  MaybeExpr result;
  for (auto riter{finish.rbegin()}; riter != finish.rend(); ++riter) {
    const parser::Expr &expr{**riter};
    result = ExprOrVariable(expr, expr.source);
    if (!result) {
      return result;
    }
  }
  return result; // last value was from analysis of "top"
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr &expr) {
  bool wasIterativelyAnalyzing{iterativelyAnalyzingSubexpressions_};
  MaybeExpr result;
  if (useSavedTypedExprs_) {
    if (expr.typedExpr) {
      return expr.typedExpr->v;
    }
    if (!wasIterativelyAnalyzing) {
      iterativelyAnalyzingSubexpressions_ = true;
      result = IterativelyAnalyzeSubexpressions(expr);
    }
  }
  if (!result) {
    result = ExprOrVariable(expr, expr.source);
  }
  iterativelyAnalyzingSubexpressions_ = wasIterativelyAnalyzing;
  return result;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Variable &variable) {
  if (useSavedTypedExprs_ && variable.typedExpr) {
    return variable.typedExpr->v;
  }
  return ExprOrVariable(variable, variable.GetSource());
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Selector &selector) {
  if (const auto *var{std::get_if<parser::Variable>(&selector.u)}) {
    if (!useSavedTypedExprs_ || !var->typedExpr) {
      parser::CharBlock source{var->GetSource()};
      auto restorer{GetContextualMessages().SetLocation(source)};
      FixMisparsedFunctionReference(context_, var->u);
      if (const auto *funcRef{
              std::get_if<common::Indirection<parser::FunctionReference>>(
                  &var->u)}) {
        // A Selector that parsed as a Variable might turn out during analysis
        // to actually be a structure constructor.  In that case, repair the
        // Variable parse tree node into an Expr
        std::optional<parser::StructureConstructor> ctor;
        if (MaybeExpr result{Analyze(funcRef->value(), &ctor)}) {
          if (ctor) {
            auto &writable{const_cast<parser::Selector &>(selector)};
            writable.u = parser::Expr{std::move(*ctor)};
            auto &expr{std::get<parser::Expr>(writable.u)};
            expr.source = source;
            SetExpr(expr, Fold(std::move(*result)));
            return expr.typedExpr->v;
          } else {
            SetExpr(*var, Fold(std::move(*result)));
            return var->typedExpr->v;
          }
        } else {
          ResetExpr(*var);
          if (context_.AnyFatalError()) {
            return std::nullopt;
          }
        }
      }
    }
  }
  // Not a Variable -> FunctionReference; handle normally as Variable or Expr
  return Analyze(selector.u);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::DataStmtConstant &x) {
  auto restorer{common::ScopedSet(inDataStmtConstant_, true)};
  return ExprOrVariable(x, x.source);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::AllocateObject &x) {
  return ExprOrVariable(x, parser::FindSourceLocation(x));
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::PointerObject &x) {
  return ExprOrVariable(x, parser::FindSourceLocation(x));
}

Expr<SubscriptInteger> ExpressionAnalyzer::AnalyzeKindSelector(
    TypeCategory category,
    const std::optional<parser::KindSelector> &selector) {
  int defaultKind{GetDefaultKind(category)};
  if (!selector) {
    return Expr<SubscriptInteger>{defaultKind};
  }
  return common::visit(
      common::visitors{
          [&](const parser::ScalarIntConstantExpr &x) {
            if (MaybeExpr kind{Analyze(x)}) {
              if (std::optional<std::int64_t> code{ToInt64(*kind)}) {
                if (CheckIntrinsicKind(category, *code)) {
                  return Expr<SubscriptInteger>{*code};
                }
              } else if (auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(*kind)}) {
                return ConvertToType<SubscriptInteger>(std::move(*intExpr));
              }
            }
            return Expr<SubscriptInteger>{defaultKind};
          },
          [&](const parser::KindSelector::StarSize &x) {
            std::intmax_t size = x.v;
            if (!CheckIntrinsicSize(category, size)) {
              size = defaultKind;
            } else if (category == TypeCategory::Complex) {
              size /= 2;
            }
            return Expr<SubscriptInteger>{size};
          },
      },
      selector->u);
}

int ExpressionAnalyzer::GetDefaultKind(common::TypeCategory category) {
  return context_.GetDefaultKind(category);
}

DynamicType ExpressionAnalyzer::GetDefaultKindOfType(
    common::TypeCategory category) {
  return {category, GetDefaultKind(category)};
}

bool ExpressionAnalyzer::CheckIntrinsicKind(
    TypeCategory category, std::int64_t kind) {
  if (foldingContext_.targetCharacteristics().IsTypeEnabled(
          category, kind)) { // C712, C714, C715, C727
    return true;
  } else if (foldingContext_.targetCharacteristics().CanSupportType(
                 category, kind)) {
    Say("%s(KIND=%jd) is not an enabled type for this targe"_warn_en_US,
        ToUpperCase(EnumToString(category)), kind);
    return true;
  } else {
    Say("%s(KIND=%jd) is not a supported type"_err_en_US,
        ToUpperCase(EnumToString(category)), kind);
    return false;
  }
}

bool ExpressionAnalyzer::CheckIntrinsicSize(
    TypeCategory category, std::int64_t size) {
  std::int64_t kind{size};
  if (category == TypeCategory::Complex) {
    // COMPLEX*16 == COMPLEX(KIND=8)
    if (size % 2 == 0) {
      kind = size / 2;
    } else {
      Say("COMPLEX*%jd is not a supported type"_err_en_US, size);
      return false;
    }
  }
  if (foldingContext_.targetCharacteristics().IsTypeEnabled(
          category, kind)) { // C712, C714, C715, C727
    return true;
  } else if (foldingContext_.targetCharacteristics().CanSupportType(
                 category, kind)) {
    Say("%s*%jd is not an enabled type for this target"_warn_en_US,
        ToUpperCase(EnumToString(category)), size);
    return true;
  } else {
    Say("%s*%jd is not a supported type"_err_en_US,
        ToUpperCase(EnumToString(category)), size);
    return false;
  }
}

bool ExpressionAnalyzer::AddImpliedDo(parser::CharBlock name, int kind) {
  return impliedDos_.insert(std::make_pair(name, kind)).second;
}

void ExpressionAnalyzer::RemoveImpliedDo(parser::CharBlock name) {
  auto iter{impliedDos_.find(name)};
  if (iter != impliedDos_.end()) {
    impliedDos_.erase(iter);
  }
}

std::optional<int> ExpressionAnalyzer::IsImpliedDo(
    parser::CharBlock name) const {
  auto iter{impliedDos_.find(name)};
  if (iter != impliedDos_.cend()) {
    return {iter->second};
  } else {
    return std::nullopt;
  }
}

bool ExpressionAnalyzer::EnforceTypeConstraint(parser::CharBlock at,
    const MaybeExpr &result, TypeCategory category, bool defaultKind) {
  if (result) {
    if (auto type{result->GetType()}) {
      if (type->category() != category) { // C885
        Say(at, "Must have %s type, but is %s"_err_en_US,
            ToUpperCase(EnumToString(category)),
            ToUpperCase(type->AsFortran()));
        return false;
      } else if (defaultKind) {
        int kind{context_.GetDefaultKind(category)};
        if (type->kind() != kind) {
          Say(at, "Must have default kind(%d) of %s type, but is %s"_err_en_US,
              kind, ToUpperCase(EnumToString(category)),
              ToUpperCase(type->AsFortran()));
          return false;
        }
      }
    } else {
      Say(at, "Must have %s type, but is typeless"_err_en_US,
          ToUpperCase(EnumToString(category)));
      return false;
    }
  }
  return true;
}

MaybeExpr ExpressionAnalyzer::MakeFunctionRef(parser::CharBlock callSite,
    ProcedureDesignator &&proc, ActualArguments &&arguments) {
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&proc.u)}) {
    if (intrinsic->characteristics.value().attrs.test(
            characteristics::Procedure::Attr::NullPointer) &&
        arguments.empty()) {
      return Expr<SomeType>{NullPointer{}};
    }
  }
  if (const Symbol *symbol{proc.GetSymbol()}) {
    if (!ResolveForward(*symbol)) {
      return std::nullopt;
    }
  }
  if (auto chars{CheckCall(callSite, proc, arguments)}) {
    if (chars->functionResult) {
      const auto &result{*chars->functionResult};
      ProcedureRef procRef{std::move(proc), std::move(arguments)};
      if (result.IsProcedurePointer()) {
        return Expr<SomeType>{std::move(procRef)};
      } else {
        // Not a procedure pointer, so type and shape are known.
        return TypedWrapper<FunctionRef, ProcedureRef>(
            DEREF(result.GetTypeAndShape()).type(), std::move(procRef));
      }
    } else {
      Say("Function result characteristics are not known"_err_en_US);
    }
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::MakeFunctionRef(
    parser::CharBlock intrinsic, ActualArguments &&arguments) {
  if (std::optional<SpecificCall> specificCall{
          context_.intrinsics().Probe(CallCharacteristics{intrinsic.ToString()},
              arguments, GetFoldingContext())}) {
    return MakeFunctionRef(intrinsic,
        ProcedureDesignator{std::move(specificCall->specificIntrinsic)},
        std::move(specificCall->arguments));
  } else {
    return std::nullopt;
  }
}

MaybeExpr ExpressionAnalyzer::AnalyzeComplex(
    MaybeExpr &&re, MaybeExpr &&im, const char *what) {
  if (re && re->Rank() > 0) {
    Say("Real part of %s is not scalar"_port_en_US, what);
  }
  if (im && im->Rank() > 0) {
    Say("Imaginary part of %s is not scalar"_port_en_US, what);
  }
  if (re && im) {
    ConformabilityCheck(GetContextualMessages(), *re, *im);
  }
  return AsMaybeExpr(ConstructComplex(GetContextualMessages(), std::move(re),
      std::move(im), GetDefaultKind(TypeCategory::Real)));
}

void ArgumentAnalyzer::Analyze(const parser::Variable &x) {
  source_.ExtendToCover(x.GetSource());
  if (MaybeExpr expr{context_.Analyze(x)}) {
    if (!IsConstantExpr(*expr)) {
      actuals_.emplace_back(std::move(*expr));
      SetArgSourceLocation(actuals_.back(), x.GetSource());
      return;
    }
    const Symbol *symbol{GetLastSymbol(*expr)};
    if (!symbol) {
      context_.SayAt(x, "Assignment to constant '%s' is not allowed"_err_en_US,
          x.GetSource());
    } else if (IsProcedure(*symbol)) {
      if (auto *msg{context_.SayAt(x,
              "Assignment to procedure '%s' is not allowed"_err_en_US,
              symbol->name())}) {
        if (auto *subp{symbol->detailsIf<semantics::SubprogramDetails>()}) {
          if (subp->isFunction()) {
            const auto &result{subp->result().name()};
            msg->Attach(result, "Function result is '%s'"_en_US, result);
          }
        }
      }
    } else {
      context_.SayAt(
          x, "Assignment to '%s' is not allowed"_err_en_US, symbol->name());
    }
  }
  fatalErrors_ = true;
}

void ArgumentAnalyzer::Analyze(
    const parser::ActualArgSpec &arg, bool isSubroutine) {
  // TODO: C1534: Don't allow a "restricted" specific intrinsic to be passed.
  std::optional<ActualArgument> actual;
  common::visit(common::visitors{
                    [&](const common::Indirection<parser::Expr> &x) {
                      actual = AnalyzeExpr(x.value());
                      SetArgSourceLocation(actual, x.value().source);
                    },
                    [&](const parser::AltReturnSpec &label) {
                      if (!isSubroutine) {
                        context_.Say(
                            "alternate return specification may not appear on"
                            " function reference"_err_en_US);
                      }
                      actual = ActualArgument(label.v);
                    },
                    [&](const parser::ActualArg::PercentRef &) {
                      context_.Say("%REF() intrinsic for arguments"_todo_en_US);
                    },
                    [&](const parser::ActualArg::PercentVal &) {
                      context_.Say("%VAL() intrinsic for arguments"_todo_en_US);
                    },
                },
      std::get<parser::ActualArg>(arg.t).u);
  if (actual) {
    if (const auto &argKW{std::get<std::optional<parser::Keyword>>(arg.t)}) {
      actual->set_keyword(argKW->v.source);
    }
    actuals_.emplace_back(std::move(*actual));
  } else {
    fatalErrors_ = true;
  }
}

bool ArgumentAnalyzer::IsIntrinsicRelational(RelationalOperator opr,
    const DynamicType &leftType, const DynamicType &rightType) const {
  CHECK(actuals_.size() == 2);
  return semantics::IsIntrinsicRelational(
      opr, leftType, GetRank(0), rightType, GetRank(1));
}

bool ArgumentAnalyzer::IsIntrinsicNumeric(NumericOperator opr) const {
  std::optional<DynamicType> leftType{GetType(0)};
  if (actuals_.size() == 1) {
    if (IsBOZLiteral(0)) {
      return opr == NumericOperator::Add; // unary '+'
    } else {
      return leftType && semantics::IsIntrinsicNumeric(*leftType);
    }
  } else {
    std::optional<DynamicType> rightType{GetType(1)};
    if (IsBOZLiteral(0) && rightType) { // BOZ opr Integer/Real
      auto cat1{rightType->category()};
      return cat1 == TypeCategory::Integer || cat1 == TypeCategory::Real;
    } else if (IsBOZLiteral(1) && leftType) { // Integer/Real opr BOZ
      auto cat0{leftType->category()};
      return cat0 == TypeCategory::Integer || cat0 == TypeCategory::Real;
    } else {
      return leftType && rightType &&
          semantics::IsIntrinsicNumeric(
              *leftType, GetRank(0), *rightType, GetRank(1));
    }
  }
}

bool ArgumentAnalyzer::IsIntrinsicLogical() const {
  if (std::optional<DynamicType> leftType{GetType(0)}) {
    if (actuals_.size() == 1) {
      return semantics::IsIntrinsicLogical(*leftType);
    } else if (std::optional<DynamicType> rightType{GetType(1)}) {
      return semantics::IsIntrinsicLogical(
          *leftType, GetRank(0), *rightType, GetRank(1));
    }
  }
  return false;
}

bool ArgumentAnalyzer::IsIntrinsicConcat() const {
  if (std::optional<DynamicType> leftType{GetType(0)}) {
    if (std::optional<DynamicType> rightType{GetType(1)}) {
      return semantics::IsIntrinsicConcat(
          *leftType, GetRank(0), *rightType, GetRank(1));
    }
  }
  return false;
}

bool ArgumentAnalyzer::CheckConformance() {
  if (actuals_.size() == 2) {
    const auto *lhs{actuals_.at(0).value().UnwrapExpr()};
    const auto *rhs{actuals_.at(1).value().UnwrapExpr()};
    if (lhs && rhs) {
      auto &foldingContext{context_.GetFoldingContext()};
      auto lhShape{GetShape(foldingContext, *lhs)};
      auto rhShape{GetShape(foldingContext, *rhs)};
      if (lhShape && rhShape) {
        if (!evaluate::CheckConformance(foldingContext.messages(), *lhShape,
                *rhShape, CheckConformanceFlags::EitherScalarExpandable,
                "left operand", "right operand")
                 .value_or(false /*fail when conformance is not known now*/)) {
          fatalErrors_ = true;
          return false;
        }
      }
    }
  }
  return true; // no proven problem
}

bool ArgumentAnalyzer::CheckAssignmentConformance() {
  if (actuals_.size() == 2) {
    const auto *lhs{actuals_.at(0).value().UnwrapExpr()};
    const auto *rhs{actuals_.at(1).value().UnwrapExpr()};
    if (lhs && rhs) {
      auto &foldingContext{context_.GetFoldingContext()};
      auto lhShape{GetShape(foldingContext, *lhs)};
      auto rhShape{GetShape(foldingContext, *rhs)};
      if (lhShape && rhShape) {
        if (!evaluate::CheckConformance(foldingContext.messages(), *lhShape,
                *rhShape, CheckConformanceFlags::RightScalarExpandable,
                "left-hand side", "right-hand side")
                 .value_or(true /*ok when conformance is not known now*/)) {
          fatalErrors_ = true;
          return false;
        }
      }
    }
  }
  return true; // no proven problem
}

bool ArgumentAnalyzer::CheckForNullPointer(const char *where) {
  for (const std::optional<ActualArgument> &arg : actuals_) {
    if (arg) {
      if (const Expr<SomeType> *expr{arg->UnwrapExpr()}) {
        if (IsNullPointer(*expr)) {
          context_.Say(
              source_, "A NULL() pointer is not allowed %s"_err_en_US, where);
          fatalErrors_ = true;
          return false;
        }
      }
    }
  }
  return true;
}

MaybeExpr ArgumentAnalyzer::TryDefinedOp(
    const char *opr, parser::MessageFixedText error, bool isUserOp) {
  if (AnyUntypedOrMissingOperand()) {
    context_.Say(error, ToUpperCase(opr), TypeAsFortran(0), TypeAsFortran(1));
    return std::nullopt;
  }
  MaybeExpr result;
  bool anyPossibilities{false};
  std::optional<parser::MessageFormattedText> inaccessible;
  std::vector<const Symbol *> hit;
  std::string oprNameString{
      isUserOp ? std::string{opr} : "operator("s + opr + ')'};
  parser::CharBlock oprName{oprNameString};
  parser::Messages hitBuffer;
  {
    parser::Messages buffer;
    auto restorer{context_.GetContextualMessages().SetMessages(buffer)};
    const auto &scope{context_.context().FindScope(source_)};
    if (Symbol *symbol{scope.FindSymbol(oprName)}) {
      anyPossibilities = true;
      parser::Name name{symbol->name(), symbol};
      result = context_.AnalyzeDefinedOp(name, GetActuals());
      if (result) {
        inaccessible = CheckAccessibleSymbol(scope, *symbol);
        if (inaccessible) {
          result.reset();
        } else {
          hit.push_back(symbol);
          hitBuffer = std::move(buffer);
        }
      }
    }
    for (std::size_t passIndex{0}; passIndex < actuals_.size(); ++passIndex) {
      buffer.clear();
      const Symbol *generic{nullptr};
      if (const Symbol *binding{
              FindBoundOp(oprName, passIndex, generic, false)}) {
        anyPossibilities = true;
        if (MaybeExpr thisResult{TryBoundOp(*binding, passIndex)}) {
          if (auto thisInaccessible{
                  CheckAccessibleSymbol(scope, DEREF(generic))}) {
            inaccessible = thisInaccessible;
          } else {
            result = std::move(thisResult);
            hit.push_back(binding);
            hitBuffer = std::move(buffer);
          }
        }
      }
    }
  }
  if (result) {
    if (hit.size() > 1) {
      if (auto *msg{context_.Say(
              "%zd matching accessible generic interfaces for %s were found"_err_en_US,
              hit.size(), ToUpperCase(opr))}) {
        for (const Symbol *symbol : hit) {
          AttachDeclaration(*msg, *symbol);
        }
      }
    }
    if (auto *msgs{context_.GetContextualMessages().messages()}) {
      msgs->Annex(std::move(hitBuffer));
    }
  } else if (inaccessible) {
    context_.Say(source_, std::move(*inaccessible));
  } else if (anyPossibilities) {
    SayNoMatch(ToUpperCase(oprNameString), false);
  } else if (actuals_.size() == 2 && !AreConformable()) {
    context_.Say(
        "Operands of %s are not conformable; have rank %d and rank %d"_err_en_US,
        ToUpperCase(opr), actuals_[0]->Rank(), actuals_[1]->Rank());
  } else if (CheckForNullPointer()) {
    context_.Say(error, ToUpperCase(opr), TypeAsFortran(0), TypeAsFortran(1));
  }
  return result;
}

MaybeExpr ArgumentAnalyzer::TryDefinedOp(
    std::vector<const char *> oprs, parser::MessageFixedText error) {
  if (oprs.size() == 1) {
    return TryDefinedOp(oprs[0], error);
  }
  MaybeExpr result;
  std::vector<const char *> hit;
  parser::Messages hitBuffer;
  {
    for (std::size_t i{0}; i < oprs.size(); ++i) {
      parser::Messages buffer;
      auto restorer{context_.GetContextualMessages().SetMessages(buffer)};
      if (MaybeExpr thisResult{TryDefinedOp(oprs[i], error)}) {
        result = std::move(thisResult);
        hit.push_back(oprs[i]);
        hitBuffer = std::move(buffer);
      }
    }
  }
  if (hit.empty()) { // for the error
    result = TryDefinedOp(oprs[0], error);
  } else if (hit.size() > 1) {
    context_.Say(
        "Matching accessible definitions were found with %zd variant spellings of the generic operator ('%s', '%s')"_err_en_US,
        hit.size(), ToUpperCase(hit[0]), ToUpperCase(hit[1]));
  } else { // one hit; preserve errors
    context_.context().messages().Annex(std::move(hitBuffer));
  }
  return result;
}

MaybeExpr ArgumentAnalyzer::TryBoundOp(const Symbol &symbol, int passIndex) {
  ActualArguments localActuals{actuals_};
  const Symbol *proc{GetBindingResolution(GetType(passIndex), symbol)};
  if (!proc) {
    proc = &symbol;
    localActuals.at(passIndex).value().set_isPassedObject();
  }
  CheckConformance();
  return context_.MakeFunctionRef(
      source_, ProcedureDesignator{*proc}, std::move(localActuals));
}

std::optional<ProcedureRef> ArgumentAnalyzer::TryDefinedAssignment() {
  using semantics::Tristate;
  const Expr<SomeType> &lhs{GetExpr(0)};
  const Expr<SomeType> &rhs{GetExpr(1)};
  std::optional<DynamicType> lhsType{lhs.GetType()};
  std::optional<DynamicType> rhsType{rhs.GetType()};
  int lhsRank{lhs.Rank()};
  int rhsRank{rhs.Rank()};
  Tristate isDefined{
      semantics::IsDefinedAssignment(lhsType, lhsRank, rhsType, rhsRank)};
  if (isDefined == Tristate::No) {
    if (lhsType && rhsType) {
      AddAssignmentConversion(*lhsType, *rhsType);
    }
    if (!fatalErrors_) {
      CheckAssignmentConformance();
    }
    return std::nullopt; // user-defined assignment not allowed for these args
  }
  auto restorer{context_.GetContextualMessages().SetLocation(source_)};
  if (std::optional<ProcedureRef> procRef{GetDefinedAssignmentProc()}) {
    if (context_.inWhereBody() && !procRef->proc().IsElemental()) { // C1032
      context_.Say(
          "Defined assignment in WHERE must be elemental, but '%s' is not"_err_en_US,
          DEREF(procRef->proc().GetSymbol()).name());
    }
    context_.CheckCall(source_, procRef->proc(), procRef->arguments());
    return std::move(*procRef);
  }
  if (isDefined == Tristate::Yes) {
    if (!lhsType || !rhsType || (lhsRank != rhsRank && rhsRank != 0) ||
        !OkLogicalIntegerAssignment(lhsType->category(), rhsType->category())) {
      SayNoMatch("ASSIGNMENT(=)", true);
    }
  }
  return std::nullopt;
}

bool ArgumentAnalyzer::OkLogicalIntegerAssignment(
    TypeCategory lhs, TypeCategory rhs) {
  if (!context_.context().languageFeatures().IsEnabled(
          common::LanguageFeature::LogicalIntegerAssignment)) {
    return false;
  }
  std::optional<parser::MessageFixedText> msg;
  if (lhs == TypeCategory::Integer && rhs == TypeCategory::Logical) {
    // allow assignment to LOGICAL from INTEGER as a legacy extension
    msg = "assignment of LOGICAL to INTEGER"_port_en_US;
  } else if (lhs == TypeCategory::Logical && rhs == TypeCategory::Integer) {
    // ... and assignment to LOGICAL from INTEGER
    msg = "assignment of INTEGER to LOGICAL"_port_en_US;
  } else {
    return false;
  }
  if (context_.context().ShouldWarn(
          common::LanguageFeature::LogicalIntegerAssignment)) {
    context_.Say(std::move(*msg));
  }
  return true;
}

std::optional<ProcedureRef> ArgumentAnalyzer::GetDefinedAssignmentProc() {
  const Symbol *proc{nullptr};
  std::optional<int> passedObjectIndex;
  std::string oprNameString{"assignment(=)"};
  parser::CharBlock oprName{oprNameString};
  const auto &scope{context_.context().FindScope(source_)};
  // If multiple resolutions were possible, they will have been already
  // diagnosed.
  {
    auto restorer{context_.GetContextualMessages().DiscardMessages()};
    if (const Symbol *symbol{scope.FindSymbol(oprName)}) {
      ExpressionAnalyzer::AdjustActuals noAdjustment;
      proc =
          context_.ResolveGeneric(*symbol, actuals_, noAdjustment, true).first;
    }
    for (std::size_t i{0}; !proc && i < actuals_.size(); ++i) {
      const Symbol *generic{nullptr};
      if (const Symbol *binding{FindBoundOp(oprName, i, generic, true)}) {
        if (CheckAccessibleSymbol(scope, DEREF(generic))) {
          // ignore inaccessible type-bound ASSIGNMENT(=) generic
        } else if (const Symbol *
            resolution{GetBindingResolution(GetType(i), *binding)}) {
          proc = resolution;
        } else {
          proc = binding;
          passedObjectIndex = i;
        }
      }
    }
  }
  if (!proc) {
    return std::nullopt;
  }
  ActualArguments actualsCopy{actuals_};
  // Ensure that the RHS argument is not passed as a variable unless
  // the dummy argument has the VALUE attribute.
  if (evaluate::IsVariable(actualsCopy.at(1).value().UnwrapExpr())) {
    auto chars{evaluate::characteristics::Procedure::Characterize(
        *proc, context_.GetFoldingContext())};
    const auto *rhsDummy{chars && chars->dummyArguments.size() == 2
            ? std::get_if<evaluate::characteristics::DummyDataObject>(
                  &chars->dummyArguments.at(1).u)
            : nullptr};
    if (!rhsDummy ||
        !rhsDummy->attrs.test(
            evaluate::characteristics::DummyDataObject::Attr::Value)) {
      actualsCopy.at(1).value().Parenthesize();
    }
  }
  if (passedObjectIndex) {
    actualsCopy[*passedObjectIndex]->set_isPassedObject();
  }
  return ProcedureRef{ProcedureDesignator{*proc}, std::move(actualsCopy)};
}

void ArgumentAnalyzer::Dump(llvm::raw_ostream &os) {
  os << "source_: " << source_.ToString() << " fatalErrors_ = " << fatalErrors_
     << '\n';
  for (const auto &actual : actuals_) {
    if (!actual.has_value()) {
      os << "- error\n";
    } else if (const Symbol *symbol{actual->GetAssumedTypeDummy()}) {
      os << "- assumed type: " << symbol->name().ToString() << '\n';
    } else if (const Expr<SomeType> *expr{actual->UnwrapExpr()}) {
      expr->AsFortran(os << "- expr: ") << '\n';
    } else {
      DIE("bad ActualArgument");
    }
  }
}

std::optional<ActualArgument> ArgumentAnalyzer::AnalyzeExpr(
    const parser::Expr &expr) {
  source_.ExtendToCover(expr.source);
  if (const Symbol *assumedTypeDummy{AssumedTypeDummy(expr)}) {
    ResetExpr(expr);
    if (isProcedureCall_) {
      ActualArgument arg{ActualArgument::AssumedType{*assumedTypeDummy}};
      SetArgSourceLocation(arg, expr.source);
      return std::move(arg);
    }
    context_.SayAt(expr.source,
        "TYPE(*) dummy argument may only be used as an actual argument"_err_en_US);
  } else if (MaybeExpr argExpr{AnalyzeExprOrWholeAssumedSizeArray(expr)}) {
    if (isProcedureCall_ || !IsProcedure(*argExpr)) {
      ActualArgument arg{std::move(*argExpr)};
      SetArgSourceLocation(arg, expr.source);
      return std::move(arg);
    }
    context_.SayAt(expr.source,
        IsFunction(*argExpr) ? "Function call must have argument list"_err_en_US
                             : "Subroutine name is not allowed here"_err_en_US);
  }
  return std::nullopt;
}

MaybeExpr ArgumentAnalyzer::AnalyzeExprOrWholeAssumedSizeArray(
    const parser::Expr &expr) {
  // If an expression's parse tree is a whole assumed-size array:
  //   Expr -> Designator -> DataRef -> Name
  // treat it as a special case for argument passing and bypass
  // the C1002/C1014 constraint checking in expression semantics.
  if (const auto *name{parser::Unwrap<parser::Name>(expr)}) {
    if (name->symbol && semantics::IsAssumedSizeArray(*name->symbol)) {
      auto restorer{context_.AllowWholeAssumedSizeArray()};
      return context_.Analyze(expr);
    }
  }
  auto restorer{context_.AllowNullPointer()};
  return context_.Analyze(expr);
}

bool ArgumentAnalyzer::AreConformable() const {
  CHECK(actuals_.size() == 2);
  return actuals_[0] && actuals_[1] &&
      evaluate::AreConformable(*actuals_[0], *actuals_[1]);
}

// Look for a type-bound operator in the type of arg number passIndex.
const Symbol *ArgumentAnalyzer::FindBoundOp(parser::CharBlock oprName,
    int passIndex, const Symbol *&generic, bool isSubroutine) {
  const auto *type{GetDerivedTypeSpec(GetType(passIndex))};
  const semantics::Scope *scope{type ? type->scope() : nullptr};
  if (scope) {
    // Use the original type definition's scope, since PDT
    // instantiations don't have redundant copies of bindings or
    // generics.
    scope = DEREF(scope->derivedTypeSpec()).typeSymbol().scope();
  }
  generic = scope ? scope->FindComponent(oprName) : nullptr;
  if (generic) {
    ExpressionAnalyzer::AdjustActuals adjustment{
        [&](const Symbol &proc, ActualArguments &) {
          return passIndex == GetPassIndex(proc).value_or(-1);
        }};
    auto pair{
        context_.ResolveGeneric(*generic, actuals_, adjustment, isSubroutine)};
    if (const Symbol *binding{pair.first}) {
      CHECK(binding->has<semantics::ProcBindingDetails>());
      // Use the most recent override of the binding, if any
      return scope->FindComponent(binding->name());
    } else {
      context_.EmitGenericResolutionError(*generic, pair.second, isSubroutine);
    }
  }
  return nullptr;
}

// If there is an implicit conversion between intrinsic types, make it explicit
void ArgumentAnalyzer::AddAssignmentConversion(
    const DynamicType &lhsType, const DynamicType &rhsType) {
  if (lhsType.category() == rhsType.category() &&
      (lhsType.category() == TypeCategory::Derived ||
          lhsType.kind() == rhsType.kind())) {
    // no conversion necessary
  } else if (auto rhsExpr{evaluate::Fold(context_.GetFoldingContext(),
                 evaluate::ConvertToType(lhsType, MoveExpr(1)))}) {
    std::optional<parser::CharBlock> source;
    if (actuals_[1]) {
      source = actuals_[1]->sourceLocation();
    }
    actuals_[1] = ActualArgument{*rhsExpr};
    SetArgSourceLocation(actuals_[1], source);
  } else {
    actuals_[1] = std::nullopt;
  }
}

std::optional<DynamicType> ArgumentAnalyzer::GetType(std::size_t i) const {
  return i < actuals_.size() ? actuals_[i].value().GetType() : std::nullopt;
}
int ArgumentAnalyzer::GetRank(std::size_t i) const {
  return i < actuals_.size() ? actuals_[i].value().Rank() : 0;
}

// If the argument at index i is a BOZ literal, convert its type to match the
// otherType.  If it's REAL convert to REAL, otherwise convert to INTEGER.
// Note that IBM supports comparing BOZ literals to CHARACTER operands.  That
// is not currently supported.
void ArgumentAnalyzer::ConvertBOZ(std::optional<DynamicType> &thisType,
    std::size_t i, std::optional<DynamicType> otherType) {
  if (IsBOZLiteral(i)) {
    Expr<SomeType> &&argExpr{MoveExpr(i)};
    auto *boz{std::get_if<BOZLiteralConstant>(&argExpr.u)};
    if (otherType && otherType->category() == TypeCategory::Real) {
      int kind{context_.context().GetDefaultKind(TypeCategory::Real)};
      MaybeExpr realExpr{
          ConvertToKind<TypeCategory::Real>(kind, std::move(*boz))};
      actuals_[i] = std::move(*realExpr);
      thisType.emplace(TypeCategory::Real, kind);
    } else {
      int kind{context_.context().GetDefaultKind(TypeCategory::Integer)};
      MaybeExpr intExpr{
          ConvertToKind<TypeCategory::Integer>(kind, std::move(*boz))};
      actuals_[i] = std::move(*intExpr);
      thisType.emplace(TypeCategory::Integer, kind);
    }
  }
}

// Report error resolving opr when there is a user-defined one available
void ArgumentAnalyzer::SayNoMatch(const std::string &opr, bool isAssignment) {
  std::string type0{TypeAsFortran(0)};
  auto rank0{actuals_[0]->Rank()};
  if (actuals_.size() == 1) {
    if (rank0 > 0) {
      context_.Say("No intrinsic or user-defined %s matches "
                   "rank %d array of %s"_err_en_US,
          opr, rank0, type0);
    } else {
      context_.Say("No intrinsic or user-defined %s matches "
                   "operand type %s"_err_en_US,
          opr, type0);
    }
  } else {
    std::string type1{TypeAsFortran(1)};
    auto rank1{actuals_[1]->Rank()};
    if (rank0 > 0 && rank1 > 0 && rank0 != rank1) {
      context_.Say("No intrinsic or user-defined %s matches "
                   "rank %d array of %s and rank %d array of %s"_err_en_US,
          opr, rank0, type0, rank1, type1);
    } else if (isAssignment && rank0 != rank1) {
      if (rank0 == 0) {
        context_.Say("No intrinsic or user-defined %s matches "
                     "scalar %s and rank %d array of %s"_err_en_US,
            opr, type0, rank1, type1);
      } else {
        context_.Say("No intrinsic or user-defined %s matches "
                     "rank %d array of %s and scalar %s"_err_en_US,
            opr, rank0, type0, type1);
      }
    } else {
      context_.Say("No intrinsic or user-defined %s matches "
                   "operand types %s and %s"_err_en_US,
          opr, type0, type1);
    }
  }
}

std::string ArgumentAnalyzer::TypeAsFortran(std::size_t i) {
  if (i >= actuals_.size() || !actuals_[i]) {
    return "missing argument";
  } else if (std::optional<DynamicType> type{GetType(i)}) {
    return type->IsAssumedType()         ? "TYPE(*)"s
        : type->IsUnlimitedPolymorphic() ? "CLASS(*)"s
        : type->IsPolymorphic()          ? type->AsFortran()
        : type->category() == TypeCategory::Derived
        ? "TYPE("s + type->AsFortran() + ')'
        : type->category() == TypeCategory::Character
        ? "CHARACTER(KIND="s + std::to_string(type->kind()) + ')'
        : ToUpperCase(type->AsFortran());
  } else {
    return "untyped";
  }
}

bool ArgumentAnalyzer::AnyUntypedOrMissingOperand() {
  for (const auto &actual : actuals_) {
    if (!actual ||
        (!actual->GetType() && !IsBareNullPointer(actual->UnwrapExpr()))) {
      return true;
    }
  }
  return false;
}
} // namespace Fortran::evaluate

namespace Fortran::semantics {
evaluate::Expr<evaluate::SubscriptInteger> AnalyzeKindSelector(
    SemanticsContext &context, common::TypeCategory category,
    const std::optional<parser::KindSelector> &selector) {
  evaluate::ExpressionAnalyzer analyzer{context};
  auto restorer{
      analyzer.GetContextualMessages().SetLocation(context.location().value())};
  return analyzer.AnalyzeKindSelector(category, selector);
}

ExprChecker::ExprChecker(SemanticsContext &context) : context_{context} {}

bool ExprChecker::Pre(const parser::DataStmtObject &obj) {
  exprAnalyzer_.set_inDataStmtObject(true);
  return true;
}

void ExprChecker::Post(const parser::DataStmtObject &obj) {
  exprAnalyzer_.set_inDataStmtObject(false);
}

bool ExprChecker::Pre(const parser::DataImpliedDo &ido) {
  parser::Walk(std::get<parser::DataImpliedDo::Bounds>(ido.t), *this);
  const auto &bounds{std::get<parser::DataImpliedDo::Bounds>(ido.t)};
  auto name{bounds.name.thing.thing};
  int kind{evaluate::ResultType<evaluate::ImpliedDoIndex>::kind};
  if (const auto dynamicType{evaluate::DynamicType::From(*name.symbol)}) {
    if (dynamicType->category() == TypeCategory::Integer) {
      kind = dynamicType->kind();
    }
  }
  exprAnalyzer_.AddImpliedDo(name.source, kind);
  parser::Walk(std::get<std::list<parser::DataIDoObject>>(ido.t), *this);
  exprAnalyzer_.RemoveImpliedDo(name.source);
  return false;
}

bool ExprChecker::Walk(const parser::Program &program) {
  parser::Walk(program, *this);
  return !context_.AnyFatalError();
}
} // namespace Fortran::semantics
