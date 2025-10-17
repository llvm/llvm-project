//===-- lib/Semantics/resolve-names-utils.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "resolve-names-utils.h"
#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/tools.h"
#include "flang/Support/Fortran-features.h"
#include "flang/Support/Fortran.h"
#include <initializer_list>
#include <variant>

namespace Fortran::semantics {

using common::LanguageFeature;
using common::LogicalOperator;
using common::NumericOperator;
using common::RelationalOperator;
using IntrinsicOperator = parser::DefinedOperator::IntrinsicOperator;

static GenericKind MapIntrinsicOperator(IntrinsicOperator);

Symbol *Resolve(const parser::Name &name, Symbol *symbol) {
  if (symbol && !name.symbol) {
    name.symbol = symbol;
  }
  return symbol;
}
Symbol &Resolve(const parser::Name &name, Symbol &symbol) {
  return *Resolve(name, &symbol);
}

parser::MessageFixedText WithSeverity(
    const parser::MessageFixedText &msg, parser::Severity severity) {
  return parser::MessageFixedText{
      msg.text().begin(), msg.text().size(), severity};
}

bool IsIntrinsicOperator(
    const SemanticsContext &context, const SourceName &name) {
  std::string str{name.ToString()};
  for (int i{0}; i != common::LogicalOperator_enumSize; ++i) {
    auto names{context.languageFeatures().GetNames(LogicalOperator{i})};
    if (llvm::is_contained(names, str)) {
      return true;
    }
  }
  for (int i{0}; i != common::RelationalOperator_enumSize; ++i) {
    auto names{context.languageFeatures().GetNames(RelationalOperator{i})};
    if (llvm::is_contained(names, str)) {
      return true;
    }
  }
  return false;
}

bool IsLogicalConstant(
    const SemanticsContext &context, const SourceName &name) {
  std::string str{name.ToString()};
  return str == ".true." || str == ".false." ||
      (context.IsEnabled(LanguageFeature::LogicalAbbreviations) &&
          (str == ".t" || str == ".f."));
}

void GenericSpecInfo::Resolve(Symbol *symbol) const {
  if (symbol) {
    if (auto *details{symbol->detailsIf<GenericDetails>()}) {
      details->set_kind(kind_);
    }
    if (parseName_) {
      semantics::Resolve(*parseName_, symbol);
    }
  }
}

void GenericSpecInfo::Analyze(const parser::DefinedOpName &name) {
  kind_ = GenericKind::OtherKind::DefinedOp;
  parseName_ = &name.v;
  symbolName_ = name.v.source;
}

void GenericSpecInfo::Analyze(const parser::GenericSpec &x) {
  symbolName_ = x.source;
  kind_ = common::visit(
      common::visitors{
          [&](const parser::Name &y) -> GenericKind {
            parseName_ = &y;
            symbolName_ = y.source;
            return GenericKind::OtherKind::Name;
          },
          [&](const parser::DefinedOperator &y) {
            return common::visit(
                common::visitors{
                    [&](const parser::DefinedOpName &z) -> GenericKind {
                      Analyze(z);
                      return GenericKind::OtherKind::DefinedOp;
                    },
                    [&](const IntrinsicOperator &z) {
                      return MapIntrinsicOperator(z);
                    },
                },
                y.u);
          },
          [&](const parser::GenericSpec::Assignment &) -> GenericKind {
            return GenericKind::OtherKind::Assignment;
          },
          [&](const parser::GenericSpec::ReadFormatted &) -> GenericKind {
            return common::DefinedIo::ReadFormatted;
          },
          [&](const parser::GenericSpec::ReadUnformatted &) -> GenericKind {
            return common::DefinedIo::ReadUnformatted;
          },
          [&](const parser::GenericSpec::WriteFormatted &) -> GenericKind {
            return common::DefinedIo::WriteFormatted;
          },
          [&](const parser::GenericSpec::WriteUnformatted &) -> GenericKind {
            return common::DefinedIo::WriteUnformatted;
          },
      },
      x.u);
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, const GenericSpecInfo &info) {
  os << "GenericSpecInfo: kind=" << info.kind_.ToString();
  os << " parseName="
     << (info.parseName_ ? info.parseName_->ToString() : "null");
  os << " symbolName="
     << (info.symbolName_ ? info.symbolName_->ToString() : "null");
  return os;
}

// parser::DefinedOperator::IntrinsicOperator -> GenericKind
static GenericKind MapIntrinsicOperator(IntrinsicOperator op) {
  switch (op) {
    SWITCH_COVERS_ALL_CASES
  case IntrinsicOperator::Concat:
    return GenericKind::OtherKind::Concat;
  case IntrinsicOperator::Power:
    return NumericOperator::Power;
  case IntrinsicOperator::Multiply:
    return NumericOperator::Multiply;
  case IntrinsicOperator::Divide:
    return NumericOperator::Divide;
  case IntrinsicOperator::Add:
    return NumericOperator::Add;
  case IntrinsicOperator::Subtract:
    return NumericOperator::Subtract;
  case IntrinsicOperator::AND:
    return LogicalOperator::And;
  case IntrinsicOperator::OR:
    return LogicalOperator::Or;
  case IntrinsicOperator::EQV:
    return LogicalOperator::Eqv;
  case IntrinsicOperator::NEQV:
    return LogicalOperator::Neqv;
  case IntrinsicOperator::NOT:
    return LogicalOperator::Not;
  case IntrinsicOperator::LT:
    return RelationalOperator::LT;
  case IntrinsicOperator::LE:
    return RelationalOperator::LE;
  case IntrinsicOperator::EQ:
    return RelationalOperator::EQ;
  case IntrinsicOperator::NE:
    return RelationalOperator::NE;
  case IntrinsicOperator::GE:
    return RelationalOperator::GE;
  case IntrinsicOperator::GT:
    return RelationalOperator::GT;
  }
}

class ArraySpecAnalyzer {
public:
  ArraySpecAnalyzer(SemanticsContext &context) : context_{context} {}
  ArraySpec Analyze(const parser::ArraySpec &);
  ArraySpec AnalyzeDeferredShapeSpecList(const parser::DeferredShapeSpecList &);
  ArraySpec Analyze(const parser::ComponentArraySpec &);
  ArraySpec Analyze(const parser::CoarraySpec &);

private:
  SemanticsContext &context_;
  ArraySpec arraySpec_;

  template <typename T> void Analyze(const std::list<T> &list) {
    for (const auto &elem : list) {
      Analyze(elem);
    }
  }
  void Analyze(const parser::AssumedShapeSpec &);
  void Analyze(const parser::ExplicitShapeSpec &);
  void Analyze(const parser::AssumedImpliedSpec &);
  void Analyze(const parser::DeferredShapeSpecList &);
  void Analyze(const parser::AssumedRankSpec &);
  void MakeExplicit(const std::optional<parser::SpecificationExpr> &,
      const parser::SpecificationExpr &);
  void MakeImplied(const std::optional<parser::SpecificationExpr> &);
  void MakeDeferred(int);
  Bound GetBound(const std::optional<parser::SpecificationExpr> &);
  Bound GetBound(const parser::SpecificationExpr &);
};

ArraySpec AnalyzeArraySpec(
    SemanticsContext &context, const parser::ArraySpec &arraySpec) {
  return ArraySpecAnalyzer{context}.Analyze(arraySpec);
}
ArraySpec AnalyzeArraySpec(
    SemanticsContext &context, const parser::ComponentArraySpec &arraySpec) {
  return ArraySpecAnalyzer{context}.Analyze(arraySpec);
}
ArraySpec AnalyzeDeferredShapeSpecList(SemanticsContext &context,
    const parser::DeferredShapeSpecList &deferredShapeSpecs) {
  return ArraySpecAnalyzer{context}.AnalyzeDeferredShapeSpecList(
      deferredShapeSpecs);
}
ArraySpec AnalyzeCoarraySpec(
    SemanticsContext &context, const parser::CoarraySpec &coarraySpec) {
  return ArraySpecAnalyzer{context}.Analyze(coarraySpec);
}

ArraySpec ArraySpecAnalyzer::Analyze(const parser::ComponentArraySpec &x) {
  common::visit([this](const auto &y) { Analyze(y); }, x.u);
  CHECK(!arraySpec_.empty());
  return arraySpec_;
}
ArraySpec ArraySpecAnalyzer::Analyze(const parser::ArraySpec &x) {
  common::visit(common::visitors{
                    [&](const parser::AssumedSizeSpec &y) {
                      Analyze(
                          std::get<std::list<parser::ExplicitShapeSpec>>(y.t));
                      Analyze(std::get<parser::AssumedImpliedSpec>(y.t));
                    },
                    [&](const parser::ImpliedShapeSpec &y) { Analyze(y.v); },
                    [&](const auto &y) { Analyze(y); },
                },
      x.u);
  CHECK(!arraySpec_.empty());
  return arraySpec_;
}
ArraySpec ArraySpecAnalyzer::AnalyzeDeferredShapeSpecList(
    const parser::DeferredShapeSpecList &x) {
  Analyze(x);
  CHECK(!arraySpec_.empty());
  return arraySpec_;
}
ArraySpec ArraySpecAnalyzer::Analyze(const parser::CoarraySpec &x) {
  common::visit(
      common::visitors{
          [&](const parser::DeferredCoshapeSpecList &y) { MakeDeferred(y.v); },
          [&](const parser::ExplicitCoshapeSpec &y) {
            Analyze(std::get<std::list<parser::ExplicitShapeSpec>>(y.t));
            MakeImplied(
                std::get<std::optional<parser::SpecificationExpr>>(y.t));
          },
      },
      x.u);
  CHECK(!arraySpec_.empty());
  return arraySpec_;
}

void ArraySpecAnalyzer::Analyze(const parser::AssumedShapeSpec &x) {
  arraySpec_.push_back(ShapeSpec::MakeAssumedShape(GetBound(x.v)));
}
void ArraySpecAnalyzer::Analyze(const parser::ExplicitShapeSpec &x) {
  MakeExplicit(std::get<std::optional<parser::SpecificationExpr>>(x.t),
      std::get<parser::SpecificationExpr>(x.t));
}
void ArraySpecAnalyzer::Analyze(const parser::AssumedImpliedSpec &x) {
  MakeImplied(x.v);
}
void ArraySpecAnalyzer::Analyze(const parser::DeferredShapeSpecList &x) {
  MakeDeferred(x.v);
}
void ArraySpecAnalyzer::Analyze(const parser::AssumedRankSpec &) {
  arraySpec_.push_back(ShapeSpec::MakeAssumedRank());
}

void ArraySpecAnalyzer::MakeExplicit(
    const std::optional<parser::SpecificationExpr> &lb,
    const parser::SpecificationExpr &ub) {
  arraySpec_.push_back(ShapeSpec::MakeExplicit(GetBound(lb), GetBound(ub)));
}
void ArraySpecAnalyzer::MakeImplied(
    const std::optional<parser::SpecificationExpr> &lb) {
  arraySpec_.push_back(ShapeSpec::MakeImplied(GetBound(lb)));
}
void ArraySpecAnalyzer::MakeDeferred(int n) {
  for (int i = 0; i < n; ++i) {
    arraySpec_.push_back(ShapeSpec::MakeDeferred());
  }
}

Bound ArraySpecAnalyzer::GetBound(
    const std::optional<parser::SpecificationExpr> &x) {
  return x ? GetBound(*x) : Bound{1};
}
Bound ArraySpecAnalyzer::GetBound(const parser::SpecificationExpr &x) {
  MaybeSubscriptIntExpr expr;
  if (MaybeExpr maybeExpr{AnalyzeExpr(context_, x.v)}) {
    if (auto *intExpr{evaluate::UnwrapExpr<SomeIntExpr>(*maybeExpr)}) {
      expr = evaluate::Fold(context_.foldingContext(),
          evaluate::ConvertToType<evaluate::SubscriptInteger>(
              std::move(*intExpr)));
    }
  }
  return Bound{std::move(expr)};
}

// If src is SAVE (explicitly or implicitly),
// set SAVE attribute on all members of dst.
static void PropagateSaveAttr(
    const EquivalenceObject &src, EquivalenceSet &dst) {
  if (IsSaved(src.symbol)) {
    for (auto &obj : dst) {
      if (!obj.symbol.attrs().test(Attr::SAVE)) {
        obj.symbol.attrs().set(Attr::SAVE);
        // If the other equivalenced symbol itself is not SAVE,
        // then adding SAVE here implies that it has to be implicit.
        obj.symbol.implicitAttrs().set(Attr::SAVE);
      }
    }
  }
}
static void PropagateSaveAttr(const EquivalenceSet &src, EquivalenceSet &dst) {
  if (!src.empty()) {
    PropagateSaveAttr(src.front(), dst);
  }
}

void EquivalenceSets::AddToSet(const parser::Designator &designator) {
  if (CheckDesignator(designator)) {
    if (Symbol * symbol{currObject_.symbol}) {
      if (!currSet_.empty()) {
        // check this symbol against first of set for compatibility
        Symbol &first{currSet_.front().symbol};
        CheckCanEquivalence(designator.source, first, *symbol) &&
            CheckCanEquivalence(designator.source, *symbol, first);
      }
      auto subscripts{currObject_.subscripts};
      if (subscripts.empty()) {
        if (const ArraySpec * shape{symbol->GetShape()};
            shape && shape->IsExplicitShape()) {
          // record a whole array as its first element
          for (const ShapeSpec &spec : *shape) {
            if (auto lbound{spec.lbound().GetExplicit()}) {
              if (auto lbValue{evaluate::ToInt64(*lbound)}) {
                subscripts.push_back(*lbValue);
                continue;
              }
            }
            subscripts.clear(); // error recovery
            break;
          }
        }
      }
      auto substringStart{currObject_.substringStart};
      currSet_.emplace_back(
          *symbol, subscripts, substringStart, designator.source);
      PropagateSaveAttr(currSet_.back(), currSet_);
    }
  }
  currObject_ = {};
}

void EquivalenceSets::FinishSet(const parser::CharBlock &source) {
  std::set<std::size_t> existing; // indices of sets intersecting this one
  for (auto &obj : currSet_) {
    auto it{objectToSet_.find(obj)};
    if (it != objectToSet_.end()) {
      existing.insert(it->second); // symbol already in this set
    }
  }
  if (existing.empty()) {
    sets_.push_back({}); // create a new equivalence set
    MergeInto(source, currSet_, sets_.size() - 1);
  } else {
    auto it{existing.begin()};
    std::size_t dstIndex{*it};
    MergeInto(source, currSet_, dstIndex);
    while (++it != existing.end()) {
      MergeInto(source, sets_[*it], dstIndex);
    }
  }
  currSet_.clear();
}

// Report an error or warning if sym1 and sym2 cannot be in the same equivalence
// set.
bool EquivalenceSets::CheckCanEquivalence(
    const parser::CharBlock &source, const Symbol &sym1, const Symbol &sym2) {
  std::optional<common::LanguageFeature> feature;
  std::optional<parser::MessageFixedText> msg;
  const DeclTypeSpec *type1{sym1.GetType()};
  const DeclTypeSpec *type2{sym2.GetType()};
  bool isDefaultNum1{IsDefaultNumericSequenceType(type1)};
  bool isAnyNum1{IsAnyNumericSequenceType(type1)};
  bool isDefaultNum2{IsDefaultNumericSequenceType(type2)};
  bool isAnyNum2{IsAnyNumericSequenceType(type2)};
  bool isChar1{IsCharacterSequenceType(type1)};
  bool isChar2{IsCharacterSequenceType(type2)};
  if (sym1.attrs().test(Attr::PROTECTED) &&
      !sym2.attrs().test(Attr::PROTECTED)) { // C8114
    msg = "Equivalence set cannot contain '%s'"
          " with PROTECTED attribute and '%s' without"_err_en_US;
  } else if ((isDefaultNum1 && isDefaultNum2) || (isChar1 && isChar2)) {
    // ok & standard conforming
  } else if (!(isAnyNum1 || isChar1) &&
      !(isAnyNum2 || isChar2)) { // C8110 - C8113
    if (AreTkCompatibleTypes(type1, type2)) {
      msg =
          "nonstandard: Equivalence set contains '%s' and '%s' with same type that is neither numeric nor character sequence type"_port_en_US;
      feature = LanguageFeature::EquivalenceSameNonSequence;
    } else {
      msg = "Equivalence set cannot contain '%s' and '%s' with distinct types "
            "that are not both numeric or character sequence types"_err_en_US;
    }
  } else if (isAnyNum1) {
    if (isChar2) {
      msg =
          "nonstandard: Equivalence set contains '%s' that is numeric sequence type and '%s' that is character"_port_en_US;
      feature = LanguageFeature::EquivalenceNumericWithCharacter;
    } else if (isAnyNum2) {
      if (isDefaultNum1) {
        msg =
            "nonstandard: Equivalence set contains '%s' that is a default "
            "numeric sequence type and '%s' that is numeric with non-default kind"_port_en_US;
      } else if (!isDefaultNum2) {
        msg = "nonstandard: Equivalence set contains '%s' and '%s' that are "
              "numeric sequence types with non-default kinds"_port_en_US;
      }
      feature = LanguageFeature::EquivalenceNonDefaultNumeric;
    }
  }
  if (msg) {
    if (feature) {
      context_.Warn(
          *feature, source, std::move(*msg), sym1.name(), sym2.name());
    } else {
      context_.Say(source, std::move(*msg), sym1.name(), sym2.name());
    }
    return false;
  }
  return true;
}

// Move objects from src to sets_[dstIndex]
void EquivalenceSets::MergeInto(const parser::CharBlock &source,
    EquivalenceSet &src, std::size_t dstIndex) {
  EquivalenceSet &dst{sets_[dstIndex]};
  PropagateSaveAttr(dst, src);
  for (const auto &obj : src) {
    dst.push_back(obj);
    objectToSet_[obj] = dstIndex;
  }
  PropagateSaveAttr(src, dst);
  src.clear();
}

// If set has an object with this symbol, return it.
const EquivalenceObject *EquivalenceSets::Find(
    const EquivalenceSet &set, const Symbol &symbol) {
  for (const auto &obj : set) {
    if (obj.symbol == symbol) {
      return &obj;
    }
  }
  return nullptr;
}

bool EquivalenceSets::CheckDesignator(const parser::Designator &designator) {
  return common::visit(
      common::visitors{
          [&](const parser::DataRef &x) {
            return CheckDataRef(designator.source, x);
          },
          [&](const parser::Substring &x) {
            const auto &dataRef{std::get<parser::DataRef>(x.t)};
            const auto &range{std::get<parser::SubstringRange>(x.t)};
            bool ok{CheckDataRef(designator.source, dataRef)};
            if (const auto &lb{std::get<0>(range.t)}) {
              ok &= CheckSubstringBound(
                  parser::UnwrapRef<parser::Expr>(lb), true);
            } else {
              currObject_.substringStart = 1;
            }
            if (const auto &ub{std::get<1>(range.t)}) {
              ok &= CheckSubstringBound(
                  parser::UnwrapRef<parser::Expr>(ub), false);
            }
            return ok;
          },
      },
      designator.u);
}

bool EquivalenceSets::CheckDataRef(
    const parser::CharBlock &source, const parser::DataRef &x) {
  return common::visit(
      common::visitors{
          [&](const parser::Name &name) { return CheckObject(name); },
          [&](const common::Indirection<parser::StructureComponent> &) {
            context_.Say(source, // C8107
                "Derived type component '%s' is not allowed in an equivalence set"_err_en_US,
                source);
            return false;
          },
          [&](const common::Indirection<parser::ArrayElement> &elem) {
            bool ok{CheckDataRef(source, elem.value().base)};
            for (const auto &subscript : elem.value().subscripts) {
              ok &= common::visit(
                  common::visitors{
                      [&](const parser::SubscriptTriplet &) {
                        context_.Say(source, // C924, R872
                            "Array section '%s' is not allowed in an equivalence set"_err_en_US,
                            source);
                        return false;
                      },
                      [&](const parser::IntExpr &y) {
                        return CheckArrayBound(
                            parser::UnwrapRef<parser::Expr>(y));
                      },
                  },
                  subscript.u);
            }
            return ok;
          },
          [&](const common::Indirection<parser::CoindexedNamedObject> &) {
            context_.Say(source, // C924 (R872)
                "Coindexed object '%s' is not allowed in an equivalence set"_err_en_US,
                source);
            return false;
          },
      },
      x.u);
}

bool EquivalenceSets::CheckObject(const parser::Name &name) {
  currObject_.symbol = name.symbol;
  return currObject_.symbol != nullptr;
}

bool EquivalenceSets::CheckArrayBound(const parser::Expr &bound) {
  MaybeExpr expr{
      evaluate::Fold(context_.foldingContext(), AnalyzeExpr(context_, bound))};
  if (!expr) {
    return false;
  }
  if (expr->Rank() > 0) {
    context_.Say(bound.source, // C924, R872
        "Array with vector subscript '%s' is not allowed in an equivalence set"_err_en_US,
        bound.source);
    return false;
  }
  auto subscript{evaluate::ToInt64(*expr)};
  if (!subscript) {
    context_.Say(bound.source, // C8109
        "Array with nonconstant subscript '%s' is not allowed in an equivalence set"_err_en_US,
        bound.source);
    return false;
  }
  currObject_.subscripts.push_back(*subscript);
  return true;
}

bool EquivalenceSets::CheckSubstringBound(
    const parser::Expr &bound, bool isStart) {
  MaybeExpr expr{
      evaluate::Fold(context_.foldingContext(), AnalyzeExpr(context_, bound))};
  if (!expr) {
    return false;
  }
  auto subscript{evaluate::ToInt64(*expr)};
  if (!subscript) {
    context_.Say(bound.source, // C8109
        "Substring with nonconstant bound '%s' is not allowed in an equivalence set"_err_en_US,
        bound.source);
    return false;
  }
  if (!isStart) {
    auto start{currObject_.substringStart};
    if (*subscript < (start ? *start : 1)) {
      context_.Say(bound.source, // C8116
          "Substring with zero length is not allowed in an equivalence set"_err_en_US);
      return false;
    }
  } else if (*subscript != 1) {
    currObject_.substringStart = *subscript;
  }
  return true;
}

bool EquivalenceSets::IsCharacterSequenceType(const DeclTypeSpec *type) {
  return IsSequenceType(type, [&](const IntrinsicTypeSpec &type) {
    auto kind{evaluate::ToInt64(type.kind())};
    return type.category() == TypeCategory::Character && kind &&
        kind.value() == context_.GetDefaultKind(TypeCategory::Character);
  });
}

// Numeric or logical type of default kind or DOUBLE PRECISION or DOUBLE COMPLEX
bool EquivalenceSets::IsDefaultKindNumericType(const IntrinsicTypeSpec &type) {
  if (auto kind{evaluate::ToInt64(type.kind())}) {
    switch (type.category()) {
    case TypeCategory::Integer:
    case TypeCategory::Logical:
      return *kind == context_.GetDefaultKind(TypeCategory::Integer);
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return *kind == context_.GetDefaultKind(TypeCategory::Real) ||
          *kind == context_.doublePrecisionKind();
    default:
      return false;
    }
  }
  return false;
}

bool EquivalenceSets::IsDefaultNumericSequenceType(const DeclTypeSpec *type) {
  return IsSequenceType(type, [&](const IntrinsicTypeSpec &type) {
    return IsDefaultKindNumericType(type);
  });
}

bool EquivalenceSets::IsAnyNumericSequenceType(const DeclTypeSpec *type) {
  return IsSequenceType(type, [&](const IntrinsicTypeSpec &type) {
    return type.category() == TypeCategory::Logical ||
        common::IsNumericTypeCategory(type.category());
  });
}

// Is type an intrinsic type that satisfies predicate or a sequence type
// whose components do.
bool EquivalenceSets::IsSequenceType(const DeclTypeSpec *type,
    std::function<bool(const IntrinsicTypeSpec &)> predicate) {
  if (!type) {
    return false;
  } else if (const IntrinsicTypeSpec * intrinsic{type->AsIntrinsic()}) {
    return predicate(*intrinsic);
  } else if (const DerivedTypeSpec * derived{type->AsDerived()}) {
    for (const auto &pair : *derived->typeSymbol().scope()) {
      const Symbol &component{*pair.second};
      if (IsAllocatableOrPointer(component) ||
          !IsSequenceType(component.GetType(), predicate)) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

// MapSubprogramToNewSymbols() relies on the following recursive symbol/scope
// copying infrastructure to duplicate an interface's symbols and map all
// of the symbol references in their contained expressions and interfaces
// to the new symbols.

struct SymbolAndTypeMappings {
  std::map<const Symbol *, const Symbol *> symbolMap;
  std::map<const DeclTypeSpec *, const DeclTypeSpec *> typeMap;
};

class SymbolMapper : public evaluate::AnyTraverse<SymbolMapper, bool> {
public:
  using Base = evaluate::AnyTraverse<SymbolMapper, bool>;
  SymbolMapper(Scope &scope, SymbolAndTypeMappings &map)
      : Base{*this}, scope_{scope}, map_{map} {}
  using Base::operator();
  bool operator()(const SymbolRef &ref) {
    if (const Symbol *mapped{MapSymbol(*ref)}) {
      const_cast<SymbolRef &>(ref) = *mapped;
    } else if (ref->has<UseDetails>()) {
      CopySymbol(&*ref);
    }
    return false;
  }
  bool operator()(const Symbol &x) {
    if (MapSymbol(x)) {
      DIE("SymbolMapper hit symbol outside SymbolRef");
    }
    return false;
  }
  void MapSymbolExprs(Symbol &);
  Symbol *CopySymbol(const Symbol *);

private:
  void MapParamValue(ParamValue &param) { (*this)(param.GetExplicit()); }
  void MapBound(Bound &bound) { (*this)(bound.GetExplicit()); }
  void MapShapeSpec(ShapeSpec &spec) {
    MapBound(spec.lbound());
    MapBound(spec.ubound());
  }
  const Symbol *MapSymbol(const Symbol &) const;
  const Symbol *MapSymbol(const Symbol *) const;
  const DeclTypeSpec *MapType(const DeclTypeSpec &);
  const DeclTypeSpec *MapType(const DeclTypeSpec *);
  const Symbol *MapInterface(const Symbol *);

  Scope &scope_;
  SymbolAndTypeMappings &map_;
};

Symbol *SymbolMapper::CopySymbol(const Symbol *symbol) {
  if (symbol) {
    if (auto *subp{symbol->detailsIf<SubprogramDetails>()}) {
      if (subp->isInterface()) {
        if (auto pair{scope_.try_emplace(symbol->name(), symbol->attrs())};
            pair.second) {
          Symbol &copy{*pair.first->second};
          map_.symbolMap[symbol] = &copy;
          copy.set(symbol->test(Symbol::Flag::Subroutine)
                  ? Symbol::Flag::Subroutine
                  : Symbol::Flag::Function);
          Scope &newScope{scope_.MakeScope(Scope::Kind::Subprogram, &copy)};
          copy.set_scope(&newScope);
          copy.set_details(SubprogramDetails{});
          auto &newSubp{copy.get<SubprogramDetails>()};
          newSubp.set_isInterface(true);
          newSubp.set_isDummy(subp->isDummy());
          newSubp.set_defaultIgnoreTKR(subp->defaultIgnoreTKR());
          MapSubprogramToNewSymbols(*symbol, copy, newScope, &map_);
          return &copy;
        }
      }
    } else if (Symbol * copy{scope_.CopySymbol(*symbol)}) {
      map_.symbolMap[symbol] = copy;
      return copy;
    }
  }
  return nullptr;
}

void SymbolMapper::MapSymbolExprs(Symbol &symbol) {
  common::visit(
      common::visitors{[&](ObjectEntityDetails &object) {
                         if (const DeclTypeSpec * type{object.type()}) {
                           if (const DeclTypeSpec * newType{MapType(*type)}) {
                             object.ReplaceType(*newType);
                           }
                         }
                         for (ShapeSpec &spec : object.shape()) {
                           MapShapeSpec(spec);
                         }
                         for (ShapeSpec &spec : object.coshape()) {
                           MapShapeSpec(spec);
                         }
                       },
          [&](ProcEntityDetails &proc) {
            if (const Symbol *
                mappedSymbol{MapInterface(proc.rawProcInterface())}) {
              proc.set_procInterfaces(
                  *mappedSymbol, BypassGeneric(mappedSymbol->GetUltimate()));
            } else if (const DeclTypeSpec * mappedType{MapType(proc.type())}) {
              if (proc.type()) {
                CHECK(*proc.type() == *mappedType);
              } else {
                proc.set_type(*mappedType);
              }
            }
            if (proc.init()) {
              if (const Symbol * mapped{MapSymbol(*proc.init())}) {
                proc.set_init(*mapped);
              }
            }
          },
          [&](const HostAssocDetails &hostAssoc) {
            if (const Symbol * mapped{MapSymbol(hostAssoc.symbol())}) {
              symbol.set_details(HostAssocDetails{*mapped});
            }
          },
          [](const auto &) {}},
      symbol.details());
}

const Symbol *SymbolMapper::MapSymbol(const Symbol &symbol) const {
  if (auto iter{map_.symbolMap.find(&symbol)}; iter != map_.symbolMap.end()) {
    return iter->second;
  }
  return nullptr;
}

const Symbol *SymbolMapper::MapSymbol(const Symbol *symbol) const {
  return symbol ? MapSymbol(*symbol) : nullptr;
}

const DeclTypeSpec *SymbolMapper::MapType(const DeclTypeSpec &type) {
  if (auto iter{map_.typeMap.find(&type)}; iter != map_.typeMap.end()) {
    return iter->second;
  }
  const DeclTypeSpec *newType{nullptr};
  if (type.category() == DeclTypeSpec::Category::Character) {
    const CharacterTypeSpec &charType{type.characterTypeSpec()};
    if (charType.length().GetExplicit()) {
      ParamValue newLen{charType.length()};
      (*this)(newLen.GetExplicit());
      newType = &scope_.MakeCharacterType(
          std::move(newLen), KindExpr{charType.kind()});
    }
  } else if (const DerivedTypeSpec *derived{type.AsDerived()}) {
    if (!derived->parameters().empty()) {
      DerivedTypeSpec newDerived{derived->name(), derived->typeSymbol()};
      newDerived.CookParameters(scope_.context().foldingContext());
      for (const auto &[paramName, paramValue] : derived->parameters()) {
        ParamValue newParamValue{paramValue};
        MapParamValue(newParamValue);
        newDerived.AddParamValue(paramName, std::move(newParamValue));
      }
      // Scope::InstantiateDerivedTypes() instantiates it later.
      newType = &scope_.MakeDerivedType(type.category(), std::move(newDerived));
    }
  }
  if (newType) {
    map_.typeMap[&type] = newType;
  }
  return newType;
}

const DeclTypeSpec *SymbolMapper::MapType(const DeclTypeSpec *type) {
  return type ? MapType(*type) : nullptr;
}

const Symbol *SymbolMapper::MapInterface(const Symbol *interface) {
  if (const Symbol *mapped{MapSymbol(interface)}) {
    return mapped;
  }
  if (interface) {
    if (&interface->owner() != &scope_) {
      return interface;
    } else if (const auto *subp{interface->detailsIf<SubprogramDetails>()};
               subp && subp->isInterface()) {
      return CopySymbol(interface);
    }
  }
  return nullptr;
}

void MapSubprogramToNewSymbols(const Symbol &oldSymbol, Symbol &newSymbol,
    Scope &newScope, SymbolAndTypeMappings *mappings) {
  SymbolAndTypeMappings newMappings;
  if (!mappings) {
    mappings = &newMappings;
  }
  mappings->symbolMap[&oldSymbol] = &newSymbol;
  const auto &oldDetails{oldSymbol.get<SubprogramDetails>()};
  auto &newDetails{newSymbol.get<SubprogramDetails>()};
  SymbolMapper mapper{newScope, *mappings};
  for (const Symbol *dummyArg : oldDetails.dummyArgs()) {
    if (!dummyArg) {
      newDetails.add_alternateReturn();
    } else if (Symbol * copy{mapper.CopySymbol(dummyArg)}) {
      copy->set(Symbol::Flag::Implicit, false);
      newDetails.add_dummyArg(*copy);
      mappings->symbolMap[dummyArg] = copy;
    }
  }
  if (oldDetails.isFunction()) {
    newScope.erase(newSymbol.name());
    const Symbol &result{oldDetails.result()};
    if (Symbol * copy{mapper.CopySymbol(&result)}) {
      newDetails.set_result(*copy);
      mappings->symbolMap[&result] = copy;
    }
  }
  for (auto &[_, ref] : newScope) {
    mapper.MapSymbolExprs(*ref);
  }
  newScope.InstantiateDerivedTypes();
}

} // namespace Fortran::semantics
