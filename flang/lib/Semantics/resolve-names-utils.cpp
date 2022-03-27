//===-- lib/Semantics/resolve-names-utils.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "resolve-names-utils.h"
#include "flang/Common/Fortran-features.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/tools.h"
#include <initializer_list>
#include <variant>

namespace Fortran::semantics {

using common::LanguageFeature;
using common::LogicalOperator;
using common::NumericOperator;
using common::RelationalOperator;
using IntrinsicOperator = parser::DefinedOperator::IntrinsicOperator;

static constexpr const char *operatorPrefix{"operator("};

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
    if (std::find(names.begin(), names.end(), str) != names.end()) {
      return true;
    }
  }
  for (int i{0}; i != common::RelationalOperator_enumSize; ++i) {
    auto names{context.languageFeatures().GetNames(RelationalOperator{i})};
    if (std::find(names.begin(), names.end(), str) != names.end()) {
      return true;
    }
  }
  return false;
}

template <typename E>
std::forward_list<std::string> GetOperatorNames(
    const SemanticsContext &context, E opr) {
  std::forward_list<std::string> result;
  for (const char *name : context.languageFeatures().GetNames(opr)) {
    result.emplace_front(std::string{operatorPrefix} + name + ')');
  }
  return result;
}

std::forward_list<std::string> GetAllNames(
    const SemanticsContext &context, const SourceName &name) {
  std::string str{name.ToString()};
  if (!name.empty() && name.end()[-1] == ')' &&
      name.ToString().rfind(std::string{operatorPrefix}, 0) == 0) {
    for (int i{0}; i != common::LogicalOperator_enumSize; ++i) {
      auto names{GetOperatorNames(context, LogicalOperator{i})};
      if (std::find(names.begin(), names.end(), str) != names.end()) {
        return names;
      }
    }
    for (int i{0}; i != common::RelationalOperator_enumSize; ++i) {
      auto names{GetOperatorNames(context, RelationalOperator{i})};
      if (std::find(names.begin(), names.end(), str) != names.end()) {
        return names;
      }
    }
  }
  return {str};
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
            return GenericKind::DefinedIo::ReadFormatted;
          },
          [&](const parser::GenericSpec::ReadUnformatted &) -> GenericKind {
            return GenericKind::DefinedIo::ReadUnformatted;
          },
          [&](const parser::GenericSpec::WriteFormatted &) -> GenericKind {
            return GenericKind::DefinedIo::WriteFormatted;
          },
          [&](const parser::GenericSpec::WriteUnformatted &) -> GenericKind {
            return GenericKind::DefinedIo::WriteUnformatted;
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

// If SAVE is set on src, set it on all members of dst
static void PropagateSaveAttr(
    const EquivalenceObject &src, EquivalenceSet &dst) {
  if (src.symbol.attrs().test(Attr::SAVE)) {
    for (auto &obj : dst) {
      obj.symbol.attrs().set(Attr::SAVE);
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
    Symbol &symbol{*currObject_.symbol};
    if (!currSet_.empty()) {
      // check this symbol against first of set for compatibility
      Symbol &first{currSet_.front().symbol};
      CheckCanEquivalence(designator.source, first, symbol) &&
          CheckCanEquivalence(designator.source, symbol, first);
    }
    auto subscripts{currObject_.subscripts};
    if (subscripts.empty() && symbol.IsObjectArray()) {
      // record a whole array as its first element
      for (const ShapeSpec &spec : symbol.get<ObjectEntityDetails>().shape()) {
        auto &lbound{spec.lbound().GetExplicit().value()};
        subscripts.push_back(evaluate::ToInt64(lbound).value());
      }
    }
    auto substringStart{currObject_.substringStart};
    currSet_.emplace_back(
        symbol, subscripts, substringStart, designator.source);
    PropagateSaveAttr(currSet_.back(), currSet_);
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
      if (context_.ShouldWarn(LanguageFeature::EquivalenceSameNonSequence)) {
        msg =
            "nonstandard: Equivalence set contains '%s' and '%s' with same "
            "type that is neither numeric nor character sequence type"_port_en_US;
      }
    } else {
      msg = "Equivalence set cannot contain '%s' and '%s' with distinct types "
            "that are not both numeric or character sequence types"_err_en_US;
    }
  } else if (isAnyNum1) {
    if (isChar2) {
      if (context_.ShouldWarn(
              LanguageFeature::EquivalenceNumericWithCharacter)) {
        msg = "nonstandard: Equivalence set contains '%s' that is numeric "
              "sequence type and '%s' that is character"_port_en_US;
      }
    } else if (isAnyNum2 &&
        context_.ShouldWarn(LanguageFeature::EquivalenceNonDefaultNumeric)) {
      if (isDefaultNum1) {
        msg =
            "nonstandard: Equivalence set contains '%s' that is a default "
            "numeric sequence type and '%s' that is numeric with non-default kind"_port_en_US;
      } else if (!isDefaultNum2) {
        msg = "nonstandard: Equivalence set contains '%s' and '%s' that are "
              "numeric sequence types with non-default kinds"_port_en_US;
      }
    }
  }
  if (msg) {
    context_.Say(source, std::move(*msg), sym1.name(), sym2.name());
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
              ok &= CheckSubstringBound(lb->thing.thing.value(), true);
            } else {
              currObject_.substringStart = 1;
            }
            if (const auto &ub{std::get<1>(range.t)}) {
              ok &= CheckSubstringBound(ub->thing.thing.value(), false);
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
                        return CheckArrayBound(y.thing.value());
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

static bool InCommonWithBind(const Symbol &symbol) {
  if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    const Symbol *commonBlock{details->commonBlock()};
    return commonBlock && commonBlock->attrs().test(Attr::BIND_C);
  } else {
    return false;
  }
}

// If symbol can't be in equivalence set report error and return false;
bool EquivalenceSets::CheckObject(const parser::Name &name) {
  if (!name.symbol) {
    return false; // an error has already occurred
  }
  currObject_.symbol = name.symbol;
  parser::MessageFixedText msg;
  const Symbol &symbol{*name.symbol};
  if (symbol.owner().IsDerivedType()) { // C8107
    msg = "Derived type component '%s'"
          " is not allowed in an equivalence set"_err_en_US;
  } else if (IsDummy(symbol)) { // C8106
    msg = "Dummy argument '%s' is not allowed in an equivalence set"_err_en_US;
  } else if (symbol.IsFuncResult()) { // C8106
    msg = "Function result '%s' is not allow in an equivalence set"_err_en_US;
  } else if (IsPointer(symbol)) { // C8106
    msg = "Pointer '%s' is not allowed in an equivalence set"_err_en_US;
  } else if (IsAllocatable(symbol)) { // C8106
    msg = "Allocatable variable '%s'"
          " is not allowed in an equivalence set"_err_en_US;
  } else if (symbol.Corank() > 0) { // C8106
    msg = "Coarray '%s' is not allowed in an equivalence set"_err_en_US;
  } else if (symbol.has<UseDetails>()) { // C8115
    msg = "Use-associated variable '%s'"
          " is not allowed in an equivalence set"_err_en_US;
  } else if (symbol.attrs().test(Attr::BIND_C)) { // C8106
    msg = "Variable '%s' with BIND attribute"
          " is not allowed in an equivalence set"_err_en_US;
  } else if (symbol.attrs().test(Attr::TARGET)) { // C8108
    msg = "Variable '%s' with TARGET attribute"
          " is not allowed in an equivalence set"_err_en_US;
  } else if (IsNamedConstant(symbol)) { // C8106
    msg = "Named constant '%s' is not allowed in an equivalence set"_err_en_US;
  } else if (InCommonWithBind(symbol)) { // C8106
    msg = "Variable '%s' in common block with BIND attribute"
          " is not allowed in an equivalence set"_err_en_US;
  } else if (const auto *type{symbol.GetType()}) {
    if (const auto *derived{type->AsDerived()}) {
      if (const auto *comp{FindUltimateComponent(
              *derived, IsAllocatableOrPointer)}) { // C8106
        msg = IsPointer(*comp)
            ? "Derived type object '%s' with pointer ultimate component"
              " is not allowed in an equivalence set"_err_en_US
            : "Derived type object '%s' with allocatable ultimate component"
              " is not allowed in an equivalence set"_err_en_US;
      } else if (!derived->typeSymbol().get<DerivedTypeDetails>().sequence()) {
        msg = "Nonsequence derived type object '%s'"
              " is not allowed in an equivalence set"_err_en_US;
      }
    } else if (IsAutomatic(symbol)) {
      msg = "Automatic object '%s'"
            " is not allowed in an equivalence set"_err_en_US;
    }
  }
  if (!msg.text().empty()) {
    context_.Say(name.source, std::move(msg), name.source);
    return false;
  }
  return true;
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

} // namespace Fortran::semantics
