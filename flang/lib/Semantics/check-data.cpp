//===-- lib/Semantics/check-data.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DATA statement semantic analysis.
// - Applies static semantic checks to the variables in each data-stmt-set with
//   class DataVarChecker;
// - Applies specific checks to each scalar element initialization with a
//   constant value or pointer tareg with class DataInitializationCompiler;
// - Collects the elemental initializations for each symbol and converts them
//   into a single init() expression with member function
//   DataChecker::ConstructInitializer().

#include "check-data.h"
#include "pointer-assignment.h"
#include "flang/Evaluate/fold-designator.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

// Ensures that references to an implied DO loop control variable are
// represented as such in the "body" of the implied DO loop.
void DataChecker::Enter(const parser::DataImpliedDo &x) {
  auto name{std::get<parser::DataImpliedDo::Bounds>(x.t).name.thing.thing};
  int kind{evaluate::ResultType<evaluate::ImpliedDoIndex>::kind};
  if (const auto dynamicType{evaluate::DynamicType::From(*name.symbol)}) {
    if (dynamicType->category() == TypeCategory::Integer) {
      kind = dynamicType->kind();
    }
  }
  exprAnalyzer_.AddImpliedDo(name.source, kind);
}

void DataChecker::Leave(const parser::DataImpliedDo &x) {
  auto name{std::get<parser::DataImpliedDo::Bounds>(x.t).name.thing.thing};
  exprAnalyzer_.RemoveImpliedDo(name.source);
}

// DataVarChecker applies static checks once to each variable that appears
// in a data-stmt-set.  These checks are independent of the values that
// correspond to the variables.
class DataVarChecker : public evaluate::AllTraverse<DataVarChecker, true> {
public:
  using Base = evaluate::AllTraverse<DataVarChecker, true>;
  DataVarChecker(SemanticsContext &c, parser::CharBlock src)
      : Base{*this}, context_{c}, source_{src} {}
  using Base::operator();
  bool HasComponentWithoutSubscripts() const {
    return hasComponent_ && !hasSubscript_;
  }
  bool operator()(const Symbol &symbol) { // C876
    // 8.6.7p(2) - precludes non-pointers of derived types with
    // default component values
    const Scope &scope{context_.FindScope(source_)};
    bool isFirstSymbol{isFirstSymbol_};
    isFirstSymbol_ = false;
    if (const char *whyNot{IsAutomatic(symbol) ? "Automatic variable"
                : IsDummy(symbol)              ? "Dummy argument"
                : IsFunctionResult(symbol)     ? "Function result"
                : IsAllocatable(symbol)        ? "Allocatable"
                : IsInitialized(symbol, true)  ? "Default-initialized"
                : IsInBlankCommon(symbol)      ? "Blank COMMON object"
                : IsProcedure(symbol) && !IsPointer(symbol) ? "Procedure"
                // remaining checks don't apply to components
                : !isFirstSymbol                  ? nullptr
                : IsHostAssociated(symbol, scope) ? "Host-associated object"
                : IsUseAssociated(symbol, scope)  ? "USE-associated object"
                                                  : nullptr}) {
      context_.Say(source_,
          "%s '%s' must not be initialized in a DATA statement"_err_en_US,
          whyNot, symbol.name());
      return false;
    } else if (IsProcedurePointer(symbol)) {
      context_.Say(source_,
          "Procedure pointer '%s' in a DATA statement is not standard"_en_US,
          symbol.name());
    }
    return true;
  }
  bool operator()(const evaluate::Component &component) {
    hasComponent_ = true;
    const Symbol &lastSymbol{component.GetLastSymbol()};
    if (isPointerAllowed_) {
      if (IsPointer(lastSymbol) && hasSubscript_) { // C877
        context_.Say(source_,
            "Rightmost data object pointer '%s' must not be subscripted"_err_en_US,
            lastSymbol.name().ToString());
        return false;
      }
      RestrictPointer();
    } else {
      if (IsPointer(lastSymbol)) { // C877
        context_.Say(source_,
            "Data object must not contain pointer '%s' as a non-rightmost part"_err_en_US,
            lastSymbol.name().ToString());
        return false;
      }
    }
    return (*this)(component.base()) && (*this)(lastSymbol);
  }
  bool operator()(const evaluate::ArrayRef &arrayRef) {
    hasSubscript_ = true;
    return (*this)(arrayRef.base()) && (*this)(arrayRef.subscript());
  }
  bool operator()(const evaluate::Substring &substring) {
    hasSubscript_ = true;
    return (*this)(substring.parent()) && (*this)(substring.lower()) &&
        (*this)(substring.upper());
  }
  bool operator()(const evaluate::CoarrayRef &) { // C874
    context_.Say(
        source_, "Data object must not be a coindexed variable"_err_en_US);
    return false;
  }
  bool operator()(const evaluate::Subscript &subs) {
    DataVarChecker subscriptChecker{context_, source_};
    subscriptChecker.RestrictPointer();
    return std::visit(
               common::visitors{
                   [&](const evaluate::IndirectSubscriptIntegerExpr &expr) {
                     return CheckSubscriptExpr(expr);
                   },
                   [&](const evaluate::Triplet &triplet) {
                     return CheckSubscriptExpr(triplet.lower()) &&
                         CheckSubscriptExpr(triplet.upper()) &&
                         CheckSubscriptExpr(triplet.stride());
                   },
               },
               subs.u) &&
        subscriptChecker(subs.u);
  }
  template <typename T>
  bool operator()(const evaluate::FunctionRef<T> &) const { // C875
    context_.Say(source_,
        "Data object variable must not be a function reference"_err_en_US);
    return false;
  }
  void RestrictPointer() { isPointerAllowed_ = false; }

private:
  bool CheckSubscriptExpr(
      const std::optional<evaluate::IndirectSubscriptIntegerExpr> &x) const {
    return !x || CheckSubscriptExpr(*x);
  }
  bool CheckSubscriptExpr(
      const evaluate::IndirectSubscriptIntegerExpr &expr) const {
    return CheckSubscriptExpr(expr.value());
  }
  bool CheckSubscriptExpr(
      const evaluate::Expr<evaluate::SubscriptInteger> &expr) const {
    if (!evaluate::IsConstantExpr(expr)) { // C875,C881
      context_.Say(
          source_, "Data object must have constant subscripts"_err_en_US);
      return false;
    } else {
      return true;
    }
  }

  SemanticsContext &context_;
  parser::CharBlock source_;
  bool hasComponent_{false};
  bool hasSubscript_{false};
  bool isPointerAllowed_{true};
  bool isFirstSymbol_{true};
};

void DataChecker::Leave(const parser::DataIDoObject &object) {
  if (const auto *designator{
          std::get_if<parser::Scalar<common::Indirection<parser::Designator>>>(
              &object.u)}) {
    if (MaybeExpr expr{exprAnalyzer_.Analyze(*designator)}) {
      auto source{designator->thing.value().source};
      if (evaluate::IsConstantExpr(*expr)) { // C878,C879
        exprAnalyzer_.context().Say(
            source, "Data implied do object must be a variable"_err_en_US);
      } else {
        DataVarChecker checker{exprAnalyzer_.context(), source};
        if (checker(*expr)) {
          if (checker.HasComponentWithoutSubscripts()) { // C880
            exprAnalyzer_.context().Say(source,
                "Data implied do structure component must be subscripted"_err_en_US);
          } else {
            return;
          }
        }
      }
    }
  }
  currentSetHasFatalErrors_ = true;
}

void DataChecker::Leave(const parser::DataStmtObject &dataObject) {
  std::visit(common::visitors{
                 [](const parser::DataImpliedDo &) { // has own Enter()/Leave()
                 },
                 [&](const auto &var) {
                   auto expr{exprAnalyzer_.Analyze(var)};
                   if (!expr ||
                       !DataVarChecker{exprAnalyzer_.context(),
                           parser::FindSourceLocation(dataObject)}(*expr)) {
                     currentSetHasFatalErrors_ = true;
                   }
                 },
             },
      dataObject.u);
}

// Steps through a list of values in a DATA statement set; implements
// repetition.
class ValueListIterator {
public:
  explicit ValueListIterator(const parser::DataStmtSet &set)
      : end_{std::get<std::list<parser::DataStmtValue>>(set.t).end()},
        at_{std::get<std::list<parser::DataStmtValue>>(set.t).begin()} {
    SetRepetitionCount();
  }
  bool hasFatalError() const { return hasFatalError_; }
  bool IsAtEnd() const { return at_ == end_; }
  const SomeExpr *operator*() const { return GetExpr(GetConstant()); }
  parser::CharBlock LocateSource() const { return GetConstant().source; }
  ValueListIterator &operator++() {
    if (repetitionsRemaining_ > 0) {
      --repetitionsRemaining_;
    } else if (at_ != end_) {
      ++at_;
      SetRepetitionCount();
    }
    return *this;
  }

private:
  using listIterator = std::list<parser::DataStmtValue>::const_iterator;
  void SetRepetitionCount();
  const parser::DataStmtConstant &GetConstant() const {
    return std::get<parser::DataStmtConstant>(at_->t);
  }

  listIterator end_;
  listIterator at_;
  ConstantSubscript repetitionsRemaining_{0};
  bool hasFatalError_{false};
};

void ValueListIterator::SetRepetitionCount() {
  for (repetitionsRemaining_ = 1; at_ != end_; ++at_) {
    if (at_->repetitions < 0) {
      hasFatalError_ = true;
    }
    if (at_->repetitions > 0) {
      repetitionsRemaining_ = at_->repetitions - 1;
      return;
    }
  }
  repetitionsRemaining_ = 0;
}

// Collects all of the elemental initializations from DATA statements
// into a single image for each symbol that appears in any DATA.
// Expands the implied DO loops and array references.
// Applies checks that validate each distinct elemental initialization
// of the variables in a data-stmt-set, as well as those that apply
// to the corresponding values being use to initialize each element.
class DataInitializationCompiler {
public:
  DataInitializationCompiler(DataInitializations &inits,
      evaluate::ExpressionAnalyzer &a, const parser::DataStmtSet &set)
      : inits_{inits}, exprAnalyzer_{a}, values_{set} {}
  const DataInitializations &inits() const { return inits_; }
  bool HasSurplusValues() const { return !values_.IsAtEnd(); }
  bool Scan(const parser::DataStmtObject &);

private:
  bool Scan(const parser::Variable &);
  bool Scan(const parser::Designator &);
  bool Scan(const parser::DataImpliedDo &);
  bool Scan(const parser::DataIDoObject &);

  // Initializes all elements of a designator, which can be an array or section.
  bool InitDesignator(const SomeExpr &);
  // Initializes a single object.
  bool InitElement(const evaluate::OffsetSymbol &, const SomeExpr &designator);

  DataInitializations &inits_;
  evaluate::ExpressionAnalyzer &exprAnalyzer_;
  ValueListIterator values_;
};

bool DataInitializationCompiler::Scan(const parser::DataStmtObject &object) {
  return std::visit(
      common::visitors{
          [&](const common::Indirection<parser::Variable> &var) {
            return Scan(var.value());
          },
          [&](const parser::DataImpliedDo &ido) { return Scan(ido); },
      },
      object.u);
}

bool DataInitializationCompiler::Scan(const parser::Variable &var) {
  if (const auto *expr{GetExpr(var)}) {
    exprAnalyzer_.GetFoldingContext().messages().SetLocation(var.GetSource());
    if (InitDesignator(*expr)) {
      return true;
    }
  }
  return false;
}

bool DataInitializationCompiler::Scan(const parser::Designator &designator) {
  if (auto expr{exprAnalyzer_.Analyze(designator)}) {
    exprAnalyzer_.GetFoldingContext().messages().SetLocation(
        parser::FindSourceLocation(designator));
    if (InitDesignator(*expr)) {
      return true;
    }
  }
  return false;
}

bool DataInitializationCompiler::Scan(const parser::DataImpliedDo &ido) {
  const auto &bounds{std::get<parser::DataImpliedDo::Bounds>(ido.t)};
  auto name{bounds.name.thing.thing};
  const auto *lowerExpr{GetExpr(bounds.lower.thing.thing)};
  const auto *upperExpr{GetExpr(bounds.upper.thing.thing)};
  const auto *stepExpr{
      bounds.step ? GetExpr(bounds.step->thing.thing) : nullptr};
  if (lowerExpr && upperExpr) {
    auto lower{ToInt64(*lowerExpr)};
    auto upper{ToInt64(*upperExpr)};
    auto step{stepExpr ? ToInt64(*stepExpr) : std::nullopt};
    auto stepVal{step.value_or(1)};
    if (stepVal == 0) {
      exprAnalyzer_.Say(name.source,
          "DATA statement implied DO loop has a step value of zero"_err_en_US);
    } else if (lower && upper) {
      int kind{evaluate::ResultType<evaluate::ImpliedDoIndex>::kind};
      if (const auto dynamicType{evaluate::DynamicType::From(*name.symbol)}) {
        if (dynamicType->category() == TypeCategory::Integer) {
          kind = dynamicType->kind();
        }
      }
      if (exprAnalyzer_.AddImpliedDo(name.source, kind)) {
        auto &value{exprAnalyzer_.GetFoldingContext().StartImpliedDo(
            name.source, *lower)};
        bool result{true};
        for (auto n{(*upper - value + stepVal) / stepVal}; n > 0;
             --n, value += stepVal) {
          for (const auto &object :
              std::get<std::list<parser::DataIDoObject>>(ido.t)) {
            if (!Scan(object)) {
              result = false;
              break;
            }
          }
        }
        exprAnalyzer_.GetFoldingContext().EndImpliedDo(name.source);
        exprAnalyzer_.RemoveImpliedDo(name.source);
        return result;
      }
    }
  }
  return false;
}

bool DataInitializationCompiler::Scan(const parser::DataIDoObject &object) {
  return std::visit(
      common::visitors{
          [&](const parser::Scalar<common::Indirection<parser::Designator>>
                  &var) { return Scan(var.thing.value()); },
          [&](const common::Indirection<parser::DataImpliedDo> &ido) {
            return Scan(ido.value());
          },
      },
      object.u);
}

bool DataInitializationCompiler::InitDesignator(const SomeExpr &designator) {
  evaluate::FoldingContext &context{exprAnalyzer_.GetFoldingContext()};
  evaluate::DesignatorFolder folder{context};
  while (auto offsetSymbol{folder.FoldDesignator(designator)}) {
    if (folder.isOutOfRange()) {
      if (auto bad{evaluate::OffsetToDesignator(context, *offsetSymbol)}) {
        exprAnalyzer_.context().Say(
            "DATA statement designator '%s' is out of range"_err_en_US,
            bad->AsFortran());
      } else {
        exprAnalyzer_.context().Say(
            "DATA statement designator '%s' is out of range"_err_en_US,
            designator.AsFortran());
      }
      return false;
    } else if (!InitElement(*offsetSymbol, designator)) {
      return false;
    } else {
      ++values_;
    }
  }
  return folder.isEmpty();
}

bool DataInitializationCompiler::InitElement(
    const evaluate::OffsetSymbol &offsetSymbol, const SomeExpr &designator) {
  const Symbol &symbol{offsetSymbol.symbol()};
  const Symbol *lastSymbol{GetLastSymbol(designator)};
  bool isPointer{lastSymbol && IsPointer(*lastSymbol)};
  bool isProcPointer{lastSymbol && IsProcedurePointer(*lastSymbol)};
  evaluate::FoldingContext &context{exprAnalyzer_.GetFoldingContext()};

  const auto DescribeElement{[&]() {
    if (auto badDesignator{
            evaluate::OffsetToDesignator(context, offsetSymbol)}) {
      return badDesignator->AsFortran();
    } else {
      // Error recovery
      std::string buf;
      llvm::raw_string_ostream ss{buf};
      ss << offsetSymbol.symbol().name() << " offset " << offsetSymbol.offset()
         << " bytes for " << offsetSymbol.size() << " bytes";
      return ss.str();
    }
  }};
  const auto GetImage{[&]() -> evaluate::InitialImage & {
    auto &symbolInit{inits_.emplace(symbol, symbol.size()).first->second};
    symbolInit.inits.emplace_back(offsetSymbol.offset(), offsetSymbol.size());
    return symbolInit.image;
  }};
  const auto OutOfRangeError{[&]() {
    evaluate::AttachDeclaration(
        exprAnalyzer_.context().Say(
            "DATA statement designator '%s' is out of range for its variable '%s'"_err_en_US,
            DescribeElement(), symbol.name()),
        symbol);
  }};

  if (values_.hasFatalError()) {
    return false;
  } else if (values_.IsAtEnd()) {
    exprAnalyzer_.context().Say(
        "DATA statement set has no value for '%s'"_err_en_US,
        DescribeElement());
    return false;
  } else if (static_cast<std::size_t>(
                 offsetSymbol.offset() + offsetSymbol.size()) > symbol.size()) {
    OutOfRangeError();
    return false;
  }

  const SomeExpr *expr{*values_};
  if (!expr) {
    CHECK(exprAnalyzer_.context().AnyFatalError());
  } else if (isPointer) {
    if (static_cast<std::size_t>(offsetSymbol.offset() + offsetSymbol.size()) >
        symbol.size()) {
      OutOfRangeError();
    } else if (evaluate::IsNullPointer(*expr)) {
      // nothing to do; rely on zero initialization
      return true;
    } else if (evaluate::IsProcedure(*expr)) {
      if (isProcPointer) {
        if (CheckPointerAssignment(context, designator, *expr)) {
          GetImage().AddPointer(offsetSymbol.offset(), *expr);
          return true;
        }
      } else {
        exprAnalyzer_.Say(values_.LocateSource(),
            "Procedure '%s' may not be used to initialize '%s', which is not a procedure pointer"_err_en_US,
            expr->AsFortran(), DescribeElement());
      }
    } else if (isProcPointer) {
      exprAnalyzer_.Say(values_.LocateSource(),
          "Data object '%s' may not be used to initialize '%s', which is a procedure pointer"_err_en_US,
          expr->AsFortran(), DescribeElement());
    } else if (CheckInitialTarget(context, designator, *expr)) {
      GetImage().AddPointer(offsetSymbol.offset(), *expr);
      return true;
    }
  } else if (evaluate::IsNullPointer(*expr)) {
    exprAnalyzer_.Say(values_.LocateSource(),
        "Initializer for '%s' must not be a pointer"_err_en_US,
        DescribeElement());
  } else if (evaluate::IsProcedure(*expr)) {
    exprAnalyzer_.Say(values_.LocateSource(),
        "Initializer for '%s' must not be a procedure"_err_en_US,
        DescribeElement());
  } else if (auto designatorType{designator.GetType()}) {
    if (auto converted{
            evaluate::ConvertToType(*designatorType, SomeExpr{*expr})}) {
      // value non-pointer initialization
      if (std::holds_alternative<evaluate::BOZLiteralConstant>(expr->u) &&
          designatorType->category() != TypeCategory::Integer) { // 8.6.7(11)
        exprAnalyzer_.Say(values_.LocateSource(),
            "BOZ literal should appear in a DATA statement only as a value for an integer object, but '%s' is '%s'"_en_US,
            DescribeElement(), designatorType->AsFortran());
      }
      auto folded{evaluate::Fold(context, std::move(*converted))};
      switch (
          GetImage().Add(offsetSymbol.offset(), offsetSymbol.size(), folded)) {
      case evaluate::InitialImage::Ok:
        return true;
      case evaluate::InitialImage::NotAConstant:
        exprAnalyzer_.Say(values_.LocateSource(),
            "DATA statement value '%s' for '%s' is not a constant"_err_en_US,
            folded.AsFortran(), DescribeElement());
        break;
      case evaluate::InitialImage::OutOfRange:
        OutOfRangeError();
        break;
      default:
        CHECK(exprAnalyzer_.context().AnyFatalError());
        break;
      }
    } else {
      exprAnalyzer_.context().Say(
          "DATA statement value could not be converted to the type '%s' of the object '%s'"_err_en_US,
          designatorType->AsFortran(), DescribeElement());
    }
  } else {
    CHECK(exprAnalyzer_.context().AnyFatalError());
  }
  return false;
}

void DataChecker::Leave(const parser::DataStmtSet &set) {
  if (!currentSetHasFatalErrors_) {
    DataInitializationCompiler scanner{inits_, exprAnalyzer_, set};
    for (const auto &object :
        std::get<std::list<parser::DataStmtObject>>(set.t)) {
      if (!scanner.Scan(object)) {
        return;
      }
    }
    if (scanner.HasSurplusValues()) {
      exprAnalyzer_.context().Say(
          "DATA statement set has more values than objects"_err_en_US);
    }
  }
  currentSetHasFatalErrors_ = false;
}

// Converts the initialization image for all the DATA statement appearances of
// a single symbol into an init() expression in the symbol table entry.
void DataChecker::ConstructInitializer(
    const Symbol &symbol, SymbolDataInitialization &initialization) {
  auto &context{exprAnalyzer_.GetFoldingContext()};
  initialization.inits.sort();
  ConstantSubscript next{0};
  for (const auto &init : initialization.inits) {
    if (init.start() < next) {
      auto badDesignator{evaluate::OffsetToDesignator(
          context, symbol, init.start(), init.size())};
      CHECK(badDesignator);
      exprAnalyzer_.Say(symbol.name(),
          "DATA statement initializations affect '%s' more than once"_err_en_US,
          badDesignator->AsFortran());
    }
    next = init.start() + init.size();
    CHECK(next <= static_cast<ConstantSubscript>(initialization.image.size()));
  }
  if (const auto *proc{symbol.detailsIf<ProcEntityDetails>()}) {
    CHECK(IsProcedurePointer(symbol));
    const auto &procDesignator{initialization.image.AsConstantProcPointer()};
    CHECK(!procDesignator.GetComponent());
    auto &mutableProc{const_cast<ProcEntityDetails &>(*proc)};
    mutableProc.set_init(DEREF(procDesignator.GetSymbol()));
  } else if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (auto symbolType{evaluate::DynamicType::From(symbol)}) {
      auto &mutableObject{const_cast<ObjectEntityDetails &>(*object)};
      if (IsPointer(symbol)) {
        mutableObject.set_init(
            initialization.image.AsConstantDataPointer(*symbolType));
        mutableObject.set_initWasValidated();
      } else {
        if (auto extents{evaluate::GetConstantExtents(context, symbol)}) {
          mutableObject.set_init(
              initialization.image.AsConstant(context, *symbolType, *extents));
          mutableObject.set_initWasValidated();
        } else {
          exprAnalyzer_.Say(symbol.name(),
              "internal: unknown shape for '%s' while constructing initializer from DATA"_err_en_US,
              symbol.name());
          return;
        }
      }
    } else {
      exprAnalyzer_.Say(symbol.name(),
          "internal: no type for '%s' while constructing initializer from DATA"_err_en_US,
          symbol.name());
      return;
    }
    if (!object->init()) {
      exprAnalyzer_.Say(symbol.name(),
          "internal: could not construct an initializer from DATA statements for '%s'"_err_en_US,
          symbol.name());
    }
  } else {
    CHECK(exprAnalyzer_.context().AnyFatalError());
  }
}

void DataChecker::CompileDataInitializationsIntoInitializers() {
  for (auto &[symbolRef, initialization] : inits_) {
    ConstructInitializer(*symbolRef, initialization);
  }
}

} // namespace Fortran::semantics
