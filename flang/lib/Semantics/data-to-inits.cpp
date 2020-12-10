//===-- lib/Semantics/data-to-inits.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DATA statement object/value checking and conversion to static
// initializers
// - Applies specific checks to each scalar element initialization with a
//   constant value or pointer target with class DataInitializationCompiler;
// - Collects the elemental initializations for each symbol and converts them
//   into a single init() expression with member function
//   DataChecker::ConstructInitializer().

#include "data-to-inits.h"
#include "pointer-assignment.h"
#include "flang/Evaluate/fold-designator.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

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
  // If the returned flag is true, emit a warning about CHARACTER misusage.
  std::optional<std::pair<SomeExpr, bool>> ConvertElement(
      const SomeExpr &, const evaluate::DynamicType &);

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

std::optional<std::pair<SomeExpr, bool>>
DataInitializationCompiler::ConvertElement(
    const SomeExpr &expr, const evaluate::DynamicType &type) {
  if (auto converted{evaluate::ConvertToType(type, SomeExpr{expr})}) {
    return {std::make_pair(std::move(*converted), false)};
  }
  if (std::optional<std::string> chValue{evaluate::GetScalarConstantValue<
          evaluate::Type<TypeCategory::Character, 1>>(expr)}) {
    // Allow DATA initialization with Hollerith and kind=1 CHARACTER like
    // (most) other Fortran compilers do.  Pad on the right with spaces
    // when short, truncate the right if long.
    // TODO: big-endian targets
    std::size_t bytes{static_cast<std::size_t>(evaluate::ToInt64(
        type.MeasureSizeInBytes(&exprAnalyzer_.GetFoldingContext()))
                                                   .value())};
    evaluate::BOZLiteralConstant bits{0};
    for (std::size_t j{0}; j < bytes; ++j) {
      char ch{j >= chValue->size() ? ' ' : chValue->at(j)};
      evaluate::BOZLiteralConstant chBOZ{static_cast<unsigned char>(ch)};
      bits = bits.IOR(chBOZ.SHIFTL(8 * j));
    }
    if (auto converted{evaluate::ConvertToType(type, SomeExpr{bits})}) {
      return {std::make_pair(std::move(*converted), true)};
    }
  }
  return std::nullopt;
}

bool DataInitializationCompiler::InitElement(
    const evaluate::OffsetSymbol &offsetSymbol, const SomeExpr &designator) {
  const Symbol &symbol{offsetSymbol.symbol()};
  const Symbol *lastSymbol{GetLastSymbol(designator)};
  bool isPointer{lastSymbol && IsPointer(*lastSymbol)};
  bool isProcPointer{lastSymbol && IsProcedurePointer(*lastSymbol)};
  evaluate::FoldingContext &context{exprAnalyzer_.GetFoldingContext()};
  auto restorer{context.messages().SetLocation(values_.LocateSource())};

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
    auto &symbolInit{inits_.emplace(&symbol, symbol.size()).first->second};
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
    } else if (isProcPointer) {
      if (evaluate::IsProcedure(*expr)) {
        if (CheckPointerAssignment(context, designator, *expr)) {
          GetImage().AddPointer(offsetSymbol.offset(), *expr);
          return true;
        }
      } else {
        exprAnalyzer_.Say(
            "Data object '%s' may not be used to initialize '%s', which is a procedure pointer"_err_en_US,
            expr->AsFortran(), DescribeElement());
      }
    } else if (evaluate::IsProcedure(*expr)) {
      exprAnalyzer_.Say(
          "Procedure '%s' may not be used to initialize '%s', which is not a procedure pointer"_err_en_US,
          expr->AsFortran(), DescribeElement());
    } else if (CheckInitialTarget(context, designator, *expr)) {
      GetImage().AddPointer(offsetSymbol.offset(), *expr);
      return true;
    }
  } else if (evaluate::IsNullPointer(*expr)) {
    exprAnalyzer_.Say("Initializer for '%s' must not be a pointer"_err_en_US,
        DescribeElement());
  } else if (evaluate::IsProcedure(*expr)) {
    exprAnalyzer_.Say("Initializer for '%s' must not be a procedure"_err_en_US,
        DescribeElement());
  } else if (auto designatorType{designator.GetType()}) {
    if (auto converted{ConvertElement(*expr, *designatorType)}) {
      // value non-pointer initialization
      if (std::holds_alternative<evaluate::BOZLiteralConstant>(expr->u) &&
          designatorType->category() != TypeCategory::Integer) { // 8.6.7(11)
        exprAnalyzer_.Say(
            "BOZ literal should appear in a DATA statement only as a value for an integer object, but '%s' is '%s'"_en_US,
            DescribeElement(), designatorType->AsFortran());
      } else if (converted->second) {
        exprAnalyzer_.context().Say(
            "DATA statement value initializes '%s' of type '%s' with CHARACTER"_en_US,
            DescribeElement(), designatorType->AsFortran());
      }
      auto folded{evaluate::Fold(context, std::move(converted->first))};
      switch (
          GetImage().Add(offsetSymbol.offset(), offsetSymbol.size(), folded)) {
      case evaluate::InitialImage::Ok:
        return true;
      case evaluate::InitialImage::NotAConstant:
        exprAnalyzer_.Say(
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

void AccumulateDataInitializations(DataInitializations &inits,
    evaluate::ExpressionAnalyzer &exprAnalyzer,
    const parser::DataStmtSet &set) {
  DataInitializationCompiler scanner{inits, exprAnalyzer, set};
  for (const auto &object :
      std::get<std::list<parser::DataStmtObject>>(set.t)) {
    if (!scanner.Scan(object)) {
      return;
    }
  }
  if (scanner.HasSurplusValues()) {
    exprAnalyzer.context().Say(
        "DATA statement set has more values than objects"_err_en_US);
  }
}

static bool CombineSomeEquivalencedInits(
    DataInitializations &inits, evaluate::ExpressionAnalyzer &exprAnalyzer) {
  auto end{inits.end()};
  for (auto iter{inits.begin()}; iter != end; ++iter) {
    const Symbol &symbol{*iter->first};
    Scope &scope{const_cast<Scope &>(symbol.owner())};
    if (scope.equivalenceSets().empty()) {
      continue; // no problem to solve here
    }
    const auto *commonBlock{FindCommonBlockContaining(symbol)};
    // Sweep following DATA initializations in search of overlapping
    // objects, accumulating into a vector; iterate to a fixed point.
    std::vector<const Symbol *> conflicts;
    auto minStart{symbol.offset()};
    auto maxEnd{symbol.offset() + symbol.size()};
    std::size_t minElementBytes{1};
    while (true) {
      auto prevCount{conflicts.size()};
      conflicts.clear();
      for (auto scan{iter}; ++scan != end;) {
        const Symbol &other{*scan->first};
        const Scope &otherScope{other.owner()};
        if (&otherScope == &scope &&
            FindCommonBlockContaining(other) == commonBlock &&
            maxEnd > other.offset() &&
            other.offset() + other.size() > minStart) {
          // "other" conflicts with "symbol" or another conflict
          conflicts.push_back(&other);
          minStart = std::min(minStart, other.offset());
          maxEnd = std::max(maxEnd, other.offset() + other.size());
        }
      }
      if (conflicts.size() == prevCount) {
        break;
      }
    }
    if (conflicts.empty()) {
      continue;
    }
    // Compute the minimum common granularity
    if (auto dyType{evaluate::DynamicType::From(symbol)}) {
      minElementBytes = evaluate::ToInt64(
          dyType->MeasureSizeInBytes(&exprAnalyzer.GetFoldingContext()))
                            .value_or(1);
    }
    for (const Symbol *s : conflicts) {
      if (auto dyType{evaluate::DynamicType::From(*s)}) {
        minElementBytes = std::min(minElementBytes,
            static_cast<std::size_t>(evaluate::ToInt64(
                dyType->MeasureSizeInBytes(&exprAnalyzer.GetFoldingContext()))
                                         .value_or(1)));
      } else {
        minElementBytes = 1;
      }
    }
    CHECK(minElementBytes > 0);
    CHECK((minElementBytes & (minElementBytes - 1)) == 0);
    auto bytes{static_cast<common::ConstantSubscript>(maxEnd - minStart)};
    CHECK(bytes % minElementBytes == 0);
    const DeclTypeSpec &typeSpec{scope.MakeNumericType(
        TypeCategory::Integer, KindExpr{minElementBytes})};
    // Combine "symbol" and "conflicts[]" into a compiler array temp
    // that overlaps all of them, and merge their initial values into
    // the temp's initializer.
    SourceName name{exprAnalyzer.context().GetTempName(scope)};
    auto emplaced{
        scope.try_emplace(name, Attrs{Attr::SAVE}, ObjectEntityDetails{})};
    CHECK(emplaced.second);
    Symbol &combinedSymbol{*emplaced.first->second};
    auto &details{combinedSymbol.get<ObjectEntityDetails>()};
    combinedSymbol.set_offset(minStart);
    combinedSymbol.set_size(bytes);
    details.set_type(typeSpec);
    ArraySpec arraySpec;
    arraySpec.emplace_back(ShapeSpec::MakeExplicit(Bound{
        bytes / static_cast<common::ConstantSubscript>(minElementBytes)}));
    details.set_shape(arraySpec);
    if (commonBlock) {
      details.set_commonBlock(*commonBlock);
    }
    // Merge these EQUIVALENCE'd DATA initializations, and remove the
    // original initializations from the map.
    auto combinedInit{
        inits.emplace(&combinedSymbol, static_cast<std::size_t>(bytes))};
    evaluate::InitialImage &combined{combinedInit.first->second.image};
    combined.Incorporate(symbol.offset() - minStart, iter->second.image);
    inits.erase(iter);
    for (const Symbol *s : conflicts) {
      auto sIter{inits.find(s)};
      CHECK(sIter != inits.end());
      combined.Incorporate(s->offset() - minStart, sIter->second.image);
      inits.erase(sIter);
    }
    return true; // got one
  }
  return false; // no remaining EQUIVALENCE'd DATA initializations
}

// Converts the initialization image for all the DATA statement appearances of
// a single symbol into an init() expression in the symbol table entry.
void ConstructInitializer(const Symbol &symbol,
    SymbolDataInitialization &initialization,
    evaluate::ExpressionAnalyzer &exprAnalyzer) {
  auto &context{exprAnalyzer.GetFoldingContext()};
  initialization.inits.sort();
  ConstantSubscript next{0};
  for (const auto &init : initialization.inits) {
    if (init.start() < next) {
      auto badDesignator{evaluate::OffsetToDesignator(
          context, symbol, init.start(), init.size())};
      CHECK(badDesignator);
      exprAnalyzer.Say(symbol.name(),
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
      } else {
        if (auto extents{evaluate::GetConstantExtents(context, symbol)}) {
          mutableObject.set_init(
              initialization.image.AsConstant(context, *symbolType, *extents));
        } else {
          exprAnalyzer.Say(symbol.name(),
              "internal: unknown shape for '%s' while constructing initializer from DATA"_err_en_US,
              symbol.name());
          return;
        }
      }
    } else {
      exprAnalyzer.Say(symbol.name(),
          "internal: no type for '%s' while constructing initializer from DATA"_err_en_US,
          symbol.name());
      return;
    }
    if (!object->init()) {
      exprAnalyzer.Say(symbol.name(),
          "internal: could not construct an initializer from DATA statements for '%s'"_err_en_US,
          symbol.name());
    }
  } else {
    CHECK(exprAnalyzer.context().AnyFatalError());
  }
}

void ConvertToInitializers(
    DataInitializations &inits, evaluate::ExpressionAnalyzer &exprAnalyzer) {
  while (CombineSomeEquivalencedInits(inits, exprAnalyzer)) {
  }
  for (auto &[symbolPtr, initialization] : inits) {
    ConstructInitializer(*symbolPtr, initialization, exprAnalyzer);
  }
}
} // namespace Fortran::semantics
