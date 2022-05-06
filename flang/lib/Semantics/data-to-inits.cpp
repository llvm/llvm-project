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
#include "flang/Evaluate/tools.h"
#include "flang/Semantics/tools.h"

// The job of generating explicit static initializers for objects that don't
// have them in order to implement default component initialization is now being
// done in lowering, so don't do it here in semantics; but the code remains here
// in case we change our minds.
static constexpr bool makeDefaultInitializationExplicit{false};

// Whether to delete the original "init()" initializers from storage-associated
// objects and pointers.
static constexpr bool removeOriginalInits{false};

namespace Fortran::semantics {

// Steps through a list of values in a DATA statement set; implements
// repetition.
template <typename DSV = parser::DataStmtValue> class ValueListIterator {
public:
  ValueListIterator(SemanticsContext &context, const std::list<DSV> &list)
      : context_{context}, end_{list.end()}, at_{list.begin()} {
    SetRepetitionCount();
  }
  bool hasFatalError() const { return hasFatalError_; }
  bool IsAtEnd() const { return at_ == end_; }
  const SomeExpr *operator*() const { return GetExpr(context_, GetConstant()); }
  std::optional<parser::CharBlock> LocateSource() const {
    if (!hasFatalError_) {
      return GetConstant().source;
    }
    return {};
  }
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
  using listIterator = typename std::list<DSV>::const_iterator;
  void SetRepetitionCount();
  const parser::DataStmtValue &GetValue() const {
    return DEREF(common::Unwrap<const parser::DataStmtValue>(*at_));
  }
  const parser::DataStmtConstant &GetConstant() const {
    return std::get<parser::DataStmtConstant>(GetValue().t);
  }

  SemanticsContext &context_;
  listIterator end_, at_;
  ConstantSubscript repetitionsRemaining_{0};
  bool hasFatalError_{false};
};

template <typename DSV> void ValueListIterator<DSV>::SetRepetitionCount() {
  for (repetitionsRemaining_ = 1; at_ != end_; ++at_) {
    auto repetitions{GetValue().repetitions};
    if (repetitions < 0) {
      hasFatalError_ = true;
    } else if (repetitions > 0) {
      repetitionsRemaining_ = repetitions - 1;
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
// to the corresponding values being used to initialize each element.
template <typename DSV = parser::DataStmtValue>
class DataInitializationCompiler {
public:
  DataInitializationCompiler(DataInitializations &inits,
      evaluate::ExpressionAnalyzer &a, const std::list<DSV> &list)
      : inits_{inits}, exprAnalyzer_{a}, values_{a.context(), list} {}
  const DataInitializations &inits() const { return inits_; }
  bool HasSurplusValues() const { return !values_.IsAtEnd(); }
  bool Scan(const parser::DataStmtObject &);
  // Initializes all elements of whole variable or component
  bool Scan(const Symbol &);

private:
  bool Scan(const parser::Variable &);
  bool Scan(const parser::Designator &);
  bool Scan(const parser::DataImpliedDo &);
  bool Scan(const parser::DataIDoObject &);

  // Initializes all elements of a designator, which can be an array or section.
  bool InitDesignator(const SomeExpr &);
  // Initializes a single scalar object.
  bool InitElement(const evaluate::OffsetSymbol &, const SomeExpr &designator);
  // If the returned flag is true, emit a warning about CHARACTER misusage.
  std::optional<std::pair<SomeExpr, bool>> ConvertElement(
      const SomeExpr &, const evaluate::DynamicType &);

  DataInitializations &inits_;
  evaluate::ExpressionAnalyzer &exprAnalyzer_;
  ValueListIterator<DSV> values_;
};

template <typename DSV>
bool DataInitializationCompiler<DSV>::Scan(
    const parser::DataStmtObject &object) {
  return common::visit(
      common::visitors{
          [&](const common::Indirection<parser::Variable> &var) {
            return Scan(var.value());
          },
          [&](const parser::DataImpliedDo &ido) { return Scan(ido); },
      },
      object.u);
}

template <typename DSV>
bool DataInitializationCompiler<DSV>::Scan(const parser::Variable &var) {
  if (const auto *expr{GetExpr(exprAnalyzer_.context(), var)}) {
    exprAnalyzer_.GetFoldingContext().messages().SetLocation(var.GetSource());
    if (InitDesignator(*expr)) {
      return true;
    }
  }
  return false;
}

template <typename DSV>
bool DataInitializationCompiler<DSV>::Scan(
    const parser::Designator &designator) {
  if (auto expr{exprAnalyzer_.Analyze(designator)}) {
    exprAnalyzer_.GetFoldingContext().messages().SetLocation(
        parser::FindSourceLocation(designator));
    if (InitDesignator(*expr)) {
      return true;
    }
  }
  return false;
}

template <typename DSV>
bool DataInitializationCompiler<DSV>::Scan(const parser::DataImpliedDo &ido) {
  const auto &bounds{std::get<parser::DataImpliedDo::Bounds>(ido.t)};
  auto name{bounds.name.thing.thing};
  const auto *lowerExpr{
      GetExpr(exprAnalyzer_.context(), bounds.lower.thing.thing)};
  const auto *upperExpr{
      GetExpr(exprAnalyzer_.context(), bounds.upper.thing.thing)};
  const auto *stepExpr{bounds.step
          ? GetExpr(exprAnalyzer_.context(), bounds.step->thing.thing)
          : nullptr};
  if (lowerExpr && upperExpr) {
    // Fold the bounds expressions (again) in case any of them depend
    // on outer implied DO loops.
    evaluate::FoldingContext &context{exprAnalyzer_.GetFoldingContext()};
    std::int64_t stepVal{1};
    if (stepExpr) {
      auto foldedStep{evaluate::Fold(context, SomeExpr{*stepExpr})};
      stepVal = ToInt64(foldedStep).value_or(1);
      if (stepVal == 0) {
        exprAnalyzer_.Say(name.source,
            "DATA statement implied DO loop has a step value of zero"_err_en_US);
        return false;
      }
    }
    auto foldedLower{evaluate::Fold(context, SomeExpr{*lowerExpr})};
    auto lower{ToInt64(foldedLower)};
    auto foldedUpper{evaluate::Fold(context, SomeExpr{*upperExpr})};
    auto upper{ToInt64(foldedUpper)};
    if (lower && upper) {
      int kind{evaluate::ResultType<evaluate::ImpliedDoIndex>::kind};
      if (const auto dynamicType{evaluate::DynamicType::From(*name.symbol)}) {
        if (dynamicType->category() == TypeCategory::Integer) {
          kind = dynamicType->kind();
        }
      }
      if (exprAnalyzer_.AddImpliedDo(name.source, kind)) {
        auto &value{context.StartImpliedDo(name.source, *lower)};
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
        context.EndImpliedDo(name.source);
        exprAnalyzer_.RemoveImpliedDo(name.source);
        return result;
      }
    }
  }
  return false;
}

template <typename DSV>
bool DataInitializationCompiler<DSV>::Scan(
    const parser::DataIDoObject &object) {
  return common::visit(
      common::visitors{
          [&](const parser::Scalar<common::Indirection<parser::Designator>>
                  &var) { return Scan(var.thing.value()); },
          [&](const common::Indirection<parser::DataImpliedDo> &ido) {
            return Scan(ido.value());
          },
      },
      object.u);
}

template <typename DSV>
bool DataInitializationCompiler<DSV>::Scan(const Symbol &symbol) {
  auto designator{exprAnalyzer_.Designate(evaluate::DataRef{symbol})};
  CHECK(designator.has_value());
  return InitDesignator(*designator);
}

template <typename DSV>
bool DataInitializationCompiler<DSV>::InitDesignator(
    const SomeExpr &designator) {
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

template <typename DSV>
std::optional<std::pair<SomeExpr, bool>>
DataInitializationCompiler<DSV>::ConvertElement(
    const SomeExpr &expr, const evaluate::DynamicType &type) {
  if (auto converted{evaluate::ConvertToType(type, SomeExpr{expr})}) {
    return {std::make_pair(std::move(*converted), false)};
  }
  if (std::optional<std::string> chValue{
          evaluate::GetScalarConstantValue<evaluate::Ascii>(expr)}) {
    // Allow DATA initialization with Hollerith and kind=1 CHARACTER like
    // (most) other Fortran compilers do.  Pad on the right with spaces
    // when short, truncate the right if long.
    // TODO: big-endian targets
    auto bytes{static_cast<std::size_t>(evaluate::ToInt64(
        type.MeasureSizeInBytes(exprAnalyzer_.GetFoldingContext(), false))
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
  SemanticsContext &context{exprAnalyzer_.context()};
  if (context.IsEnabled(common::LanguageFeature::LogicalIntegerAssignment)) {
    if (MaybeExpr converted{evaluate::DataConstantConversionExtension(
            exprAnalyzer_.GetFoldingContext(), type, expr)}) {
      if (context.ShouldWarn(
              common::LanguageFeature::LogicalIntegerAssignment)) {
        context.Say(
            "nonstandard usage: initialization of %s with %s"_port_en_US,
            type.AsFortran(), expr.GetType().value().AsFortran());
      }
      return {std::make_pair(std::move(*converted), false)};
    }
  }
  return std::nullopt;
}

template <typename DSV>
bool DataInitializationCompiler<DSV>::InitElement(
    const evaluate::OffsetSymbol &offsetSymbol, const SomeExpr &designator) {
  const Symbol &symbol{offsetSymbol.symbol()};
  const Symbol *lastSymbol{GetLastSymbol(designator)};
  bool isPointer{lastSymbol && IsPointer(*lastSymbol)};
  bool isProcPointer{lastSymbol && IsProcedurePointer(*lastSymbol)};
  evaluate::FoldingContext &context{exprAnalyzer_.GetFoldingContext()};
  auto &messages{context.messages()};
  auto restorer{
      messages.SetLocation(values_.LocateSource().value_or(messages.at()))};

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
    auto iter{inits_.emplace(&symbol, symbol.size())};
    auto &symbolInit{iter.first->second};
    symbolInit.initializedRanges.emplace_back(
        offsetSymbol.offset(), offsetSymbol.size());
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
    if (expr->Rank() > 0) {
      // Because initial-data-target is ambiguous with scalar-constant and
      // scalar-constant-subobject at parse time, enforcement of scalar-*
      // must be deferred to here.
      exprAnalyzer_.Say(
          "DATA statement value initializes '%s' with an array"_err_en_US,
          DescribeElement());
    } else if (auto converted{ConvertElement(*expr, *designatorType)}) {
      // value non-pointer initialization
      if (IsBOZLiteral(*expr) &&
          designatorType->category() != TypeCategory::Integer) { // 8.6.7(11)
        exprAnalyzer_.Say(
            "BOZ literal should appear in a DATA statement only as a value for an integer object, but '%s' is '%s'"_port_en_US,
            DescribeElement(), designatorType->AsFortran());
      } else if (converted->second) {
        exprAnalyzer_.context().Say(
            "DATA statement value initializes '%s' of type '%s' with CHARACTER"_port_en_US,
            DescribeElement(), designatorType->AsFortran());
      }
      auto folded{evaluate::Fold(context, std::move(converted->first))};
      switch (GetImage().Add(
          offsetSymbol.offset(), offsetSymbol.size(), folded, context)) {
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
  DataInitializationCompiler scanner{
      inits, exprAnalyzer, std::get<std::list<parser::DataStmtValue>>(set.t)};
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

void AccumulateDataInitializations(DataInitializations &inits,
    evaluate::ExpressionAnalyzer &exprAnalyzer, const Symbol &symbol,
    const std::list<common::Indirection<parser::DataStmtValue>> &list) {
  DataInitializationCompiler<common::Indirection<parser::DataStmtValue>>
      scanner{inits, exprAnalyzer, list};
  if (scanner.Scan(symbol) && scanner.HasSurplusValues()) {
    exprAnalyzer.context().Say(
        "DATA statement set has more values than objects"_err_en_US);
  }
}

// Looks for default derived type component initialization -- but
// *not* allocatables.
static const DerivedTypeSpec *HasDefaultInitialization(const Symbol &symbol) {
  if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (object->init().has_value()) {
      return nullptr; // init is explicit, not default
    } else if (!object->isDummy() && object->type()) {
      if (const DerivedTypeSpec * derived{object->type()->AsDerived()}) {
        DirectComponentIterator directs{*derived};
        if (std::find_if(
                directs.begin(), directs.end(), [](const Symbol &component) {
                  return !IsAllocatable(component) &&
                      HasDeclarationInitializer(component);
                })) {
          return derived;
        }
      }
    }
  }
  return nullptr;
}

// PopulateWithComponentDefaults() adds initializations to an instance
// of SymbolDataInitialization containing all of the default component
// initializers

static void PopulateWithComponentDefaults(SymbolDataInitialization &init,
    std::size_t offset, const DerivedTypeSpec &derived,
    evaluate::FoldingContext &foldingContext);

static void PopulateWithComponentDefaults(SymbolDataInitialization &init,
    std::size_t offset, const DerivedTypeSpec &derived,
    evaluate::FoldingContext &foldingContext, const Symbol &symbol) {
  if (auto extents{evaluate::GetConstantExtents(foldingContext, symbol)}) {
    const Scope &scope{derived.scope() ? *derived.scope()
                                       : DEREF(derived.typeSymbol().scope())};
    std::size_t stride{scope.size()};
    if (std::size_t alignment{scope.alignment().value_or(0)}) {
      stride = ((stride + alignment - 1) / alignment) * alignment;
    }
    for (auto elements{evaluate::GetSize(*extents)}; elements-- > 0;
         offset += stride) {
      PopulateWithComponentDefaults(init, offset, derived, foldingContext);
    }
  }
}

// F'2018 19.5.3(10) allows storage-associated default component initialization
// when the values are identical.
static void PopulateWithComponentDefaults(SymbolDataInitialization &init,
    std::size_t offset, const DerivedTypeSpec &derived,
    evaluate::FoldingContext &foldingContext) {
  const Scope &scope{
      derived.scope() ? *derived.scope() : DEREF(derived.typeSymbol().scope())};
  for (const auto &pair : scope) {
    const Symbol &component{*pair.second};
    std::size_t componentOffset{offset + component.offset()};
    if (const auto *object{component.detailsIf<ObjectEntityDetails>()}) {
      if (!IsAllocatable(component) && !IsAutomatic(component)) {
        bool initialized{false};
        if (object->init()) {
          initialized = true;
          if (IsPointer(component)) {
            if (auto extant{init.image.AsConstantPointer(componentOffset)}) {
              initialized = !(*extant == *object->init());
            }
            if (initialized) {
              init.image.AddPointer(componentOffset, *object->init());
            }
          } else { // data, not pointer
            if (auto dyType{evaluate::DynamicType::From(component)}) {
              if (auto extents{evaluate::GetConstantExtents(
                      foldingContext, component)}) {
                if (auto extant{init.image.AsConstant(
                        foldingContext, *dyType, *extents, componentOffset)}) {
                  initialized = !(*extant == *object->init());
                }
              }
            }
            if (initialized) {
              init.image.Add(componentOffset, component.size(), *object->init(),
                  foldingContext);
            }
          }
        } else if (const DeclTypeSpec * type{component.GetType()}) {
          if (const DerivedTypeSpec * componentDerived{type->AsDerived()}) {
            PopulateWithComponentDefaults(init, componentOffset,
                *componentDerived, foldingContext, component);
          }
        }
        if (initialized) {
          init.initializedRanges.emplace_back(
              componentOffset, component.size());
        }
      }
    } else if (const auto *proc{component.detailsIf<ProcEntityDetails>()}) {
      if (proc->init() && *proc->init()) {
        SomeExpr procPtrInit{evaluate::ProcedureDesignator{**proc->init()}};
        auto extant{init.image.AsConstantPointer(componentOffset)};
        if (!extant || !(*extant == procPtrInit)) {
          init.initializedRanges.emplace_back(
              componentOffset, component.size());
          init.image.AddPointer(componentOffset, std::move(procPtrInit));
        }
      }
    }
  }
}

static bool CheckForOverlappingInitialization(
    const std::list<SymbolRef> &symbols,
    SymbolDataInitialization &initialization,
    evaluate::ExpressionAnalyzer &exprAnalyzer, const std::string &what) {
  bool result{true};
  auto &context{exprAnalyzer.GetFoldingContext()};
  initialization.initializedRanges.sort();
  ConstantSubscript next{0};
  for (const auto &range : initialization.initializedRanges) {
    if (range.start() < next) {
      result = false; // error: overlap
      bool hit{false};
      for (const Symbol &symbol : symbols) {
        auto offset{range.start() -
            static_cast<ConstantSubscript>(
                symbol.offset() - symbols.front()->offset())};
        if (offset >= 0) {
          if (auto badDesignator{evaluate::OffsetToDesignator(
                  context, symbol, offset, range.size())}) {
            hit = true;
            exprAnalyzer.Say(symbol.name(),
                "%s affect '%s' more than once"_err_en_US, what,
                badDesignator->AsFortran());
          }
        }
      }
      CHECK(hit);
    }
    next = range.start() + range.size();
    CHECK(next <= static_cast<ConstantSubscript>(initialization.image.size()));
  }
  return result;
}

static void IncorporateExplicitInitialization(
    SymbolDataInitialization &combined, DataInitializations &inits,
    const Symbol &symbol, ConstantSubscript firstOffset,
    evaluate::FoldingContext &foldingContext) {
  auto iter{inits.find(&symbol)};
  const auto offset{symbol.offset() - firstOffset};
  if (iter != inits.end()) { // DATA statement initialization
    for (const auto &range : iter->second.initializedRanges) {
      auto at{offset + range.start()};
      combined.initializedRanges.emplace_back(at, range.size());
      combined.image.Incorporate(
          at, iter->second.image, range.start(), range.size());
    }
    if (removeOriginalInits) {
      inits.erase(iter);
    }
  } else { // Declaration initialization
    Symbol &mutableSymbol{const_cast<Symbol &>(symbol)};
    if (IsPointer(mutableSymbol)) {
      if (auto *object{mutableSymbol.detailsIf<ObjectEntityDetails>()}) {
        if (object->init()) {
          combined.initializedRanges.emplace_back(offset, mutableSymbol.size());
          combined.image.AddPointer(offset, *object->init());
          if (removeOriginalInits) {
            object->init().reset();
          }
        }
      } else if (auto *proc{mutableSymbol.detailsIf<ProcEntityDetails>()}) {
        if (proc->init() && *proc->init()) {
          combined.initializedRanges.emplace_back(offset, mutableSymbol.size());
          combined.image.AddPointer(
              offset, SomeExpr{evaluate::ProcedureDesignator{**proc->init()}});
          if (removeOriginalInits) {
            proc->init().reset();
          }
        }
      }
    } else if (auto *object{mutableSymbol.detailsIf<ObjectEntityDetails>()}) {
      if (!IsNamedConstant(mutableSymbol) && object->init()) {
        combined.initializedRanges.emplace_back(offset, mutableSymbol.size());
        combined.image.Add(
            offset, mutableSymbol.size(), *object->init(), foldingContext);
        if (removeOriginalInits) {
          object->init().reset();
        }
      }
    }
  }
}

// Finds the size of the smallest element type in a list of
// storage-associated objects.
static std::size_t ComputeMinElementBytes(
    const std::list<SymbolRef> &associated,
    evaluate::FoldingContext &foldingContext) {
  std::size_t minElementBytes{1};
  const Symbol &first{*associated.front()};
  for (const Symbol &s : associated) {
    if (auto dyType{evaluate::DynamicType::From(s)}) {
      auto size{static_cast<std::size_t>(
          evaluate::ToInt64(dyType->MeasureSizeInBytes(foldingContext, true))
              .value_or(1))};
      if (std::size_t alignment{dyType->GetAlignment(foldingContext)}) {
        size = ((size + alignment - 1) / alignment) * alignment;
      }
      if (&s == &first) {
        minElementBytes = size;
      } else {
        minElementBytes = std::min(minElementBytes, size);
      }
    } else {
      minElementBytes = 1;
    }
  }
  return minElementBytes;
}

// Checks for overlapping initialization errors in a list of
// storage-associated objects.  Default component initializations
// are allowed to be overridden by explicit initializations.
// If the objects are static, save the combined initializer as
// a compiler-created object that covers all of them.
static bool CombineEquivalencedInitialization(
    const std::list<SymbolRef> &associated,
    evaluate::ExpressionAnalyzer &exprAnalyzer, DataInitializations &inits) {
  // Compute the minimum common granularity and total size
  const Symbol &first{*associated.front()};
  std::size_t maxLimit{0};
  for (const Symbol &s : associated) {
    CHECK(s.offset() >= first.offset());
    auto limit{s.offset() + s.size()};
    if (limit > maxLimit) {
      maxLimit = limit;
    }
  }
  auto bytes{static_cast<common::ConstantSubscript>(maxLimit - first.offset())};
  Scope &scope{const_cast<Scope &>(first.owner())};
  // Combine the initializations of the associated objects.
  // Apply all default initializations first.
  SymbolDataInitialization combined{static_cast<std::size_t>(bytes)};
  auto &foldingContext{exprAnalyzer.GetFoldingContext()};
  for (const Symbol &s : associated) {
    if (!IsNamedConstant(s)) {
      if (const auto *derived{HasDefaultInitialization(s)}) {
        PopulateWithComponentDefaults(
            combined, s.offset() - first.offset(), *derived, foldingContext, s);
      }
    }
  }
  if (!CheckForOverlappingInitialization(associated, combined, exprAnalyzer,
          "Distinct default component initializations of equivalenced objects"s)) {
    return false;
  }
  // Don't complain about overlap between explicit initializations and
  // default initializations.
  combined.initializedRanges.clear();
  // Now overlay all explicit initializations from DATA statements and
  // from initializers in declarations.
  for (const Symbol &symbol : associated) {
    IncorporateExplicitInitialization(
        combined, inits, symbol, first.offset(), foldingContext);
  }
  if (!CheckForOverlappingInitialization(associated, combined, exprAnalyzer,
          "Explicit initializations of equivalenced objects"s)) {
    return false;
  }
  // If the items are in static storage, save the final initialization.
  if (std::find_if(associated.begin(), associated.end(),
          [](SymbolRef ref) { return IsSaved(*ref); }) != associated.end()) {
    // Create a compiler array temp that overlaps all the items.
    SourceName name{exprAnalyzer.context().GetTempName(scope)};
    auto emplaced{
        scope.try_emplace(name, Attrs{Attr::SAVE}, ObjectEntityDetails{})};
    CHECK(emplaced.second);
    Symbol &combinedSymbol{*emplaced.first->second};
    combinedSymbol.set(Symbol::Flag::CompilerCreated);
    inits.emplace(&combinedSymbol, std::move(combined));
    auto &details{combinedSymbol.get<ObjectEntityDetails>()};
    combinedSymbol.set_offset(first.offset());
    combinedSymbol.set_size(bytes);
    std::size_t minElementBytes{
        ComputeMinElementBytes(associated, foldingContext)};
    if (!evaluate::IsValidKindOfIntrinsicType(
            TypeCategory::Integer, minElementBytes) ||
        (bytes % minElementBytes) != 0) {
      minElementBytes = 1;
    }
    const DeclTypeSpec &typeSpec{scope.MakeNumericType(
        TypeCategory::Integer, KindExpr{minElementBytes})};
    details.set_type(typeSpec);
    ArraySpec arraySpec;
    arraySpec.emplace_back(ShapeSpec::MakeExplicit(Bound{
        bytes / static_cast<common::ConstantSubscript>(minElementBytes)}));
    details.set_shape(arraySpec);
    if (const auto *commonBlock{FindCommonBlockContaining(first)}) {
      details.set_commonBlock(*commonBlock);
    }
    // Add an EQUIVALENCE set to the scope so that the new object appears in
    // the results of GetStorageAssociations().
    auto &newSet{scope.equivalenceSets().emplace_back()};
    newSet.emplace_back(combinedSymbol);
    newSet.emplace_back(const_cast<Symbol &>(first));
  }
  return true;
}

// When a statically-allocated derived type variable has no explicit
// initialization, but its type has at least one nonallocatable ultimate
// component with default initialization, make its initialization explicit.
[[maybe_unused]] static void MakeDefaultInitializationExplicit(
    const Scope &scope, const std::list<std::list<SymbolRef>> &associations,
    evaluate::FoldingContext &foldingContext, DataInitializations &inits) {
  UnorderedSymbolSet equivalenced;
  for (const std::list<SymbolRef> &association : associations) {
    for (const Symbol &symbol : association) {
      equivalenced.emplace(symbol);
    }
  }
  for (const auto &pair : scope) {
    const Symbol &symbol{*pair.second};
    if (!symbol.test(Symbol::Flag::InDataStmt) &&
        !HasDeclarationInitializer(symbol) && IsSaved(symbol) &&
        equivalenced.find(symbol) == equivalenced.end()) {
      // Static object, no local storage association, no explicit initialization
      if (const DerivedTypeSpec * derived{HasDefaultInitialization(symbol)}) {
        auto newInitIter{inits.emplace(&symbol, symbol.size())};
        CHECK(newInitIter.second);
        auto &newInit{newInitIter.first->second};
        PopulateWithComponentDefaults(
            newInit, 0, *derived, foldingContext, symbol);
      }
    }
  }
}

// Traverses the Scopes to:
// 1) combine initialization of equivalenced objects, &
// 2) optionally make initialization explicit for otherwise uninitialized static
//    objects of derived types with default component initialization
// Returns false on error.
static bool ProcessScopes(const Scope &scope,
    evaluate::ExpressionAnalyzer &exprAnalyzer, DataInitializations &inits) {
  bool result{true}; // no error
  switch (scope.kind()) {
  case Scope::Kind::Global:
  case Scope::Kind::Module:
  case Scope::Kind::MainProgram:
  case Scope::Kind::Subprogram:
  case Scope::Kind::BlockData:
  case Scope::Kind::Block: {
    std::list<std::list<SymbolRef>> associations{GetStorageAssociations(scope)};
    for (const std::list<SymbolRef> &associated : associations) {
      if (std::find_if(associated.begin(), associated.end(), [](SymbolRef ref) {
            return IsInitialized(*ref);
          }) != associated.end()) {
        result &=
            CombineEquivalencedInitialization(associated, exprAnalyzer, inits);
      }
    }
    if constexpr (makeDefaultInitializationExplicit) {
      MakeDefaultInitializationExplicit(
          scope, associations, exprAnalyzer.GetFoldingContext(), inits);
    }
    for (const Scope &child : scope.children()) {
      result &= ProcessScopes(child, exprAnalyzer, inits);
    }
  } break;
  default:;
  }
  return result;
}

// Converts the static initialization image for a single symbol with
// one or more DATA statement appearances.
void ConstructInitializer(const Symbol &symbol,
    SymbolDataInitialization &initialization,
    evaluate::ExpressionAnalyzer &exprAnalyzer) {
  std::list<SymbolRef> symbols{symbol};
  CheckForOverlappingInitialization(
      symbols, initialization, exprAnalyzer, "DATA statement initializations"s);
  auto &context{exprAnalyzer.GetFoldingContext()};
  if (const auto *proc{symbol.detailsIf<ProcEntityDetails>()}) {
    CHECK(IsProcedurePointer(symbol));
    auto &mutableProc{const_cast<ProcEntityDetails &>(*proc)};
    if (MaybeExpr expr{initialization.image.AsConstantPointer()}) {
      if (const auto *procDesignator{
              std::get_if<evaluate::ProcedureDesignator>(&expr->u)}) {
        CHECK(!procDesignator->GetComponent());
        mutableProc.set_init(DEREF(procDesignator->GetSymbol()));
      } else {
        CHECK(evaluate::IsNullPointer(*expr));
        mutableProc.set_init(nullptr);
      }
    } else {
      mutableProc.set_init(nullptr);
    }
  } else if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    auto &mutableObject{const_cast<ObjectEntityDetails &>(*object)};
    if (IsPointer(symbol)) {
      if (auto ptr{initialization.image.AsConstantPointer()}) {
        mutableObject.set_init(*ptr);
      } else {
        mutableObject.set_init(SomeExpr{evaluate::NullPointer{}});
      }
    } else if (auto symbolType{evaluate::DynamicType::From(symbol)}) {
      if (auto extents{evaluate::GetConstantExtents(context, symbol)}) {
        mutableObject.set_init(
            initialization.image.AsConstant(context, *symbolType, *extents));
      } else {
        exprAnalyzer.Say(symbol.name(),
            "internal: unknown shape for '%s' while constructing initializer from DATA"_err_en_US,
            symbol.name());
        return;
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
  if (ProcessScopes(
          exprAnalyzer.context().globalScope(), exprAnalyzer, inits)) {
    for (auto &[symbolPtr, initialization] : inits) {
      ConstructInitializer(*symbolPtr, initialization, exprAnalyzer);
    }
  }
}
} // namespace Fortran::semantics
