//===-- lib/Evaluate/characteristics.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/characteristics.h"
#include "flang/Common/indirection.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/message.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/raw_ostream.h"
#include <initializer_list>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate::characteristics {

// Copy attributes from a symbol to dst based on the mapping in pairs.
template <typename A, typename B>
static void CopyAttrs(const semantics::Symbol &src, A &dst,
    const std::initializer_list<std::pair<semantics::Attr, B>> &pairs) {
  for (const auto &pair : pairs) {
    if (src.attrs().test(pair.first)) {
      dst.attrs.set(pair.second);
    }
  }
}

// Shapes of function results and dummy arguments have to have
// the same rank, the same deferred dimensions, and the same
// values for explicit dimensions when constant.
bool ShapesAreCompatible(const Shape &x, const Shape &y) {
  if (x.size() != y.size()) {
    return false;
  }
  auto yIter{y.begin()};
  for (const auto &xDim : x) {
    const auto &yDim{*yIter++};
    if (xDim) {
      if (!yDim || ToInt64(*xDim) != ToInt64(*yDim)) {
        return false;
      }
    } else if (yDim) {
      return false;
    }
  }
  return true;
}

bool TypeAndShape::operator==(const TypeAndShape &that) const {
  return type_ == that.type_ && ShapesAreCompatible(shape_, that.shape_) &&
      attrs_ == that.attrs_ && corank_ == that.corank_;
}

TypeAndShape &TypeAndShape::Rewrite(FoldingContext &context) {
  LEN_ = Fold(context, std::move(LEN_));
  shape_ = Fold(context, std::move(shape_));
  return *this;
}

std::optional<TypeAndShape> TypeAndShape::Characterize(
    const semantics::Symbol &symbol, FoldingContext &context) {
  const auto &ultimate{symbol.GetUltimate()};
  return common::visit(
      common::visitors{
          [&](const semantics::ProcEntityDetails &proc) {
            const semantics::ProcInterface &interface { proc.interface() };
            if (interface.type()) {
              return Characterize(*interface.type(), context);
            } else if (interface.symbol()) {
              return Characterize(*interface.symbol(), context);
            } else {
              return std::optional<TypeAndShape>{};
            }
          },
          [&](const semantics::AssocEntityDetails &assoc) {
            return Characterize(assoc, context);
          },
          [&](const semantics::ProcBindingDetails &binding) {
            return Characterize(binding.symbol(), context);
          },
          [&](const auto &x) -> std::optional<TypeAndShape> {
            using Ty = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<Ty, semantics::EntityDetails> ||
                std::is_same_v<Ty, semantics::ObjectEntityDetails> ||
                std::is_same_v<Ty, semantics::TypeParamDetails>) {
              if (const semantics::DeclTypeSpec * type{ultimate.GetType()}) {
                if (auto dyType{DynamicType::From(*type)}) {
                  TypeAndShape result{
                      std::move(*dyType), GetShape(context, ultimate)};
                  result.AcquireAttrs(ultimate);
                  result.AcquireLEN(ultimate);
                  return std::move(result.Rewrite(context));
                }
              }
            }
            return std::nullopt;
          },
      },
      // GetUltimate() used here, not ResolveAssociations(), because
      // we need the type/rank of an associate entity from TYPE IS,
      // CLASS IS, or RANK statement.
      ultimate.details());
}

std::optional<TypeAndShape> TypeAndShape::Characterize(
    const semantics::AssocEntityDetails &assoc, FoldingContext &context) {
  std::optional<TypeAndShape> result;
  if (auto type{DynamicType::From(assoc.type())}) {
    if (auto rank{assoc.rank()}) {
      if (*rank >= 0 && *rank <= common::maxRank) {
        result = TypeAndShape{std::move(*type), Shape(*rank)};
      }
    } else if (auto shape{GetShape(context, assoc.expr())}) {
      result = TypeAndShape{std::move(*type), std::move(*shape)};
    }
    if (result && type->category() == TypeCategory::Character) {
      if (const auto *chExpr{UnwrapExpr<Expr<SomeCharacter>>(assoc.expr())}) {
        if (auto len{chExpr->LEN()}) {
          result->set_LEN(std::move(*len));
        }
      }
    }
  }
  return Fold(context, std::move(result));
}

std::optional<TypeAndShape> TypeAndShape::Characterize(
    const semantics::DeclTypeSpec &spec, FoldingContext &context) {
  if (auto type{DynamicType::From(spec)}) {
    return Fold(context, TypeAndShape{std::move(*type)});
  } else {
    return std::nullopt;
  }
}

std::optional<TypeAndShape> TypeAndShape::Characterize(
    const ActualArgument &arg, FoldingContext &context) {
  return Characterize(arg.UnwrapExpr(), context);
}

bool TypeAndShape::IsCompatibleWith(parser::ContextualMessages &messages,
    const TypeAndShape &that, const char *thisIs, const char *thatIs,
    bool omitShapeConformanceCheck,
    enum CheckConformanceFlags::Flags flags) const {
  if (!type_.IsTkCompatibleWith(that.type_)) {
    messages.Say(
        "%1$s type '%2$s' is not compatible with %3$s type '%4$s'"_err_en_US,
        thatIs, that.AsFortran(), thisIs, AsFortran());
    return false;
  }
  return omitShapeConformanceCheck ||
      CheckConformance(messages, shape_, that.shape_, flags, thisIs, thatIs)
          .value_or(true /*fail only when nonconformance is known now*/);
}

std::optional<Expr<SubscriptInteger>> TypeAndShape::MeasureElementSizeInBytes(
    FoldingContext &foldingContext, bool align) const {
  if (LEN_) {
    CHECK(type_.category() == TypeCategory::Character);
    return Fold(foldingContext,
        Expr<SubscriptInteger>{
            foldingContext.targetCharacteristics().GetByteSize(
                type_.category(), type_.kind())} *
            Expr<SubscriptInteger>{*LEN_});
  }
  if (auto elementBytes{type_.MeasureSizeInBytes(foldingContext, align)}) {
    return Fold(foldingContext, std::move(*elementBytes));
  }
  return std::nullopt;
}

std::optional<Expr<SubscriptInteger>> TypeAndShape::MeasureSizeInBytes(
    FoldingContext &foldingContext) const {
  if (auto elements{GetSize(Shape{shape_})}) {
    // Sizes of arrays (even with single elements) are multiples of
    // their alignments.
    if (auto elementBytes{
            MeasureElementSizeInBytes(foldingContext, GetRank(shape_) > 0)}) {
      return Fold(
          foldingContext, std::move(*elements) * std::move(*elementBytes));
    }
  }
  return std::nullopt;
}

void TypeAndShape::AcquireAttrs(const semantics::Symbol &symbol) {
  if (IsAssumedShape(symbol)) {
    attrs_.set(Attr::AssumedShape);
  }
  if (IsDeferredShape(symbol)) {
    attrs_.set(Attr::DeferredShape);
  }
  if (const auto *object{
          symbol.GetUltimate().detailsIf<semantics::ObjectEntityDetails>()}) {
    corank_ = object->coshape().Rank();
    if (object->IsAssumedRank()) {
      attrs_.set(Attr::AssumedRank);
    }
    if (object->IsAssumedSize()) {
      attrs_.set(Attr::AssumedSize);
    }
    if (object->IsCoarray()) {
      attrs_.set(Attr::Coarray);
    }
  }
}

void TypeAndShape::AcquireLEN() {
  if (auto len{type_.GetCharLength()}) {
    LEN_ = std::move(len);
  }
}

void TypeAndShape::AcquireLEN(const semantics::Symbol &symbol) {
  if (type_.category() == TypeCategory::Character) {
    if (auto len{DataRef{symbol}.LEN()}) {
      LEN_ = std::move(*len);
    }
  }
}

std::string TypeAndShape::AsFortran() const {
  return type_.AsFortran(LEN_ ? LEN_->AsFortran() : "");
}

llvm::raw_ostream &TypeAndShape::Dump(llvm::raw_ostream &o) const {
  o << type_.AsFortran(LEN_ ? LEN_->AsFortran() : "");
  attrs_.Dump(o, EnumToString);
  if (!shape_.empty()) {
    o << " dimension";
    char sep{'('};
    for (const auto &expr : shape_) {
      o << sep;
      sep = ',';
      if (expr) {
        expr->AsFortran(o);
      } else {
        o << ':';
      }
    }
    o << ')';
  }
  return o;
}

bool DummyDataObject::operator==(const DummyDataObject &that) const {
  return type == that.type && attrs == that.attrs && intent == that.intent &&
      coshape == that.coshape;
}

static bool AreCompatibleDummyDataObjectShapes(const Shape &x, const Shape &y) {
  // TODO: Validate more than just compatible ranks
  return GetRank(x) == GetRank(y);
}

bool DummyDataObject::IsCompatibleWith(
    const DummyDataObject &actual, std::string *whyNot) const {
  if (!AreCompatibleDummyDataObjectShapes(type.shape(), actual.type.shape())) {
    if (whyNot) {
      *whyNot = "incompatible dummy data object shapes";
    }
    return false;
  }
  if (!type.type().IsTkCompatibleWith(actual.type.type())) {
    if (whyNot) {
      *whyNot = "incompatible dummy data object types: "s +
          type.type().AsFortran() + " vs " + actual.type.type().AsFortran();
    }
    return false;
  }
  if (attrs != actual.attrs) {
    if (whyNot) {
      *whyNot = "incompatible dummy data object attributes";
    }
    return false;
  }
  if (intent != actual.intent) {
    if (whyNot) {
      *whyNot = "incompatible dummy data object intents";
    }
    return false;
  }
  if (coshape != actual.coshape) {
    if (whyNot) {
      *whyNot = "incompatible dummy data object coshapes";
    }
    return false;
  }
  return true;
}

static common::Intent GetIntent(const semantics::Attrs &attrs) {
  if (attrs.test(semantics::Attr::INTENT_IN)) {
    return common::Intent::In;
  } else if (attrs.test(semantics::Attr::INTENT_OUT)) {
    return common::Intent::Out;
  } else if (attrs.test(semantics::Attr::INTENT_INOUT)) {
    return common::Intent::InOut;
  } else {
    return common::Intent::Default;
  }
}

std::optional<DummyDataObject> DummyDataObject::Characterize(
    const semantics::Symbol &symbol, FoldingContext &context) {
  if (symbol.has<semantics::ObjectEntityDetails>() ||
      symbol.has<semantics::EntityDetails>()) {
    if (auto type{TypeAndShape::Characterize(symbol, context)}) {
      std::optional<DummyDataObject> result{std::move(*type)};
      using semantics::Attr;
      CopyAttrs<DummyDataObject, DummyDataObject::Attr>(symbol, *result,
          {
              {Attr::OPTIONAL, DummyDataObject::Attr::Optional},
              {Attr::ALLOCATABLE, DummyDataObject::Attr::Allocatable},
              {Attr::ASYNCHRONOUS, DummyDataObject::Attr::Asynchronous},
              {Attr::CONTIGUOUS, DummyDataObject::Attr::Contiguous},
              {Attr::VALUE, DummyDataObject::Attr::Value},
              {Attr::VOLATILE, DummyDataObject::Attr::Volatile},
              {Attr::POINTER, DummyDataObject::Attr::Pointer},
              {Attr::TARGET, DummyDataObject::Attr::Target},
          });
      result->intent = GetIntent(symbol.attrs());
      return result;
    }
  }
  return std::nullopt;
}

bool DummyDataObject::CanBePassedViaImplicitInterface() const {
  if ((attrs &
          Attrs{Attr::Allocatable, Attr::Asynchronous, Attr::Optional,
              Attr::Pointer, Attr::Target, Attr::Value, Attr::Volatile})
          .any()) {
    return false; // 15.4.2.2(3)(a)
  } else if ((type.attrs() &
                 TypeAndShape::Attrs{TypeAndShape::Attr::AssumedShape,
                     TypeAndShape::Attr::AssumedRank,
                     TypeAndShape::Attr::Coarray})
                 .any()) {
    return false; // 15.4.2.2(3)(b-d)
  } else if (type.type().IsPolymorphic()) {
    return false; // 15.4.2.2(3)(f)
  } else if (const auto *derived{GetDerivedTypeSpec(type.type())}) {
    return derived->parameters().empty(); // 15.4.2.2(3)(e)
  } else {
    return true;
  }
}

llvm::raw_ostream &DummyDataObject::Dump(llvm::raw_ostream &o) const {
  attrs.Dump(o, EnumToString);
  if (intent != common::Intent::Default) {
    o << "INTENT(" << common::EnumToString(intent) << ')';
  }
  type.Dump(o);
  if (!coshape.empty()) {
    char sep{'['};
    for (const auto &expr : coshape) {
      expr.AsFortran(o << sep);
      sep = ',';
    }
  }
  return o;
}

DummyProcedure::DummyProcedure(Procedure &&p)
    : procedure{new Procedure{std::move(p)}} {}

bool DummyProcedure::operator==(const DummyProcedure &that) const {
  return attrs == that.attrs && intent == that.intent &&
      procedure.value() == that.procedure.value();
}

bool DummyProcedure::IsCompatibleWith(
    const DummyProcedure &actual, std::string *whyNot) const {
  if (attrs != actual.attrs) {
    if (whyNot) {
      *whyNot = "incompatible dummy procedure attributes";
    }
    return false;
  }
  if (intent != actual.intent) {
    if (whyNot) {
      *whyNot = "incompatible dummy procedure intents";
    }
    return false;
  }
  if (!procedure.value().IsCompatibleWith(actual.procedure.value(), whyNot)) {
    if (whyNot) {
      *whyNot = "incompatible dummy procedure interfaces: "s + *whyNot;
    }
    return false;
  }
  return true;
}

static std::string GetSeenProcs(
    const semantics::UnorderedSymbolSet &seenProcs) {
  // Sort the symbols so that they appear in the same order on all platforms
  auto ordered{semantics::OrderBySourcePosition(seenProcs)};
  std::string result;
  llvm::interleave(
      ordered,
      [&](const SymbolRef p) { result += '\'' + p->name().ToString() + '\''; },
      [&]() { result += ", "; });
  return result;
}

// These functions with arguments of type UnorderedSymbolSet are used with
// mutually recursive calls when characterizing a Procedure, a DummyArgument,
// or a DummyProcedure to detect circularly defined procedures as required by
// 15.4.3.6, paragraph 2.
static std::optional<DummyArgument> CharacterizeDummyArgument(
    const semantics::Symbol &symbol, FoldingContext &context,
    semantics::UnorderedSymbolSet seenProcs);
static std::optional<FunctionResult> CharacterizeFunctionResult(
    const semantics::Symbol &symbol, FoldingContext &context,
    semantics::UnorderedSymbolSet seenProcs);

static std::optional<Procedure> CharacterizeProcedure(
    const semantics::Symbol &original, FoldingContext &context,
    semantics::UnorderedSymbolSet seenProcs) {
  Procedure result;
  const auto &symbol{ResolveAssociations(original)};
  if (seenProcs.find(symbol) != seenProcs.end()) {
    std::string procsList{GetSeenProcs(seenProcs)};
    context.messages().Say(symbol.name(),
        "Procedure '%s' is recursively defined.  Procedures in the cycle:"
        " %s"_err_en_US,
        symbol.name(), procsList);
    return std::nullopt;
  }
  seenProcs.insert(symbol);
  if (IsElementalProcedure(symbol)) {
    result.attrs.set(Procedure::Attr::Elemental);
  }
  CopyAttrs<Procedure, Procedure::Attr>(symbol, result,
      {
          {semantics::Attr::BIND_C, Procedure::Attr::BindC},
      });
  if (IsPureProcedure(symbol) || // works for ENTRY too
      (!symbol.attrs().test(semantics::Attr::IMPURE) &&
          result.attrs.test(Procedure::Attr::Elemental))) {
    result.attrs.set(Procedure::Attr::Pure);
  }
  return common::visit(
      common::visitors{
          [&](const semantics::SubprogramDetails &subp)
              -> std::optional<Procedure> {
            if (subp.isFunction()) {
              if (auto fr{CharacterizeFunctionResult(
                      subp.result(), context, seenProcs)}) {
                result.functionResult = std::move(fr);
              } else {
                return std::nullopt;
              }
            } else {
              result.attrs.set(Procedure::Attr::Subroutine);
            }
            for (const semantics::Symbol *arg : subp.dummyArgs()) {
              if (!arg) {
                if (subp.isFunction()) {
                  return std::nullopt;
                } else {
                  result.dummyArguments.emplace_back(AlternateReturn{});
                }
              } else if (auto argCharacteristics{CharacterizeDummyArgument(
                             *arg, context, seenProcs)}) {
                result.dummyArguments.emplace_back(
                    std::move(argCharacteristics.value()));
              } else {
                return std::nullopt;
              }
            }
            return result;
          },
          [&](const semantics::ProcEntityDetails &proc)
              -> std::optional<Procedure> {
            if (symbol.attrs().test(semantics::Attr::INTRINSIC)) {
              // Fails when the intrinsic is not a specific intrinsic function
              // from F'2018 table 16.2.  In order to handle forward references,
              // attempts to use impermissible intrinsic procedures as the
              // interfaces of procedure pointers are caught and flagged in
              // declaration checking in Semantics.
              auto intrinsic{context.intrinsics().IsSpecificIntrinsicFunction(
                  symbol.name().ToString())};
              if (intrinsic && intrinsic->isRestrictedSpecific) {
                intrinsic.reset(); // Exclude intrinsics from table 16.3.
              }
              return intrinsic;
            }
            const semantics::ProcInterface &interface { proc.interface() };
            if (const semantics::Symbol * interfaceSymbol{interface.symbol()}) {
              auto interface {
                CharacterizeProcedure(*interfaceSymbol, context, seenProcs)
              };
              if (interface && IsPointer(symbol)) {
                interface->attrs.reset(Procedure::Attr::Elemental);
              }
              return interface;
            } else {
              result.attrs.set(Procedure::Attr::ImplicitInterface);
              const semantics::DeclTypeSpec *type{interface.type()};
              if (symbol.test(semantics::Symbol::Flag::Subroutine)) {
                // ignore any implicit typing
                result.attrs.set(Procedure::Attr::Subroutine);
              } else if (type) {
                if (auto resultType{DynamicType::From(*type)}) {
                  result.functionResult = FunctionResult{*resultType};
                } else {
                  return std::nullopt;
                }
              } else if (symbol.test(semantics::Symbol::Flag::Function)) {
                return std::nullopt;
              }
              // The PASS name, if any, is not a characteristic.
              return result;
            }
          },
          [&](const semantics::ProcBindingDetails &binding) {
            if (auto result{CharacterizeProcedure(
                    binding.symbol(), context, seenProcs)}) {
              if (binding.symbol().attrs().test(semantics::Attr::INTRINSIC)) {
                result->attrs.reset(Procedure::Attr::Elemental);
              }
              if (!symbol.attrs().test(semantics::Attr::NOPASS)) {
                auto passName{binding.passName()};
                for (auto &dummy : result->dummyArguments) {
                  if (!passName || dummy.name.c_str() == *passName) {
                    dummy.pass = true;
                    break;
                  }
                }
              }
              return result;
            } else {
              return std::optional<Procedure>{};
            }
          },
          [&](const semantics::UseDetails &use) {
            return CharacterizeProcedure(use.symbol(), context, seenProcs);
          },
          [](const semantics::UseErrorDetails &) {
            // Ambiguous use-association will be handled later during symbol
            // checks, ignore UseErrorDetails here without actual symbol usage.
            return std::optional<Procedure>{};
          },
          [&](const semantics::HostAssocDetails &assoc) {
            return CharacterizeProcedure(assoc.symbol(), context, seenProcs);
          },
          [&](const semantics::EntityDetails &) {
            context.messages().Say(
                "Procedure '%s' is referenced before being sufficiently defined in a context where it must be so"_err_en_US,
                symbol.name());
            return std::optional<Procedure>{};
          },
          [&](const semantics::SubprogramNameDetails &) {
            context.messages().Say(
                "Procedure '%s' is referenced before being sufficiently defined in a context where it must be so"_err_en_US,
                symbol.name());
            return std::optional<Procedure>{};
          },
          [&](const auto &) {
            context.messages().Say(
                "'%s' is not a procedure"_err_en_US, symbol.name());
            return std::optional<Procedure>{};
          },
      },
      symbol.details());
}

static std::optional<DummyProcedure> CharacterizeDummyProcedure(
    const semantics::Symbol &symbol, FoldingContext &context,
    semantics::UnorderedSymbolSet seenProcs) {
  if (auto procedure{CharacterizeProcedure(symbol, context, seenProcs)}) {
    // Dummy procedures may not be elemental.  Elemental dummy procedure
    // interfaces are errors when the interface is not intrinsic, and that
    // error is caught elsewhere.  Elemental intrinsic interfaces are
    // made non-elemental.
    procedure->attrs.reset(Procedure::Attr::Elemental);
    DummyProcedure result{std::move(procedure.value())};
    CopyAttrs<DummyProcedure, DummyProcedure::Attr>(symbol, result,
        {
            {semantics::Attr::OPTIONAL, DummyProcedure::Attr::Optional},
            {semantics::Attr::POINTER, DummyProcedure::Attr::Pointer},
        });
    result.intent = GetIntent(symbol.attrs());
    return result;
  } else {
    return std::nullopt;
  }
}

llvm::raw_ostream &DummyProcedure::Dump(llvm::raw_ostream &o) const {
  attrs.Dump(o, EnumToString);
  if (intent != common::Intent::Default) {
    o << "INTENT(" << common::EnumToString(intent) << ')';
  }
  procedure.value().Dump(o);
  return o;
}

llvm::raw_ostream &AlternateReturn::Dump(llvm::raw_ostream &o) const {
  return o << '*';
}

DummyArgument::~DummyArgument() {}

bool DummyArgument::operator==(const DummyArgument &that) const {
  return u == that.u; // name and passed-object usage are not characteristics
}

bool DummyArgument::IsCompatibleWith(
    const DummyArgument &actual, std::string *whyNot) const {
  if (const auto *ifaceData{std::get_if<DummyDataObject>(&u)}) {
    if (const auto *actualData{std::get_if<DummyDataObject>(&actual.u)}) {
      return ifaceData->IsCompatibleWith(*actualData, whyNot);
    }
    if (whyNot) {
      *whyNot = "one dummy argument is an object, the other is not";
    }
  } else if (const auto *ifaceProc{std::get_if<DummyProcedure>(&u)}) {
    if (const auto *actualProc{std::get_if<DummyProcedure>(&actual.u)}) {
      return ifaceProc->IsCompatibleWith(*actualProc, whyNot);
    }
    if (whyNot) {
      *whyNot = "one dummy argument is a procedure, the other is not";
    }
  } else {
    CHECK(std::holds_alternative<AlternateReturn>(u));
    if (std::holds_alternative<AlternateReturn>(actual.u)) {
      return true;
    }
    if (whyNot) {
      *whyNot = "one dummy argument is an alternate return, the other is not";
    }
  }
  return false;
}

static std::optional<DummyArgument> CharacterizeDummyArgument(
    const semantics::Symbol &symbol, FoldingContext &context,
    semantics::UnorderedSymbolSet seenProcs) {
  auto name{symbol.name().ToString()};
  if (symbol.has<semantics::ObjectEntityDetails>() ||
      symbol.has<semantics::EntityDetails>()) {
    if (auto obj{DummyDataObject::Characterize(symbol, context)}) {
      return DummyArgument{std::move(name), std::move(obj.value())};
    }
  } else if (auto proc{
                 CharacterizeDummyProcedure(symbol, context, seenProcs)}) {
    return DummyArgument{std::move(name), std::move(proc.value())};
  }
  return std::nullopt;
}

std::optional<DummyArgument> DummyArgument::FromActual(
    std::string &&name, const Expr<SomeType> &expr, FoldingContext &context) {
  return common::visit(
      common::visitors{
          [&](const BOZLiteralConstant &) {
            return std::make_optional<DummyArgument>(std::move(name),
                DummyDataObject{
                    TypeAndShape{DynamicType::TypelessIntrinsicArgument()}});
          },
          [&](const NullPointer &) {
            return std::make_optional<DummyArgument>(std::move(name),
                DummyDataObject{
                    TypeAndShape{DynamicType::TypelessIntrinsicArgument()}});
          },
          [&](const ProcedureDesignator &designator) {
            if (auto proc{Procedure::Characterize(designator, context)}) {
              return std::make_optional<DummyArgument>(
                  std::move(name), DummyProcedure{std::move(*proc)});
            } else {
              return std::optional<DummyArgument>{};
            }
          },
          [&](const ProcedureRef &call) {
            if (auto proc{Procedure::Characterize(call, context)}) {
              return std::make_optional<DummyArgument>(
                  std::move(name), DummyProcedure{std::move(*proc)});
            } else {
              return std::optional<DummyArgument>{};
            }
          },
          [&](const auto &) {
            if (auto type{TypeAndShape::Characterize(expr, context)}) {
              return std::make_optional<DummyArgument>(
                  std::move(name), DummyDataObject{std::move(*type)});
            } else {
              return std::optional<DummyArgument>{};
            }
          },
      },
      expr.u);
}

bool DummyArgument::IsOptional() const {
  return common::visit(
      common::visitors{
          [](const DummyDataObject &data) {
            return data.attrs.test(DummyDataObject::Attr::Optional);
          },
          [](const DummyProcedure &proc) {
            return proc.attrs.test(DummyProcedure::Attr::Optional);
          },
          [](const AlternateReturn &) { return false; },
      },
      u);
}

void DummyArgument::SetOptional(bool value) {
  common::visit(common::visitors{
                    [value](DummyDataObject &data) {
                      data.attrs.set(DummyDataObject::Attr::Optional, value);
                    },
                    [value](DummyProcedure &proc) {
                      proc.attrs.set(DummyProcedure::Attr::Optional, value);
                    },
                    [](AlternateReturn &) { DIE("cannot set optional"); },
                },
      u);
}

void DummyArgument::SetIntent(common::Intent intent) {
  common::visit(common::visitors{
                    [intent](DummyDataObject &data) { data.intent = intent; },
                    [intent](DummyProcedure &proc) { proc.intent = intent; },
                    [](AlternateReturn &) { DIE("cannot set intent"); },
                },
      u);
}

common::Intent DummyArgument::GetIntent() const {
  return common::visit(
      common::visitors{
          [](const DummyDataObject &data) { return data.intent; },
          [](const DummyProcedure &proc) { return proc.intent; },
          [](const AlternateReturn &) -> common::Intent {
            DIE("Alternate returns have no intent");
          },
      },
      u);
}

bool DummyArgument::CanBePassedViaImplicitInterface() const {
  if (const auto *object{std::get_if<DummyDataObject>(&u)}) {
    return object->CanBePassedViaImplicitInterface();
  } else {
    return true;
  }
}

bool DummyArgument::IsTypelessIntrinsicDummy() const {
  const auto *argObj{std::get_if<characteristics::DummyDataObject>(&u)};
  return argObj && argObj->type.type().IsTypelessIntrinsicArgument();
}

llvm::raw_ostream &DummyArgument::Dump(llvm::raw_ostream &o) const {
  if (!name.empty()) {
    o << name << '=';
  }
  if (pass) {
    o << " PASS";
  }
  common::visit([&](const auto &x) { x.Dump(o); }, u);
  return o;
}

FunctionResult::FunctionResult(DynamicType t) : u{TypeAndShape{t}} {}
FunctionResult::FunctionResult(TypeAndShape &&t) : u{std::move(t)} {}
FunctionResult::FunctionResult(Procedure &&p) : u{std::move(p)} {}
FunctionResult::~FunctionResult() {}

bool FunctionResult::operator==(const FunctionResult &that) const {
  return attrs == that.attrs && u == that.u;
}

static std::optional<FunctionResult> CharacterizeFunctionResult(
    const semantics::Symbol &symbol, FoldingContext &context,
    semantics::UnorderedSymbolSet seenProcs) {
  if (symbol.has<semantics::ObjectEntityDetails>()) {
    if (auto type{TypeAndShape::Characterize(symbol, context)}) {
      FunctionResult result{std::move(*type)};
      CopyAttrs<FunctionResult, FunctionResult::Attr>(symbol, result,
          {
              {semantics::Attr::ALLOCATABLE, FunctionResult::Attr::Allocatable},
              {semantics::Attr::CONTIGUOUS, FunctionResult::Attr::Contiguous},
              {semantics::Attr::POINTER, FunctionResult::Attr::Pointer},
          });
      return result;
    }
  } else if (auto maybeProc{
                 CharacterizeProcedure(symbol, context, seenProcs)}) {
    FunctionResult result{std::move(*maybeProc)};
    result.attrs.set(FunctionResult::Attr::Pointer);
    return result;
  }
  return std::nullopt;
}

std::optional<FunctionResult> FunctionResult::Characterize(
    const Symbol &symbol, FoldingContext &context) {
  semantics::UnorderedSymbolSet seenProcs;
  return CharacterizeFunctionResult(symbol, context, seenProcs);
}

bool FunctionResult::IsAssumedLengthCharacter() const {
  if (const auto *ts{std::get_if<TypeAndShape>(&u)}) {
    return ts->type().IsAssumedLengthCharacter();
  } else {
    return false;
  }
}

bool FunctionResult::CanBeReturnedViaImplicitInterface() const {
  if (attrs.test(Attr::Pointer) || attrs.test(Attr::Allocatable)) {
    return false; // 15.4.2.2(4)(b)
  } else if (const auto *typeAndShape{GetTypeAndShape()}) {
    if (typeAndShape->Rank() > 0) {
      return false; // 15.4.2.2(4)(a)
    } else {
      const DynamicType &type{typeAndShape->type()};
      switch (type.category()) {
      case TypeCategory::Character:
        if (type.knownLength()) {
          return true;
        } else if (const auto *param{type.charLengthParamValue()}) {
          if (const auto &expr{param->GetExplicit()}) {
            return IsConstantExpr(*expr); // 15.4.2.2(4)(c)
          } else if (param->isAssumed()) {
            return true;
          }
        }
        return false;
      case TypeCategory::Derived:
        if (!type.IsPolymorphic()) {
          const auto &spec{type.GetDerivedTypeSpec()};
          for (const auto &pair : spec.parameters()) {
            if (const auto &expr{pair.second.GetExplicit()}) {
              if (!IsConstantExpr(*expr)) {
                return false; // 15.4.2.2(4)(c)
              }
            }
          }
          return true;
        }
        return false;
      default:
        return true;
      }
    }
  } else {
    return false; // 15.4.2.2(4)(b) - procedure pointer
  }
}

bool FunctionResult::IsCompatibleWith(
    const FunctionResult &actual, std::string *whyNot) const {
  Attrs actualAttrs{actual.attrs};
  if (!attrs.test(Attr::Contiguous)) {
    actualAttrs.reset(Attr::Contiguous);
  }
  if (attrs != actualAttrs) {
    if (whyNot) {
      *whyNot = "function results have incompatible attributes";
    }
  } else if (const auto *ifaceTypeShape{std::get_if<TypeAndShape>(&u)}) {
    if (const auto *actualTypeShape{std::get_if<TypeAndShape>(&actual.u)}) {
      if (ifaceTypeShape->Rank() != actualTypeShape->Rank()) {
        if (whyNot) {
          *whyNot = "function results have distinct ranks";
        }
      } else if (!attrs.test(Attr::Allocatable) && !attrs.test(Attr::Pointer) &&
          ifaceTypeShape->shape() != actualTypeShape->shape()) {
        if (whyNot) {
          *whyNot = "function results have distinct extents";
        }
      } else if (!ifaceTypeShape->type().IsTkCompatibleWith(
                     actualTypeShape->type())) {
        if (whyNot) {
          *whyNot = "function results have incompatible types: "s +
              ifaceTypeShape->type().AsFortran() + " vs "s +
              actualTypeShape->type().AsFortran();
        }
      } else {
        return true;
      }
    } else {
      if (whyNot) {
        *whyNot = "function result type and shape are not known";
      }
    }
  } else {
    const auto *ifaceProc{std::get_if<CopyableIndirection<Procedure>>(&u)};
    CHECK(ifaceProc != nullptr);
    if (const auto *actualProc{
            std::get_if<CopyableIndirection<Procedure>>(&actual.u)}) {
      if (ifaceProc->value().IsCompatibleWith(actualProc->value(), whyNot)) {
        return true;
      }
      if (whyNot) {
        *whyNot =
            "function results are incompatible procedure pointers: "s + *whyNot;
      }
    } else {
      if (whyNot) {
        *whyNot =
            "one function result is a procedure pointer, the other is not";
      }
    }
  }
  return false;
}

llvm::raw_ostream &FunctionResult::Dump(llvm::raw_ostream &o) const {
  attrs.Dump(o, EnumToString);
  common::visit(common::visitors{
                    [&](const TypeAndShape &ts) { ts.Dump(o); },
                    [&](const CopyableIndirection<Procedure> &p) {
                      p.value().Dump(o << " procedure(") << ')';
                    },
                },
      u);
  return o;
}

Procedure::Procedure(FunctionResult &&fr, DummyArguments &&args, Attrs a)
    : functionResult{std::move(fr)}, dummyArguments{std::move(args)}, attrs{a} {
}
Procedure::Procedure(DummyArguments &&args, Attrs a)
    : dummyArguments{std::move(args)}, attrs{a} {}
Procedure::~Procedure() {}

bool Procedure::operator==(const Procedure &that) const {
  return attrs == that.attrs && functionResult == that.functionResult &&
      dummyArguments == that.dummyArguments;
}

bool Procedure::IsCompatibleWith(const Procedure &actual, std::string *whyNot,
    const SpecificIntrinsic *specificIntrinsic) const {
  // 15.5.2.9(1): if dummy is not pure, actual need not be.
  // Ditto with elemental.
  Attrs actualAttrs{actual.attrs};
  if (!attrs.test(Attr::Pure)) {
    actualAttrs.reset(Attr::Pure);
  }
  if (!attrs.test(Attr::Elemental) && specificIntrinsic) {
    actualAttrs.reset(Attr::Elemental);
  }
  Attrs differences{attrs ^ actualAttrs};
  differences.reset(Attr::Subroutine); // dealt with specifically later
  if (!differences.empty()) {
    if (whyNot) {
      auto sep{": "s};
      *whyNot = "incompatible procedure attributes";
      differences.IterateOverMembers([&](Attr x) {
        *whyNot += sep + EnumToString(x);
        sep = ", ";
      });
    }
  } else if ((IsFunction() && actual.IsSubroutine()) ||
      (IsSubroutine() && actual.IsFunction())) {
    if (whyNot) {
      *whyNot =
          "incompatible procedures: one is a function, the other a subroutine";
    }
  } else if (functionResult && actual.functionResult &&
      !functionResult->IsCompatibleWith(*actual.functionResult, whyNot)) {
  } else if (dummyArguments.size() != actual.dummyArguments.size()) {
    if (whyNot) {
      *whyNot = "distinct numbers of dummy arguments";
    }
  } else {
    for (std::size_t j{0}; j < dummyArguments.size(); ++j) {
      if (!dummyArguments[j].IsCompatibleWith(
              actual.dummyArguments[j], whyNot)) {
        if (whyNot) {
          *whyNot = "incompatible dummy argument #"s + std::to_string(j + 1) +
              ": "s + *whyNot;
        }
        return false;
      }
    }
    return true;
  }
  return false;
}

int Procedure::FindPassIndex(std::optional<parser::CharBlock> name) const {
  int argCount{static_cast<int>(dummyArguments.size())};
  int index{0};
  if (name) {
    while (index < argCount && *name != dummyArguments[index].name.c_str()) {
      ++index;
    }
  }
  CHECK(index < argCount);
  return index;
}

bool Procedure::CanOverride(
    const Procedure &that, std::optional<int> passIndex) const {
  // A pure procedure may override an impure one (7.5.7.3(2))
  if ((that.attrs.test(Attr::Pure) && !attrs.test(Attr::Pure)) ||
      that.attrs.test(Attr::Elemental) != attrs.test(Attr::Elemental) ||
      functionResult != that.functionResult) {
    return false;
  }
  int argCount{static_cast<int>(dummyArguments.size())};
  if (argCount != static_cast<int>(that.dummyArguments.size())) {
    return false;
  }
  for (int j{0}; j < argCount; ++j) {
    if ((!passIndex || j != *passIndex) &&
        dummyArguments[j] != that.dummyArguments[j]) {
      return false;
    }
  }
  return true;
}

std::optional<Procedure> Procedure::Characterize(
    const semantics::Symbol &original, FoldingContext &context) {
  semantics::UnorderedSymbolSet seenProcs;
  return CharacterizeProcedure(original, context, seenProcs);
}

std::optional<Procedure> Procedure::Characterize(
    const ProcedureDesignator &proc, FoldingContext &context) {
  if (const auto *symbol{proc.GetSymbol()}) {
    if (auto result{
            characteristics::Procedure::Characterize(*symbol, context)}) {
      return result;
    }
  } else if (const auto *intrinsic{proc.GetSpecificIntrinsic()}) {
    return intrinsic->characteristics.value();
  }
  return std::nullopt;
}

std::optional<Procedure> Procedure::Characterize(
    const ProcedureRef &ref, FoldingContext &context) {
  if (auto callee{Characterize(ref.proc(), context)}) {
    if (callee->functionResult) {
      if (const Procedure *
          proc{callee->functionResult->IsProcedurePointer()}) {
        return {*proc};
      }
    }
  }
  return std::nullopt;
}

bool Procedure::CanBeCalledViaImplicitInterface() const {
  // TODO: Pass back information on why we return false
  if (attrs.test(Attr::Elemental) || attrs.test(Attr::BindC)) {
    return false; // 15.4.2.2(5,6)
  } else if (IsFunction() &&
      !functionResult->CanBeReturnedViaImplicitInterface()) {
    return false;
  } else {
    for (const DummyArgument &arg : dummyArguments) {
      if (!arg.CanBePassedViaImplicitInterface()) {
        return false;
      }
    }
    return true;
  }
}

llvm::raw_ostream &Procedure::Dump(llvm::raw_ostream &o) const {
  attrs.Dump(o, EnumToString);
  if (functionResult) {
    functionResult->Dump(o << "TYPE(") << ") FUNCTION";
  } else if (attrs.test(Attr::Subroutine)) {
    o << "SUBROUTINE";
  } else {
    o << "EXTERNAL";
  }
  char sep{'('};
  for (const auto &dummy : dummyArguments) {
    dummy.Dump(o << sep);
    sep = ',';
  }
  return o << (sep == '(' ? "()" : ")");
}

// Utility class to determine if Procedures, etc. are distinguishable
class DistinguishUtils {
public:
  explicit DistinguishUtils(const common::LanguageFeatureControl &features)
      : features_{features} {}

  // Are these procedures distinguishable for a generic name?
  bool Distinguishable(const Procedure &, const Procedure &) const;
  // Are these procedures distinguishable for a generic operator or assignment?
  bool DistinguishableOpOrAssign(const Procedure &, const Procedure &) const;

private:
  struct CountDummyProcedures {
    CountDummyProcedures(const DummyArguments &args) {
      for (const DummyArgument &arg : args) {
        if (std::holds_alternative<DummyProcedure>(arg.u)) {
          total += 1;
          notOptional += !arg.IsOptional();
        }
      }
    }
    int total{0};
    int notOptional{0};
  };

  bool Rule3Distinguishable(const Procedure &, const Procedure &) const;
  const DummyArgument *Rule1DistinguishingArg(
      const DummyArguments &, const DummyArguments &) const;
  int FindFirstToDistinguishByPosition(
      const DummyArguments &, const DummyArguments &) const;
  int FindLastToDistinguishByName(
      const DummyArguments &, const DummyArguments &) const;
  int CountCompatibleWith(const DummyArgument &, const DummyArguments &) const;
  int CountNotDistinguishableFrom(
      const DummyArgument &, const DummyArguments &) const;
  bool Distinguishable(const DummyArgument &, const DummyArgument &) const;
  bool Distinguishable(const DummyDataObject &, const DummyDataObject &) const;
  bool Distinguishable(const DummyProcedure &, const DummyProcedure &) const;
  bool Distinguishable(const FunctionResult &, const FunctionResult &) const;
  bool Distinguishable(const TypeAndShape &, const TypeAndShape &) const;
  bool IsTkrCompatible(const DummyArgument &, const DummyArgument &) const;
  bool IsTkrCompatible(const TypeAndShape &, const TypeAndShape &) const;
  const DummyArgument *GetAtEffectivePosition(
      const DummyArguments &, int) const;
  const DummyArgument *GetPassArg(const Procedure &) const;

  const common::LanguageFeatureControl &features_;
};

// Simpler distinguishability rules for operators and assignment
bool DistinguishUtils::DistinguishableOpOrAssign(
    const Procedure &proc1, const Procedure &proc2) const {
  auto &args1{proc1.dummyArguments};
  auto &args2{proc2.dummyArguments};
  if (args1.size() != args2.size()) {
    return true; // C1511: distinguishable based on number of arguments
  }
  for (std::size_t i{0}; i < args1.size(); ++i) {
    if (Distinguishable(args1[i], args2[i])) {
      return true; // C1511, C1512: distinguishable based on this arg
    }
  }
  return false;
}

bool DistinguishUtils::Distinguishable(
    const Procedure &proc1, const Procedure &proc2) const {
  auto &args1{proc1.dummyArguments};
  auto &args2{proc2.dummyArguments};
  auto count1{CountDummyProcedures(args1)};
  auto count2{CountDummyProcedures(args2)};
  if (count1.notOptional > count2.total || count2.notOptional > count1.total) {
    return true; // distinguishable based on C1514 rule 2
  }
  if (Rule3Distinguishable(proc1, proc2)) {
    return true; // distinguishable based on C1514 rule 3
  }
  if (Rule1DistinguishingArg(args1, args2)) {
    return true; // distinguishable based on C1514 rule 1
  }
  int pos1{FindFirstToDistinguishByPosition(args1, args2)};
  int name1{FindLastToDistinguishByName(args1, args2)};
  if (pos1 >= 0 && pos1 <= name1) {
    return true; // distinguishable based on C1514 rule 4
  }
  int pos2{FindFirstToDistinguishByPosition(args2, args1)};
  int name2{FindLastToDistinguishByName(args2, args1)};
  if (pos2 >= 0 && pos2 <= name2) {
    return true; // distinguishable based on C1514 rule 4
  }
  return false;
}

// C1514 rule 3: Procedures are distinguishable if both have a passed-object
// dummy argument and those are distinguishable.
bool DistinguishUtils::Rule3Distinguishable(
    const Procedure &proc1, const Procedure &proc2) const {
  const DummyArgument *pass1{GetPassArg(proc1)};
  const DummyArgument *pass2{GetPassArg(proc2)};
  return pass1 && pass2 && Distinguishable(*pass1, *pass2);
}

// Find a non-passed-object dummy data object in one of the argument lists
// that satisfies C1514 rule 1. I.e. x such that:
// - m is the number of dummy data objects in one that are nonoptional,
//   are not passed-object, that x is TKR compatible with
// - n is the number of non-passed-object dummy data objects, in the other
//   that are not distinguishable from x
// - m is greater than n
const DummyArgument *DistinguishUtils::Rule1DistinguishingArg(
    const DummyArguments &args1, const DummyArguments &args2) const {
  auto size1{args1.size()};
  auto size2{args2.size()};
  for (std::size_t i{0}; i < size1 + size2; ++i) {
    const DummyArgument &x{i < size1 ? args1[i] : args2[i - size1]};
    if (!x.pass && std::holds_alternative<DummyDataObject>(x.u)) {
      if (CountCompatibleWith(x, args1) >
              CountNotDistinguishableFrom(x, args2) ||
          CountCompatibleWith(x, args2) >
              CountNotDistinguishableFrom(x, args1)) {
        return &x;
      }
    }
  }
  return nullptr;
}

// Find the index of the first nonoptional non-passed-object dummy argument
// in args1 at an effective position such that either:
// - args2 has no dummy argument at that effective position
// - the dummy argument at that position is distinguishable from it
int DistinguishUtils::FindFirstToDistinguishByPosition(
    const DummyArguments &args1, const DummyArguments &args2) const {
  int effective{0}; // position of arg1 in list, ignoring passed arg
  for (std::size_t i{0}; i < args1.size(); ++i) {
    const DummyArgument &arg1{args1.at(i)};
    if (!arg1.pass && !arg1.IsOptional()) {
      const DummyArgument *arg2{GetAtEffectivePosition(args2, effective)};
      if (!arg2 || Distinguishable(arg1, *arg2)) {
        return i;
      }
    }
    effective += !arg1.pass;
  }
  return -1;
}

// Find the index of the last nonoptional non-passed-object dummy argument
// in args1 whose name is such that either:
// - args2 has no dummy argument with that name
// - the dummy argument with that name is distinguishable from it
int DistinguishUtils::FindLastToDistinguishByName(
    const DummyArguments &args1, const DummyArguments &args2) const {
  std::map<std::string, const DummyArgument *> nameToArg;
  for (const auto &arg2 : args2) {
    nameToArg.emplace(arg2.name, &arg2);
  }
  for (int i = args1.size() - 1; i >= 0; --i) {
    const DummyArgument &arg1{args1.at(i)};
    if (!arg1.pass && !arg1.IsOptional()) {
      auto it{nameToArg.find(arg1.name)};
      if (it == nameToArg.end() || Distinguishable(arg1, *it->second)) {
        return i;
      }
    }
  }
  return -1;
}

// Count the dummy data objects in args that are nonoptional, are not
// passed-object, and that x is TKR compatible with
int DistinguishUtils::CountCompatibleWith(
    const DummyArgument &x, const DummyArguments &args) const {
  return std::count_if(args.begin(), args.end(), [&](const DummyArgument &y) {
    return !y.pass && !y.IsOptional() && IsTkrCompatible(x, y);
  });
}

// Return the number of dummy data objects in args that are not
// distinguishable from x and not passed-object.
int DistinguishUtils::CountNotDistinguishableFrom(
    const DummyArgument &x, const DummyArguments &args) const {
  return std::count_if(args.begin(), args.end(), [&](const DummyArgument &y) {
    return !y.pass && std::holds_alternative<DummyDataObject>(y.u) &&
        !Distinguishable(y, x);
  });
}

bool DistinguishUtils::Distinguishable(
    const DummyArgument &x, const DummyArgument &y) const {
  if (x.u.index() != y.u.index()) {
    return true; // different kind: data/proc/alt-return
  }
  return common::visit(
      common::visitors{
          [&](const DummyDataObject &z) {
            return Distinguishable(z, std::get<DummyDataObject>(y.u));
          },
          [&](const DummyProcedure &z) {
            return Distinguishable(z, std::get<DummyProcedure>(y.u));
          },
          [&](const AlternateReturn &) { return false; },
      },
      x.u);
}

bool DistinguishUtils::Distinguishable(
    const DummyDataObject &x, const DummyDataObject &y) const {
  using Attr = DummyDataObject::Attr;
  if (Distinguishable(x.type, y.type)) {
    return true;
  } else if (x.attrs.test(Attr::Allocatable) && y.attrs.test(Attr::Pointer) &&
      y.intent != common::Intent::In) {
    return true;
  } else if (y.attrs.test(Attr::Allocatable) && x.attrs.test(Attr::Pointer) &&
      x.intent != common::Intent::In) {
    return true;
  } else if (features_.IsEnabled(
                 common::LanguageFeature::DistinguishableSpecifics) &&
      (x.attrs.test(Attr::Allocatable) || x.attrs.test(Attr::Pointer)) &&
      (y.attrs.test(Attr::Allocatable) || y.attrs.test(Attr::Pointer)) &&
      (x.type.type().IsUnlimitedPolymorphic() !=
              y.type.type().IsUnlimitedPolymorphic() ||
          x.type.type().IsPolymorphic() != y.type.type().IsPolymorphic())) {
    // Extension: Per 15.5.2.5(2), an allocatable/pointer dummy and its
    // corresponding actual argument must both or neither be polymorphic,
    // and must both or neither be unlimited polymorphic.  So when exactly
    // one of two dummy arguments is polymorphic or unlimited polymorphic,
    // any actual argument that is admissible to one of them cannot also match
    // the other one.
    return true;
  } else {
    return false;
  }
}

bool DistinguishUtils::Distinguishable(
    const DummyProcedure &x, const DummyProcedure &y) const {
  const Procedure &xProc{x.procedure.value()};
  const Procedure &yProc{y.procedure.value()};
  if (Distinguishable(xProc, yProc)) {
    return true;
  } else {
    const std::optional<FunctionResult> &xResult{xProc.functionResult};
    const std::optional<FunctionResult> &yResult{yProc.functionResult};
    return xResult ? !yResult || Distinguishable(*xResult, *yResult)
                   : yResult.has_value();
  }
}

bool DistinguishUtils::Distinguishable(
    const FunctionResult &x, const FunctionResult &y) const {
  if (x.u.index() != y.u.index()) {
    return true; // one is data object, one is procedure
  }
  return common::visit(
      common::visitors{
          [&](const TypeAndShape &z) {
            return Distinguishable(z, std::get<TypeAndShape>(y.u));
          },
          [&](const CopyableIndirection<Procedure> &z) {
            return Distinguishable(z.value(),
                std::get<CopyableIndirection<Procedure>>(y.u).value());
          },
      },
      x.u);
}

bool DistinguishUtils::Distinguishable(
    const TypeAndShape &x, const TypeAndShape &y) const {
  return !IsTkrCompatible(x, y) && !IsTkrCompatible(y, x);
}

// Compatibility based on type, kind, and rank
bool DistinguishUtils::IsTkrCompatible(
    const DummyArgument &x, const DummyArgument &y) const {
  const auto *obj1{std::get_if<DummyDataObject>(&x.u)};
  const auto *obj2{std::get_if<DummyDataObject>(&y.u)};
  return obj1 && obj2 && IsTkrCompatible(obj1->type, obj2->type);
}
bool DistinguishUtils::IsTkrCompatible(
    const TypeAndShape &x, const TypeAndShape &y) const {
  return x.type().IsTkCompatibleWith(y.type()) &&
      (x.attrs().test(TypeAndShape::Attr::AssumedRank) ||
          y.attrs().test(TypeAndShape::Attr::AssumedRank) ||
          x.Rank() == y.Rank());
}

// Return the argument at the given index, ignoring the passed arg
const DummyArgument *DistinguishUtils::GetAtEffectivePosition(
    const DummyArguments &args, int index) const {
  for (const DummyArgument &arg : args) {
    if (!arg.pass) {
      if (index == 0) {
        return &arg;
      }
      --index;
    }
  }
  return nullptr;
}

// Return the passed-object dummy argument of this procedure, if any
const DummyArgument *DistinguishUtils::GetPassArg(const Procedure &proc) const {
  for (const auto &arg : proc.dummyArguments) {
    if (arg.pass) {
      return &arg;
    }
  }
  return nullptr;
}

bool Distinguishable(const common::LanguageFeatureControl &features,
    const Procedure &x, const Procedure &y) {
  return DistinguishUtils{features}.Distinguishable(x, y);
}

bool DistinguishableOpOrAssign(const common::LanguageFeatureControl &features,
    const Procedure &x, const Procedure &y) {
  return DistinguishUtils{features}.DistinguishableOpOrAssign(x, y);
}

DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(DummyArgument)
DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(DummyProcedure)
DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(FunctionResult)
DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(Procedure)
} // namespace Fortran::evaluate::characteristics

template class Fortran::common::Indirection<
    Fortran::evaluate::characteristics::Procedure, true>;
