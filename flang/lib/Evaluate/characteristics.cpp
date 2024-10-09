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
// An ASYNCHRONOUS attribute counts even if it is implied.
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
bool ShapesAreCompatible(const std::optional<Shape> &x,
    const std::optional<Shape> &y, bool *possibleWarning) {
  if (!x || !y) {
    return !x && !y;
  }
  if (x->size() != y->size()) {
    return false;
  }
  auto yIter{y->begin()};
  for (const auto &xDim : *x) {
    const auto &yDim{*yIter++};
    if (xDim && yDim) {
      if (auto equiv{AreEquivalentInInterface(*xDim, *yDim)}) {
        if (!*equiv) {
          return false;
        }
      } else if (possibleWarning) {
        *possibleWarning = true;
      }
    } else if (xDim || yDim) {
      return false;
    }
  }
  return true;
}

bool TypeAndShape::operator==(const TypeAndShape &that) const {
  return type_.IsEquivalentTo(that.type_) &&
      ShapesAreCompatible(shape_, that.shape_) && attrs_ == that.attrs_ &&
      corank_ == that.corank_;
}

TypeAndShape &TypeAndShape::Rewrite(FoldingContext &context) {
  LEN_ = Fold(context, std::move(LEN_));
  if (LEN_) {
    if (auto n{ToInt64(*LEN_)}) {
      type_ = DynamicType{type_.kind(), *n};
    }
  }
  shape_ = Fold(context, std::move(shape_));
  return *this;
}

std::optional<TypeAndShape> TypeAndShape::Characterize(
    const semantics::Symbol &symbol, FoldingContext &context,
    bool invariantOnly) {
  const auto &ultimate{symbol.GetUltimate()};
  return common::visit(
      common::visitors{
          [&](const semantics::ProcEntityDetails &proc) {
            if (proc.procInterface()) {
              return Characterize(
                  *proc.procInterface(), context, invariantOnly);
            } else if (proc.type()) {
              return Characterize(*proc.type(), context, invariantOnly);
            } else {
              return std::optional<TypeAndShape>{};
            }
          },
          [&](const semantics::AssocEntityDetails &assoc) {
            return Characterize(assoc, context, invariantOnly);
          },
          [&](const semantics::ProcBindingDetails &binding) {
            return Characterize(binding.symbol(), context, invariantOnly);
          },
          [&](const auto &x) -> std::optional<TypeAndShape> {
            using Ty = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<Ty, semantics::EntityDetails> ||
                std::is_same_v<Ty, semantics::ObjectEntityDetails> ||
                std::is_same_v<Ty, semantics::TypeParamDetails>) {
              if (const semantics::DeclTypeSpec * type{ultimate.GetType()}) {
                if (auto dyType{DynamicType::From(*type)}) {
                  TypeAndShape result{std::move(*dyType),
                      GetShape(context, ultimate, invariantOnly)};
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
    const semantics::AssocEntityDetails &assoc, FoldingContext &context,
    bool invariantOnly) {
  std::optional<TypeAndShape> result;
  if (auto type{DynamicType::From(assoc.type())}) {
    if (auto rank{assoc.rank()}) {
      if (*rank >= 0 && *rank <= common::maxRank) {
        result = TypeAndShape{std::move(*type), Shape(*rank)};
      }
    } else if (auto shape{GetShape(context, assoc.expr(), invariantOnly)}) {
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
    const semantics::DeclTypeSpec &spec, FoldingContext &context,
    bool /*invariantOnly=*/) {
  if (auto type{DynamicType::From(spec)}) {
    return Fold(context, TypeAndShape{std::move(*type)});
  } else {
    return std::nullopt;
  }
}

std::optional<TypeAndShape> TypeAndShape::Characterize(
    const ActualArgument &arg, FoldingContext &context, bool invariantOnly) {
  if (const auto *expr{arg.UnwrapExpr()}) {
    return Characterize(*expr, context, invariantOnly);
  } else if (const Symbol * assumed{arg.GetAssumedTypeDummy()}) {
    return Characterize(*assumed, context, invariantOnly);
  } else {
    return std::nullopt;
  }
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
  return omitShapeConformanceCheck || (!shape_ && !that.shape_) ||
      (shape_ && that.shape_ &&
          CheckConformance(
              messages, *shape_, *that.shape_, flags, thisIs, thatIs)
              .value_or(true /*fail only when nonconformance is known now*/));
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
  if (auto elements{GetSize(shape_)}) {
    // Sizes of arrays (even with single elements) are multiples of
    // their alignments.
    if (auto elementBytes{
            MeasureElementSizeInBytes(foldingContext, Rank() > 0)}) {
      return Fold(
          foldingContext, std::move(*elements) * std::move(*elementBytes));
    }
  }
  return std::nullopt;
}

void TypeAndShape::AcquireAttrs(const semantics::Symbol &symbol) {
  if (IsAssumedShape(symbol)) {
    attrs_.set(Attr::AssumedShape);
  } else if (IsDeferredShape(symbol)) {
    attrs_.set(Attr::DeferredShape);
  } else if (semantics::IsAssumedSizeArray(symbol)) {
    attrs_.set(Attr::AssumedSize);
  }
  if (const auto *object{
          symbol.GetUltimate().detailsIf<semantics::ObjectEntityDetails>()}) {
    corank_ = object->coshape().Rank();
    if (object->IsAssumedRank()) {
      attrs_.set(Attr::AssumedRank);
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
  if (!shape_) {
    o << " dimension(..)";
  } else if (!shape_->empty()) {
    o << " dimension";
    char sep{'('};
    for (const auto &expr : *shape_) {
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
      coshape == that.coshape && cudaDataAttr == that.cudaDataAttr;
}

bool DummyDataObject::IsCompatibleWith(const DummyDataObject &actual,
    std::string *whyNot, std::optional<std::string> *warning) const {
  bool possibleWarning{false};
  if (!ShapesAreCompatible(
          type.shape(), actual.type.shape(), &possibleWarning)) {
    if (whyNot) {
      *whyNot = "incompatible dummy data object shapes";
    }
    return false;
  } else if (warning && possibleWarning) {
    *warning = "distinct dummy data object shapes";
  }
  // Treat deduced dummy character type as if it were assumed-length character
  // to avoid useless "implicit interfaces have distinct type" warnings from
  // CALL FOO('abc'); CALL FOO('abcd').
  bool deducedAssumedLength{type.type().category() == TypeCategory::Character &&
      attrs.test(Attr::DeducedFromActual)};
  bool compatibleTypes{deducedAssumedLength
          ? type.type().IsTkCompatibleWith(actual.type.type())
          : type.type().IsTkLenCompatibleWith(actual.type.type())};
  if (!compatibleTypes) {
    if (whyNot) {
      *whyNot = "incompatible dummy data object types: "s +
          type.type().AsFortran() + " vs " + actual.type.type().AsFortran();
    }
    return false;
  }
  if (type.type().IsPolymorphic() != actual.type.type().IsPolymorphic()) {
    if (whyNot) {
      *whyNot = "incompatible dummy data object polymorphism: "s +
          type.type().AsFortran() + " vs " + actual.type.type().AsFortran();
    }
    return false;
  }
  if (type.type().category() == TypeCategory::Character &&
      !deducedAssumedLength) {
    if (actual.type.type().IsAssumedLengthCharacter() !=
        type.type().IsAssumedLengthCharacter()) {
      if (whyNot) {
        *whyNot = "assumed-length character vs explicit-length character";
      }
      return false;
    }
    if (!type.type().IsAssumedLengthCharacter() && type.LEN() &&
        actual.type.LEN()) {
      auto len{ToInt64(*type.LEN())};
      auto actualLen{ToInt64(*actual.type.LEN())};
      if (len.has_value() != actualLen.has_value()) {
        if (whyNot) {
          *whyNot = "constant-length vs non-constant-length character dummy "
                    "arguments";
        }
        return false;
      } else if (len && *len != *actualLen) {
        if (whyNot) {
          *whyNot = "character dummy arguments with distinct lengths";
        }
        return false;
      }
    }
  }
  if (!IdenticalSignificantAttrs(attrs, actual.attrs) ||
      type.attrs() != actual.type.attrs()) {
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
  if (ignoreTKR != actual.ignoreTKR) {
    if (whyNot) {
      *whyNot = "incompatible !DIR$ IGNORE_TKR directives";
    }
  }
  if (!attrs.test(Attr::Value) &&
      !common::AreCompatibleCUDADataAttrs(cudaDataAttr, actual.cudaDataAttr,
          ignoreTKR,
          /*allowUnifiedMatchingRule=*/false)) {
    if (whyNot) {
      *whyNot = "incompatible CUDA data attributes";
    }
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
  if (const auto *object{symbol.detailsIf<semantics::ObjectEntityDetails>()};
      object || symbol.has<semantics::EntityDetails>()) {
    if (auto type{TypeAndShape::Characterize(
            symbol, context, /*invariantOnly=*/false)}) {
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
      result->ignoreTKR = GetIgnoreTKR(symbol);
      if (object) {
        result->cudaDataAttr = object->cudaDataAttr();
        if (!result->cudaDataAttr &&
            !result->attrs.test(DummyDataObject::Attr::Value) &&
            semantics::IsCUDADeviceContext(&symbol.owner())) {
          result->cudaDataAttr = common::CUDADataAttr::Device;
        }
      }
      return result;
    }
  }
  return std::nullopt;
}

bool DummyDataObject::CanBePassedViaImplicitInterface(
    std::string *whyNot) const {
  if ((attrs &
          Attrs{Attr::Allocatable, Attr::Asynchronous, Attr::Optional,
              Attr::Pointer, Attr::Target, Attr::Value, Attr::Volatile})
          .any()) {
    if (whyNot) {
      *whyNot = "a dummy argument has the allocatable, asynchronous, optional, "
                "pointer, target, value, or volatile attribute";
    }
    return false; // 15.4.2.2(3)(a)
  } else if ((type.attrs() &
                 TypeAndShape::Attrs{TypeAndShape::Attr::AssumedShape,
                     TypeAndShape::Attr::AssumedRank,
                     TypeAndShape::Attr::Coarray})
                 .any()) {
    if (whyNot) {
      *whyNot = "a dummy argument is assumed-shape, assumed-rank, or a coarray";
    }
    return false; // 15.4.2.2(3)(b-d)
  } else if (type.type().IsPolymorphic()) {
    if (whyNot) {
      *whyNot = "a dummy argument is polymorphic";
    }
    return false; // 15.4.2.2(3)(f)
  } else if (cudaDataAttr) {
    if (whyNot) {
      *whyNot = "a dummy argument has a CUDA data attribute";
    }
    return false;
  } else if (const auto *derived{GetDerivedTypeSpec(type.type())}) {
    if (derived->parameters().empty()) { // 15.4.2.2(3)(e)
      return true;
    } else {
      if (whyNot) {
        *whyNot = "a dummy argument has derived type parameters";
      }
      return false;
    }
  } else {
    return true;
  }
}

bool DummyDataObject::IsPassedByDescriptor(bool isBindC) const {
  constexpr TypeAndShape::Attrs shapeRequiringBox = {
      TypeAndShape::Attr::AssumedShape, TypeAndShape::Attr::DeferredShape,
      TypeAndShape::Attr::AssumedRank, TypeAndShape::Attr::Coarray};
  if ((attrs & Attrs{Attr::Allocatable, Attr::Pointer}).any()) {
    return true;
  } else if ((type.attrs() & shapeRequiringBox).any()) {
    // Need to pass shape/coshape info in a descriptor.
    return true;
  } else if (type.type().IsPolymorphic() && !type.type().IsAssumedType()) {
    // Need to pass dynamic type info in a descriptor.
    return true;
  } else if (const auto *derived{GetDerivedTypeSpec(type.type())}) {
    if (!derived->parameters().empty()) {
      for (const auto &param : derived->parameters()) {
        if (param.second.isLen()) {
          // Need to pass length type parameters in a descriptor.
          return true;
        }
      }
    }
  } else if (isBindC && type.type().IsAssumedLengthCharacter()) {
    // Fortran 2018 18.3.6 point 2 (5)
    return true;
  }
  return false;
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
  if (cudaDataAttr) {
    o << " cudaDataAttr: " << common::EnumToString(*cudaDataAttr);
  }
  if (!ignoreTKR.empty()) {
    ignoreTKR.Dump(o << ' ', common::EnumToString);
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
  if (!procedure.value().IsCompatibleWith(actual.procedure.value(),
          /*ignoreImplicitVsExplicit=*/false, whyNot)) {
    if (whyNot) {
      *whyNot = "incompatible dummy procedure interfaces: "s + *whyNot;
    }
    return false;
  }
  return true;
}

bool DummyProcedure::CanBePassedViaImplicitInterface(
    std::string *whyNot) const {
  if ((attrs & Attrs{Attr::Optional, Attr::Pointer}).any()) {
    if (whyNot) {
      *whyNot = "a dummy procedure is optional or a pointer";
    }
    return false; // 15.4.2.2(3)(a)
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
    semantics::UnorderedSymbolSet seenProcs, bool emitError);

static std::optional<Procedure> CharacterizeProcedure(
    const semantics::Symbol &original, FoldingContext &context,
    semantics::UnorderedSymbolSet seenProcs, bool emitError) {
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
  auto CheckForNested{[&](const Symbol &symbol) {
    if (emitError) {
      context.messages().Say(
          "Procedure '%s' is referenced before being sufficiently defined in a context where it must be so"_err_en_US,
          symbol.name());
    }
  }};
  auto result{common::visit(
      common::visitors{
          [&](const semantics::SubprogramDetails &subp)
              -> std::optional<Procedure> {
            Procedure result;
            if (subp.isFunction()) {
              if (auto fr{CharacterizeFunctionResult(
                      subp.result(), context, seenProcs, emitError)}) {
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
            result.cudaSubprogramAttrs = subp.cudaSubprogramAttrs();
            return std::move(result);
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
            if (const semantics::Symbol *
                interfaceSymbol{proc.procInterface()}) {
              auto result{CharacterizeProcedure(
                  *interfaceSymbol, context, seenProcs, /*emitError=*/false)};
              if (result && (IsDummy(symbol) || IsPointer(symbol))) {
                // Dummy procedures and procedure pointers may not be
                // ELEMENTAL, but we do accept the use of elemental intrinsic
                // functions as their interfaces.
                result->attrs.reset(Procedure::Attr::Elemental);
              }
              return result;
            } else {
              Procedure result;
              result.attrs.set(Procedure::Attr::ImplicitInterface);
              const semantics::DeclTypeSpec *type{proc.type()};
              if (symbol.test(semantics::Symbol::Flag::Subroutine)) {
                // ignore any implicit typing
                result.attrs.set(Procedure::Attr::Subroutine);
                if (proc.isCUDAKernel()) {
                  result.cudaSubprogramAttrs =
                      common::CUDASubprogramAttrs::Global;
                }
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
              return std::move(result);
            }
          },
          [&](const semantics::ProcBindingDetails &binding) {
            if (auto result{CharacterizeProcedure(binding.symbol(), context,
                    seenProcs, /*emitError=*/false)}) {
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
            return CharacterizeProcedure(
                use.symbol(), context, seenProcs, /*emitError=*/false);
          },
          [](const semantics::UseErrorDetails &) {
            // Ambiguous use-association will be handled later during symbol
            // checks, ignore UseErrorDetails here without actual symbol usage.
            return std::optional<Procedure>{};
          },
          [&](const semantics::HostAssocDetails &assoc) {
            return CharacterizeProcedure(
                assoc.symbol(), context, seenProcs, /*emitError=*/false);
          },
          [&](const semantics::GenericDetails &generic) {
            if (const semantics::Symbol * specific{generic.specific()}) {
              return CharacterizeProcedure(
                  *specific, context, seenProcs, emitError);
            } else {
              return std::optional<Procedure>{};
            }
          },
          [&](const semantics::EntityDetails &) {
            CheckForNested(symbol);
            return std::optional<Procedure>{};
          },
          [&](const semantics::SubprogramNameDetails &) {
            CheckForNested(symbol);
            return std::optional<Procedure>{};
          },
          [&](const auto &) {
            context.messages().Say(
                "'%s' is not a procedure"_err_en_US, symbol.name());
            return std::optional<Procedure>{};
          },
      },
      symbol.details())};
  if (result && !symbol.has<semantics::ProcBindingDetails>()) {
    CopyAttrs<Procedure, Procedure::Attr>(symbol, *result,
        {
            {semantics::Attr::BIND_C, Procedure::Attr::BindC},
        });
    CopyAttrs<Procedure, Procedure::Attr>(DEREF(GetMainEntry(&symbol)), *result,
        {
            {semantics::Attr::ELEMENTAL, Procedure::Attr::Elemental},
        });
    if (IsPureProcedure(symbol) || // works for ENTRY too
        (!IsExplicitlyImpureProcedure(symbol) &&
            result->attrs.test(Procedure::Attr::Elemental))) {
      result->attrs.set(Procedure::Attr::Pure);
    }
  }
  return result;
}

static std::optional<DummyProcedure> CharacterizeDummyProcedure(
    const semantics::Symbol &symbol, FoldingContext &context,
    semantics::UnorderedSymbolSet seenProcs) {
  if (auto procedure{CharacterizeProcedure(
          symbol, context, seenProcs, /*emitError=*/true)}) {
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

bool DummyArgument::IsCompatibleWith(const DummyArgument &actual,
    std::string *whyNot, std::optional<std::string> *warning) const {
  if (const auto *ifaceData{std::get_if<DummyDataObject>(&u)}) {
    if (const auto *actualData{std::get_if<DummyDataObject>(&actual.u)}) {
      return ifaceData->IsCompatibleWith(*actualData, whyNot, warning);
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

std::optional<DummyArgument> DummyArgument::FromActual(std::string &&name,
    const Expr<SomeType> &expr, FoldingContext &context,
    bool forImplicitInterface) {
  return common::visit(
      common::visitors{
          [&](const BOZLiteralConstant &) {
            DummyDataObject obj{
                TypeAndShape{DynamicType::TypelessIntrinsicArgument()}};
            obj.attrs.set(DummyDataObject::Attr::DeducedFromActual);
            return std::make_optional<DummyArgument>(
                std::move(name), std::move(obj));
          },
          [&](const NullPointer &) {
            DummyDataObject obj{
                TypeAndShape{DynamicType::TypelessIntrinsicArgument()}};
            obj.attrs.set(DummyDataObject::Attr::DeducedFromActual);
            return std::make_optional<DummyArgument>(
                std::move(name), std::move(obj));
          },
          [&](const ProcedureDesignator &designator) {
            if (auto proc{Procedure::Characterize(
                    designator, context, /*emitError=*/true)}) {
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
              if (forImplicitInterface &&
                  !type->type().IsUnlimitedPolymorphic() &&
                  type->type().IsPolymorphic()) {
                // Pass the monomorphic declared type to an implicit interface
                type->set_type(DynamicType{
                    type->type().GetDerivedTypeSpec(), /*poly=*/false});
              }
              DummyDataObject obj{std::move(*type)};
              obj.attrs.set(DummyDataObject::Attr::DeducedFromActual);
              return std::make_optional<DummyArgument>(
                  std::move(name), std::move(obj));
            } else {
              return std::optional<DummyArgument>{};
            }
          },
      },
      expr.u);
}

std::optional<DummyArgument> DummyArgument::FromActual(std::string &&name,
    const ActualArgument &arg, FoldingContext &context,
    bool forImplicitInterface) {
  if (const auto *expr{arg.UnwrapExpr()}) {
    return FromActual(std::move(name), *expr, context, forImplicitInterface);
  } else if (arg.GetAssumedTypeDummy()) {
    return std::nullopt;
  } else {
    return DummyArgument{AlternateReturn{}};
  }
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

bool DummyArgument::CanBePassedViaImplicitInterface(std::string *whyNot) const {
  if (const auto *object{std::get_if<DummyDataObject>(&u)}) {
    return object->CanBePassedViaImplicitInterface(whyNot);
  } else if (const auto *proc{std::get_if<DummyProcedure>(&u)}) {
    return proc->CanBePassedViaImplicitInterface(whyNot);
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
  return attrs == that.attrs && cudaDataAttr == that.cudaDataAttr &&
      u == that.u;
}

static std::optional<FunctionResult> CharacterizeFunctionResult(
    const semantics::Symbol &symbol, FoldingContext &context,
    semantics::UnorderedSymbolSet seenProcs, bool emitError) {
  if (const auto *object{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    if (auto type{TypeAndShape::Characterize(
            symbol, context, /*invariantOnly=*/false)}) {
      FunctionResult result{std::move(*type)};
      CopyAttrs<FunctionResult, FunctionResult::Attr>(symbol, result,
          {
              {semantics::Attr::ALLOCATABLE, FunctionResult::Attr::Allocatable},
              {semantics::Attr::CONTIGUOUS, FunctionResult::Attr::Contiguous},
              {semantics::Attr::POINTER, FunctionResult::Attr::Pointer},
          });
      result.cudaDataAttr = object->cudaDataAttr();
      return result;
    }
  } else if (auto maybeProc{CharacterizeProcedure(
                 symbol, context, seenProcs, emitError)}) {
    FunctionResult result{std::move(*maybeProc)};
    result.attrs.set(FunctionResult::Attr::Pointer);
    return result;
  }
  return std::nullopt;
}

std::optional<FunctionResult> FunctionResult::Characterize(
    const Symbol &symbol, FoldingContext &context) {
  semantics::UnorderedSymbolSet seenProcs;
  return CharacterizeFunctionResult(
      symbol, context, seenProcs, /*emitError=*/false);
}

bool FunctionResult::IsAssumedLengthCharacter() const {
  if (const auto *ts{std::get_if<TypeAndShape>(&u)}) {
    return ts->type().IsAssumedLengthCharacter();
  } else {
    return false;
  }
}

bool FunctionResult::CanBeReturnedViaImplicitInterface(
    std::string *whyNot) const {
  if (attrs.test(Attr::Pointer) || attrs.test(Attr::Allocatable)) {
    if (whyNot) {
      *whyNot = "the function result is a pointer or allocatable";
    }
    return false; // 15.4.2.2(4)(b)
  } else if (cudaDataAttr) {
    if (whyNot) {
      *whyNot = "the function result has CUDA attributes";
    }
    return false;
  } else if (const auto *typeAndShape{GetTypeAndShape()}) {
    if (typeAndShape->Rank() > 0) {
      if (whyNot) {
        *whyNot = "the function result is an array";
      }
      return false; // 15.4.2.2(4)(a)
    } else {
      const DynamicType &type{typeAndShape->type()};
      switch (type.category()) {
      case TypeCategory::Character:
        if (type.knownLength()) {
          return true;
        } else if (const auto *param{type.charLengthParamValue()}) {
          if (const auto &expr{param->GetExplicit()}) {
            if (IsConstantExpr(*expr)) { // 15.4.2.2(4)(c)
              return true;
            } else {
              if (whyNot) {
                *whyNot = "the function result's length is not constant";
              }
              return false;
            }
          } else if (param->isAssumed()) {
            return true;
          }
        }
        if (whyNot) {
          *whyNot = "the function result's length is not known to the caller";
        }
        return false;
      case TypeCategory::Derived:
        if (type.IsPolymorphic()) {
          if (whyNot) {
            *whyNot = "the function result is polymorphic";
          }
          return false;
        } else {
          const auto &spec{type.GetDerivedTypeSpec()};
          for (const auto &pair : spec.parameters()) {
            if (const auto &expr{pair.second.GetExplicit()}) {
              if (!IsConstantExpr(*expr)) {
                if (whyNot) {
                  *whyNot = "the function result's derived type has a "
                            "non-constant parameter";
                }
                return false; // 15.4.2.2(4)(c)
              }
            }
          }
          return true;
        }
      default:
        return true;
      }
    }
  } else {
    if (whyNot) {
      *whyNot = "the function result has unknown type or shape";
    }
    return false; // 15.4.2.2(4)(b) - procedure pointer?
  }
}

static std::optional<std::string> AreIncompatibleFunctionResultShapes(
    const Shape &x, const Shape &y) {
  // Function results cannot be assumed-rank, hence the non optional arguments.
  int rank{GetRank(x)};
  if (int yrank{GetRank(y)}; yrank != rank) {
    return "rank "s + std::to_string(rank) + " vs " + std::to_string(yrank);
  }
  for (int j{0}; j < rank; ++j) {
    if (x[j] && y[j] && !(*x[j] == *y[j])) {
      return x[j]->AsFortran() + " vs " + y[j]->AsFortran();
    }
  }
  return std::nullopt;
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
  } else if (cudaDataAttr != actual.cudaDataAttr) {
    if (whyNot) {
      *whyNot = "function results have incompatible CUDA data attributes";
    }
  } else if (const auto *ifaceTypeShape{std::get_if<TypeAndShape>(&u)}) {
    if (const auto *actualTypeShape{std::get_if<TypeAndShape>(&actual.u)}) {
      std::optional<std::string> details;
      if (ifaceTypeShape->Rank() != actualTypeShape->Rank()) {
        if (whyNot) {
          *whyNot = "function results have distinct ranks";
        }
      } else if (!attrs.test(Attr::Allocatable) && !attrs.test(Attr::Pointer) &&
          (details = AreIncompatibleFunctionResultShapes(
               ifaceTypeShape->shape().value(),
               actualTypeShape->shape().value()))) {
        if (whyNot) {
          *whyNot = "function results have distinct extents (" + *details + ')';
        }
      } else if (ifaceTypeShape->type() != actualTypeShape->type()) {
        if (ifaceTypeShape->type().category() !=
            actualTypeShape->type().category()) {
        } else if (ifaceTypeShape->type().category() ==
            TypeCategory::Character) {
          if (ifaceTypeShape->type().kind() == actualTypeShape->type().kind()) {
            if (IsAssumedLengthCharacter() ||
                actual.IsAssumedLengthCharacter()) {
              return true;
            } else {
              auto len{ToInt64(ifaceTypeShape->LEN())};
              auto actualLen{ToInt64(actualTypeShape->LEN())};
              if (len.has_value() != actualLen.has_value()) {
                if (whyNot) {
                  *whyNot = "constant-length vs non-constant-length character "
                            "results";
                }
              } else if (len && *len != *actualLen) {
                if (whyNot) {
                  *whyNot = "character results with distinct lengths";
                }
              } else {
                const auto *ifaceLenParam{
                    ifaceTypeShape->type().charLengthParamValue()};
                const auto *actualLenParam{
                    actualTypeShape->type().charLengthParamValue()};
                if (ifaceLenParam && actualLenParam &&
                    ifaceLenParam->isExplicit() !=
                        actualLenParam->isExplicit()) {
                  if (whyNot) {
                    *whyNot =
                        "explicit-length vs deferred-length character results";
                  }
                } else {
                  return true;
                }
              }
            }
          }
        } else if (ifaceTypeShape->type().category() == TypeCategory::Derived) {
          if (ifaceTypeShape->type().IsPolymorphic() ==
                  actualTypeShape->type().IsPolymorphic() &&
              !ifaceTypeShape->type().IsUnlimitedPolymorphic() &&
              !actualTypeShape->type().IsUnlimitedPolymorphic() &&
              AreSameDerivedType(ifaceTypeShape->type().GetDerivedTypeSpec(),
                  actualTypeShape->type().GetDerivedTypeSpec())) {
            return true;
          }
        }
        if (whyNot) {
          *whyNot = "function results have distinct types: "s +
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
      if (ifaceProc->value().IsCompatibleWith(actualProc->value(),
              /*ignoreImplicitVsExplicit=*/false, whyNot)) {
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
  if (cudaDataAttr) {
    o << " cudaDataAttr: " << common::EnumToString(*cudaDataAttr);
  }
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
      dummyArguments == that.dummyArguments &&
      cudaSubprogramAttrs == that.cudaSubprogramAttrs;
}

bool Procedure::IsCompatibleWith(const Procedure &actual,
    bool ignoreImplicitVsExplicit, std::string *whyNot,
    const SpecificIntrinsic *specificIntrinsic,
    std::optional<std::string> *warning) const {
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
  if (ignoreImplicitVsExplicit) {
    differences.reset(Attr::ImplicitInterface);
  }
  if (!differences.empty()) {
    if (whyNot) {
      auto sep{": "s};
      *whyNot = "incompatible procedure attributes";
      differences.IterateOverMembers([&](Attr x) {
        *whyNot += sep + std::string{EnumToString(x)};
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
  } else if (cudaSubprogramAttrs != actual.cudaSubprogramAttrs) {
    if (whyNot) {
      *whyNot = "incompatible CUDA subprogram attributes";
    }
  } else if (dummyArguments.size() != actual.dummyArguments.size()) {
    if (whyNot) {
      *whyNot = "distinct numbers of dummy arguments";
    }
  } else {
    for (std::size_t j{0}; j < dummyArguments.size(); ++j) {
      // Subtlety: the dummy/actual distinction must be reversed for this
      // compatibility test in order to correctly check extended vs.
      // base types.  Example:
      //   subroutine s1(base); subroutine s2(extended)
      //   procedure(s1), pointer :: p
      //   p => s2 ! an error, s2 is more restricted, can't handle "base"
      std::optional<std::string> gotWarning;
      if (!actual.dummyArguments[j].IsCompatibleWith(
              dummyArguments[j], whyNot, warning ? &gotWarning : nullptr)) {
        if (whyNot) {
          *whyNot = "incompatible dummy argument #"s + std::to_string(j + 1) +
              ": "s + *whyNot;
        }
        return false;
      } else if (warning && !*warning && gotWarning) {
        *warning = "possibly incompatible dummy argument #"s +
            std::to_string(j + 1) + ": "s + std::move(*gotWarning);
      }
    }
    return true;
  }
  return false;
}

std::optional<int> Procedure::FindPassIndex(
    std::optional<parser::CharBlock> name) const {
  int argCount{static_cast<int>(dummyArguments.size())};
  if (name) {
    for (int index{0}; index < argCount; ++index) {
      if (*name == dummyArguments[index].name.c_str()) {
        return index;
      }
    }
    return std::nullopt;
  } else if (argCount > 0) {
    return 0;
  } else {
    return std::nullopt;
  }
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
    if (passIndex && j == *passIndex) {
      if (!that.dummyArguments[j].IsCompatibleWith(dummyArguments[j])) {
        return false;
      }
    } else if (dummyArguments[j] != that.dummyArguments[j]) {
      return false;
    }
  }
  return true;
}

std::optional<Procedure> Procedure::Characterize(
    const semantics::Symbol &symbol, FoldingContext &context) {
  semantics::UnorderedSymbolSet seenProcs;
  return CharacterizeProcedure(symbol, context, seenProcs, /*emitError=*/true);
}

std::optional<Procedure> Procedure::Characterize(
    const ProcedureDesignator &proc, FoldingContext &context, bool emitError) {
  if (const auto *symbol{proc.GetSymbol()}) {
    semantics::UnorderedSymbolSet seenProcs;
    return CharacterizeProcedure(*symbol, context, seenProcs, emitError);
  } else if (const auto *intrinsic{proc.GetSpecificIntrinsic()}) {
    return intrinsic->characteristics.value();
  } else {
    return std::nullopt;
  }
}

std::optional<Procedure> Procedure::Characterize(
    const ProcedureRef &ref, FoldingContext &context) {
  if (auto callee{Characterize(ref.proc(), context, /*emitError=*/true)}) {
    if (callee->functionResult) {
      if (const Procedure *
          proc{callee->functionResult->IsProcedurePointer()}) {
        return {*proc};
      }
    }
  }
  return std::nullopt;
}

std::optional<Procedure> Procedure::Characterize(
    const Expr<SomeType> &expr, FoldingContext &context) {
  if (const auto *procRef{UnwrapProcedureRef(expr)}) {
    return Characterize(*procRef, context);
  } else if (const auto *procDesignator{
                 std::get_if<ProcedureDesignator>(&expr.u)}) {
    return Characterize(*procDesignator, context, /*emitError=*/true);
  } else if (const Symbol * symbol{UnwrapWholeSymbolOrComponentDataRef(expr)}) {
    return Characterize(*symbol, context);
  } else {
    context.messages().Say(
        "Expression '%s' is not a procedure"_err_en_US, expr.AsFortran());
    return std::nullopt;
  }
}

std::optional<Procedure> Procedure::FromActuals(const ProcedureDesignator &proc,
    const ActualArguments &args, FoldingContext &context) {
  auto callee{Characterize(proc, context, /*emitError=*/true)};
  if (callee) {
    if (callee->dummyArguments.empty() &&
        callee->attrs.test(Procedure::Attr::ImplicitInterface)) {
      int j{0};
      for (const auto &arg : args) {
        ++j;
        if (arg) {
          if (auto dummy{DummyArgument::FromActual("x"s + std::to_string(j),
                  *arg, context,
                  /*forImplicitInterface=*/true)}) {
            callee->dummyArguments.emplace_back(std::move(*dummy));
            continue;
          }
        }
        callee.reset();
        break;
      }
    }
  }
  return callee;
}

bool Procedure::CanBeCalledViaImplicitInterface(std::string *whyNot) const {
  if (attrs.test(Attr::Elemental)) {
    if (whyNot) {
      *whyNot = "the procedure is elemental";
    }
    return false; // 15.4.2.2(5,6)
  } else if (attrs.test(Attr::BindC)) {
    if (whyNot) {
      *whyNot = "the procedure is BIND(C)";
    }
    return false; // 15.4.2.2(5,6)
  } else if (cudaSubprogramAttrs &&
      *cudaSubprogramAttrs != common::CUDASubprogramAttrs::Host &&
      *cudaSubprogramAttrs != common::CUDASubprogramAttrs::Global) {
    if (whyNot) {
      *whyNot = "the procedure is CUDA but neither HOST nor GLOBAL";
    }
    return false;
  } else if (IsFunction() &&
      !functionResult->CanBeReturnedViaImplicitInterface(whyNot)) {
    return false;
  } else {
    for (const DummyArgument &arg : dummyArguments) {
      if (!arg.CanBePassedViaImplicitInterface(whyNot)) {
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
  o << (sep == '(' ? "()" : ")");
  if (cudaSubprogramAttrs) {
    o << " cudaSubprogramAttrs: " << common::EnumToString(*cudaSubprogramAttrs);
  }
  return o;
}

// Utility class to determine if Procedures, etc. are distinguishable
class DistinguishUtils {
public:
  explicit DistinguishUtils(const common::LanguageFeatureControl &features)
      : features_{features} {}

  // Are these procedures distinguishable for a generic name?
  std::optional<bool> Distinguishable(
      const Procedure &, const Procedure &) const;
  // Are these procedures distinguishable for a generic operator or assignment?
  std::optional<bool> DistinguishableOpOrAssign(
      const Procedure &, const Procedure &) const;

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

  bool AnyOptionalData(const DummyArguments &) const;
  bool AnyUnlimitedPolymorphicData(const DummyArguments &) const;
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
  bool Distinguishable(
      const TypeAndShape &, const TypeAndShape &, common::IgnoreTKRSet) const;
  bool IsTkrCompatible(const DummyArgument &, const DummyArgument &) const;
  bool IsTkCompatible(const DummyDataObject &, const DummyDataObject &) const;
  const DummyArgument *GetAtEffectivePosition(
      const DummyArguments &, int) const;
  const DummyArgument *GetPassArg(const Procedure &) const;

  const common::LanguageFeatureControl &features_;
};

// Simpler distinguishability rules for operators and assignment
std::optional<bool> DistinguishUtils::DistinguishableOpOrAssign(
    const Procedure &proc1, const Procedure &proc2) const {
  if ((proc1.IsFunction() && proc2.IsSubroutine()) ||
      (proc1.IsSubroutine() && proc2.IsFunction())) {
    return true;
  }
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

std::optional<bool> DistinguishUtils::Distinguishable(
    const Procedure &proc1, const Procedure &proc2) const {
  if ((proc1.IsFunction() && proc2.IsSubroutine()) ||
      (proc1.IsSubroutine() && proc2.IsFunction())) {
    return true;
  }
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
  if (proc1.cudaSubprogramAttrs != proc2.cudaSubprogramAttrs) {
    return true;
  }
  // If there are no optional or unlimited polymorphic dummy arguments,
  // then we know the result for sure; otherwise, it's possible for
  // the procedures to be unambiguous.
  if ((AnyOptionalData(args1) || AnyUnlimitedPolymorphicData(args1)) &&
      (AnyOptionalData(args2) || AnyUnlimitedPolymorphicData(args2))) {
    return std::nullopt; // meaning "maybe"
  } else {
    return false;
  }
}

bool DistinguishUtils::AnyOptionalData(const DummyArguments &args) const {
  for (const auto &arg : args) {
    if (std::holds_alternative<DummyDataObject>(arg.u) && arg.IsOptional()) {
      return true;
    }
  }
  return false;
}

bool DistinguishUtils::AnyUnlimitedPolymorphicData(
    const DummyArguments &args) const {
  for (const auto &arg : args) {
    if (const auto *object{std::get_if<DummyDataObject>(&arg.u)}) {
      if (object->type.type().IsUnlimitedPolymorphic()) {
        return true;
      }
    }
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
  return llvm::count_if(args, [&](const DummyArgument &y) {
    return !y.pass && !y.IsOptional() && IsTkrCompatible(x, y);
  });
}

// Return the number of dummy data objects in args that are not
// distinguishable from x and not passed-object.
int DistinguishUtils::CountNotDistinguishableFrom(
    const DummyArgument &x, const DummyArguments &args) const {
  return llvm::count_if(args, [&](const DummyArgument &y) {
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
  if (Distinguishable(x.type, y.type, x.ignoreTKR | y.ignoreTKR)) {
    return true;
  } else if (x.attrs.test(Attr::Allocatable) && y.attrs.test(Attr::Pointer) &&
      y.intent != common::Intent::In) {
    return true;
  } else if (y.attrs.test(Attr::Allocatable) && x.attrs.test(Attr::Pointer) &&
      x.intent != common::Intent::In) {
    return true;
  } else if (!common::AreCompatibleCUDADataAttrs(x.cudaDataAttr, y.cudaDataAttr,
                 x.ignoreTKR | y.ignoreTKR,
                 /*allowUnifiedMatchingRule=*/false)) {
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
  if (Distinguishable(xProc, yProc).value_or(false)) {
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
  if (x.cudaDataAttr != y.cudaDataAttr) {
    return true;
  }
  return common::visit(
      common::visitors{
          [&](const TypeAndShape &z) {
            return Distinguishable(
                z, std::get<TypeAndShape>(y.u), common::IgnoreTKRSet{});
          },
          [&](const CopyableIndirection<Procedure> &z) {
            return Distinguishable(z.value(),
                std::get<CopyableIndirection<Procedure>>(y.u).value())
                .value_or(false);
          },
      },
      x.u);
}

bool DistinguishUtils::Distinguishable(const TypeAndShape &x,
    const TypeAndShape &y, common::IgnoreTKRSet ignoreTKR) const {
  if (!x.type().IsTkCompatibleWith(y.type(), ignoreTKR) &&
      !y.type().IsTkCompatibleWith(x.type(), ignoreTKR)) {
    return true;
  }
  if (ignoreTKR.test(common::IgnoreTKR::Rank)) {
  } else if (x.attrs().test(TypeAndShape::Attr::AssumedRank) ||
      y.attrs().test(TypeAndShape::Attr::AssumedRank)) {
  } else if (x.Rank() != y.Rank()) {
    return true;
  }
  return false;
}

// Compatibility based on type, kind, and rank

bool DistinguishUtils::IsTkrCompatible(
    const DummyArgument &x, const DummyArgument &y) const {
  const auto *obj1{std::get_if<DummyDataObject>(&x.u)};
  const auto *obj2{std::get_if<DummyDataObject>(&y.u)};
  return obj1 && obj2 && IsTkCompatible(*obj1, *obj2) &&
      (obj1->type.Rank() == obj2->type.Rank() ||
          obj1->type.attrs().test(TypeAndShape::Attr::AssumedRank) ||
          obj2->type.attrs().test(TypeAndShape::Attr::AssumedRank) ||
          obj1->ignoreTKR.test(common::IgnoreTKR::Rank) ||
          obj2->ignoreTKR.test(common::IgnoreTKR::Rank));
}

bool DistinguishUtils::IsTkCompatible(
    const DummyDataObject &x, const DummyDataObject &y) const {
  return x.type.type().IsTkCompatibleWith(
      y.type.type(), x.ignoreTKR | y.ignoreTKR);
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

std::optional<bool> Distinguishable(
    const common::LanguageFeatureControl &features, const Procedure &x,
    const Procedure &y) {
  return DistinguishUtils{features}.Distinguishable(x, y);
}

std::optional<bool> DistinguishableOpOrAssign(
    const common::LanguageFeatureControl &features, const Procedure &x,
    const Procedure &y) {
  return DistinguishUtils{features}.DistinguishableOpOrAssign(x, y);
}

DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(DummyArgument)
DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(DummyProcedure)
DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(FunctionResult)
DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(Procedure)
} // namespace Fortran::evaluate::characteristics

template class Fortran::common::Indirection<
    Fortran::evaluate::characteristics::Procedure, true>;
