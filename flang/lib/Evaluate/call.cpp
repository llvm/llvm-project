//===-- lib/Evaluate/call.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/call.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/tools.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Support/Fortran.h"

namespace Fortran::evaluate {

DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(ActualArgument)
ActualArgument::ActualArgument(Expr<SomeType> &&x) : u_{std::move(x)} {}
ActualArgument::ActualArgument(common::CopyableIndirection<Expr<SomeType>> &&v)
    : u_{std::move(v)} {}
ActualArgument::ActualArgument(AssumedType x) : u_{x} {}
ActualArgument::ActualArgument(common::Label x) : u_{x} {}
ActualArgument::~ActualArgument() {}

ActualArgument::AssumedType::AssumedType(const Symbol &symbol)
    : symbol_{symbol} {
  const semantics::DeclTypeSpec *type{symbol.GetType()};
  CHECK(type && type->category() == semantics::DeclTypeSpec::TypeStar);
}

int ActualArgument::AssumedType::Rank() const { return symbol_->Rank(); }

ActualArgument &ActualArgument::operator=(Expr<SomeType> &&expr) {
  u_ = std::move(expr);
  return *this;
}

std::optional<DynamicType> ActualArgument::GetType() const {
  if (const Expr<SomeType> *expr{UnwrapExpr()}) {
    return expr->GetType();
  } else if (std::holds_alternative<AssumedType>(u_)) {
    return DynamicType::AssumedType();
  } else {
    return std::nullopt;
  }
}

int ActualArgument::Rank() const {
  if (const Expr<SomeType> *expr{UnwrapExpr()}) {
    return expr->Rank();
  } else {
    return std::get<AssumedType>(u_).Rank();
  }
}

bool ActualArgument::operator==(const ActualArgument &that) const {
  return keyword_ == that.keyword_ && attrs_ == that.attrs_ && u_ == that.u_;
}

void ActualArgument::Parenthesize() {
  u_ = evaluate::Parenthesize(std::move(DEREF(UnwrapExpr())));
}

SpecificIntrinsic::SpecificIntrinsic(
    IntrinsicProcedure n, characteristics::Procedure &&chars)
    : name{n}, characteristics{
                   new characteristics::Procedure{std::move(chars)}} {}

DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(SpecificIntrinsic)

SpecificIntrinsic::~SpecificIntrinsic() {}

bool SpecificIntrinsic::operator==(const SpecificIntrinsic &that) const {
  return name == that.name && characteristics == that.characteristics;
}

ProcedureDesignator::ProcedureDesignator(Component &&c)
    : u{common::CopyableIndirection<Component>::Make(std::move(c))} {}

bool ProcedureDesignator::operator==(const ProcedureDesignator &that) const {
  return u == that.u;
}

std::optional<DynamicType> ProcedureDesignator::GetType() const {
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    if (const auto &result{intrinsic->characteristics.value().functionResult}) {
      if (const auto *typeAndShape{result->GetTypeAndShape()}) {
        return typeAndShape->type();
      }
    }
  } else {
    return DynamicType::From(GetSymbol());
  }
  return std::nullopt;
}

int ProcedureDesignator::Rank() const {
  if (const Symbol * symbol{GetSymbol()}) {
    // Subtle: will be zero for functions returning procedure pointers
    return symbol->Rank();
  }
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    if (const auto &result{intrinsic->characteristics.value().functionResult}) {
      if (const auto *typeAndShape{result->GetTypeAndShape()}) {
        CHECK(!typeAndShape->attrs().test(
            characteristics::TypeAndShape::Attr::AssumedRank));
        return typeAndShape->Rank();
      }
      // Otherwise, intrinsic returns a procedure pointer (e.g. NULL(MOLD=pptr))
    }
  }
  return 0;
}

const Symbol *ProcedureDesignator::GetInterfaceSymbol() const {
  if (const Symbol * symbol{GetSymbol()}) {
    const Symbol &ultimate{symbol->GetUltimate()};
    if (const auto *proc{ultimate.detailsIf<semantics::ProcEntityDetails>()}) {
      return proc->procInterface();
    } else if (const auto *binding{
                   ultimate.detailsIf<semantics::ProcBindingDetails>()}) {
      return &binding->symbol();
    } else if (ultimate.has<semantics::SubprogramDetails>()) {
      return &ultimate;
    }
  }
  return nullptr;
}

bool ProcedureDesignator::IsElemental() const {
  if (const Symbol * interface{GetInterfaceSymbol()}) {
    return IsElementalProcedure(*interface);
  } else if (const Symbol * symbol{GetSymbol()}) {
    return IsElementalProcedure(*symbol);
  } else if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    return intrinsic->characteristics.value().attrs.test(
        characteristics::Procedure::Attr::Elemental);
  } else {
    DIE("ProcedureDesignator::IsElemental(): no case");
  }
  return false;
}

bool ProcedureDesignator::IsPure() const {
  if (const Symbol * interface{GetInterfaceSymbol()}) {
    return IsPureProcedure(*interface);
  } else if (const Symbol * symbol{GetSymbol()}) {
    return IsPureProcedure(*symbol);
  } else if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    return intrinsic->characteristics.value().attrs.test(
        characteristics::Procedure::Attr::Pure);
  } else {
    DIE("ProcedureDesignator::IsPure(): no case");
  }
  return false;
}

const SpecificIntrinsic *ProcedureDesignator::GetSpecificIntrinsic() const {
  return std::get_if<SpecificIntrinsic>(&u);
}

const Component *ProcedureDesignator::GetComponent() const {
  if (auto *c{std::get_if<common::CopyableIndirection<Component>>(&u)}) {
    return &c->value();
  } else {
    return nullptr;
  }
}

const Symbol *ProcedureDesignator::GetSymbol() const {
  return common::visit(
      common::visitors{
          [](SymbolRef symbol) { return &*symbol; },
          [](const common::CopyableIndirection<Component> &c) {
            return &c.value().GetLastSymbol();
          },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      u);
}

const SymbolRef *ProcedureDesignator::UnwrapSymbolRef() const {
  return std::get_if<SymbolRef>(&u);
}

std::string ProcedureDesignator::GetName() const {
  return common::visit(
      common::visitors{
          [](const SpecificIntrinsic &i) { return i.name; },
          [](const Symbol &symbol) { return symbol.name().ToString(); },
          [](const common::CopyableIndirection<Component> &c) {
            return c.value().GetLastSymbol().name().ToString();
          },
      },
      u);
}

std::optional<Expr<SubscriptInteger>> ProcedureRef::LEN() const {
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&proc_.u)}) {
    if (intrinsic->name == "repeat") {
      // LEN(REPEAT(ch,n)) == LEN(ch) * n
      CHECK(arguments_.size() == 2);
      const auto *stringArg{
          UnwrapExpr<Expr<SomeCharacter>>(arguments_[0].value())};
      const auto *nCopiesArg{
          UnwrapExpr<Expr<SomeInteger>>(arguments_[1].value())};
      CHECK(stringArg && nCopiesArg);
      if (auto stringLen{stringArg->LEN()}) {
        auto converted{ConvertTo(*stringLen, common::Clone(*nCopiesArg))};
        return *std::move(stringLen) * std::move(converted);
      }
    }
    // Some other cases (e.g., LEN(CHAR(...))) are handled in
    // ProcedureDesignator::LEN() because they're independent of the
    // lengths of the actual arguments.
  }
  if (auto len{proc_.LEN()}) {
    if (IsActuallyConstant(*len)) {
      return len;
    }
    // TODO: Handle cases where the length of a function result is a
    // safe expression in terms of actual argument values, after substituting
    // actual argument expressions for INTENT(IN)/VALUE dummy arguments.
  }
  return std::nullopt;
}

int ProcedureRef::Rank() const {
  if (IsElemental()) {
    for (const auto &arg : arguments_) {
      if (arg) {
        if (int rank{arg->Rank()}; rank > 0) {
          return rank;
        }
      }
    }
    return 0;
  } else {
    return proc_.Rank();
  }
}

ProcedureRef::~ProcedureRef() {}

void ProcedureRef::Deleter(ProcedureRef *p) { delete p; }

// We don't know the dummy argument info (e.g., procedure with implicit
// interface
static void DetermineCopyInOutArgument(
    const characteristics::Procedure &procInfo, ActualArgument &actual,
    semantics::SemanticsContext &sc) {
  if (actual.isAlternateReturn()) {
    return;
  }
  if (!evaluate::IsVariable(actual)) {
    // Actual argument expressions that aren’t variables are copy-in, but
    // not copy-out.
    actual.SetMayNeedCopyIn();
  } else if (bool actualIsArray{actual.Rank() > 0}; actualIsArray &&
             !IsSimplyContiguous(actual, sc.foldingContext())) {
    // Actual arguments that are variables are copy-in when non-contiguous.
    // They are copy-out when don't have vector subscripts
    actual.SetMayNeedCopyIn();
    if (!HasVectorSubscript(actual)) {
      actual.SetMayNeedCopyOut();
    }
  } else if (ExtractCoarrayRef(actual)) {
    // Coindexed actual args need copy-in and copy-out
    actual.SetMayNeedCopyIn();
    actual.SetMayNeedCopyOut();
  }
}

static void DetermineCopyInOutArgument(
    const characteristics::Procedure &procInfo, ActualArgument &actual,
    characteristics::DummyArgument &dummy, semantics::SemanticsContext &sc) {
  assert(procInfo.HasExplicitInterface() && "expect explicit interface proc");
  if (actual.isAlternateReturn()) {
    return;
  }
  if (!evaluate::IsVariable(actual)) {
    // Actual argument expressions that aren’t variables are copy-in, but
    // not copy-out.
    actual.SetMayNeedCopyIn();
    return;
  }
  const auto *dummyObj{std::get_if<characteristics::DummyDataObject>(&dummy.u)};
  if (!dummyObj) {
    // Only DummyDataObject has the information we need
    return;
  }
  // Pass by value, always copy-in, never copy-out
  bool dummyIsValue{
      dummyObj->attrs.test(characteristics::DummyDataObject::Attr::Value)};
  if (dummyIsValue) {
    actual.SetMayNeedCopyIn();
    return;
  }
  bool dummyIntentIn{dummyObj->intent == common::Intent::In};
  bool dummyIntentOut{dummyObj->intent == common::Intent::Out};

  auto setCopyIn = [&]() {
    if (!dummyIntentOut) {
      // INTENT(OUT) never need copy-in
      actual.SetMayNeedCopyIn();
    }
  };
  auto setCopyOut = [&]() {
    if (!dummyIntentIn) {
      // INTENT(IN) never need copy-out
      actual.SetMayNeedCopyOut();
    }
  };

  bool actualIsArray{actual.Rank() > 0};
  if (!actualIsArray) {
    return;
  }

  // Check actual contiguity, unless dummy doesn't care
  bool actualTreatAsContiguous{
      dummyObj->ignoreTKR.test(common::IgnoreTKR::Contiguous) ||
      IsSimplyContiguous(actual, sc.foldingContext())};

  bool actualHasVectorSubscript{HasVectorSubscript(actual)};

  bool dummyIsArray{dummyObj->type.Rank() > 0};
  bool dummyIsExplicitShape{
      dummyIsArray ? IsExplicitShape(*dummyObj->type.shape()) : false};
  bool dummyIsAssumedSize{dummyObj->type.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedSize)};
  bool dummyNeedsContiguity{dummyIsArray &&
      (dummyIsExplicitShape || dummyIsAssumedSize ||
          dummyObj->attrs.test(
              characteristics::DummyDataObject::Attr::Contiguous))};
  if (!actualTreatAsContiguous && dummyNeedsContiguity) {
    setCopyIn();
    // Cannot do copy-out for vector subscripts: there could be repeated
    // indices, for example
    if (!actualHasVectorSubscript) {
      setCopyOut();
    }
    return;
  }

  if (!dummyObj->ignoreTKR.test(common::IgnoreTKR::Type)) {
    // flang supports limited cases of passing polymorphic to non-polimorphic.
    // These cases require temporary of non-polymorphic type.
    auto actualType{characteristics::TypeAndShape::Characterize(
        actual, sc.foldingContext())};
    bool actualIsPolymorphic{actualType->type().IsPolymorphic()};
    bool dummyIsPolymorphic{dummyObj->type.type().IsPolymorphic()};
    if (actualIsPolymorphic && !dummyIsPolymorphic) {
      setCopyIn();
      setCopyOut();
    }
  }

  // TODO: character type differences?
}

void ProcedureRef::DetermineCopyInOut() {
  if (!proc_.GetSymbol()) {
    return;
  }
  // Get folding context of the call site owner
  semantics::SemanticsContext &sc{proc_.GetSymbol()->owner().context()};
  FoldingContext &fc{sc.foldingContext()};
  auto procInfo{
      characteristics::Procedure::Characterize(proc_, fc, /*emitError=*/true)};
  if (!procInfo) {
    return;
  }
  if (!procInfo->HasExplicitInterface()) {
    for (auto &actual : arguments_) {
      if (!actual) {
        continue;
      }
      DetermineCopyInOutArgument(*procInfo, *actual, sc);
    }
    return;
  }
  // Don't change anything about actual or dummy arguments, except for
  // computing copy-in/copy-out information. If detect something wrong with
  // the arguments, stop processing and let semantic analysis generate the
  // error messages.
  size_t index{0};
  std::set<std::string> processedKeywords;
  bool seenKeyword{false};
  for (auto &actual : arguments_) {
    if (!actual) {
      continue;
    }
    if (index >= procInfo->dummyArguments.size()) {
      // More actual arguments than dummy arguments. Semantic analysis will
      // deal with the error.
      return;
    }
    if (actual->keyword()) {
      seenKeyword = true;
      auto actualName{actual->keyword()->ToString()};
      if (processedKeywords.find(actualName) != processedKeywords.end()) {
        // Actual arguments with duplicate keywords. Semantic analysis will
        // deal with the error.
        return;
      } else {
        processedKeywords.insert(actualName);
        if (auto it{std::find_if(procInfo->dummyArguments.begin(),
                procInfo->dummyArguments.end(),
                [&](const characteristics::DummyArgument &dummy) {
                  return dummy.name == actualName;
                })};
            it != procInfo->dummyArguments.end()) {
          DetermineCopyInOutArgument(*procInfo, *actual, *it, sc);
        }
      }
    } else if (seenKeyword) {
      // Non-keyword actual argument after have seen at least one keyword
      // actual argument. Semantic analysis will deal with the error.
      return;
    } else {
      // Positional argument processing
      DetermineCopyInOutArgument(
          *procInfo, *actual, procInfo->dummyArguments[index], sc);
    }
    ++index;
  }
}

} // namespace Fortran::evaluate
