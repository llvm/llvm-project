//===-- lib/Semantics/pointer-assignment.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pointer-assignment.h"
#include "definable.h"
#include "flang/Common/idioms.h"
#include "flang/Common/restorer.h"
#include "flang/Common/template.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <set>
#include <string>
#include <type_traits>

// Semantic checks for pointer assignment.

namespace Fortran::semantics {

using namespace parser::literals;
using evaluate::characteristics::DummyDataObject;
using evaluate::characteristics::FunctionResult;
using evaluate::characteristics::Procedure;
using evaluate::characteristics::TypeAndShape;
using parser::MessageFixedText;
using parser::MessageFormattedText;

class PointerAssignmentChecker {
public:
  PointerAssignmentChecker(SemanticsContext &context, const Scope &scope,
      parser::CharBlock source, const std::string &description)
      : context_{context}, scope_{scope}, source_{source}, description_{
                                                               description} {}
  PointerAssignmentChecker(
      SemanticsContext &context, const Scope &scope, const Symbol &lhs)
      : context_{context}, scope_{scope}, source_{lhs.name()},
        description_{"pointer '"s + lhs.name().ToString() + '\''}, lhs_{&lhs} {
    set_lhsType(TypeAndShape::Characterize(lhs, foldingContext_));
    set_isContiguous(lhs.attrs().test(Attr::CONTIGUOUS));
    set_isVolatile(lhs.attrs().test(Attr::VOLATILE));
  }
  PointerAssignmentChecker &set_lhsType(std::optional<TypeAndShape> &&);
  PointerAssignmentChecker &set_isContiguous(bool);
  PointerAssignmentChecker &set_isVolatile(bool);
  PointerAssignmentChecker &set_isBoundsRemapping(bool);
  PointerAssignmentChecker &set_isAssumedRank(bool);
  PointerAssignmentChecker &set_pointerComponentLHS(const Symbol *);
  bool CheckLeftHandSide(const SomeExpr &);
  bool Check(const SomeExpr &);

private:
  bool CharacterizeProcedure();
  template <typename T> bool Check(const T &);
  template <typename T> bool Check(const evaluate::Expr<T> &);
  template <typename T> bool Check(const evaluate::FunctionRef<T> &);
  template <typename T> bool Check(const evaluate::Designator<T> &);
  bool Check(const evaluate::NullPointer &);
  bool Check(const evaluate::ProcedureDesignator &);
  bool Check(const evaluate::ProcedureRef &);
  // Target is a procedure
  bool Check(parser::CharBlock rhsName, bool isCall,
      const Procedure * = nullptr,
      const evaluate::SpecificIntrinsic *specific = nullptr);
  bool LhsOkForUnlimitedPoly() const;
  template <typename... A> parser::Message *Say(A &&...);

  SemanticsContext &context_;
  evaluate::FoldingContext &foldingContext_{context_.foldingContext()};
  const Scope &scope_;
  const parser::CharBlock source_;
  const std::string description_;
  const Symbol *lhs_{nullptr};
  std::optional<TypeAndShape> lhsType_;
  std::optional<Procedure> procedure_;
  bool characterizedProcedure_{false};
  bool isContiguous_{false};
  bool isVolatile_{false};
  bool isBoundsRemapping_{false};
  bool isAssumedRank_{false};
  const Symbol *pointerComponentLHS_{nullptr};
};

PointerAssignmentChecker &PointerAssignmentChecker::set_lhsType(
    std::optional<TypeAndShape> &&lhsType) {
  lhsType_ = std::move(lhsType);
  return *this;
}

PointerAssignmentChecker &PointerAssignmentChecker::set_isContiguous(
    bool isContiguous) {
  isContiguous_ = isContiguous;
  return *this;
}

PointerAssignmentChecker &PointerAssignmentChecker::set_isVolatile(
    bool isVolatile) {
  isVolatile_ = isVolatile;
  return *this;
}

PointerAssignmentChecker &PointerAssignmentChecker::set_isBoundsRemapping(
    bool isBoundsRemapping) {
  isBoundsRemapping_ = isBoundsRemapping;
  return *this;
}

PointerAssignmentChecker &PointerAssignmentChecker::set_isAssumedRank(
    bool isAssumedRank) {
  isAssumedRank_ = isAssumedRank;
  return *this;
}

PointerAssignmentChecker &PointerAssignmentChecker::set_pointerComponentLHS(
    const Symbol *symbol) {
  pointerComponentLHS_ = symbol;
  return *this;
}

bool PointerAssignmentChecker::CharacterizeProcedure() {
  if (!characterizedProcedure_) {
    characterizedProcedure_ = true;
    if (lhs_ && IsProcedure(*lhs_)) {
      procedure_ = Procedure::Characterize(*lhs_, foldingContext_);
    }
  }
  return procedure_.has_value();
}

bool PointerAssignmentChecker::CheckLeftHandSide(const SomeExpr &lhs) {
  if (auto whyNot{WhyNotDefinable(foldingContext_.messages().at(), scope_,
          DefinabilityFlags{DefinabilityFlag::PointerDefinition}, lhs)}) {
    if (auto *msg{Say(
            "The left-hand side of a pointer assignment is not definable"_err_en_US)}) {
      msg->Attach(std::move(whyNot->set_severity(parser::Severity::Because)));
    }
    return false;
  } else if (evaluate::IsAssumedRank(lhs)) {
    Say("The left-hand side of a pointer assignment must not be an assumed-rank dummy argument"_err_en_US);
    return false;
  } else {
    return true;
  }
}

template <typename T> bool PointerAssignmentChecker::Check(const T &) {
  // Catch-all case for really bad target expression
  Say("Target associated with %s must be a designator or a call to a"
      " pointer-valued function"_err_en_US,
      description_);
  return false;
}

template <typename T>
bool PointerAssignmentChecker::Check(const evaluate::Expr<T> &x) {
  return common::visit([&](const auto &x) { return Check(x); }, x.u);
}

bool PointerAssignmentChecker::Check(const SomeExpr &rhs) {
  if (HasVectorSubscript(rhs)) { // C1025
    Say("An array section with a vector subscript may not be a pointer target"_err_en_US);
    return false;
  }
  if (ExtractCoarrayRef(rhs)) { // C1026
    Say("A coindexed object may not be a pointer target"_err_en_US);
    return false;
  }
  if (!common::visit([&](const auto &x) { return Check(x); }, rhs.u)) {
    return false;
  }
  if (IsNullPointer(rhs)) {
    return true;
  }
  if (lhs_ && IsProcedure(*lhs_)) {
    return true;
  }
  if (const auto *pureProc{FindPureProcedureContaining(scope_)}) {
    if (pointerComponentLHS_) { // C1594(4) is a hard error
      if (const Symbol * object{FindExternallyVisibleObject(rhs, *pureProc)}) {
        if (auto *msg{Say(
                "Externally visible object '%s' may not be associated with pointer component '%s' in a pure procedure"_err_en_US,
                object->name(), pointerComponentLHS_->name())}) {
          msg->Attach(object->name(), "Object declaration"_en_US)
              .Attach(
                  pointerComponentLHS_->name(), "Pointer declaration"_en_US);
        }
        return false;
      }
    } else if (const Symbol * base{GetFirstSymbol(rhs)}) {
      if (const char *why{WhyBaseObjectIsSuspicious(
              base->GetUltimate(), scope_)}) { // C1594(3)
        evaluate::SayWithDeclaration(foldingContext_.messages(), *base,
            "A pure subprogram may not use '%s' as the target of pointer assignment because it is %s"_err_en_US,
            base->name(), why);
        return false;
      }
    }
  }
  if (isContiguous_) {
    if (auto contiguous{evaluate::IsContiguous(rhs, foldingContext_)}) {
      if (!*contiguous) {
        Say("CONTIGUOUS pointer may not be associated with a discontiguous target"_err_en_US);
        return false;
      }
    } else if (context_.ShouldWarn(
                   common::UsageWarning::PointerToPossibleNoncontiguous)) {
      Say("Target of CONTIGUOUS pointer association is not known to be contiguous"_warn_en_US);
    }
  }
  // Warn about undefinable data targets
  if (context_.ShouldWarn(common::UsageWarning::PointerToUndefinable)) {
    if (auto because{WhyNotDefinable(
            foldingContext_.messages().at(), scope_, {}, rhs)}) {
      if (auto *msg{
              Say("Pointer target is not a definable variable"_warn_en_US)}) {
        msg->Attach(
            std::move(because->set_severity(parser::Severity::Because)));
      }
      return false;
    }
  }
  return true;
}

bool PointerAssignmentChecker::Check(const evaluate::NullPointer &) {
  return true; // P => NULL() without MOLD=; always OK
}

template <typename T>
bool PointerAssignmentChecker::Check(const evaluate::FunctionRef<T> &f) {
  std::string funcName;
  const auto *symbol{f.proc().GetSymbol()};
  if (symbol) {
    funcName = symbol->name().ToString();
  } else if (const auto *intrinsic{f.proc().GetSpecificIntrinsic()}) {
    funcName = intrinsic->name;
  }
  auto proc{
      Procedure::Characterize(f.proc(), foldingContext_, /*emitError=*/true)};
  if (!proc) {
    return false;
  }
  std::optional<MessageFixedText> msg;
  const auto &funcResult{proc->functionResult}; // C1025
  if (!funcResult) {
    msg = "%s is associated with the non-existent result of reference to"
          " procedure"_err_en_US;
  } else if (CharacterizeProcedure()) {
    // Shouldn't be here in this function unless lhs is an object pointer.
    msg = "Procedure %s is associated with the result of a reference to"
          " function '%s' that does not return a procedure pointer"_err_en_US;
  } else if (funcResult->IsProcedurePointer()) {
    msg = "Object %s is associated with the result of a reference to"
          " function '%s' that is a procedure pointer"_err_en_US;
  } else if (!funcResult->attrs.test(FunctionResult::Attr::Pointer)) {
    msg = "%s is associated with the result of a reference to function '%s'"
          " that is a not a pointer"_err_en_US;
  } else if (isContiguous_ &&
      !funcResult->attrs.test(FunctionResult::Attr::Contiguous)) {
    if (context_.ShouldWarn(
            common::UsageWarning::PointerToPossibleNoncontiguous)) {
      msg =
          "CONTIGUOUS %s is associated with the result of reference to function '%s' that is not known to be contiguous"_warn_en_US;
    }
  } else if (lhsType_) {
    const auto *frTypeAndShape{funcResult->GetTypeAndShape()};
    CHECK(frTypeAndShape);
    if (!lhsType_->IsCompatibleWith(foldingContext_.messages(), *frTypeAndShape,
            "pointer", "function result",
            /*omitShapeConformanceCheck=*/isBoundsRemapping_ || isAssumedRank_,
            evaluate::CheckConformanceFlags::BothDeferredShape)) {
      return false; // IsCompatibleWith() emitted message
    }
  }
  if (msg) {
    auto restorer{common::ScopedSet(lhs_, symbol)};
    Say(*msg, description_, funcName);
    return false;
  }
  return true;
}

template <typename T>
bool PointerAssignmentChecker::Check(const evaluate::Designator<T> &d) {
  const Symbol *last{d.GetLastSymbol()};
  const Symbol *base{d.GetBaseObject().symbol()};
  if (!last || !base) {
    // P => "character literal"(1:3)
    Say("Pointer target is not a named entity"_err_en_US);
    return false;
  }
  std::optional<std::variant<MessageFixedText, MessageFormattedText>> msg;
  if (CharacterizeProcedure()) {
    // Shouldn't be here in this function unless lhs is an object pointer.
    msg = "In assignment to procedure %s, the target is not a procedure or"
          " procedure pointer"_err_en_US;
  } else if (!evaluate::GetLastTarget(GetSymbolVector(d))) { // C1025
    msg = "In assignment to object %s, the target '%s' is not an object with"
          " POINTER or TARGET attributes"_err_en_US;
  } else if (auto rhsType{TypeAndShape::Characterize(d, foldingContext_)}) {
    if (!lhsType_) {
      msg = "%s associated with object '%s' with incompatible type or"
            " shape"_err_en_US;
    } else if (rhsType->corank() > 0 &&
        (isVolatile_ != last->attrs().test(Attr::VOLATILE))) { // C1020
      // TODO: what if A is VOLATILE in A%B%C?  need a better test here
      if (isVolatile_) {
        msg = "Pointer may not be VOLATILE when target is a"
              " non-VOLATILE coarray"_err_en_US;
      } else {
        msg = "Pointer must be VOLATILE when target is a"
              " VOLATILE coarray"_err_en_US;
      }
    } else if (rhsType->type().IsUnlimitedPolymorphic()) {
      if (!LhsOkForUnlimitedPoly()) {
        msg = "Pointer type must be unlimited polymorphic or non-extensible"
              " derived type when target is unlimited polymorphic"_err_en_US;
      }
    } else {
      if (!lhsType_->type().IsTkLenCompatibleWith(rhsType->type())) {
        msg = MessageFormattedText{
            "Target type %s is not compatible with pointer type %s"_err_en_US,
            rhsType->type().AsFortran(), lhsType_->type().AsFortran()};

      } else if (!isBoundsRemapping_ &&
          !lhsType_->attrs().test(TypeAndShape::Attr::AssumedRank)) {
        int lhsRank{lhsType_->Rank()};
        int rhsRank{rhsType->Rank()};
        if (lhsRank != rhsRank) {
          msg = MessageFormattedText{
              "Pointer has rank %d but target has rank %d"_err_en_US, lhsRank,
              rhsRank};
        }
      }
    }
  }
  if (msg) {
    auto restorer{common::ScopedSet(lhs_, last)};
    if (auto *m{std::get_if<MessageFixedText>(&*msg)}) {
      std::string buf;
      llvm::raw_string_ostream ss{buf};
      d.AsFortran(ss);
      Say(*m, description_, buf);
    } else {
      Say(std::get<MessageFormattedText>(*msg));
    }
    return false;
  } else {
    context_.NoteDefinedSymbol(*base);
    return true;
  }
}

// Common handling for procedure pointer right-hand sides
bool PointerAssignmentChecker::Check(parser::CharBlock rhsName, bool isCall,
    const Procedure *rhsProcedure,
    const evaluate::SpecificIntrinsic *specific) {
  std::string whyNot;
  std::optional<std::string> warning;
  CharacterizeProcedure();
  if (std::optional<MessageFixedText> msg{evaluate::CheckProcCompatibility(
          isCall, procedure_, rhsProcedure, specific, whyNot, warning,
          /*ignoreImplicitVsExplicit=*/isCall)}) {
    Say(std::move(*msg), description_, rhsName, whyNot);
    return false;
  }
  if (context_.ShouldWarn(common::UsageWarning::ProcDummyArgShapes) &&
      warning) {
    Say("%s and %s may not be completely compatible procedures: %s"_warn_en_US,
        description_, rhsName, std::move(*warning));
  }
  return true;
}

bool PointerAssignmentChecker::Check(const evaluate::ProcedureDesignator &d) {
  const Symbol *symbol{d.GetSymbol()};
  if (symbol) {
    if (const auto *subp{
            symbol->GetUltimate().detailsIf<SubprogramDetails>()}) {
      if (subp->stmtFunction()) {
        evaluate::SayWithDeclaration(foldingContext_.messages(), *symbol,
            "Statement function '%s' may not be the target of a pointer assignment"_err_en_US,
            symbol->name());
        return false;
      }
    } else if (symbol->has<ProcBindingDetails>() &&
        context_.ShouldWarn(common::LanguageFeature::BindingAsProcedure)) {
      evaluate::SayWithDeclaration(foldingContext_.messages(), *symbol,
          "Procedure binding '%s' used as target of a pointer assignment"_port_en_US,
          symbol->name());
    }
  }
  if (auto chars{
          Procedure::Characterize(d, foldingContext_, /*emitError=*/true)}) {
    // Disregard the elemental attribute of RHS intrinsics.
    if (symbol && symbol->GetUltimate().attrs().test(Attr::INTRINSIC)) {
      chars->attrs.reset(Procedure::Attr::Elemental);
    }
    return Check(d.GetName(), false, &*chars, d.GetSpecificIntrinsic());
  } else {
    return Check(d.GetName(), false);
  }
}

bool PointerAssignmentChecker::Check(const evaluate::ProcedureRef &ref) {
  auto chars{Procedure::Characterize(ref, foldingContext_)};
  return Check(ref.proc().GetName(), true, common::GetPtrFromOptional(chars));
}

// The target can be unlimited polymorphic if the pointer is, or if it is
// a non-extensible derived type.
bool PointerAssignmentChecker::LhsOkForUnlimitedPoly() const {
  const auto &type{lhsType_->type()};
  if (type.category() != TypeCategory::Derived || type.IsAssumedType()) {
    return false;
  } else if (type.IsUnlimitedPolymorphic()) {
    return true;
  } else {
    return !IsExtensibleType(&type.GetDerivedTypeSpec());
  }
}

template <typename... A>
parser::Message *PointerAssignmentChecker::Say(A &&...x) {
  auto *msg{foldingContext_.messages().Say(std::forward<A>(x)...)};
  if (msg) {
    if (lhs_) {
      return evaluate::AttachDeclaration(msg, *lhs_);
    }
    if (!source_.empty()) {
      msg->Attach(source_, "Declaration of %s"_en_US, description_);
    }
  }
  return msg;
}

// Verify that any bounds on the LHS of a pointer assignment are valid.
// Return true if it is a bound-remapping so we can perform further checks.
static bool CheckPointerBounds(
    evaluate::FoldingContext &context, const evaluate::Assignment &assignment) {
  auto &messages{context.messages()};
  const SomeExpr &lhs{assignment.lhs};
  const SomeExpr &rhs{assignment.rhs};
  bool isBoundsRemapping{false};
  std::size_t numBounds{common::visit(
      common::visitors{
          [&](const evaluate::Assignment::BoundsSpec &bounds) {
            return bounds.size();
          },
          [&](const evaluate::Assignment::BoundsRemapping &bounds) {
            isBoundsRemapping = true;
            evaluate::ExtentExpr lhsSizeExpr{1};
            for (const auto &bound : bounds) {
              lhsSizeExpr = std::move(lhsSizeExpr) *
                  (common::Clone(bound.second) - common::Clone(bound.first) +
                      evaluate::ExtentExpr{1});
            }
            if (std::optional<std::int64_t> lhsSize{evaluate::ToInt64(
                    evaluate::Fold(context, std::move(lhsSizeExpr)))}) {
              if (auto shape{evaluate::GetShape(context, rhs)}) {
                if (std::optional<std::int64_t> rhsSize{
                        evaluate::ToInt64(evaluate::Fold(
                            context, evaluate::GetSize(std::move(*shape))))}) {
                  if (*lhsSize > *rhsSize) {
                    messages.Say(
                        "Pointer bounds require %d elements but target has"
                        " only %d"_err_en_US,
                        *lhsSize, *rhsSize); // 10.2.2.3(9)
                  }
                }
              }
            }
            return bounds.size();
          },
          [](const auto &) -> std::size_t {
            DIE("not valid for pointer assignment");
          },
      },
      assignment.u)};
  if (numBounds > 0) {
    if (lhs.Rank() != static_cast<int>(numBounds)) {
      messages.Say("Pointer '%s' has rank %d but the number of bounds specified"
                   " is %d"_err_en_US,
          lhs.AsFortran(), lhs.Rank(), numBounds); // C1018
    }
  }
  if (isBoundsRemapping && rhs.Rank() != 1 &&
      !evaluate::IsSimplyContiguous(rhs, context)) {
    messages.Say("Pointer bounds remapping target must have rank 1 or be"
                 " simply contiguous"_err_en_US); // 10.2.2.3(9)
  }
  return isBoundsRemapping;
}

bool CheckPointerAssignment(SemanticsContext &context,
    const evaluate::Assignment &assignment, const Scope &scope) {
  return CheckPointerAssignment(context, assignment.lhs, assignment.rhs, scope,
      CheckPointerBounds(context.foldingContext(), assignment),
      /*isAssumedRank=*/false);
}

bool CheckPointerAssignment(SemanticsContext &context, const SomeExpr &lhs,
    const SomeExpr &rhs, const Scope &scope, bool isBoundsRemapping,
    bool isAssumedRank) {
  const Symbol *pointer{GetLastSymbol(lhs)};
  if (!pointer) {
    return false; // error was reported
  }
  PointerAssignmentChecker checker{context, scope, *pointer};
  checker.set_isBoundsRemapping(isBoundsRemapping);
  checker.set_isAssumedRank(isAssumedRank);
  bool lhsOk{checker.CheckLeftHandSide(lhs)};
  bool rhsOk{checker.Check(rhs)};
  return lhsOk && rhsOk; // don't short-circuit
}

bool CheckStructConstructorPointerComponent(SemanticsContext &context,
    const Symbol &lhs, const SomeExpr &rhs, const Scope &scope) {
  return PointerAssignmentChecker{context, scope, lhs}
      .set_pointerComponentLHS(&lhs)
      .Check(rhs);
}

bool CheckPointerAssignment(SemanticsContext &context, parser::CharBlock source,
    const std::string &description, const DummyDataObject &lhs,
    const SomeExpr &rhs, const Scope &scope, bool isAssumedRank) {
  return PointerAssignmentChecker{context, scope, source, description}
      .set_lhsType(common::Clone(lhs.type))
      .set_isContiguous(lhs.attrs.test(DummyDataObject::Attr::Contiguous))
      .set_isVolatile(lhs.attrs.test(DummyDataObject::Attr::Volatile))
      .set_isAssumedRank(isAssumedRank)
      .Check(rhs);
}

bool CheckInitialDataPointerTarget(SemanticsContext &context,
    const SomeExpr &pointer, const SomeExpr &init, const Scope &scope) {
  return evaluate::IsInitialDataTarget(
             init, &context.foldingContext().messages()) &&
      CheckPointerAssignment(context, pointer, init, scope,
          /*isBoundsRemapping=*/false,
          /*isAssumedRank=*/false);
}

} // namespace Fortran::semantics
