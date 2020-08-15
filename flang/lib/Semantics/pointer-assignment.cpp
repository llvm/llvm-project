//===-- lib/Semantics/pointer-assignment.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pointer-assignment.h"
#include "flang/Common/idioms.h"
#include "flang/Common/restorer.h"
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
  PointerAssignmentChecker(evaluate::FoldingContext &context,
      parser::CharBlock source, const std::string &description)
      : context_{context}, source_{source}, description_{description} {}
  PointerAssignmentChecker(evaluate::FoldingContext &context, const Symbol &lhs)
      : context_{context}, source_{lhs.name()},
        description_{"pointer '"s + lhs.name().ToString() + '\''}, lhs_{&lhs},
        procedure_{Procedure::Characterize(lhs, context.intrinsics())} {
    set_lhsType(TypeAndShape::Characterize(lhs, context));
    set_isContiguous(lhs.attrs().test(Attr::CONTIGUOUS));
    set_isVolatile(lhs.attrs().test(Attr::VOLATILE));
  }
  PointerAssignmentChecker &set_lhsType(std::optional<TypeAndShape> &&);
  PointerAssignmentChecker &set_isContiguous(bool);
  PointerAssignmentChecker &set_isVolatile(bool);
  PointerAssignmentChecker &set_isBoundsRemapping(bool);
  bool Check(const SomeExpr &);

private:
  template <typename T> bool Check(const T &);
  template <typename T> bool Check(const evaluate::Expr<T> &);
  template <typename T> bool Check(const evaluate::FunctionRef<T> &);
  template <typename T> bool Check(const evaluate::Designator<T> &);
  bool Check(const evaluate::NullPointer &);
  bool Check(const evaluate::ProcedureDesignator &);
  bool Check(const evaluate::ProcedureRef &);
  // Target is a procedure
  bool Check(
      parser::CharBlock rhsName, bool isCall, const Procedure * = nullptr);
  bool LhsOkForUnlimitedPoly() const;
  template <typename... A> parser::Message *Say(A &&...);

  evaluate::FoldingContext &context_;
  const parser::CharBlock source_;
  const std::string description_;
  const Symbol *lhs_{nullptr};
  std::optional<TypeAndShape> lhsType_;
  std::optional<Procedure> procedure_;
  bool isContiguous_{false};
  bool isVolatile_{false};
  bool isBoundsRemapping_{false};
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

template <typename T> bool PointerAssignmentChecker::Check(const T &) {
  // Catch-all case for really bad target expression
  Say("Target associated with %s must be a designator or a call to a"
      " pointer-valued function"_err_en_US,
      description_);
  return false;
}

template <typename T>
bool PointerAssignmentChecker::Check(const evaluate::Expr<T> &x) {
  return std::visit([&](const auto &x) { return Check(x); }, x.u);
}

bool PointerAssignmentChecker::Check(const SomeExpr &rhs) {
  if (HasVectorSubscript(rhs)) { // C1025
    Say("An array section with a vector subscript may not be a pointer target"_err_en_US);
    return false;
  } else if (ExtractCoarrayRef(rhs)) { // C1026
    Say("A coindexed object may not be a pointer target"_err_en_US);
    return false;
  } else {
    return std::visit([&](const auto &x) { return Check(x); }, rhs.u);
  }
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
  auto proc{Procedure::Characterize(f.proc(), context_.intrinsics())};
  if (!proc) {
    return false;
  }
  std::optional<MessageFixedText> msg;
  const auto &funcResult{proc->functionResult}; // C1025
  if (!funcResult) {
    msg = "%s is associated with the non-existent result of reference to"
          " procedure"_err_en_US;
  } else if (procedure_) {
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
    msg = "CONTIGUOUS %s is associated with the result of reference to"
          " function '%s' that is not contiguous"_err_en_US;
  } else if (lhsType_) {
    const auto *frTypeAndShape{funcResult->GetTypeAndShape()};
    CHECK(frTypeAndShape);
    if (!lhsType_->IsCompatibleWith(context_.messages(), *frTypeAndShape)) {
      msg = "%s is associated with the result of a reference to function '%s'"
            " whose pointer result has an incompatible type or shape"_err_en_US;
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
    context_.messages().Say("Pointer target is not a named entity"_err_en_US);
    return false;
  }
  std::optional<std::variant<MessageFixedText, MessageFormattedText>> msg;
  if (procedure_) {
    // Shouldn't be here in this function unless lhs is an object pointer.
    msg = "In assignment to procedure %s, the target is not a procedure or"
          " procedure pointer"_err_en_US;
  } else if (!evaluate::GetLastTarget(GetSymbolVector(d))) { // C1025
    msg = "In assignment to object %s, the target '%s' is not an object with"
          " POINTER or TARGET attributes"_err_en_US;
  } else if (auto rhsType{TypeAndShape::Characterize(d, context_)}) {
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
      if (!lhsType_->type().IsTypeCompatibleWith(rhsType->type())) {
        msg = MessageFormattedText{
            "Target type %s is not compatible with pointer type %s"_err_en_US,
            rhsType->type().AsFortran(), lhsType_->type().AsFortran()};

      } else if (!isBoundsRemapping_) {
        std::size_t lhsRank{lhsType_->shape().size()};
        std::size_t rhsRank{rhsType->shape().size()};
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
      Say(*m, description_, ss.str());
    } else {
      Say(std::get<MessageFormattedText>(*msg));
    }
    return false;
  }
  return true;
}

// Compare procedure characteristics for equality except that lhs may be
// Pure or Elemental when rhs is not.
static bool CharacteristicsMatch(const Procedure &lhs, const Procedure &rhs) {
  using Attr = Procedure::Attr;
  auto lhsAttrs{rhs.attrs};
  lhsAttrs.set(
      Attr::Pure, lhs.attrs.test(Attr::Pure) | rhs.attrs.test(Attr::Pure));
  lhsAttrs.set(Attr::Elemental,
      lhs.attrs.test(Attr::Elemental) | rhs.attrs.test(Attr::Elemental));
  return lhsAttrs == rhs.attrs && lhs.functionResult == rhs.functionResult &&
      lhs.dummyArguments == rhs.dummyArguments;
}

// Common handling for procedure pointer right-hand sides
bool PointerAssignmentChecker::Check(
    parser::CharBlock rhsName, bool isCall, const Procedure *rhsProcedure) {
  std::optional<MessageFixedText> msg;
  if (!procedure_) {
    msg = "In assignment to object %s, the target '%s' is a procedure"
          " designator"_err_en_US;
  } else if (!rhsProcedure) {
    msg = "In assignment to procedure %s, the characteristics of the target"
          " procedure '%s' could not be determined"_err_en_US;
  } else if (CharacteristicsMatch(*procedure_, *rhsProcedure)) {
    // OK
  } else if (isCall) {
    msg = "Procedure %s associated with result of reference to function '%s'"
          " that is an incompatible procedure pointer"_err_en_US;
  } else if (procedure_->IsPure() && !rhsProcedure->IsPure()) {
    msg = "PURE procedure %s may not be associated with non-PURE"
          " procedure designator '%s'"_err_en_US;
  } else if (procedure_->IsElemental() && !rhsProcedure->IsElemental()) {
    msg = "ELEMENTAL procedure %s may not be associated with non-ELEMENTAL"
          " procedure designator '%s'"_err_en_US;
  } else if (procedure_->IsFunction() && !rhsProcedure->IsFunction()) {
    msg = "Function %s may not be associated with subroutine"
          " designator '%s'"_err_en_US;
  } else if (!procedure_->IsFunction() && rhsProcedure->IsFunction()) {
    msg = "Subroutine %s may not be associated with function"
          " designator '%s'"_err_en_US;
  } else if (procedure_->HasExplicitInterface() &&
      !rhsProcedure->HasExplicitInterface()) {
    msg = "Procedure %s with explicit interface may not be associated with"
          " procedure designator '%s' with implicit interface"_err_en_US;
  } else if (!procedure_->HasExplicitInterface() &&
      rhsProcedure->HasExplicitInterface()) {
    msg = "Procedure %s with implicit interface may not be associated with"
          " procedure designator '%s' with explicit interface"_err_en_US;
  } else {
    msg = "Procedure %s associated with incompatible procedure"
          " designator '%s'"_err_en_US;
  }
  if (msg) {
    Say(std::move(*msg), description_, rhsName);
    return false;
  }
  return true;
}

bool PointerAssignmentChecker::Check(const evaluate::ProcedureDesignator &d) {
  if (auto chars{Procedure::Characterize(d, context_.intrinsics())}) {
    return Check(d.GetName(), false, &*chars);
  } else {
    return Check(d.GetName(), false);
  }
}

bool PointerAssignmentChecker::Check(const evaluate::ProcedureRef &ref) {
  const Procedure *procedure{nullptr};
  auto chars{Procedure::Characterize(ref, context_.intrinsics())};
  if (chars) {
    procedure = &*chars;
    if (chars->functionResult) {
      if (const auto *proc{chars->functionResult->IsProcedurePointer()}) {
        procedure = proc;
      }
    }
  }
  return Check(ref.proc().GetName(), true, procedure);
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
  auto *msg{context_.messages().Say(std::forward<A>(x)...)};
  if (lhs_) {
    return evaluate::AttachDeclaration(msg, *lhs_);
  } else if (!source_.empty()) {
    msg->Attach(source_, "Declaration of %s"_en_US, description_);
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
  std::size_t numBounds{std::visit(
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
      !evaluate::IsSimplyContiguous(rhs, context.intrinsics())) {
    messages.Say("Pointer bounds remapping target must have rank 1 or be"
                 " simply contiguous"_err_en_US); // 10.2.2.3(9)
  }
  return isBoundsRemapping;
}

bool CheckPointerAssignment(
    evaluate::FoldingContext &context, const evaluate::Assignment &assignment) {
  return CheckPointerAssignment(context, assignment.lhs, assignment.rhs,
      CheckPointerBounds(context, assignment));
}

bool CheckPointerAssignment(evaluate::FoldingContext &context,
    const SomeExpr &lhs, const SomeExpr &rhs, bool isBoundsRemapping) {
  const Symbol *pointer{GetLastSymbol(lhs)};
  if (!pointer) {
    return false; // error was reported
  }
  if (!IsPointer(*pointer)) {
    evaluate::SayWithDeclaration(context.messages(), *pointer,
        "'%s' is not a pointer"_err_en_US, pointer->name());
    return false;
  }
  if (pointer->has<ProcEntityDetails>() && evaluate::ExtractCoarrayRef(lhs)) {
    context.messages().Say( // C1027
        "Procedure pointer may not be a coindexed object"_err_en_US);
    return false;
  }
  return PointerAssignmentChecker{context, *pointer}
      .set_isBoundsRemapping(isBoundsRemapping)
      .Check(rhs);
}

bool CheckPointerAssignment(
    evaluate::FoldingContext &context, const Symbol &lhs, const SomeExpr &rhs) {
  CHECK(IsPointer(lhs));
  return PointerAssignmentChecker{context, lhs}.Check(rhs);
}

bool CheckPointerAssignment(evaluate::FoldingContext &context,
    parser::CharBlock source, const std::string &description,
    const DummyDataObject &lhs, const SomeExpr &rhs) {
  return PointerAssignmentChecker{context, source, description}
      .set_lhsType(common::Clone(lhs.type))
      .set_isContiguous(lhs.attrs.test(DummyDataObject::Attr::Contiguous))
      .set_isVolatile(lhs.attrs.test(DummyDataObject::Attr::Volatile))
      .Check(rhs);
}

bool CheckInitialTarget(evaluate::FoldingContext &context,
    const SomeExpr &pointer, const SomeExpr &init) {
  return evaluate::IsInitialDataTarget(init, &context.messages()) &&
      CheckPointerAssignment(context, pointer, init);
}

} // namespace Fortran::semantics
