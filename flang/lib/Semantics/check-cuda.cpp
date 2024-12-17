//===-- lib/Semantics/check-cuda.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-cuda.h"
#include "flang/Common/template.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"

// Once labeled DO constructs have been canonicalized and their parse subtrees
// transformed into parser::DoConstructs, scan the parser::Blocks of the program
// and merge adjacent CUFKernelDoConstructs and DoConstructs whenever the
// CUFKernelDoConstruct doesn't already have an embedded DoConstruct.  Also
// emit errors about improper or missing DoConstructs.

namespace Fortran::parser {
struct Mutator {
  template <typename A> bool Pre(A &) { return true; }
  template <typename A> void Post(A &) {}
  bool Pre(Block &);
};

bool Mutator::Pre(Block &block) {
  for (auto iter{block.begin()}; iter != block.end(); ++iter) {
    if (auto *kernel{Unwrap<CUFKernelDoConstruct>(*iter)}) {
      auto &nested{std::get<std::optional<DoConstruct>>(kernel->t)};
      if (!nested) {
        if (auto next{iter}; ++next != block.end()) {
          if (auto *doConstruct{Unwrap<DoConstruct>(*next)}) {
            nested = std::move(*doConstruct);
            block.erase(next);
          }
        }
      }
    } else {
      Walk(*iter, *this);
    }
  }
  return false;
}
} // namespace Fortran::parser

namespace Fortran::semantics {

bool CanonicalizeCUDA(parser::Program &program) {
  parser::Mutator mutator;
  parser::Walk(program, mutator);
  return true;
}

using MaybeMsg = std::optional<parser::MessageFormattedText>;

// Traverses an evaluate::Expr<> in search of unsupported operations
// on the device.

struct DeviceExprChecker
    : public evaluate::AnyTraverse<DeviceExprChecker, MaybeMsg> {
  using Result = MaybeMsg;
  using Base = evaluate::AnyTraverse<DeviceExprChecker, Result>;
  DeviceExprChecker() : Base(*this) {}
  using Base::operator();
  Result operator()(const evaluate::ProcedureDesignator &x) const {
    if (const Symbol * sym{x.GetInterfaceSymbol()}) {
      const auto *subp{
          sym->GetUltimate().detailsIf<semantics::SubprogramDetails>()};
      if (subp) {
        if (auto attrs{subp->cudaSubprogramAttrs()}) {
          if (*attrs == common::CUDASubprogramAttrs::HostDevice ||
              *attrs == common::CUDASubprogramAttrs::Device) {
            return {};
          }
        }
      }
    } else if (x.GetSpecificIntrinsic()) {
      // TODO(CUDA): Check for unsupported intrinsics here
      return {};
    }
    return parser::MessageFormattedText(
        "'%s' may not be called in device code"_err_en_US, x.GetName());
  }
};

struct FindHostArray
    : public evaluate::AnyTraverse<FindHostArray, const Symbol *> {
  using Result = const Symbol *;
  using Base = evaluate::AnyTraverse<FindHostArray, Result>;
  FindHostArray() : Base(*this) {}
  using Base::operator();
  Result operator()(const evaluate::Component &x) const {
    const Symbol &symbol{x.GetLastSymbol()};
    if (IsAllocatableOrPointer(symbol)) {
      if (Result hostArray{(*this)(symbol)}) {
        return hostArray;
      }
    }
    return (*this)(x.base());
  }
  Result operator()(const Symbol &symbol) const {
    if (const auto *details{
            symbol.GetUltimate().detailsIf<semantics::ObjectEntityDetails>()}) {
      if (details->IsArray() &&
          !symbol.attrs().test(Fortran::semantics::Attr::PARAMETER) &&
          (!details->cudaDataAttr() ||
              (details->cudaDataAttr() &&
                  *details->cudaDataAttr() != common::CUDADataAttr::Device &&
                  *details->cudaDataAttr() != common::CUDADataAttr::Managed &&
                  *details->cudaDataAttr() != common::CUDADataAttr::Unified))) {
        return &symbol;
      }
    }
    return nullptr;
  }
};

template <typename A> static MaybeMsg CheckUnwrappedExpr(const A &x) {
  if (const auto *expr{parser::Unwrap<parser::Expr>(x)}) {
    return DeviceExprChecker{}(expr->typedExpr);
  }
  return {};
}

template <typename A>
static void CheckUnwrappedExpr(
    SemanticsContext &context, SourceName at, const A &x) {
  if (const auto *expr{parser::Unwrap<parser::Expr>(x)}) {
    if (auto msg{DeviceExprChecker{}(expr->typedExpr)}) {
      context.Say(at, std::move(*msg));
    }
  }
}

template <bool CUF_KERNEL> struct ActionStmtChecker {
  template <typename A> static MaybeMsg WhyNotOk(const A &x) {
    if constexpr (ConstraintTrait<A>) {
      return WhyNotOk(x.thing);
    } else if constexpr (WrapperTrait<A>) {
      return WhyNotOk(x.v);
    } else if constexpr (UnionTrait<A>) {
      return WhyNotOk(x.u);
    } else if constexpr (TupleTrait<A>) {
      return WhyNotOk(x.t);
    } else {
      return parser::MessageFormattedText{
          "Statement may not appear in device code"_err_en_US};
    }
  }
  template <typename A>
  static MaybeMsg WhyNotOk(const common::Indirection<A> &x) {
    return WhyNotOk(x.value());
  }
  template <typename... As>
  static MaybeMsg WhyNotOk(const std::variant<As...> &x) {
    return common::visit([](const auto &x) { return WhyNotOk(x); }, x);
  }
  template <std::size_t J = 0, typename... As>
  static MaybeMsg WhyNotOk(const std::tuple<As...> &x) {
    if constexpr (J == sizeof...(As)) {
      return {};
    } else if (auto msg{WhyNotOk(std::get<J>(x))}) {
      return msg;
    } else {
      return WhyNotOk<(J + 1)>(x);
    }
  }
  template <typename A> static MaybeMsg WhyNotOk(const std::list<A> &x) {
    for (const auto &y : x) {
      if (MaybeMsg result{WhyNotOk(y)}) {
        return result;
      }
    }
    return {};
  }
  template <typename A> static MaybeMsg WhyNotOk(const std::optional<A> &x) {
    if (x) {
      return WhyNotOk(*x);
    } else {
      return {};
    }
  }
  template <typename A>
  static MaybeMsg WhyNotOk(const parser::UnlabeledStatement<A> &x) {
    return WhyNotOk(x.statement);
  }
  template <typename A>
  static MaybeMsg WhyNotOk(const parser::Statement<A> &x) {
    return WhyNotOk(x.statement);
  }
  static MaybeMsg WhyNotOk(const parser::AllocateStmt &) {
    return {}; // AllocateObjects are checked elsewhere
  }
  static MaybeMsg WhyNotOk(const parser::AllocateCoarraySpec &) {
    return parser::MessageFormattedText(
        "A coarray may not be allocated on the device"_err_en_US);
  }
  static MaybeMsg WhyNotOk(const parser::DeallocateStmt &) {
    return {}; // AllocateObjects are checked elsewhere
  }
  static MaybeMsg WhyNotOk(const parser::AssignmentStmt &x) {
    return DeviceExprChecker{}(x.typedAssignment);
  }
  static MaybeMsg WhyNotOk(const parser::CallStmt &x) {
    return DeviceExprChecker{}(x.typedCall);
  }
  static MaybeMsg WhyNotOk(const parser::ContinueStmt &) { return {}; }
  static MaybeMsg WhyNotOk(const parser::IfStmt &x) {
    if (auto result{
            CheckUnwrappedExpr(std::get<parser::ScalarLogicalExpr>(x.t))}) {
      return result;
    }
    return WhyNotOk(
        std::get<parser::UnlabeledStatement<parser::ActionStmt>>(x.t)
            .statement);
  }
  static MaybeMsg WhyNotOk(const parser::NullifyStmt &x) {
    for (const auto &y : x.v) {
      if (MaybeMsg result{DeviceExprChecker{}(y.typedExpr)}) {
        return result;
      }
    }
    return {};
  }
  static MaybeMsg WhyNotOk(const parser::PointerAssignmentStmt &x) {
    return DeviceExprChecker{}(x.typedAssignment);
  }
};

template <bool IsCUFKernelDo> class DeviceContextChecker {
public:
  explicit DeviceContextChecker(SemanticsContext &c) : context_{c} {}
  void CheckSubprogram(const parser::Name &name, const parser::Block &body) {
    if (name.symbol) {
      const auto *subp{
          name.symbol->GetUltimate().detailsIf<SubprogramDetails>()};
      if (subp && subp->moduleInterface()) {
        subp = subp->moduleInterface()
                   ->GetUltimate()
                   .detailsIf<SubprogramDetails>();
      }
      if (subp &&
          subp->cudaSubprogramAttrs().value_or(
              common::CUDASubprogramAttrs::Host) !=
              common::CUDASubprogramAttrs::Host) {
        Check(body);
      }
    }
  }
  void Check(const parser::Block &block) {
    for (const auto &epc : block) {
      Check(epc);
    }
  }

private:
  void Check(const parser::ExecutionPartConstruct &epc) {
    common::visit(
        common::visitors{
            [&](const parser::ExecutableConstruct &x) { Check(x); },
            [&](const parser::Statement<common::Indirection<parser::EntryStmt>>
                    &x) {
              context_.Say(x.source,
                  "Device code may not contain an ENTRY statement"_err_en_US);
            },
            [](const parser::Statement<common::Indirection<parser::FormatStmt>>
                    &) {},
            [](const parser::Statement<common::Indirection<parser::DataStmt>>
                    &) {},
            [](const parser::Statement<
                common::Indirection<parser::NamelistStmt>> &) {},
            [](const parser::ErrorRecovery &) {},
        },
        epc.u);
  }
  void Check(const parser::ExecutableConstruct &ec) {
    common::visit(
        common::visitors{
            [&](const parser::Statement<parser::ActionStmt> &stmt) {
              Check(stmt.statement, stmt.source);
            },
            [&](const common::Indirection<parser::DoConstruct> &x) {
              if (const std::optional<parser::LoopControl> &control{
                      x.value().GetLoopControl()}) {
                common::visit([&](const auto &y) { Check(y); }, control->u);
              }
              Check(std::get<parser::Block>(x.value().t));
            },
            [&](const common::Indirection<parser::BlockConstruct> &x) {
              Check(std::get<parser::Block>(x.value().t));
            },
            [&](const common::Indirection<parser::IfConstruct> &x) {
              Check(x.value());
            },
            [&](const auto &x) {
              if (auto source{parser::GetSource(x)}) {
                context_.Say(*source,
                    "Statement may not appear in device code"_err_en_US);
              }
            },
        },
        ec.u);
  }
  template <typename SEEK, typename A>
  static const SEEK *GetIOControl(const A &stmt) {
    for (const auto &spec : stmt.controls) {
      if (const auto *result{std::get_if<SEEK>(&spec.u)}) {
        return result;
      }
    }
    return nullptr;
  }
  template <typename A> static bool IsInternalIO(const A &stmt) {
    if (stmt.iounit.has_value()) {
      return std::holds_alternative<Fortran::parser::Variable>(stmt.iounit->u);
    }
    if (auto *unit{GetIOControl<Fortran::parser::IoUnit>(stmt)}) {
      return std::holds_alternative<Fortran::parser::Variable>(unit->u);
    }
    return false;
  }
  void WarnOnIoStmt(const parser::CharBlock &source) {
    context_.Warn(common::UsageWarning::CUDAUsage, source,
        "I/O statement might not be supported on device"_warn_en_US);
  }
  template <typename A>
  void WarnIfNotInternal(const A &stmt, const parser::CharBlock &source) {
    if (!IsInternalIO(stmt)) {
      WarnOnIoStmt(source);
    }
  }
  template <typename A>
  void ErrorIfHostSymbol(const A &expr, parser::CharBlock source) {
    if (const Symbol * hostArray{FindHostArray{}(expr)}) {
      context_.Say(source,
          "Host array '%s' cannot be present in device context"_err_en_US,
          hostArray->name());
    }
  }
  void Check(const parser::ActionStmt &stmt, const parser::CharBlock &source) {
    common::visit(
        common::visitors{
            [&](const common::Indirection<parser::PrintStmt> &) {},
            [&](const common::Indirection<parser::WriteStmt> &x) {
              if (x.value().format) { // Formatted write to '*' or '6'
                if (std::holds_alternative<Fortran::parser::Star>(
                        x.value().format->u)) {
                  if (x.value().iounit) {
                    if (std::holds_alternative<Fortran::parser::Star>(
                            x.value().iounit->u)) {
                      return;
                    }
                  }
                }
              }
              WarnIfNotInternal(x.value(), source);
            },
            [&](const common::Indirection<parser::CloseStmt> &x) {
              WarnOnIoStmt(source);
            },
            [&](const common::Indirection<parser::EndfileStmt> &x) {
              WarnOnIoStmt(source);
            },
            [&](const common::Indirection<parser::OpenStmt> &x) {
              WarnOnIoStmt(source);
            },
            [&](const common::Indirection<parser::ReadStmt> &x) {
              WarnIfNotInternal(x.value(), source);
            },
            [&](const common::Indirection<parser::InquireStmt> &x) {
              WarnOnIoStmt(source);
            },
            [&](const common::Indirection<parser::RewindStmt> &x) {
              WarnOnIoStmt(source);
            },
            [&](const common::Indirection<parser::BackspaceStmt> &x) {
              WarnOnIoStmt(source);
            },
            [&](const common::Indirection<parser::IfStmt> &x) {
              Check(x.value());
            },
            [&](const common::Indirection<parser::AssignmentStmt> &x) {
              if (const evaluate::Assignment *
                  assign{semantics::GetAssignment(x.value())}) {
                ErrorIfHostSymbol(assign->lhs, source);
                ErrorIfHostSymbol(assign->rhs, source);
              }
              if (auto msg{ActionStmtChecker<IsCUFKernelDo>::WhyNotOk(x)}) {
                context_.Say(source, std::move(*msg));
              }
            },
            [&](const auto &x) {
              if (auto msg{ActionStmtChecker<IsCUFKernelDo>::WhyNotOk(x)}) {
                context_.Say(source, std::move(*msg));
              }
            },
        },
        stmt.u);
  }
  void Check(const parser::IfConstruct &ic) {
    const auto &ifS{std::get<parser::Statement<parser::IfThenStmt>>(ic.t)};
    CheckUnwrappedExpr(context_, ifS.source,
        std::get<parser::ScalarLogicalExpr>(ifS.statement.t));
    Check(std::get<parser::Block>(ic.t));
    for (const auto &eib :
        std::get<std::list<parser::IfConstruct::ElseIfBlock>>(ic.t)) {
      const auto &eIfS{std::get<parser::Statement<parser::ElseIfStmt>>(eib.t)};
      CheckUnwrappedExpr(context_, eIfS.source,
          std::get<parser::ScalarLogicalExpr>(eIfS.statement.t));
      Check(std::get<parser::Block>(eib.t));
    }
    if (const auto &eb{
            std::get<std::optional<parser::IfConstruct::ElseBlock>>(ic.t)}) {
      Check(std::get<parser::Block>(eb->t));
    }
  }
  void Check(const parser::IfStmt &is) {
    const auto &uS{
        std::get<parser::UnlabeledStatement<parser::ActionStmt>>(is.t)};
    CheckUnwrappedExpr(
        context_, uS.source, std::get<parser::ScalarLogicalExpr>(is.t));
    Check(uS.statement, uS.source);
  }
  void Check(const parser::LoopControl::Bounds &bounds) {
    Check(bounds.lower);
    Check(bounds.upper);
    if (bounds.step) {
      Check(*bounds.step);
    }
  }
  void Check(const parser::LoopControl::Concurrent &x) {
    const auto &header{std::get<parser::ConcurrentHeader>(x.t)};
    for (const auto &cc :
        std::get<std::list<parser::ConcurrentControl>>(header.t)) {
      Check(std::get<1>(cc.t));
      Check(std::get<2>(cc.t));
      if (const auto &step{
              std::get<std::optional<parser::ScalarIntExpr>>(cc.t)}) {
        Check(*step);
      }
    }
    if (const auto &mask{
            std::get<std::optional<parser::ScalarLogicalExpr>>(header.t)}) {
      Check(*mask);
    }
  }
  void Check(const parser::ScalarLogicalExpr &x) {
    Check(DEREF(parser::Unwrap<parser::Expr>(x)));
  }
  void Check(const parser::ScalarIntExpr &x) {
    Check(DEREF(parser::Unwrap<parser::Expr>(x)));
  }
  void Check(const parser::ScalarExpr &x) {
    Check(DEREF(parser::Unwrap<parser::Expr>(x)));
  }
  void Check(const parser::Expr &expr) {
    if (MaybeMsg msg{DeviceExprChecker{}(expr.typedExpr)}) {
      context_.Say(expr.source, std::move(*msg));
    }
  }

  SemanticsContext &context_;
};

void CUDAChecker::Enter(const parser::SubroutineSubprogram &x) {
  DeviceContextChecker<false>{context_}.CheckSubprogram(
      std::get<parser::Name>(
          std::get<parser::Statement<parser::SubroutineStmt>>(x.t).statement.t),
      std::get<parser::ExecutionPart>(x.t).v);
}

void CUDAChecker::Enter(const parser::FunctionSubprogram &x) {
  DeviceContextChecker<false>{context_}.CheckSubprogram(
      std::get<parser::Name>(
          std::get<parser::Statement<parser::FunctionStmt>>(x.t).statement.t),
      std::get<parser::ExecutionPart>(x.t).v);
}

void CUDAChecker::Enter(const parser::SeparateModuleSubprogram &x) {
  DeviceContextChecker<false>{context_}.CheckSubprogram(
      std::get<parser::Statement<parser::MpSubprogramStmt>>(x.t).statement.v,
      std::get<parser::ExecutionPart>(x.t).v);
}

// !$CUF KERNEL DO semantic checks

static int DoConstructTightNesting(
    const parser::DoConstruct *doConstruct, const parser::Block *&innerBlock) {
  if (!doConstruct || !doConstruct->IsDoNormal()) {
    return 0;
  }
  innerBlock = &std::get<parser::Block>(doConstruct->t);
  if (innerBlock->size() == 1) {
    if (const auto *execConstruct{
            std::get_if<parser::ExecutableConstruct>(&innerBlock->front().u)}) {
      if (const auto *next{
              std::get_if<common::Indirection<parser::DoConstruct>>(
                  &execConstruct->u)}) {
        return 1 + DoConstructTightNesting(&next->value(), innerBlock);
      }
    }
  }
  return 1;
}

static void CheckReduce(
    SemanticsContext &context, const parser::CUFReduction &reduce) {
  auto op{std::get<parser::CUFReduction::Operator>(reduce.t).v};
  for (const auto &var :
      std::get<std::list<parser::Scalar<parser::Variable>>>(reduce.t)) {
    if (const auto &typedExprPtr{var.thing.typedExpr};
        typedExprPtr && typedExprPtr->v) {
      const auto &expr{*typedExprPtr->v};
      if (auto type{expr.GetType()}) {
        auto cat{type->category()};
        bool isOk{false};
        switch (op) {
        case parser::ReductionOperator::Operator::Plus:
        case parser::ReductionOperator::Operator::Multiply:
        case parser::ReductionOperator::Operator::Max:
        case parser::ReductionOperator::Operator::Min:
          isOk = cat == TypeCategory::Integer || cat == TypeCategory::Real;
          break;
        case parser::ReductionOperator::Operator::Iand:
        case parser::ReductionOperator::Operator::Ior:
        case parser::ReductionOperator::Operator::Ieor:
          isOk = cat == TypeCategory::Integer;
          break;
        case parser::ReductionOperator::Operator::And:
        case parser::ReductionOperator::Operator::Or:
        case parser::ReductionOperator::Operator::Eqv:
        case parser::ReductionOperator::Operator::Neqv:
          isOk = cat == TypeCategory::Logical;
          break;
        }
        if (!isOk) {
          context.Say(var.thing.GetSource(),
              "!$CUF KERNEL DO REDUCE operation is not acceptable for a variable with type %s"_err_en_US,
              type->AsFortran());
        }
      }
    }
  }
}

void CUDAChecker::Enter(const parser::CUFKernelDoConstruct &x) {
  auto source{std::get<parser::CUFKernelDoConstruct::Directive>(x.t).source};
  const auto &directive{std::get<parser::CUFKernelDoConstruct::Directive>(x.t)};
  std::int64_t depth{1};
  if (auto expr{AnalyzeExpr(context_,
          std::get<std::optional<parser::ScalarIntConstantExpr>>(
              directive.t))}) {
    depth = evaluate::ToInt64(expr).value_or(0);
    if (depth <= 0) {
      context_.Say(source,
          "!$CUF KERNEL DO (%jd): loop nesting depth must be positive"_err_en_US,
          std::intmax_t{depth});
      depth = 1;
    }
  }
  const parser::DoConstruct *doConstruct{common::GetPtrFromOptional(
      std::get<std::optional<parser::DoConstruct>>(x.t))};
  const parser::Block *innerBlock{nullptr};
  if (DoConstructTightNesting(doConstruct, innerBlock) < depth) {
    context_.Say(source,
        "!$CUF KERNEL DO (%jd) must be followed by a DO construct with tightly nested outer levels of counted DO loops"_err_en_US,
        std::intmax_t{depth});
  }
  if (innerBlock) {
    DeviceContextChecker<true>{context_}.Check(*innerBlock);
  }
  for (const auto &reduce :
      std::get<std::list<parser::CUFReduction>>(directive.t)) {
    CheckReduce(context_, reduce);
  }
  inCUFKernelDoConstruct_ = true;
}

void CUDAChecker::Leave(const parser::CUFKernelDoConstruct &) {
  inCUFKernelDoConstruct_ = false;
}

void CUDAChecker::Enter(const parser::AssignmentStmt &x) {
  auto lhsLoc{std::get<parser::Variable>(x.t).GetSource()};
  const auto &scope{context_.FindScope(lhsLoc)};
  const Scope &progUnit{GetProgramUnitContaining(scope)};
  if (IsCUDADeviceContext(&progUnit) || inCUFKernelDoConstruct_) {
    return; // Data transfer with assignment is only perform on host.
  }

  const evaluate::Assignment *assign{semantics::GetAssignment(x)};
  if (!assign) {
    return;
  }

  int nbLhs{evaluate::GetNbOfCUDADeviceSymbols(assign->lhs)};
  int nbRhs{evaluate::GetNbOfCUDADeviceSymbols(assign->rhs)};

  // device to host transfer with more than one device object on the rhs is not
  // legal.
  if (nbLhs == 0 && nbRhs > 1) {
    context_.Say(lhsLoc,
        "More than one reference to a CUDA object on the right hand side of the assigment"_err_en_US);
  }
}

} // namespace Fortran::semantics
