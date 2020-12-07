//===-- lib/Parser/parse-tree.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/parse-tree.h"
#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Parser/tools.h"
#include "flang/Parser/user-state.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace Fortran::parser {

// R867
ImportStmt::ImportStmt(common::ImportKind &&k, std::list<Name> &&n)
    : kind{k}, names(std::move(n)) {
  CHECK(kind == common::ImportKind::Default ||
      kind == common::ImportKind::Only || names.empty());
}

// R873
CommonStmt::CommonStmt(std::optional<Name> &&name,
    std::list<CommonBlockObject> &&objects, std::list<Block> &&others) {
  blocks.emplace_front(std::move(name), std::move(objects));
  blocks.splice(blocks.end(), std::move(others));
}

// R901 designator
bool Designator::EndsInBareName() const {
  return std::visit(
      common::visitors{
          [](const DataRef &dr) {
            return std::holds_alternative<Name>(dr.u) ||
                std::holds_alternative<common::Indirection<StructureComponent>>(
                    dr.u);
          },
          [](const Substring &) { return false; },
      },
      u);
}

// R911 data-ref -> part-ref [% part-ref]...
DataRef::DataRef(std::list<PartRef> &&prl) : u{std::move(prl.front().name)} {
  for (bool first{true}; !prl.empty(); first = false, prl.pop_front()) {
    PartRef &pr{prl.front()};
    if (!first) {
      u = common::Indirection<StructureComponent>::Make(
          std::move(*this), std::move(pr.name));
    }
    if (!pr.subscripts.empty()) {
      u = common::Indirection<ArrayElement>::Make(
          std::move(*this), std::move(pr.subscripts));
    }
    if (pr.imageSelector) {
      u = common::Indirection<CoindexedNamedObject>::Make(
          std::move(*this), std::move(*pr.imageSelector));
    }
  }
}

// R1001 - R1022 expression
Expr::Expr(Designator &&x)
    : u{common::Indirection<Designator>::Make(std::move(x))} {}
Expr::Expr(FunctionReference &&x)
    : u{common::Indirection<FunctionReference>::Make(std::move(x))} {}

const std::optional<LoopControl> &DoConstruct::GetLoopControl() const {
  const NonLabelDoStmt &doStmt{
      std::get<Statement<NonLabelDoStmt>>(t).statement};
  const std::optional<LoopControl> &control{
      std::get<std::optional<LoopControl>>(doStmt.t)};
  return control;
}

bool DoConstruct::IsDoNormal() const {
  const std::optional<LoopControl> &control{GetLoopControl()};
  return control && std::holds_alternative<LoopControl::Bounds>(control->u);
}

bool DoConstruct::IsDoWhile() const {
  const std::optional<LoopControl> &control{GetLoopControl()};
  return control && std::holds_alternative<ScalarLogicalExpr>(control->u);
}

bool DoConstruct::IsDoConcurrent() const {
  const std::optional<LoopControl> &control{GetLoopControl()};
  return control && std::holds_alternative<LoopControl::Concurrent>(control->u);
}

static Designator MakeArrayElementRef(
    const Name &name, std::list<Expr> &&subscripts) {
  ArrayElement arrayElement{DataRef{Name{name}}, std::list<SectionSubscript>{}};
  for (Expr &expr : subscripts) {
    arrayElement.subscripts.push_back(
        SectionSubscript{Integer{common::Indirection{std::move(expr)}}});
  }
  return Designator{DataRef{common::Indirection{std::move(arrayElement)}}};
}

static Designator MakeArrayElementRef(
    StructureComponent &&sc, std::list<Expr> &&subscripts) {
  ArrayElement arrayElement{DataRef{common::Indirection{std::move(sc)}},
      std::list<SectionSubscript>{}};
  for (Expr &expr : subscripts) {
    arrayElement.subscripts.push_back(
        SectionSubscript{Integer{common::Indirection{std::move(expr)}}});
  }
  return Designator{DataRef{common::Indirection{std::move(arrayElement)}}};
}

// Set source in any type of node that has it.
template <typename T> T WithSource(CharBlock source, T &&x) {
  x.source = source;
  return std::move(x);
}

static Expr ActualArgToExpr(ActualArgSpec &arg) {
  return std::visit(
      common::visitors{
          [&](common::Indirection<Expr> &y) { return std::move(y.value()); },
          [&](common::Indirection<Variable> &y) {
            return std::visit(
                common::visitors{
                    [&](common::Indirection<Designator> &z) {
                      return WithSource(
                          z.value().source, Expr{std::move(z.value())});
                    },
                    [&](common::Indirection<FunctionReference> &z) {
                      return WithSource(
                          z.value().v.source, Expr{std::move(z.value())});
                    },
                },
                y.value().u);
          },
          [&](auto &) -> Expr { common::die("unexpected type"); },
      },
      std::get<ActualArg>(arg.t).u);
}

Designator FunctionReference::ConvertToArrayElementRef() {
  std::list<Expr> args;
  for (auto &arg : std::get<std::list<ActualArgSpec>>(v.t)) {
    args.emplace_back(ActualArgToExpr(arg));
  }
  return std::visit(
      common::visitors{
          [&](const Name &name) {
            return WithSource(
                v.source, MakeArrayElementRef(name, std::move(args)));
          },
          [&](ProcComponentRef &pcr) {
            return WithSource(v.source,
                MakeArrayElementRef(std::move(pcr.v.thing), std::move(args)));
          },
      },
      std::get<ProcedureDesignator>(v.t).u);
}

StructureConstructor FunctionReference::ConvertToStructureConstructor(
    const semantics::DerivedTypeSpec &derived) {
  Name name{std::get<parser::Name>(std::get<ProcedureDesignator>(v.t).u)};
  std::list<ComponentSpec> components;
  for (auto &arg : std::get<std::list<ActualArgSpec>>(v.t)) {
    std::optional<Keyword> keyword;
    if (auto &kw{std::get<std::optional<Keyword>>(arg.t)}) {
      keyword.emplace(Keyword{Name{kw->v}});
    }
    components.emplace_back(
        std::move(keyword), ComponentDataSource{ActualArgToExpr(arg)});
  }
  DerivedTypeSpec spec{std::move(name), std::list<TypeParamSpec>{}};
  spec.derivedTypeSpec = &derived;
  return StructureConstructor{std::move(spec), std::move(components)};
}

StructureConstructor ArrayElement::ConvertToStructureConstructor(
    const semantics::DerivedTypeSpec &derived) {
  Name name{std::get<parser::Name>(base.u)};
  std::list<ComponentSpec> components;
  for (auto &subscript : subscripts) {
    components.emplace_back(std::optional<Keyword>{},
        ComponentDataSource{std::move(*Unwrap<Expr>(subscript))});
  }
  DerivedTypeSpec spec{std::move(name), std::list<TypeParamSpec>{}};
  spec.derivedTypeSpec = &derived;
  return StructureConstructor{std::move(spec), std::move(components)};
}

Substring ArrayElement::ConvertToSubstring() {
  auto iter{subscripts.begin()};
  CHECK(iter != subscripts.end());
  auto &triplet{std::get<SubscriptTriplet>(iter->u)};
  CHECK(!std::get<2>(triplet.t));
  CHECK(++iter == subscripts.end());
  return Substring{std::move(base),
      SubstringRange{std::get<0>(std::move(triplet.t)),
          std::get<1>(std::move(triplet.t))}};
}

// R1544 stmt-function-stmt
// Convert this stmt-function-stmt to an array element assignment statement.
Statement<ActionStmt> StmtFunctionStmt::ConvertToAssignment() {
  auto &funcName{std::get<Name>(t)};
  auto &funcArgs{std::get<std::list<Name>>(t)};
  auto &funcExpr{std::get<Scalar<Expr>>(t).thing};
  CharBlock source{funcName.source};
  std::list<Expr> subscripts;
  for (Name &arg : funcArgs) {
    subscripts.push_back(WithSource(arg.source,
        Expr{common::Indirection{
            WithSource(arg.source, Designator{DataRef{Name{arg}}})}}));
    source.ExtendToCover(arg.source);
  }
  // extend source to include closing paren
  if (funcArgs.empty()) {
    CHECK(*source.end() == '(');
    source = CharBlock{source.begin(), source.end() + 1};
  }
  CHECK(*source.end() == ')');
  source = CharBlock{source.begin(), source.end() + 1};
  auto variable{Variable{common::Indirection{WithSource(
      source, MakeArrayElementRef(funcName, std::move(subscripts)))}}};
  return Statement{std::nullopt,
      ActionStmt{common::Indirection{
          AssignmentStmt{std::move(variable), std::move(funcExpr)}}}};
}

CharBlock Variable::GetSource() const {
  return std::visit(
      common::visitors{
          [&](const common::Indirection<Designator> &des) {
            return des.value().source;
          },
          [&](const common::Indirection<parser::FunctionReference> &call) {
            return call.value().v.source;
          },
      },
      u);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Name &x) {
  return os << x.ToString();
}
} // namespace Fortran::parser
