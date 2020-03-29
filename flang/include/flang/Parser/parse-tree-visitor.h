//===-- include/flang/Parser/parse-tree-visitor.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_PARSE_TREE_VISITOR_H_
#define FORTRAN_PARSER_PARSE_TREE_VISITOR_H_

#include "parse-tree.h"
#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>

/// Parse tree visitor
/// Call Walk(x, visitor) to visit x and, by default, each node under x.
/// If x is non-const, the visitor member functions can modify the tree.
///
/// visitor.Pre(x) is called before visiting x and its children are not
/// visited if it returns false.
///
/// visitor.Post(x) is called after visiting x.

namespace Fortran::parser {

// Default case for visitation of non-class data members, strings, and
// any other non-decomposable values.
template <typename A, typename V>
std::enable_if_t<!std::is_class_v<A> || std::is_same_v<std::string, A> ||
    std::is_same_v<CharBlock, A>>
Walk(const A &x, V &visitor) {
  if (visitor.Pre(x)) {
    visitor.Post(x);
  }
}
template <typename A, typename M>
std::enable_if_t<!std::is_class_v<A> || std::is_same_v<std::string, A> ||
    std::is_same_v<CharBlock, A>>
Walk(A &x, M &mutator) {
  if (mutator.Pre(x)) {
    mutator.Post(x);
  }
}

template <typename V> void Walk(const format::ControlEditDesc &, V &);
template <typename M> void Walk(format::ControlEditDesc &, M &);
template <typename V> void Walk(const format::DerivedTypeDataEditDesc &, V &);
template <typename M> void Walk(format::DerivedTypeDataEditDesc &, M &);
template <typename V> void Walk(const format::FormatItem &, V &);
template <typename M> void Walk(format::FormatItem &, M &);
template <typename V> void Walk(const format::FormatSpecification &, V &);
template <typename M> void Walk(format::FormatSpecification &, M &);
template <typename V> void Walk(const format::IntrinsicTypeDataEditDesc &, V &);
template <typename M> void Walk(format::IntrinsicTypeDataEditDesc &, M &);

// Traversal of needed STL template classes (optional, list, tuple, variant)
template <typename T, typename V>
void Walk(const std::optional<T> &x, V &visitor) {
  if (x) {
    Walk(*x, visitor);
  }
}
template <typename T, typename M> void Walk(std::optional<T> &x, M &mutator) {
  if (x) {
    Walk(*x, mutator);
  }
}
// For most lists, just traverse the elements; but when a list constitutes
// a Block (i.e., std::list<ExecutionPartConstruct>), also invoke the
// visitor/mutator on the list itself.
template <typename T, typename V> void Walk(const std::list<T> &x, V &visitor) {
  for (const auto &elem : x) {
    Walk(elem, visitor);
  }
}
template <typename T, typename M> void Walk(std::list<T> &x, M &mutator) {
  for (auto &elem : x) {
    Walk(elem, mutator);
  }
}
template <typename V> void Walk(const Block &x, V &visitor) {
  if (visitor.Pre(x)) {
    for (const auto &elem : x) {
      Walk(elem, visitor);
    }
    visitor.Post(x);
  }
}
template <typename M> void Walk(Block &x, M &mutator) {
  if (mutator.Pre(x)) {
    for (auto &elem : x) {
      Walk(elem, mutator);
    }
    mutator.Post(x);
  }
}
template <std::size_t I = 0, typename Func, typename T>
void ForEachInTuple(const T &tuple, Func func) {
  func(std::get<I>(tuple));
  if constexpr (I + 1 < std::tuple_size_v<T>) {
    ForEachInTuple<I + 1>(tuple, func);
  }
}
template <typename V, typename... A>
void Walk(const std::tuple<A...> &x, V &visitor) {
  if (sizeof...(A) > 0) {
    if (visitor.Pre(x)) {
      ForEachInTuple(x, [&](const auto &y) { Walk(y, visitor); });
      visitor.Post(x);
    }
  }
}
template <std::size_t I = 0, typename Func, typename T>
void ForEachInTuple(T &tuple, Func func) {
  func(std::get<I>(tuple));
  if constexpr (I + 1 < std::tuple_size_v<T>) {
    ForEachInTuple<I + 1>(tuple, func);
  }
}
template <typename M, typename... A>
void Walk(std::tuple<A...> &x, M &mutator) {
  if (sizeof...(A) > 0) {
    if (mutator.Pre(x)) {
      ForEachInTuple(x, [&](auto &y) { Walk(y, mutator); });
      mutator.Post(x);
    }
  }
}
template <typename V, typename... A>
void Walk(const std::variant<A...> &x, V &visitor) {
  if (visitor.Pre(x)) {
    std::visit([&](const auto &y) { Walk(y, visitor); }, x);
    visitor.Post(x);
  }
}
template <typename M, typename... A>
void Walk(std::variant<A...> &x, M &mutator) {
  if (mutator.Pre(x)) {
    std::visit([&](auto &y) { Walk(y, mutator); }, x);
    mutator.Post(x);
  }
}
template <typename A, typename B, typename V>
void Walk(const std::pair<A, B> &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.first, visitor);
    Walk(x.second, visitor);
  }
}
template <typename A, typename B, typename M>
void Walk(std::pair<A, B> &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.first, mutator);
    Walk(x.second, mutator);
  }
}

// Trait-determined traversal of empty, tuple, union, wrapper,
// and constraint-checking classes.
template <typename A, typename V>
std::enable_if_t<EmptyTrait<A>> Walk(const A &x, V &visitor) {
  if (visitor.Pre(x)) {
    visitor.Post(x);
  }
}
template <typename A, typename M>
std::enable_if_t<EmptyTrait<A>> Walk(A &x, M &mutator) {
  if (mutator.Pre(x)) {
    mutator.Post(x);
  }
}

template <typename A, typename V>
std::enable_if_t<TupleTrait<A>> Walk(const A &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.t, visitor);
    visitor.Post(x);
  }
}
template <typename A, typename M>
std::enable_if_t<TupleTrait<A>> Walk(A &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.t, mutator);
    mutator.Post(x);
  }
}

template <typename A, typename V>
std::enable_if_t<UnionTrait<A>> Walk(const A &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.u, visitor);
    visitor.Post(x);
  }
}
template <typename A, typename M>
std::enable_if_t<UnionTrait<A>> Walk(A &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.u, mutator);
    mutator.Post(x);
  }
}

template <typename A, typename V>
std::enable_if_t<WrapperTrait<A>> Walk(const A &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.v, visitor);
    visitor.Post(x);
  }
}
template <typename A, typename M>
std::enable_if_t<WrapperTrait<A>> Walk(A &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.v, mutator);
    mutator.Post(x);
  }
}

template <typename A, typename V>
std::enable_if_t<ConstraintTrait<A>> Walk(const A &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.thing, visitor);
    visitor.Post(x);
  }
}
template <typename A, typename M>
std::enable_if_t<ConstraintTrait<A>> Walk(A &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.thing, mutator);
    mutator.Post(x);
  }
}

template <typename T, typename V>
void Walk(const common::Indirection<T> &x, V &visitor) {
  Walk(x.value(), visitor);
}
template <typename T, typename M>
void Walk(common::Indirection<T> &x, M &mutator) {
  Walk(x.value(), mutator);
}

template <typename T, typename V> void Walk(const Statement<T> &x, V &visitor) {
  if (visitor.Pre(x)) {
    // N.B. The label, if any, is not visited.
    Walk(x.source, visitor);
    Walk(x.statement, visitor);
    visitor.Post(x);
  }
}
template <typename T, typename M> void Walk(Statement<T> &x, M &mutator) {
  if (mutator.Pre(x)) {
    // N.B. The label, if any, is not visited.
    Walk(x.source, mutator);
    Walk(x.statement, mutator);
    mutator.Post(x);
  }
}

template <typename T, typename V>
void Walk(const UnlabeledStatement<T> &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.source, visitor);
    Walk(x.statement, visitor);
    visitor.Post(x);
  }
}
template <typename T, typename M>
void Walk(UnlabeledStatement<T> &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.source, mutator);
    Walk(x.statement, mutator);
    mutator.Post(x);
  }
}

template <typename V> void Walk(const Name &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.source, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(Name &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.source, mutator);
    mutator.Post(x);
  }
}

template <typename V> void Walk(const AcSpec &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.type, visitor);
    Walk(x.values, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(AcSpec &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.type, mutator);
    Walk(x.values, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const ArrayElement &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.base, visitor);
    Walk(x.subscripts, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(ArrayElement &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.base, mutator);
    Walk(x.subscripts, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const CharSelector::LengthAndKind &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.length, visitor);
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(CharSelector::LengthAndKind &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.length, mutator);
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const CaseValueRange::Range &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.lower, visitor);
    Walk(x.upper, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(CaseValueRange::Range &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.lower, mutator);
    Walk(x.upper, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const CoindexedNamedObject &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.base, visitor);
    Walk(x.imageSelector, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(CoindexedNamedObject &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.base, mutator);
    Walk(x.imageSelector, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const DeclarationTypeSpec::Class &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.derived, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(DeclarationTypeSpec::Class &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.derived, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const DeclarationTypeSpec::Type &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.derived, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(DeclarationTypeSpec::Type &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.derived, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const ImportStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.names, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(ImportStmt &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.names, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const IntrinsicTypeSpec::Character &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.selector, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(IntrinsicTypeSpec::Character &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.selector, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const IntrinsicTypeSpec::Complex &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(IntrinsicTypeSpec::Complex &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const IntrinsicTypeSpec::Logical &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(IntrinsicTypeSpec::Logical &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const IntrinsicTypeSpec::Real &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(IntrinsicTypeSpec::Real &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template <typename A, typename B, typename V>
void Walk(const LoopBounds<A, B> &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.name, visitor);
    Walk(x.lower, visitor);
    Walk(x.upper, visitor);
    Walk(x.step, visitor);
    visitor.Post(x);
  }
}
template <typename A, typename B, typename M>
void Walk(LoopBounds<A, B> &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.name, mutator);
    Walk(x.lower, mutator);
    Walk(x.upper, mutator);
    Walk(x.step, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const CommonStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.blocks, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(CommonStmt &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.blocks, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const Expr &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.source, visitor);
    Walk(x.u, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(Expr &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.source, mutator);
    Walk(x.u, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const Designator &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.source, visitor);
    Walk(x.u, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(Designator &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.source, mutator);
    Walk(x.u, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const Call &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.source, visitor);
    Walk(x.t, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(Call &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.source, mutator);
    Walk(x.t, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const PartRef &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.name, visitor);
    Walk(x.subscripts, visitor);
    Walk(x.imageSelector, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(PartRef &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.name, mutator);
    Walk(x.subscripts, mutator);
    Walk(x.imageSelector, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const ReadStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.iounit, visitor);
    Walk(x.format, visitor);
    Walk(x.controls, visitor);
    Walk(x.items, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(ReadStmt &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.iounit, mutator);
    Walk(x.format, mutator);
    Walk(x.controls, mutator);
    Walk(x.items, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const SignedIntLiteralConstant &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.source, visitor);
    Walk(x.t, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(SignedIntLiteralConstant &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.source, mutator);
    Walk(x.t, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const RealLiteralConstant &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.real, visitor);
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(RealLiteralConstant &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.real, mutator);
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const RealLiteralConstant::Real &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.source, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(RealLiteralConstant::Real &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.source, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const StructureComponent &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.base, visitor);
    Walk(x.component, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(StructureComponent &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.base, mutator);
    Walk(x.component, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const Suffix &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.binding, visitor);
    Walk(x.resultName, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(Suffix &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.binding, mutator);
    Walk(x.resultName, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const TypeBoundProcedureStmt::WithInterface &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.interfaceName, visitor);
    Walk(x.attributes, visitor);
    Walk(x.bindingNames, visitor);
    visitor.Post(x);
  }
}
template <typename M>
void Walk(TypeBoundProcedureStmt::WithInterface &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.interfaceName, mutator);
    Walk(x.attributes, mutator);
    Walk(x.bindingNames, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const TypeBoundProcedureStmt::WithoutInterface &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.attributes, visitor);
    Walk(x.declarations, visitor);
    visitor.Post(x);
  }
}
template <typename M>
void Walk(TypeBoundProcedureStmt::WithoutInterface &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.attributes, mutator);
    Walk(x.declarations, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const UseStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.nature, visitor);
    Walk(x.moduleName, visitor);
    Walk(x.u, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(UseStmt &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.nature, mutator);
    Walk(x.moduleName, mutator);
    Walk(x.u, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const WriteStmt &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.iounit, visitor);
    Walk(x.format, visitor);
    Walk(x.controls, visitor);
    Walk(x.items, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(WriteStmt &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.iounit, mutator);
    Walk(x.format, mutator);
    Walk(x.controls, mutator);
    Walk(x.items, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const format::ControlEditDesc &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(format::ControlEditDesc &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.kind, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const format::DerivedTypeDataEditDesc &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.type, visitor);
    Walk(x.parameters, visitor);
    visitor.Post(x);
  }
}
template <typename M>
void Walk(format::DerivedTypeDataEditDesc &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.type, mutator);
    Walk(x.parameters, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const format::FormatItem &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.repeatCount, visitor);
    Walk(x.u, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(format::FormatItem &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.repeatCount, mutator);
    Walk(x.u, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const format::FormatSpecification &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.items, visitor);
    Walk(x.unlimitedItems, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(format::FormatSpecification &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.items, mutator);
    Walk(x.unlimitedItems, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const format::IntrinsicTypeDataEditDesc &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.kind, visitor);
    Walk(x.width, visitor);
    Walk(x.digits, visitor);
    Walk(x.exponentWidth, visitor);
    visitor.Post(x);
  }
}
template <typename M>
void Walk(format::IntrinsicTypeDataEditDesc &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.kind, mutator);
    Walk(x.width, mutator);
    Walk(x.digits, mutator);
    Walk(x.exponentWidth, mutator);
    mutator.Post(x);
  }
}
template <typename V> void Walk(const CompilerDirective &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.source, visitor);
    Walk(x.u, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(CompilerDirective &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.source, mutator);
    Walk(x.u, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const OmpLinearClause::WithModifier &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.modifier, visitor);
    Walk(x.names, visitor);
    Walk(x.step, visitor);
    visitor.Post(x);
  }
}
template <typename M> void Walk(OmpLinearClause::WithModifier &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.modifier, mutator);
    Walk(x.names, mutator);
    Walk(x.step, mutator);
    mutator.Post(x);
  }
}
template <typename V>
void Walk(const OmpLinearClause::WithoutModifier &x, V &visitor) {
  if (visitor.Pre(x)) {
    Walk(x.names, visitor);
    Walk(x.step, visitor);
    visitor.Post(x);
  }
}
template <typename M>
void Walk(OmpLinearClause::WithoutModifier &x, M &mutator) {
  if (mutator.Pre(x)) {
    Walk(x.names, mutator);
    Walk(x.step, mutator);
    mutator.Post(x);
  }
}
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_PARSE_TREE_VISITOR_H_
