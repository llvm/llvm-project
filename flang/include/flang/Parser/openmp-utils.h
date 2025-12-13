//===-- flang/Parser/openmp-utils.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common OpenMP utilities.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_OPENMP_UTILS_H
#define FORTRAN_PARSER_OPENMP_UTILS_H

#include "flang/Common/indirection.h"
#include "flang/Common/template.h"
#include "flang/Parser/parse-tree.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace Fortran::parser::omp {

template <typename T> constexpr auto addr_if(std::optional<T> &x) {
  return x ? &*x : nullptr;
}
template <typename T> constexpr auto addr_if(const std::optional<T> &x) {
  return x ? &*x : nullptr;
}

namespace detail {
struct DirectiveNameScope {
  static OmpDirectiveName MakeName(CharBlock source = {},
      llvm::omp::Directive id = llvm::omp::Directive::OMPD_unknown) {
    OmpDirectiveName name;
    name.source = source;
    name.v = id;
    return name;
  }

  static OmpDirectiveName GetOmpDirectiveName(const OmpDirectiveName &x) {
    return x;
  }

  static OmpDirectiveName GetOmpDirectiveName(const OmpBeginLoopDirective &x) {
    return x.DirName();
  }

  static OmpDirectiveName GetOmpDirectiveName(const OpenMPSectionConstruct &x) {
    if (auto &spec{std::get<std::optional<OmpDirectiveSpecification>>(x.t)}) {
      return spec->DirName();
    } else {
      return MakeName({}, llvm::omp::Directive::OMPD_section);
    }
  }

  static OmpDirectiveName GetOmpDirectiveName(
      const OmpBeginSectionsDirective &x) {
    return x.DirName();
  }

  template <typename T>
  static OmpDirectiveName GetOmpDirectiveName(const T &x) {
    if constexpr (WrapperTrait<T>) {
      return GetOmpDirectiveName(x.v);
    } else if constexpr (TupleTrait<T>) {
      if constexpr (std::is_base_of_v<OmpBlockConstruct, T>) {
        return std::get<OmpBeginDirective>(x.t).DirName();
      } else {
        return GetFromTuple(
            x.t, std::make_index_sequence<std::tuple_size_v<decltype(x.t)>>{});
      }
    } else if constexpr (UnionTrait<T>) {
      return common::visit(
          [](auto &&s) { return GetOmpDirectiveName(s); }, x.u);
    } else {
      return MakeName();
    }
  }

  template <typename... Ts, size_t... Is>
  static OmpDirectiveName GetFromTuple(
      const std::tuple<Ts...> &t, std::index_sequence<Is...>) {
    OmpDirectiveName name = MakeName();
    auto accumulate = [&](const OmpDirectiveName &n) {
      if (name.v == llvm::omp::Directive::OMPD_unknown) {
        name = n;
      } else {
        assert(
            n.v == llvm::omp::Directive::OMPD_unknown && "Conflicting names");
      }
    };
    (accumulate(GetOmpDirectiveName(std::get<Is>(t))), ...);
    return name;
  }

  template <typename T>
  static OmpDirectiveName GetOmpDirectiveName(const common::Indirection<T> &x) {
    return GetOmpDirectiveName(x.value());
  }
};
} // namespace detail

template <typename T> OmpDirectiveName GetOmpDirectiveName(const T &x) {
  return detail::DirectiveNameScope::GetOmpDirectiveName(x);
}

const OpenMPDeclarativeConstruct *GetOmp(const DeclarationConstruct &x);
const OpenMPConstruct *GetOmp(const ExecutionPartConstruct &x);

const OpenMPLoopConstruct *GetOmpLoop(const ExecutionPartConstruct &x);
const DoConstruct *GetDoConstruct(const ExecutionPartConstruct &x);

// Is the template argument "Statement<T>" for some T?
template <typename T> struct IsStatement {
  static constexpr bool value{false};
};
template <typename T> struct IsStatement<Statement<T>> {
  static constexpr bool value{true};
};

std::optional<Label> GetStatementLabel(const ExecutionPartConstruct &x);
std::optional<Label> GetFinalLabel(const OpenMPConstruct &x);

namespace detail {
// Clauses with flangClass = "OmpObjectList".
using MemberObjectListClauses =
    std::tuple<OmpClause::Copyin, OmpClause::Copyprivate, OmpClause::Exclusive,
        OmpClause::Firstprivate, OmpClause::HasDeviceAddr, OmpClause::Inclusive,
        OmpClause::IsDevicePtr, OmpClause::Link, OmpClause::Private,
        OmpClause::Shared, OmpClause::UseDeviceAddr, OmpClause::UseDevicePtr>;

// Clauses with flangClass = "OmpSomeClause", and OmpObjectList a
// member of tuple OmpSomeClause::t.
using TupleObjectListClauses = std::tuple<OmpClause::AdjustArgs,
    OmpClause::Affinity, OmpClause::Aligned, OmpClause::Allocate,
    OmpClause::Enter, OmpClause::From, OmpClause::InReduction,
    OmpClause::Lastprivate, OmpClause::Linear, OmpClause::Map,
    OmpClause::Reduction, OmpClause::TaskReduction, OmpClause::To>;

// Does U have WrapperTrait (i.e. has a member 'v'), and if so, is T the
// type of v?
template <typename T, typename U, bool IsWrapper> struct WrappedInType {
  static constexpr bool value{false};
};

template <typename T, typename U> struct WrappedInType<T, U, true> {
  static constexpr bool value{std::is_same_v<T, decltype(U::v)>};
};

// Same as WrappedInType, but with a list of types Us. Satisfied if any
// type U in Us satisfies WrappedInType<T, U>.
template <typename...> struct WrappedInTypes;

template <typename T> struct WrappedInTypes<T> {
  static constexpr bool value{false};
};

template <typename T, typename U, typename... Us>
struct WrappedInTypes<T, U, Us...> {
  static constexpr bool value{WrappedInType<T, U, WrapperTrait<U>>::value ||
      WrappedInTypes<T, Us...>::value};
};

// Same as WrappedInTypes, but takes type list in a form of a tuple or
// a variant.
template <typename...> struct WrappedInTupleOrVariant {
  static constexpr bool value{false};
};
template <typename T, typename... Us>
struct WrappedInTupleOrVariant<T, std::tuple<Us...>> {
  static constexpr bool value{WrappedInTypes<T, Us...>::value};
};
template <typename T, typename... Us>
struct WrappedInTupleOrVariant<T, std::variant<Us...>> {
  static constexpr bool value{WrappedInTypes<T, Us...>::value};
};
template <typename T, typename U>
constexpr bool WrappedInTupleOrVariantV{WrappedInTupleOrVariant<T, U>::value};
} // namespace detail

template <typename T> const OmpObjectList *GetOmpObjectList(const T &clause) {
  using namespace detail;
  static_assert(std::is_class_v<T>, "Unexpected argument type");

  if constexpr (common::HasMember<T, decltype(OmpClause::u)>) {
    if constexpr (common::HasMember<T, MemberObjectListClauses>) {
      return &clause.v;
    } else if constexpr (common::HasMember<T, TupleObjectListClauses>) {
      return &std::get<OmpObjectList>(clause.v.t);
    } else {
      return nullptr;
    }
  } else if constexpr (WrappedInTupleOrVariantV<T, TupleObjectListClauses>) {
    return &std::get<OmpObjectList>(clause.t);
  } else if constexpr (WrappedInTupleOrVariantV<T, decltype(OmpClause::u)>) {
    return nullptr;
  } else {
    // The condition should be type-dependent, but it should always be false.
    static_assert(sizeof(T) < 0 && "Unexpected argument type");
  }
}

const OmpObjectList *GetOmpObjectList(const OmpClause &clause);
const OmpObjectList *GetOmpObjectList(const OmpClause::Depend &clause);
const OmpObjectList *GetOmpObjectList(const OmpDependClause::TaskDep &x);

template <typename T>
const T *GetFirstArgument(const OmpDirectiveSpecification &spec) {
  for (const OmpArgument &arg : spec.Arguments().v) {
    if (auto *t{std::get_if<T>(&arg.u)}) {
      return t;
    }
  }
  return nullptr;
}

const BlockConstruct *GetFortranBlockConstruct(
    const ExecutionPartConstruct &epc);
const Block &GetInnermostExecPart(const Block &block);
bool IsStrictlyStructuredBlock(const Block &block);

const OmpCombinerExpression *GetCombinerExpr(
    const OmpReductionSpecifier &rspec);
const OmpInitializerExpression *GetInitializerExpr(const OmpClause &init);

struct OmpAllocateInfo {
  std::vector<const OmpAllocateDirective *> dirs;
  const ExecutionPartConstruct *body{nullptr};
};

OmpAllocateInfo SplitOmpAllocate(const OmpAllocateDirective &x);

namespace detail {
template <bool IsConst, typename T> struct ConstIf {
  using type = std::conditional_t<IsConst, std::add_const_t<T>, T>;
};

template <bool IsConst, typename T>
using ConstIfT = typename ConstIf<IsConst, T>::type;
} // namespace detail

template <bool IsConst> struct LoopRange {
  using QualBlock = detail::ConstIfT<IsConst, Block>;
  using QualReference = decltype(std::declval<QualBlock>().front());
  using QualPointer = std::remove_reference_t<QualReference> *;

  LoopRange(QualBlock &x) { Initialize(x); }
  LoopRange(QualReference x);

  LoopRange(detail::ConstIfT<IsConst, OpenMPLoopConstruct> &x)
      : LoopRange(std::get<Block>(x.t)) {}
  LoopRange(detail::ConstIfT<IsConst, DoConstruct> &x)
      : LoopRange(std::get<Block>(x.t)) {}

  size_t size() const { return items.size(); }
  bool empty() const { return items.size() == 0; }

  struct iterator;

  iterator begin();
  iterator end();

private:
  void Initialize(QualBlock &body);

  std::vector<QualPointer> items;
};

template <typename T> LoopRange(T &x) -> LoopRange<std::is_const_v<T>>;

template <bool IsConst> struct LoopRange<IsConst>::iterator {
  QualReference operator*() { return **at; }

  bool operator==(const iterator &other) const { return at == other.at; }
  bool operator!=(const iterator &other) const { return at != other.at; }

  iterator &operator++() {
    ++at;
    return *this;
  }
  iterator &operator--() {
    --at;
    return *this;
  }
  iterator operator++(int);
  iterator operator--(int);

private:
  friend struct LoopRange;
  typename decltype(LoopRange::items)::iterator at;
};

template <bool IsConst> inline auto LoopRange<IsConst>::begin() -> iterator {
  iterator x;
  x.at = items.begin();
  return x;
}

template <bool IsConst> inline auto LoopRange<IsConst>::end() -> iterator {
  iterator x;
  x.at = items.end();
  return x;
}

using ConstLoopRange = LoopRange<true>;

extern template struct LoopRange<true>;
extern template struct LoopRange<false>;

} // namespace Fortran::parser::omp

#endif // FORTRAN_PARSER_OPENMP_UTILS_H
