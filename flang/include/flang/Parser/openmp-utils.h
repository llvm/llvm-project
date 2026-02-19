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
#include "llvm/ADT/iterator_range.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include <cassert>
#include <iterator>
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

const OmpCombinerExpression *GetCombinerExpr(const OmpReductionSpecifier &x);
const OmpCombinerExpression *GetCombinerExpr(const OmpClause &x);
const OmpInitializerExpression *GetInitializerExpr(const OmpClause &x);

struct OmpAllocateInfo {
  std::vector<const OmpAllocateDirective *> dirs;
  const ExecutionPartConstruct *body{nullptr};
};

OmpAllocateInfo SplitOmpAllocate(const OmpAllocateDirective &x);

// Iterate over a range of parser::Block::const_iterator's. When the end
// of the range is reached, the iterator becomes invalid.
// Treat BLOCK constructs as if they were transparent, i.e. as if the
// BLOCK/ENDBLOCK statements, and the specification part contained within
// were removed. The stepping determines whether the iterator steps "into"
// DO loops and OpenMP loop constructs, or steps "over" them.
//
// Example: consecutive locations of the iterator:
//
//    Step::Into                  Step::Over
//          block                       block
//    1 =>    stmt1               1 =>    stmt1
//            block                       block
//              integer :: x                integer :: x
//    2 =>      stmt2             2 =>      stmt2
//              block                       block
//              end block                   end block
//            end block                   end block
//    3 =>    do i = 1, n         3 =>    do i = 1, n
//    4 =>      continue                    continue
//            end do                      end do
//    5 =>    stmt3               4 =>    stmt3
//          end block                   end block
//
//    6 =>  <invalid>             5 =>  <invalid>
//
// The iterator is in a legal state (position) if it's at an
// ExecutionPartConstruct that is not a BlockConstruct, or is invalid.
struct ExecutionPartIterator {
  enum class Step {
    Into,
    Over,
    Default = Into,
  };

  using IteratorType = Block::const_iterator;
  using IteratorRange = llvm::iterator_range<IteratorType>;

  struct Construct {
    Construct(IteratorType b, IteratorType e, const ExecutionPartConstruct *c)
        : range(b, e), owner(c) {}
    template <typename R>
    Construct(const R &r, const ExecutionPartConstruct *c)
        : range(r), owner(c) {}
    Construct(const Construct &c) = default;
    IteratorRange range;
    const ExecutionPartConstruct *owner;
  };

  ExecutionPartIterator() = default;

  ExecutionPartIterator(IteratorType b, IteratorType e, Step s = Step::Default,
      const ExecutionPartConstruct *c = nullptr)
      : stepping_(s) {
    stack_.emplace_back(b, e, c);
    adjust();
  }
  template <typename R, //
      typename = decltype(std::declval<R>().begin()),
      typename = decltype(std::declval<R>().end())>
  ExecutionPartIterator(const R &range, Step stepping = Step::Default,
      const ExecutionPartConstruct *construct = nullptr)
      : ExecutionPartIterator(range.begin(), range.end(), stepping, construct) {
  }

  // Advance the iterator to the next legal position. If the current position
  // is a DO-loop or a loop construct, step into the contained Block.
  void step();

  // Advance the iterator to the next legal position. If the current position
  // is a DO-loop or a loop construct, step to the next legal position following
  // the DO-loop or loop construct.
  void next();

  bool valid() const { return !stack_.empty(); }

  decltype(auto) operator*() const { return *at(); }
  bool operator==(const ExecutionPartIterator &other) const {
    if (valid() != other.valid()) {
      return false;
    }
    // Invalid iterators are considered equal.
    return !valid() ||
        stack_.back().range.begin() == other.stack_.back().range.begin();
  }
  bool operator!=(const ExecutionPartIterator &other) const {
    return !(*this == other);
  }

  ExecutionPartIterator &operator++() {
    if (stepping_ == Step::Into) {
      step();
    } else {
      assert(stepping_ == Step::Over && "Unexpected stepping");
      next();
    }
    return *this;
  }

  ExecutionPartIterator operator++(int) {
    ExecutionPartIterator copy{*this};
    operator++();
    return copy;
  }

  using difference_type = IteratorType::difference_type;
  using value_type = IteratorType::value_type;
  using reference = IteratorType::reference;
  using pointer = IteratorType::pointer;
  using iterator_category = std::forward_iterator_tag;

private:
  IteratorType at() const { return stack_.back().range.begin(); };

  // If the iterator is not at a legal location, keep advancing it until
  // it lands at a legal location or becomes invalid.
  void adjust();

  const Step stepping_ = Step::Default;
  std::vector<Construct> stack_;
};

template <typename Iterator = ExecutionPartIterator> struct ExecutionPartRange {
  using Step = typename Iterator::Step;

  ExecutionPartRange(Block::const_iterator begin, Block::const_iterator end,
      Step stepping = Step::Default,
      const ExecutionPartConstruct *owner = nullptr)
      : begin_(begin, end, stepping, owner), end_() {}
  template <typename R, //
      typename = decltype(std::declval<R>().begin()),
      typename = decltype(std::declval<R>().end())>
  ExecutionPartRange(const R &range, Step stepping = Step::Default,
      const ExecutionPartConstruct *owner = nullptr)
      : ExecutionPartRange(range.begin(), range.end(), stepping, owner) {}

  Iterator begin() const { return begin_; }
  Iterator end() const { return end_; }

private:
  Iterator begin_, end_;
};

struct LoopNestIterator : public ExecutionPartIterator {
  LoopNestIterator() = default;

  LoopNestIterator(IteratorType b, IteratorType e, Step s = Step::Default,
      const ExecutionPartConstruct *c = nullptr)
      : ExecutionPartIterator(b, e, s, c) {
    adjust();
  }
  template <typename R, //
      typename = decltype(std::declval<R>().begin()),
      typename = decltype(std::declval<R>().end())>
  LoopNestIterator(const R &range, Step stepping = Step::Default,
      const ExecutionPartConstruct *construct = nullptr)
      : LoopNestIterator(range.begin(), range.end(), stepping, construct) {}

  LoopNestIterator &operator++() {
    ExecutionPartIterator::operator++();
    adjust();
    return *this;
  }

  LoopNestIterator operator++(int) {
    LoopNestIterator copy{*this};
    operator++();
    return copy;
  }

private:
  static bool isLoop(const ExecutionPartConstruct &c);

  void adjust() {
    while (valid() && !isLoop(**this)) {
      ExecutionPartIterator::operator++();
    }
  }
};

using BlockRange = ExecutionPartRange<ExecutionPartIterator>;
using LoopRange = ExecutionPartRange<LoopNestIterator>;

} // namespace Fortran::parser::omp

#endif // FORTRAN_PARSER_OPENMP_UTILS_H
