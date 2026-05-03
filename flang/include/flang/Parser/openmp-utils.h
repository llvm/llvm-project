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

const parser::Designator *GetDesignatorFromObj(const parser::OmpObject &object);
const parser::DataRef *GetDataRefFromObj(const parser::OmpObject &object);
const parser::ArrayElement *GetArrayElementFromObj(
    const parser::OmpObject &object);
std::optional<parser::CharBlock> GetObjectSource(
    const parser::OmpObject &object);
const parser::OmpObject *GetArgumentObject(const parser::OmpArgument &argument);

const OmpDirectiveSpecification &GetOmpDirectiveSpecification(
    const OpenMPConstruct &x);
const OmpDirectiveSpecification &GetOmpDirectiveSpecification(
    const OpenMPDeclarativeConstruct &x);

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

std::string GetUpperName(llvm::omp::Clause id, unsigned version);
std::string GetUpperName(llvm::omp::Directive id, unsigned version);

const OpenMPDeclarativeConstruct *GetOmp(const DeclarationConstruct &x);
const OpenMPConstruct *GetOmp(const ExecutionPartConstruct &x);

const OpenMPLoopConstruct *GetOmpLoop(const ExecutionPartConstruct &x);
const DoConstruct *GetDoConstruct(const ExecutionPartConstruct &x);

namespace detail {
struct OmpObjectListScope {
  template <typename T> static const OmpObjectList *Get(const T &x) {
    if constexpr (std::is_same_v<OmpObjectList, T>) {
      return &x;
    } else if constexpr (WrapperTrait<T>) {
      return Get(x.v);
    } else if constexpr (UnionTrait<T>) {
      return std::visit([](auto &&s) { return Get(s); }, x.u);
    } else if constexpr (TupleTrait<T>) {
      return GetFromTuple(
          x.t, std::make_index_sequence<std::tuple_size_v<decltype(x.t)>>{});
    } else if constexpr (ConstraintTrait<T>) {
      return Get(x.thing);
    } else {
      return nullptr;
    }
  }

  template <typename T>
  static const OmpObjectList *Get(const common::Indirection<T> &x) {
    return Get(x.value());
  }

  template <typename... Ts, size_t... Is>
  static const OmpObjectList *GetFromTuple(
      const std::tuple<Ts...> &t, std::index_sequence<Is...>) {
    const OmpObjectList *objects{nullptr};
    ((objects = objects ? objects : Get(std::get<Is>(t))), ...);
    return objects;
  }
};
} // namespace detail

template <typename T> const OmpObjectList *GetOmpObjectList(const T &clause) {
  static_assert(std::is_class_v<T>, "Unexpected argument type");
  return detail::OmpObjectListScope::Get(clause);
}

template <typename T>
const T *GetFirstArgument(const OmpDirectiveSpecification &spec) {
  for (const OmpArgument &arg : spec.Arguments().v) {
    if (auto *t{std::get_if<T>(&arg.u)}) {
      return t;
    }
  }
  return nullptr;
}

const OmpClause *FindClause(
    const OmpDirectiveSpecification &spec, llvm::omp::Clause clauseId);

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

template <typename R, typename = void, typename = void> struct is_range {
  static constexpr bool value{false};
};

template <typename R>
struct is_range<R, //
    std::void_t<decltype(std::declval<R>().begin())>,
    std::void_t<decltype(std::declval<R>().end())>> {
  static constexpr bool value{true};
};

template <typename R> constexpr bool is_range_v = is_range<R>::value;

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

  // An iterator range with a third iterator indicating a position inside
  // the range.
  struct IteratorGauge : public IteratorRange {
    IteratorGauge(IteratorType b, IteratorType e)
        : IteratorRange(b, e), at(b) {}
    IteratorGauge(IteratorRange r) : IteratorRange(r), at(r.begin()) {}

    bool atEnd() const { return at == end(); }
    IteratorType at;
  };

  struct Construct {
    Construct(IteratorType b, IteratorType e, const ExecutionPartConstruct *c)
        : location(b, e), owner(c) {}
    template <typename R>
    Construct(const R &r, const ExecutionPartConstruct *c)
        : location(r), owner(c) {}
    Construct(const Construct &c) = default;
    // The original range of the construct with the current position in it.
    // The location.at is the construct currently being pointed at, or
    // stepped into.
    IteratorGauge location;
    const ExecutionPartConstruct *owner;
  };

  ExecutionPartIterator() = default;

  ExecutionPartIterator(IteratorType b, IteratorType e, Step s = Step::Default,
      const ExecutionPartConstruct *c = nullptr)
      : stepping_(s) {
    stack_.emplace_back(b, e, c);
    adjust();
  }
  template <typename R, typename = std::enable_if_t<is_range_v<R>>>
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

  const std::vector<Construct> &stack() const { return stack_; }
  decltype(auto) operator*() const { return *at(); }
  bool operator==(const ExecutionPartIterator &other) const {
    if (valid() != other.valid()) {
      return false;
    }
    // Invalid iterators are considered equal.
    return !valid() ||
        stack_.back().location.at == other.stack_.back().location.at;
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
  IteratorType at() const { return stack_.back().location.at; };

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
  template <typename R, typename = std::enable_if_t<is_range_v<R>>>
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
  template <typename R, typename = std::enable_if_t<is_range_v<R>>>
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
