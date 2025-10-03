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
#include "flang/Parser/parse-tree.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace Fortran::parser::omp {

namespace detail {
using D = llvm::omp::Directive;

template <typename Construct> //
struct ConstructId {
  static constexpr llvm::omp::Directive id{D::OMPD_unknown};
};

#define MAKE_CONSTR_ID(Construct, Id) \
  template <> struct ConstructId<Construct> { \
    static constexpr llvm::omp::Directive id{Id}; \
  }

MAKE_CONSTR_ID(OpenMPDeclarativeAllocate, D::OMPD_allocate);
MAKE_CONSTR_ID(OpenMPExecutableAllocate, D::OMPD_allocate);

#undef MAKE_CONSTR_ID

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
      if constexpr (std::is_same_v<T, OpenMPCancelConstruct> ||
          std::is_same_v<T, OpenMPCancellationPointConstruct> ||
          std::is_same_v<T, OpenMPDepobjConstruct> ||
          std::is_same_v<T, OpenMPFlushConstruct> ||
          std::is_same_v<T, OpenMPInteropConstruct> ||
          std::is_same_v<T, OpenMPSimpleStandaloneConstruct> ||
          std::is_same_v<T, OpenMPGroupprivate>) {
        return x.v.DirName();
      } else {
        return GetOmpDirectiveName(x.v);
      }
    } else if constexpr (TupleTrait<T>) {
      if constexpr (std::is_base_of_v<OmpBlockConstruct, T>) {
        return std::get<OmpBeginDirective>(x.t).DirName();
      } else if constexpr (std::is_same_v<T, OpenMPDeclarativeAllocate> ||
          std::is_same_v<T, OpenMPExecutableAllocate>) {
        return MakeName(std::get<Verbatim>(x.t).source, ConstructId<T>::id);
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

const OmpObjectList *GetOmpObjectList(const OmpClause &clause);
const BlockConstruct *GetFortranBlockConstruct(
    const ExecutionPartConstruct &epc);

} // namespace Fortran::parser::omp

#endif // FORTRAN_PARSER_OPENMP_UTILS_H
