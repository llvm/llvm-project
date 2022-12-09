//===-- include/flang/Semantics/runtime-type-info.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// BuildRuntimeDerivedTypeTables() translates the scopes of derived types
// and parameterized derived type instantiations into the type descriptions
// defined in module/__fortran_type_info.f90, packaging these descriptions
// as static initializers for compiler-created objects.

#ifndef FORTRAN_SEMANTICS_RUNTIME_TYPE_INFO_H_
#define FORTRAN_SEMANTICS_RUNTIME_TYPE_INFO_H_

#include <set>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace Fortran::semantics {
class Scope;
class SemanticsContext;
class Symbol;

struct RuntimeDerivedTypeTables {
  Scope *schemata{nullptr};
  std::set<std::string> names;
};

RuntimeDerivedTypeTables BuildRuntimeDerivedTypeTables(SemanticsContext &);

/// Name of the builtin module that defines builtin derived types meant
/// to describe other derived types at runtime in flang descriptor.
constexpr char typeInfoBuiltinModule[]{"__fortran_type_info"};

std::vector<const Symbol *> CollectBindings(const Scope &dtScope);

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_RUNTIME_TYPE_INFO_H_
