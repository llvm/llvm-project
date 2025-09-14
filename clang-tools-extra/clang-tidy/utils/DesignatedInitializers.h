//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utilities for designated initializers.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/Expr.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"

namespace clang::tidy::utils {

/// Get designators describing the elements of a (syntactic) init list.
///
/// Given for example the type
/// \code
/// struct S { int i, j; };
/// \endcode
/// and the definition
/// \code
///  S s{1, 2};
/// \endcode
/// calling `getUnwrittenDesignators` for the initializer list expression
/// `{1, 2}` would produce the map `{loc(1): ".i", loc(2): ".j"}`.
///
/// It does not produce designators for any explicitly-written nested lists,
/// e.g. `{1, .j=2}` would only return `{loc(1): ".i"}`.
///
/// It also considers structs with fields of record types like
/// `struct T { S s; };`. In this case, there would be designators of the
/// form `.s.i` and `.s.j` in the returned map.
llvm::DenseMap<clang::SourceLocation, std::string>
getUnwrittenDesignators(const clang::InitListExpr *Syn);

} // namespace clang::tidy::utils
