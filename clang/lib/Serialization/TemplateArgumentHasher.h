//===- TemplateArgumentHasher.h - Hash Template Arguments -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/TemplateBase.h"

namespace clang {
namespace serialization {

/// Calculate a stable hash value for template arguments. We guarantee that
/// the same template arguments must have the same hashed values. But we don't
/// guarantee that the template arguments with the same hashed value are the
/// same template arguments.
///
/// ODR hashing may not be the best mechanism to hash the template
/// arguments. ODR hashing is (or perhaps, should be) about determining whether
/// two things are spelled the same way and have the same meaning (as required
/// by the C++ ODR), whereas what we want here is whether they have the same
/// meaning regardless of spelling. Maybe we can get away with reusing ODR
/// hashing anyway, on the basis that any canonical, non-dependent template
/// argument should have the same (invented) spelling in every translation
/// unit, but it is not sure that's true in all cases. There may still be cases
/// where the canonical type includes some aspect of "whatever we saw first",
/// in which case the ODR hash can differ across translation units for
/// non-dependent, canonical template arguments that are spelled differently
/// but have the same meaning. But it is not easy to raise examples.
unsigned StableHashForTemplateArguments(llvm::ArrayRef<TemplateArgument> Args);

} // namespace serialization
} // namespace clang
