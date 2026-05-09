//===- UnsafeBufferUsageTest.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions and data structures that help UnsafeBufferUsage unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGETEST_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGETEST_H

#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/EntitySummary.h"
#include "llvm/Support/Error.h"

namespace clang::ssaf {

llvm::Expected<std::unique_ptr<EntitySummary>> serializeDeserializeRoundTrip(
    const UnsafeBufferUsageEntitySummary &S,
    std::function<uint64_t(EntityId)> IdToIntFn,
    std::function<llvm::Expected<EntityId>(uint64_t)> IdFromIntFn);

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_UNSAFEBUFFERUSAGE_UNSAFEBUFFERUSAGETEST_H
