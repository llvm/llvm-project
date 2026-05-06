//===- UnsafeBufferUsage.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"

using namespace clang;
using namespace ssaf;

UnsafeBufferUsageEntitySummary
ssaf::buildUnsafeBufferUsageEntitySummary(EntityPointerLevelSet UnsafeBuffers) {
  return UnsafeBufferUsageEntitySummary(std::move(UnsafeBuffers));
}

llvm::iterator_range<EntityPointerLevelSet::const_iterator>
ssaf::getUnsafeBuffers(const UnsafeBufferUsageEntitySummary &S) {
  return llvm::make_range(S.UnsafeBuffers.begin(), S.UnsafeBuffers.end());
}
