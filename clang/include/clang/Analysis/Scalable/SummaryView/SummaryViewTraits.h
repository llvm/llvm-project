//===- SummaryViewTraits.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type traits for SummaryView subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWTRAITS_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWTRAITS_H

#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include <type_traits>

namespace clang::ssaf {

/// Type trait that checks whether \p T has a static summaryName() method
/// returning SummaryName. Used to enforce the convention on SummaryView
/// subclasses at instantiation time.
template <typename T, typename = void>
struct HasSummaryName : std::false_type {};

template <typename T>
struct HasSummaryName<T, std::void_t<decltype(T::summaryName())>>
    : std::is_same<decltype(T::summaryName()), SummaryName> {};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWTRAITS_H
