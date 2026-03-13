//===- SummaryDataTraits.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type traits for SummaryData subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATATRAITS_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATATRAITS_H

#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include <type_traits>

namespace clang::ssaf {

/// Type trait that checks whether \p T has a static \c summaryName() method
/// returning \c SummaryName. Used to enforce the convention on SummaryData
/// subclasses at instantiation time.
///
/// The expression \c T::summaryName() is only well-formed for static methods —
/// calling a non-static member without an object is ill-formed and causes the
/// partial specialization to be discarded via SFINAE, so non-static overloads
/// are correctly rejected.
template <typename T, typename = void>
struct HasSummaryName : std::false_type {};

template <typename T>
struct HasSummaryName<T, std::void_t<decltype(T::summaryName())>>
    : std::is_same<decltype(T::summaryName()), SummaryName> {};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATATRAITS_H
