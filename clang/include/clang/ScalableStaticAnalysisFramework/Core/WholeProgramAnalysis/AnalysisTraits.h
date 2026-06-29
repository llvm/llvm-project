//===- AnalysisTraits.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type traits for AnalysisResult subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISTRAITS_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISTRAITS_H

#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include <type_traits>

namespace clang::ssaf {

/// Type trait that checks whether \p T has a static \c analysisName() method
/// returning \c AnalysisName. Used to enforce the convention on AnalysisResult
/// subclasses and analysis classes at instantiation time.
template <typename T, typename = void>
struct HasAnalysisName : std::false_type {};

template <typename T>
struct HasAnalysisName<T, std::void_t<decltype(T::analysisName())>>
    : std::is_same<decltype(T::analysisName()), AnalysisName> {};

template <typename T>
inline constexpr bool HasAnalysisName_v = HasAnalysisName<T>::value;

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISTRAITS_H
