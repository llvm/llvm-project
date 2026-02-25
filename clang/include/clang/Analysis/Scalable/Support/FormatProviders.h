//===- FormatProviders.h - llvm::formatv support for SSAF model types -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides llvm::format_provider specialisations for SSAF model
// types, enabling them to be used directly with llvm::formatv and ErrorBuilder.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_SUPPORT_FORMATPROVIDERS_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_SUPPORT_FORMATPROVIDERS_H

#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "llvm/Support/FormatProviders.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

template <> struct format_provider<clang::ssaf::EntityId> {
  static void format(const clang::ssaf::EntityId &Val, raw_ostream &OS,
                     StringRef Style) {
    OS << Val;
  }
};

template <> struct format_provider<clang::ssaf::EntityLinkage> {
  static void format(clang::ssaf::EntityLinkage Val, raw_ostream &OS,
                     StringRef Style) {
    OS << Val;
  }
};

template <> struct format_provider<clang::ssaf::BuildNamespace> {
  static void format(const clang::ssaf::BuildNamespace &Val, raw_ostream &OS,
                     StringRef Style) {
    OS << Val;
  }
};

template <> struct format_provider<clang::ssaf::NestedBuildNamespace> {
  static void format(const clang::ssaf::NestedBuildNamespace &Val,
                     raw_ostream &OS, StringRef Style) {
    OS << Val;
  }
};

template <> struct format_provider<clang::ssaf::EntityName> {
  static void format(const clang::ssaf::EntityName &Val, raw_ostream &OS,
                     StringRef Style) {
    OS << Val;
  }
};

template <> struct format_provider<clang::ssaf::SummaryName> {
  static void format(const clang::ssaf::SummaryName &Val, raw_ostream &OS,
                     StringRef Style) {
    OS << Val;
  }
};

} // namespace llvm

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_SUPPORT_FORMATPROVIDERS_H
