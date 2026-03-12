//===- TestFixture.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_TESTFIXTURE_H
#define LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_TESTFIXTURE_H

#include "clang/Analysis/Scalable/EntityLinker/LUSummary.h"
#include "clang/Analysis/Scalable/EntityLinker/LUSummaryEncoding.h"
#include "clang/Analysis/Scalable/EntityLinker/TUSummaryEncoding.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "gtest/gtest.h"
#include <iosfwd>

namespace clang::ssaf {

class TestFixture : public ::testing::Test {
protected:
#define FIELD(CLASS, FIELD_NAME)                                               \
  static const auto &get##FIELD_NAME(const CLASS &X) { return X.FIELD_NAME; }  \
  static auto &get##FIELD_NAME(CLASS &X) { return X.FIELD_NAME; }
#include "clang/Analysis/Scalable/Model/PrivateFieldNames.def"
};

void PrintTo(const BuildNamespace &BN, std::ostream *OS);
void PrintTo(const EntityId &E, std::ostream *OS);
void PrintTo(const EntityLinkage &EL, std::ostream *OS);
void PrintTo(const EntityName &EN, std::ostream *OS);
void PrintTo(const NestedBuildNamespace &NBN, std::ostream *OS);
void PrintTo(const SummaryName &N, std::ostream *OS);

} // namespace clang::ssaf

#endif // LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_TESTFIXTURE_H
