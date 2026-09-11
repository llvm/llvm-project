//===- TestFixture.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_TESTFIXTURE_H
#define LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_TESTFIXTURE_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/MultiArchSharedLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/MultiArchStaticLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/StaticLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "gtest/gtest.h"
#include <iosfwd>

namespace clang::ssaf {

class TestFixture : public ::testing::Test {
protected:
  static WPASuite makeWPASuite() { return WPASuite(); }

#define FIELD(CLASS, FIELD_NAME)                                               \
  static const auto &get##FIELD_NAME(const CLASS &X) { return X.FIELD_NAME; }  \
  static auto &get##FIELD_NAME(CLASS &X) { return X.FIELD_NAME; }
#include "clang/ScalableStaticAnalysis/Core/Model/PrivateFieldNames.def"
};

void PrintTo(const BuildNamespace &BN, std::ostream *OS);
void PrintTo(const EntityId &E, std::ostream *OS);
void PrintTo(const EntityLinkage &EL, std::ostream *OS);
void PrintTo(const EntityName &EN, std::ostream *OS);
void PrintTo(const NestedBuildNamespace &NBN, std::ostream *OS);
void PrintTo(const SummaryName &N, std::ostream *OS);

} // namespace clang::ssaf

#endif // LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_TESTFIXTURE_H
