//===-- examples/flang-omp-report-plugin/flang-omp-report-visitor.h -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_FLANG_OMP_REPORT_VISITOR_H
#define FORTRAN_FLANG_OMP_REPORT_VISITOR_H

#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/parsing.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace Fortran {
namespace parser {
struct ClauseInfo {
  std::string clause;
  std::string clauseDetails;
  ClauseInfo() {}
  ClauseInfo(const std::string &c, const std::string &cd)
      : clause{c}, clauseDetails{cd} {}
  ClauseInfo(const std::pair<std::string, std::string> &p)
      : clause{std::get<0>(p)}, clauseDetails{std::get<1>(p)} {}
};
bool operator<(const ClauseInfo &a, const ClauseInfo &b);
bool operator==(const ClauseInfo &a, const ClauseInfo &b);
bool operator!=(const ClauseInfo &a, const ClauseInfo &b);

struct LogRecord {
  std::string file;
  int line;
  std::string construct;
  llvm::SmallVector<ClauseInfo> clauses;
};
bool operator==(const LogRecord &a, const LogRecord &b);
bool operator!=(const LogRecord &a, const LogRecord &b);

using OmpWrapperType =
    std::variant<const OpenMPConstruct *, const OpenMPDeclarativeConstruct *>;

struct OpenMPCounterVisitor {
  std::string normalize_construct_name(std::string s);
  ClauseInfo normalize_clause_name(const llvm::StringRef s);
  SourcePosition getLocation(const OmpWrapperType &w);
  SourcePosition getLocation(const OpenMPDeclarativeConstruct &c);
  SourcePosition getLocation(const OpenMPConstruct &c);

  std::string getName(const OmpWrapperType &w);
  std::string getName(const OpenMPDeclarativeConstruct &c);
  std::string getName(const OpenMPConstruct &c);

  template <typename A> bool Pre(const A &) { return true; }
  template <typename A> void Post(const A &) {}
  bool Pre(const OpenMPDeclarativeConstruct &c);
  bool Pre(const OpenMPConstruct &c);

  void Post(const OpenMPDeclarativeConstruct &);
  void Post(const OpenMPConstruct &);
  void PostConstructsCommon();

  void Post(const OmpProcBindClause::Type &c);
  void Post(const OmpDefaultClause::Type &c);
  void Post(const OmpDefaultmapClause::ImplicitBehavior &c);
  void Post(const OmpDefaultmapClause::VariableCategory &c);
  void Post(const OmpScheduleModifierType::ModType &c);
  void Post(const OmpLinearModifier::Type &c);
  void Post(const OmpDependenceType::Type &c);
  void Post(const OmpMapType::Type &c);
  void Post(const OmpScheduleClause::ScheduleType &c);
  void Post(const OmpIfClause::DirectiveNameModifier &c);
  void Post(const OmpCancelType::Type &c);
  void Post(const OmpClause &c);
  void PostClauseCommon(const ClauseInfo &ci);

  std::string clauseDetails{""};
  llvm::SmallVector<LogRecord> constructClauses;
  llvm::SmallVector<OmpWrapperType *> ompWrapperStack;
  llvm::DenseMap<OmpWrapperType *, llvm::SmallVector<ClauseInfo>> clauseStrings;
  Parsing *parsing{nullptr};
};
} // namespace parser
} // namespace Fortran

#endif /* FORTRAN_FLANG_OMP_REPORT_VISITOR_H */
