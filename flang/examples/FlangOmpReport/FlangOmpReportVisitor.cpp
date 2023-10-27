//===-- examples/flang-omp-report-plugin/flang-omp-report-visitor.cpp -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FlangOmpReportVisitor.h"
#include "llvm/ADT/StringExtras.h"

namespace Fortran {
namespace parser {
bool operator<(const ClauseInfo &a, const ClauseInfo &b) {
  return a.clause < b.clause;
}
bool operator==(const ClauseInfo &a, const ClauseInfo &b) {
  return a.clause == b.clause && a.clauseDetails == b.clauseDetails;
}
bool operator!=(const ClauseInfo &a, const ClauseInfo &b) { return !(a == b); }

bool operator==(const LogRecord &a, const LogRecord &b) {
  return a.file == b.file && a.line == b.line && a.construct == b.construct &&
      a.clauses == b.clauses;
}
bool operator!=(const LogRecord &a, const LogRecord &b) { return !(a == b); }

std::string OpenMPCounterVisitor::normalize_construct_name(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
      [](unsigned char c) { return llvm::toLower(c); });
  return s;
}
ClauseInfo OpenMPCounterVisitor::normalize_clause_name(
    const llvm::StringRef s) {
  std::size_t start = s.find('(');
  std::size_t end = s.find(')');
  std::string clauseName;
  if (start != llvm::StringRef::npos && end != llvm::StringRef::npos) {
    clauseName = s.substr(0, start);
    clauseDetails = s.substr(start + 1, end - start - 1);
  } else {
    clauseName = s;
  }
  std::transform(clauseName.begin(), clauseName.end(), clauseName.begin(),
      [](unsigned char c) { return llvm::toLower(c); });
  std::transform(clauseDetails.begin(), clauseDetails.end(),
      clauseDetails.begin(), [](unsigned char c) { return llvm::toLower(c); });
  return ClauseInfo{clauseName, clauseDetails};
}
SourcePosition OpenMPCounterVisitor::getLocation(const OmpWrapperType &w) {
  if (auto *val = std::get_if<const OpenMPConstruct *>(&w)) {
    const OpenMPConstruct *o{*val};
    return getLocation(*o);
  }
  return getLocation(*std::get<const OpenMPDeclarativeConstruct *>(w));
}
SourcePosition OpenMPCounterVisitor::getLocation(
    const OpenMPDeclarativeConstruct &c) {
  return std::visit(
      [&](const auto &o) -> SourcePosition {
        return parsing->allCooked().GetSourcePositionRange(o.source)->first;
      },
      c.u);
}
SourcePosition OpenMPCounterVisitor::getLocation(const OpenMPConstruct &c) {
  return std::visit(
      Fortran::common::visitors{
          [&](const OpenMPStandaloneConstruct &c) -> SourcePosition {
            return parsing->allCooked().GetSourcePositionRange(c.source)->first;
          },
          // OpenMPSectionsConstruct, OpenMPLoopConstruct,
          // OpenMPBlockConstruct, OpenMPCriticalConstruct Get the source from
          // the directive field.
          [&](const auto &c) -> SourcePosition {
            const CharBlock &source{std::get<0>(c.t).source};
            return (parsing->allCooked().GetSourcePositionRange(source))->first;
          },
          [&](const OpenMPAtomicConstruct &c) -> SourcePosition {
            return std::visit(
                [&](const auto &o) -> SourcePosition {
                  const CharBlock &source{std::get<Verbatim>(o.t).source};
                  return parsing->allCooked()
                      .GetSourcePositionRange(source)
                      ->first;
                },
                c.u);
          },
          [&](const OpenMPSectionConstruct &c) -> SourcePosition {
            const CharBlock &source{c.source};
            return (parsing->allCooked().GetSourcePositionRange(source))->first;
          },
      },
      c.u);
}

std::string OpenMPCounterVisitor::getName(const OmpWrapperType &w) {
  if (auto *val = std::get_if<const OpenMPConstruct *>(&w)) {
    const OpenMPConstruct *o{*val};
    return getName(*o);
  }
  return getName(*std::get<const OpenMPDeclarativeConstruct *>(w));
}
std::string OpenMPCounterVisitor::getName(const OpenMPDeclarativeConstruct &c) {
  return std::visit(
      [&](const auto &o) -> std::string {
        const CharBlock &source{std::get<Verbatim>(o.t).source};
        return normalize_construct_name(source.ToString());
      },
      c.u);
}
std::string OpenMPCounterVisitor::getName(const OpenMPConstruct &c) {
  return std::visit(
      Fortran::common::visitors{
          [&](const OpenMPStandaloneConstruct &c) -> std::string {
            return std::visit(
                [&](const auto &c) {
                  // Get source from the directive or verbatim fields
                  const CharBlock &source{std::get<0>(c.t).source};
                  return normalize_construct_name(source.ToString());
                },
                c.u);
          },
          [&](const OpenMPExecutableAllocate &c) -> std::string {
            const CharBlock &source{std::get<0>(c.t).source};
            return normalize_construct_name(source.ToString());
          },
          [&](const OpenMPDeclarativeAllocate &c) -> std::string {
            const CharBlock &source{std::get<0>(c.t).source};
            return normalize_construct_name(source.ToString());
          },
          [&](const OpenMPAllocatorsConstruct &c) -> std::string {
            const CharBlock &source{std::get<0>(c.t).source};
            return normalize_construct_name(source.ToString());
          },
          [&](const OpenMPAtomicConstruct &c) -> std::string {
            return std::visit(
                [&](const auto &c) {
                  // Get source from the verbatim fields
                  const CharBlock &source{std::get<Verbatim>(c.t).source};
                  return "atomic-" +
                      normalize_construct_name(source.ToString());
                },
                c.u);
          },
          [&](const OpenMPSectionConstruct &c) -> std::string {
            return "section";
          },
          // OpenMPSectionsConstruct, OpenMPLoopConstruct,
          // OpenMPBlockConstruct, OpenMPCriticalConstruct Get the source from
          // the directive field of the begin directive or from the verbatim
          // field of the begin directive in Critical
          [&](const auto &c) -> std::string {
            const CharBlock &source{std::get<0>(std::get<0>(c.t).t).source};
            return normalize_construct_name(source.ToString());
          },
      },
      c.u);
}

bool OpenMPCounterVisitor::Pre(const OpenMPDeclarativeConstruct &c) {
  OmpWrapperType *ow{new OmpWrapperType(&c)};
  ompWrapperStack.push_back(ow);
  return true;
}
bool OpenMPCounterVisitor::Pre(const OpenMPConstruct &c) {
  OmpWrapperType *ow{new OmpWrapperType(&c)};
  ompWrapperStack.push_back(ow);
  return true;
}

void OpenMPCounterVisitor::Post(const OpenMPDeclarativeConstruct &) {
  PostConstructsCommon();
}
void OpenMPCounterVisitor::Post(const OpenMPConstruct &) {
  PostConstructsCommon();
}
void OpenMPCounterVisitor::PostConstructsCommon() {
  OmpWrapperType *curConstruct = ompWrapperStack.back();
  std::sort(
      clauseStrings[curConstruct].begin(), clauseStrings[curConstruct].end());

  SourcePosition s{getLocation(*curConstruct)};
  LogRecord r{
      s.path, s.line, getName(*curConstruct), clauseStrings[curConstruct]};
  constructClauses.push_back(r);

  auto it = clauseStrings.find(curConstruct);
  clauseStrings.erase(it);
  ompWrapperStack.pop_back();
  delete curConstruct;
}

void OpenMPCounterVisitor::Post(const OmpProcBindClause::Type &c) {
  clauseDetails +=
      "type=" + std::string{OmpProcBindClause::EnumToString(c)} + ";";
}
void OpenMPCounterVisitor::Post(const OmpDefaultClause::Type &c) {
  clauseDetails +=
      "type=" + std::string{OmpDefaultClause::EnumToString(c)} + ";";
}
void OpenMPCounterVisitor::Post(const OmpDeviceTypeClause::Type &c) {
  clauseDetails +=
      "type=" + std::string{OmpDeviceTypeClause::EnumToString(c)} + ";";
}
void OpenMPCounterVisitor::Post(
    const OmpDefaultmapClause::ImplicitBehavior &c) {
  clauseDetails +=
      "implicit_behavior=" + std::string{OmpDefaultmapClause::EnumToString(c)} +
      ";";
}
void OpenMPCounterVisitor::Post(
    const OmpDefaultmapClause::VariableCategory &c) {
  clauseDetails +=
      "variable_category=" + std::string{OmpDefaultmapClause::EnumToString(c)} +
      ";";
}
void OpenMPCounterVisitor::Post(const OmpScheduleModifierType::ModType &c) {
  clauseDetails +=
      "modifier=" + std::string{OmpScheduleModifierType::EnumToString(c)} + ";";
}
void OpenMPCounterVisitor::Post(const OmpLinearModifier::Type &c) {
  clauseDetails +=
      "modifier=" + std::string{OmpLinearModifier::EnumToString(c)} + ";";
}
void OpenMPCounterVisitor::Post(const OmpDependenceType::Type &c) {
  clauseDetails +=
      "type=" + std::string{OmpDependenceType::EnumToString(c)} + ";";
}
void OpenMPCounterVisitor::Post(const OmpMapType::Type &c) {
  clauseDetails += "type=" + std::string{OmpMapType::EnumToString(c)} + ";";
}
void OpenMPCounterVisitor::Post(const OmpScheduleClause::ScheduleType &c) {
  clauseDetails +=
      "type=" + std::string{OmpScheduleClause::EnumToString(c)} + ";";
}
void OpenMPCounterVisitor::Post(const OmpIfClause::DirectiveNameModifier &c) {
  clauseDetails +=
      "name_modifier=" + std::string{OmpIfClause::EnumToString(c)} + ";";
}
void OpenMPCounterVisitor::Post(const OmpCancelType::Type &c) {
  clauseDetails += "type=" + std::string{OmpCancelType::EnumToString(c)} + ";";
}
void OpenMPCounterVisitor::Post(const OmpClause &c) {
  PostClauseCommon(normalize_clause_name(c.source.ToString()));
  clauseDetails.clear();
}
void OpenMPCounterVisitor::PostClauseCommon(const ClauseInfo &ci) {
  assert(
      !ompWrapperStack.empty() && "Construct should be visited before clause");
  clauseStrings[ompWrapperStack.back()].push_back(ci);
}
} // namespace parser
} // namespace Fortran
