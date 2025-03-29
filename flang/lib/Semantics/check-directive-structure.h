//===-- lib/Semantics/check-directive-structure.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Directive structure validity checks common to OpenMP, OpenACC and other
// directive language.

#ifndef FORTRAN_SEMANTICS_CHECK_DIRECTIVE_STRUCTURE_H_
#define FORTRAN_SEMANTICS_CHECK_DIRECTIVE_STRUCTURE_H_

#include "flang/Common/enum-set.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/iterator_range.h"

#include <unordered_map>

namespace Fortran::semantics {

template <typename C, std::size_t ClauseEnumSize> struct DirectiveClauses {
  const common::EnumSet<C, ClauseEnumSize> allowed;
  const common::EnumSet<C, ClauseEnumSize> allowedOnce;
  const common::EnumSet<C, ClauseEnumSize> allowedExclusive;
  const common::EnumSet<C, ClauseEnumSize> requiredOneOf;
};

// Generic branching checker for invalid branching out of OpenMP/OpenACC
// directive.
// typename D is the directive enumeration.
template <typename D> class NoBranchingEnforce {
public:
  NoBranchingEnforce(SemanticsContext &context,
      parser::CharBlock sourcePosition, D directive,
      std::string &&upperCaseDirName)
      : context_{context}, sourcePosition_{sourcePosition},
        upperCaseDirName_{std::move(upperCaseDirName)},
        currentDirective_{directive}, numDoConstruct_{0} {}
  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  template <typename T> bool Pre(const parser::Statement<T> &statement) {
    currentStatementSourcePosition_ = statement.source;
    return true;
  }

  bool Pre(const parser::DoConstruct &) {
    numDoConstruct_++;
    return true;
  }
  void Post(const parser::DoConstruct &) { numDoConstruct_--; }
  void Post(const parser::ReturnStmt &) { EmitBranchOutError("RETURN"); }
  void Post(const parser::ExitStmt &exitStmt) {
    if (const auto &exitName{exitStmt.v}) {
      CheckConstructNameBranching("EXIT", exitName.value());
    } else {
      CheckConstructNameBranching("EXIT");
    }
  }
  void Post(const parser::CycleStmt &cycleStmt) {
    if (const auto &cycleName{cycleStmt.v}) {
      CheckConstructNameBranching("CYCLE", cycleName.value());
    } else {
      if constexpr (std::is_same_v<D, llvm::omp::Directive>) {
        switch ((llvm::omp::Directive)currentDirective_) {
        // exclude directives which do not need a check for unlabelled CYCLES
        case llvm::omp::Directive::OMPD_do:
        case llvm::omp::Directive::OMPD_simd:
        case llvm::omp::Directive::OMPD_parallel_do:
        case llvm::omp::Directive::OMPD_parallel_do_simd:
        case llvm::omp::Directive::OMPD_distribute_parallel_do:
        case llvm::omp::Directive::OMPD_distribute_parallel_do_simd:
        case llvm::omp::Directive::OMPD_distribute_parallel_for:
        case llvm::omp::Directive::OMPD_distribute_simd:
        case llvm::omp::Directive::OMPD_distribute_parallel_for_simd:
        case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do:
        case llvm::omp::Directive::
            OMPD_target_teams_distribute_parallel_do_simd:
          return;
        default:
          break;
        }
      } else if constexpr (std::is_same_v<D, llvm::acc::Directive>) {
        switch ((llvm::acc::Directive)currentDirective_) {
        // exclude loop directives which do not need a check for unlabelled
        // CYCLES
        case llvm::acc::Directive::ACCD_loop:
        case llvm::acc::Directive::ACCD_kernels_loop:
        case llvm::acc::Directive::ACCD_parallel_loop:
        case llvm::acc::Directive::ACCD_serial_loop:
          return;
        default:
          break;
        }
      }
      CheckConstructNameBranching("CYCLE");
    }
  }

private:
  parser::MessageFormattedText GetEnclosingMsg() const {
    return {"Enclosing %s construct"_en_US, upperCaseDirName_};
  }

  void EmitBranchOutError(const char *stmt) const {
    context_
        .Say(currentStatementSourcePosition_,
            "%s statement is not allowed in a %s construct"_err_en_US, stmt,
            upperCaseDirName_)
        .Attach(sourcePosition_, GetEnclosingMsg());
  }

  inline void EmitUnlabelledBranchOutError(const char *stmt) {
    context_
        .Say(currentStatementSourcePosition_,
            "%s to construct outside of %s construct is not allowed"_err_en_US,
            stmt, upperCaseDirName_)
        .Attach(sourcePosition_, GetEnclosingMsg());
  }

  void EmitBranchOutErrorWithName(
      const char *stmt, const parser::Name &toName) const {
    const std::string branchingToName{toName.ToString()};
    context_
        .Say(currentStatementSourcePosition_,
            "%s to construct '%s' outside of %s construct is not allowed"_err_en_US,
            stmt, branchingToName, upperCaseDirName_)
        .Attach(sourcePosition_, GetEnclosingMsg());
  }

  // Current semantic checker is not following OpenACC/OpenMP constructs as they
  // are not Fortran constructs. Hence the ConstructStack doesn't capture
  // OpenACC/OpenMP constructs. Apply an inverse way to figure out if a
  // construct-name is branching out of an OpenACC/OpenMP construct. The control
  // flow goes out of an OpenACC/OpenMP construct, if a construct-name from
  // statement is found in ConstructStack.
  void CheckConstructNameBranching(
      const char *stmt, const parser::Name &stmtName) {
    const ConstructStack &stack{context_.constructStack()};
    for (auto iter{stack.cend()}; iter-- != stack.cbegin();) {
      const ConstructNode &construct{*iter};
      const auto &constructName{MaybeGetNodeName(construct)};
      if (constructName) {
        if (stmtName.source == constructName->source) {
          EmitBranchOutErrorWithName(stmt, stmtName);
          return;
        }
      }
    }
  }

  // Check branching for unlabelled CYCLES and EXITs
  void CheckConstructNameBranching(const char *stmt) {
    // found an enclosing looping construct for the unlabelled EXIT/CYCLE
    if (numDoConstruct_ > 0) {
      return;
    }
    // did not found an enclosing looping construct within the OpenMP/OpenACC
    // directive
    EmitUnlabelledBranchOutError(stmt);
  }

  SemanticsContext &context_;
  parser::CharBlock currentStatementSourcePosition_;
  parser::CharBlock sourcePosition_;
  std::string upperCaseDirName_;
  D currentDirective_;
  int numDoConstruct_; // tracks number of DoConstruct found AFTER encountering
                       // an OpenMP/OpenACC directive
};

// Generic structure checker for directives/clauses language such as OpenMP
// and OpenACC.
// typename D is the directive enumeration.
// typename C is the clause enumeration.
// typename PC is the parser class defined in parse-tree.h for the clauses.
template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
class DirectiveStructureChecker : public virtual BaseChecker {
protected:
  DirectiveStructureChecker(SemanticsContext &context,
      const std::unordered_map<D, DirectiveClauses<C, ClauseEnumSize>>
          &directiveClausesMap)
      : context_{context}, directiveClausesMap_(directiveClausesMap) {}
  virtual ~DirectiveStructureChecker() {}

  using ClauseMapTy = std::multimap<C, const PC *>;
  struct DirectiveContext {
    DirectiveContext(parser::CharBlock source, D d)
        : directiveSource{source}, directive{d} {}

    parser::CharBlock directiveSource{nullptr};
    parser::CharBlock clauseSource{nullptr};
    D directive;
    common::EnumSet<C, ClauseEnumSize> allowedClauses{};
    common::EnumSet<C, ClauseEnumSize> allowedOnceClauses{};
    common::EnumSet<C, ClauseEnumSize> allowedExclusiveClauses{};
    common::EnumSet<C, ClauseEnumSize> requiredClauses{};

    const PC *clause{nullptr};
    ClauseMapTy clauseInfo;
    std::list<C> actualClauses;
    std::list<C> crtGroup;
    Symbol *loopIV{nullptr};
  };

  void SetLoopIv(Symbol *symbol) { GetContext().loopIV = symbol; }

  // back() is the top of the stack
  DirectiveContext &GetContext() {
    CHECK(!dirContext_.empty());
    return dirContext_.back();
  }

  DirectiveContext &GetContextParent() {
    CHECK(dirContext_.size() >= 2);
    return dirContext_[dirContext_.size() - 2];
  }

  void SetContextClause(const PC &clause) {
    GetContext().clauseSource = clause.source;
    GetContext().clause = &clause;
  }

  void ResetPartialContext(const parser::CharBlock &source) {
    CHECK(!dirContext_.empty());
    SetContextDirectiveSource(source);
    GetContext().allowedClauses = {};
    GetContext().allowedOnceClauses = {};
    GetContext().allowedExclusiveClauses = {};
    GetContext().requiredClauses = {};
    GetContext().clauseInfo = {};
    GetContext().loopIV = {nullptr};
  }

  void SetContextDirectiveSource(const parser::CharBlock &directive) {
    GetContext().directiveSource = directive;
  }

  void SetContextDirectiveEnum(D dir) { GetContext().directive = dir; }

  void SetContextAllowed(const common::EnumSet<C, ClauseEnumSize> &allowed) {
    GetContext().allowedClauses = allowed;
  }

  void SetContextAllowedOnce(
      const common::EnumSet<C, ClauseEnumSize> &allowedOnce) {
    GetContext().allowedOnceClauses = allowedOnce;
  }

  void SetContextAllowedExclusive(
      const common::EnumSet<C, ClauseEnumSize> &allowedExclusive) {
    GetContext().allowedExclusiveClauses = allowedExclusive;
  }

  void SetContextRequired(const common::EnumSet<C, ClauseEnumSize> &required) {
    GetContext().requiredClauses = required;
  }

  void SetContextClauseInfo(C type) {
    GetContext().clauseInfo.emplace(type, GetContext().clause);
  }

  void AddClauseToCrtContext(C type) {
    GetContext().actualClauses.push_back(type);
  }

  void AddClauseToCrtGroupInContext(C type) {
    GetContext().crtGroup.push_back(type);
  }

  void ResetCrtGroup() { GetContext().crtGroup.clear(); }

  // Check if the given clause is present in the current context
  const PC *FindClause(C type) { return FindClause(GetContext(), type); }

  // Check if the given clause is present in the given context
  const PC *FindClause(DirectiveContext &context, C type) {
    auto it{context.clauseInfo.find(type)};
    if (it != context.clauseInfo.end()) {
      return it->second;
    }
    return nullptr;
  }

  // Check if the given clause is present in the parent context
  const PC *FindClauseParent(C type) {
    auto it{GetContextParent().clauseInfo.find(type)};
    if (it != GetContextParent().clauseInfo.end()) {
      return it->second;
    }
    return nullptr;
  }

  llvm::iterator_range<typename ClauseMapTy::iterator> FindClauses(C type) {
    auto it{GetContext().clauseInfo.equal_range(type)};
    return llvm::make_range(it);
  }

  DirectiveContext *GetEnclosingDirContext() {
    CHECK(!dirContext_.empty());
    auto it{dirContext_.rbegin()};
    if (++it != dirContext_.rend()) {
      return &(*it);
    }
    return nullptr;
  }

  void PushContext(const parser::CharBlock &source, D dir) {
    dirContext_.emplace_back(source, dir);
  }

  DirectiveContext *GetEnclosingContextWithDir(D dir) {
    CHECK(!dirContext_.empty());
    auto it{dirContext_.rbegin()};
    while (++it != dirContext_.rend()) {
      if (it->directive == dir) {
        return &(*it);
      }
    }
    return nullptr;
  }

  bool CurrentDirectiveIsNested() { return dirContext_.size() > 1; };

  void SetClauseSets(D dir) {
    dirContext_.back().allowedClauses = directiveClausesMap_[dir].allowed;
    dirContext_.back().allowedOnceClauses =
        directiveClausesMap_[dir].allowedOnce;
    dirContext_.back().allowedExclusiveClauses =
        directiveClausesMap_[dir].allowedExclusive;
    dirContext_.back().requiredClauses =
        directiveClausesMap_[dir].requiredOneOf;
  }
  void PushContextAndClauseSets(const parser::CharBlock &source, D dir) {
    PushContext(source, dir);
    SetClauseSets(dir);
  }

  void SayNotMatching(const parser::CharBlock &, const parser::CharBlock &);

  template <typename B> void CheckMatching(const B &beginDir, const B &endDir) {
    const auto &begin{beginDir.v};
    const auto &end{endDir.v};
    if (begin != end) {
      SayNotMatching(beginDir.source, endDir.source);
    }
  }
  // Check illegal branching out of `Parser::Block` for `Parser::Name` based
  // nodes (example `Parser::ExitStmt`)
  void CheckNoBranching(const parser::Block &block, D directive,
      const parser::CharBlock &directiveSource);

  // Check that only clauses in set are after the specific clauses.
  void CheckOnlyAllowedAfter(C clause, common::EnumSet<C, ClauseEnumSize> set);

  void CheckRequireAtLeastOneOf(bool warnInsteadOfError = false);

  // Check if a clause is allowed on a directive. Returns true if is and
  // false otherwise.
  bool CheckAllowed(C clause, bool warnInsteadOfError = false);

  // Check that the clause appears only once. The counter is reset when the
  // separator clause appears.
  void CheckAllowedOncePerGroup(C clause, C separator);

  void CheckMutuallyExclusivePerGroup(
      C clause, C separator, common::EnumSet<C, ClauseEnumSize> set);

  void CheckAtLeastOneClause();

  void CheckNotAllowedIfClause(
      C clause, common::EnumSet<C, ClauseEnumSize> set);

  std::string ContextDirectiveAsFortran();

  void RequiresConstantPositiveParameter(
      const C &clause, const parser::ScalarIntConstantExpr &i);

  void RequiresPositiveParameter(const C &clause,
      const parser::ScalarIntExpr &i, llvm::StringRef paramName = "parameter");

  void OptionalConstantPositiveParameter(
      const C &clause, const std::optional<parser::ScalarIntConstantExpr> &o);

  virtual llvm::StringRef getClauseName(C clause) { return ""; };

  virtual llvm::StringRef getDirectiveName(D directive) { return ""; };

  SemanticsContext &context_;
  std::vector<DirectiveContext> dirContext_; // used as a stack
  std::unordered_map<D, DirectiveClauses<C, ClauseEnumSize>>
      directiveClausesMap_;

  std::string ClauseSetToString(const common::EnumSet<C, ClauseEnumSize> set);
};

template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
void DirectiveStructureChecker<D, C, PC, ClauseEnumSize>::CheckNoBranching(
    const parser::Block &block, D directive,
    const parser::CharBlock &directiveSource) {
  NoBranchingEnforce<D> noBranchingEnforce{
      context_, directiveSource, directive, ContextDirectiveAsFortran()};
  parser::Walk(block, noBranchingEnforce);
}

// Check that only clauses included in the given set are present after the given
// clause.
template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
void DirectiveStructureChecker<D, C, PC, ClauseEnumSize>::CheckOnlyAllowedAfter(
    C clause, common::EnumSet<C, ClauseEnumSize> set) {
  bool enforceCheck = false;
  for (auto cl : GetContext().actualClauses) {
    if (cl == clause) {
      enforceCheck = true;
      continue;
    } else if (enforceCheck && !set.test(cl)) {
      auto parserClause = GetContext().clauseInfo.find(cl);
      context_.Say(parserClause->second->source,
          "Clause %s is not allowed after clause %s on the %s "
          "directive"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(cl).str()),
          parser::ToUpperCaseLetters(getClauseName(clause).str()),
          ContextDirectiveAsFortran());
    }
  }
}

// Check that at least one clause is attached to the directive.
template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
void DirectiveStructureChecker<D, C, PC,
    ClauseEnumSize>::CheckAtLeastOneClause() {
  if (GetContext().actualClauses.empty()) {
    context_.Say(GetContext().directiveSource,
        "At least one clause is required on the %s directive"_err_en_US,
        ContextDirectiveAsFortran());
  }
}

template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
std::string
DirectiveStructureChecker<D, C, PC, ClauseEnumSize>::ClauseSetToString(
    const common::EnumSet<C, ClauseEnumSize> set) {
  std::string list;
  set.IterateOverMembers([&](C o) {
    if (!list.empty())
      list.append(", ");
    list.append(parser::ToUpperCaseLetters(getClauseName(o).str()));
  });
  return list;
}

// Check that at least one clause in the required set is present on the
// directive.
template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
void DirectiveStructureChecker<D, C, PC,
    ClauseEnumSize>::CheckRequireAtLeastOneOf(bool warnInsteadOfError) {
  if (GetContext().requiredClauses.empty()) {
    return;
  }
  for (auto cl : GetContext().actualClauses) {
    if (GetContext().requiredClauses.test(cl)) {
      return;
    }
  }
  // No clause matched in the actual clauses list
  if (warnInsteadOfError) {
    context_.Warn(common::UsageWarning::Portability,
        GetContext().directiveSource,
        "At least one of %s clause should appear on the %s directive"_port_en_US,
        ClauseSetToString(GetContext().requiredClauses),
        ContextDirectiveAsFortran());
  } else {
    context_.Say(GetContext().directiveSource,
        "At least one of %s clause must appear on the %s directive"_err_en_US,
        ClauseSetToString(GetContext().requiredClauses),
        ContextDirectiveAsFortran());
  }
}

template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
std::string DirectiveStructureChecker<D, C, PC,
    ClauseEnumSize>::ContextDirectiveAsFortran() {
  return parser::ToUpperCaseLetters(
      getDirectiveName(GetContext().directive).str());
}

// Check that clauses present on the directive are allowed clauses.
template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
bool DirectiveStructureChecker<D, C, PC, ClauseEnumSize>::CheckAllowed(
    C clause, bool warnInsteadOfError) {
  if (!GetContext().allowedClauses.test(clause) &&
      !GetContext().allowedOnceClauses.test(clause) &&
      !GetContext().allowedExclusiveClauses.test(clause) &&
      !GetContext().requiredClauses.test(clause)) {
    if (warnInsteadOfError) {
      context_.Warn(common::UsageWarning::Portability,
          GetContext().clauseSource,
          "%s clause is not allowed on the %s directive and will be ignored"_port_en_US,
          parser::ToUpperCaseLetters(getClauseName(clause).str()),
          parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    } else {
      context_.Say(GetContext().clauseSource,
          "%s clause is not allowed on the %s directive"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clause).str()),
          parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    }
    return false;
  }
  if ((GetContext().allowedOnceClauses.test(clause) ||
          GetContext().allowedExclusiveClauses.test(clause)) &&
      FindClause(clause)) {
    context_.Say(GetContext().clauseSource,
        "At most one %s clause can appear on the %s directive"_err_en_US,
        parser::ToUpperCaseLetters(getClauseName(clause).str()),
        parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    return false;
  }
  if (GetContext().allowedExclusiveClauses.test(clause)) {
    std::vector<C> others;
    GetContext().allowedExclusiveClauses.IterateOverMembers([&](C o) {
      if (FindClause(o)) {
        others.emplace_back(o);
      }
    });
    for (const auto &e : others) {
      context_.Say(GetContext().clauseSource,
          "%s and %s clauses are mutually exclusive and may not appear on the "
          "same %s directive"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clause).str()),
          parser::ToUpperCaseLetters(getClauseName(e).str()),
          parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    }
    if (!others.empty()) {
      return false;
    }
  }
  SetContextClauseInfo(clause);
  AddClauseToCrtContext(clause);
  AddClauseToCrtGroupInContext(clause);
  return true;
}

// Enforce restriction where clauses in the given set are not allowed if the
// given clause appears.
template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
void DirectiveStructureChecker<D, C, PC,
    ClauseEnumSize>::CheckNotAllowedIfClause(C clause,
    common::EnumSet<C, ClauseEnumSize> set) {
  if (!llvm::is_contained(GetContext().actualClauses, clause)) {
    return; // Clause is not present
  }

  for (auto cl : GetContext().actualClauses) {
    if (set.test(cl)) {
      context_.Say(GetContext().directiveSource,
          "Clause %s is not allowed if clause %s appears on the %s directive"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(cl).str()),
          parser::ToUpperCaseLetters(getClauseName(clause).str()),
          ContextDirectiveAsFortran());
    }
  }
}

template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
void DirectiveStructureChecker<D, C, PC,
    ClauseEnumSize>::CheckAllowedOncePerGroup(C clause, C separator) {
  bool clauseIsPresent = false;
  for (auto cl : GetContext().actualClauses) {
    if (cl == clause) {
      if (clauseIsPresent) {
        context_.Say(GetContext().clauseSource,
            "At most one %s clause can appear on the %s directive or in group separated by the %s clause"_err_en_US,
            parser::ToUpperCaseLetters(getClauseName(clause).str()),
            parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()),
            parser::ToUpperCaseLetters(getClauseName(separator).str()));
      } else {
        clauseIsPresent = true;
      }
    }
    if (cl == separator)
      clauseIsPresent = false;
  }
}

template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
void DirectiveStructureChecker<D, C, PC,
    ClauseEnumSize>::CheckMutuallyExclusivePerGroup(C clause, C separator,
    common::EnumSet<C, ClauseEnumSize> set) {

  // Checking of there is any offending clauses before the first separator.
  for (auto cl : GetContext().actualClauses) {
    if (cl == separator) {
      break;
    }
    if (set.test(cl)) {
      context_.Say(GetContext().directiveSource,
          "Clause %s is not allowed if clause %s appears on the %s directive"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clause).str()),
          parser::ToUpperCaseLetters(getClauseName(cl).str()),
          ContextDirectiveAsFortran());
    }
  }

  // Checking for mutually exclusive clauses in the current group.
  for (auto cl : GetContext().crtGroup) {
    if (set.test(cl)) {
      context_.Say(GetContext().directiveSource,
          "Clause %s is not allowed if clause %s appears on the %s directive"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clause).str()),
          parser::ToUpperCaseLetters(getClauseName(cl).str()),
          ContextDirectiveAsFortran());
    }
  }
}

// Check the value of the clause is a constant positive integer.
template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
void DirectiveStructureChecker<D, C, PC,
    ClauseEnumSize>::RequiresConstantPositiveParameter(const C &clause,
    const parser::ScalarIntConstantExpr &i) {
  if (const auto v{GetIntValue(i)}) {
    if (*v <= 0) {
      context_.Say(GetContext().clauseSource,
          "The parameter of the %s clause must be "
          "a constant positive integer expression"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clause).str()));
    }
  }
}

// Check the value of the clause is a constant positive parameter.
template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
void DirectiveStructureChecker<D, C, PC,
    ClauseEnumSize>::OptionalConstantPositiveParameter(const C &clause,
    const std::optional<parser::ScalarIntConstantExpr> &o) {
  if (o != std::nullopt) {
    RequiresConstantPositiveParameter(clause, o.value());
  }
}

template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
void DirectiveStructureChecker<D, C, PC, ClauseEnumSize>::SayNotMatching(
    const parser::CharBlock &beginSource, const parser::CharBlock &endSource) {
  context_
      .Say(endSource, "Unmatched %s directive"_err_en_US,
          parser::ToUpperCaseLetters(endSource.ToString()))
      .Attach(beginSource, "Does not match directive"_en_US);
}

// Check the value of the clause is a positive parameter.
template <typename D, typename C, typename PC, std::size_t ClauseEnumSize>
void DirectiveStructureChecker<D, C, PC,
    ClauseEnumSize>::RequiresPositiveParameter(const C &clause,
    const parser::ScalarIntExpr &i, llvm::StringRef paramName) {
  if (const auto v{GetIntValue(i)}) {
    if (*v < 0) {
      context_.Say(GetContext().clauseSource,
          "The %s of the %s clause must be "
          "a positive integer expression"_err_en_US,
          paramName.str(),
          parser::ToUpperCaseLetters(getClauseName(clause).str()));
    }
  }
}

} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_CHECK_DIRECTIVE_STRUCTURE_H_
