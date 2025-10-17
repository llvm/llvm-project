//===-- lib/Semantics/check-omp-structure.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// OpenMP structure validity check list
//    1. invalid clauses on directive
//    2. invalid repeated clauses on directive
//    3. TODO: invalid nesting of regions

#ifndef FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_
#define FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_

#include "check-directive-structure.h"
#include "flang/Common/enum-set.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/semantics.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using OmpClauseSet =
    Fortran::common::EnumSet<llvm::omp::Clause, llvm::omp::Clause_enumSize>;

#define GEN_FLANG_DIRECTIVE_CLAUSE_SETS
#include "llvm/Frontend/OpenMP/OMP.inc"

namespace llvm {
namespace omp {
static OmpClauseSet privateSet{
    Clause::OMPC_private, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate};
static OmpClauseSet privateReductionSet{
    OmpClauseSet{Clause::OMPC_reduction} | privateSet};
// omp.td cannot differentiate allowed/not allowed clause list for few
// directives for fortran. nowait is not allowed on begin directive clause list
// for below list of directives. Directives with conflicting list of clauses are
// included in below list.
static const OmpDirectiveSet noWaitClauseNotAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_do_simd,
    Directive::OMPD_sections,
    Directive::OMPD_single,
    Directive::OMPD_workshare,
};
} // namespace omp
} // namespace llvm

namespace Fortran::semantics {
struct AnalyzedCondStmt;

// Mapping from 'Symbol' to 'Source' to keep track of the variables
// used in multiple clauses
using SymbolSourceMap = std::multimap<const Symbol *, parser::CharBlock>;
// Multimap to check the triple <current_dir, enclosing_dir, enclosing_clause>
using DirectivesClauseTriple = std::multimap<llvm::omp::Directive,
    std::pair<llvm::omp::Directive, const OmpClauseSet>>;

class OmpStructureChecker
    : public DirectiveStructureChecker<llvm::omp::Directive, llvm::omp::Clause,
          parser::OmpClause, llvm::omp::Clause_enumSize> {
public:
  using Base = DirectiveStructureChecker<llvm::omp::Directive,
      llvm::omp::Clause, parser::OmpClause, llvm::omp::Clause_enumSize>;

  OmpStructureChecker(SemanticsContext &context)
      : DirectiveStructureChecker(context,
#define GEN_FLANG_DIRECTIVE_CLAUSE_MAP
#include "llvm/Frontend/OpenMP/OMP.inc"
        ) {
  }
  using llvmOmpClause = const llvm::omp::Clause;

  void Enter(const parser::OpenMPConstruct &);
  void Leave(const parser::OpenMPConstruct &);
  void Enter(const parser::OpenMPInteropConstruct &);
  void Leave(const parser::OpenMPInteropConstruct &);
  void Enter(const parser::OpenMPDeclarativeConstruct &);
  void Leave(const parser::OpenMPDeclarativeConstruct &);

  void Enter(const parser::OpenMPLoopConstruct &);
  void Leave(const parser::OpenMPLoopConstruct &);
  void Enter(const parser::OmpEndLoopDirective &);
  void Leave(const parser::OmpEndLoopDirective &);

  void Enter(const parser::OpenMPAssumeConstruct &);
  void Leave(const parser::OpenMPAssumeConstruct &);
  void Enter(const parser::OpenMPDeclarativeAssumes &);
  void Leave(const parser::OpenMPDeclarativeAssumes &);
  void Enter(const parser::OmpBlockConstruct &);
  void Leave(const parser::OmpBlockConstruct &);
  void Leave(const parser::OmpBeginDirective &);
  void Enter(const parser::OmpEndDirective &);
  void Leave(const parser::OmpEndDirective &);

  void Enter(const parser::OpenMPSectionsConstruct &);
  void Leave(const parser::OpenMPSectionsConstruct &);
  void Enter(const parser::OmpEndSectionsDirective &);
  void Leave(const parser::OmpEndSectionsDirective &);

  void Enter(const parser::OmpDeclareVariantDirective &);
  void Leave(const parser::OmpDeclareVariantDirective &);
  void Enter(const parser::OpenMPDeclareSimdConstruct &);
  void Leave(const parser::OpenMPDeclareSimdConstruct &);
  void Enter(const parser::OpenMPDeclarativeAllocate &);
  void Leave(const parser::OpenMPDeclarativeAllocate &);
  void Enter(const parser::OpenMPDeclareMapperConstruct &);
  void Leave(const parser::OpenMPDeclareMapperConstruct &);
  void Enter(const parser::OpenMPDeclareReductionConstruct &);
  void Leave(const parser::OpenMPDeclareReductionConstruct &);
  void Enter(const parser::OpenMPDeclareTargetConstruct &);
  void Leave(const parser::OpenMPDeclareTargetConstruct &);
  void Enter(const parser::OpenMPDepobjConstruct &);
  void Leave(const parser::OpenMPDepobjConstruct &);
  void Enter(const parser::OpenMPDispatchConstruct &);
  void Leave(const parser::OpenMPDispatchConstruct &);
  void Enter(const parser::OmpErrorDirective &);
  void Leave(const parser::OmpErrorDirective &);
  void Enter(const parser::OmpNothingDirective &);
  void Leave(const parser::OmpNothingDirective &);
  void Enter(const parser::OpenMPExecutableAllocate &);
  void Leave(const parser::OpenMPExecutableAllocate &);
  void Enter(const parser::OpenMPAllocatorsConstruct &);
  void Leave(const parser::OpenMPAllocatorsConstruct &);
  void Enter(const parser::OpenMPRequiresConstruct &);
  void Leave(const parser::OpenMPRequiresConstruct &);
  void Enter(const parser::OpenMPGroupprivate &);
  void Leave(const parser::OpenMPGroupprivate &);
  void Enter(const parser::OpenMPThreadprivate &);
  void Leave(const parser::OpenMPThreadprivate &);

  void Enter(const parser::OpenMPSimpleStandaloneConstruct &);
  void Leave(const parser::OpenMPSimpleStandaloneConstruct &);
  void Enter(const parser::OpenMPFlushConstruct &);
  void Leave(const parser::OpenMPFlushConstruct &);
  void Enter(const parser::OpenMPCancelConstruct &);
  void Leave(const parser::OpenMPCancelConstruct &);
  void Enter(const parser::OpenMPCancellationPointConstruct &);
  void Leave(const parser::OpenMPCancellationPointConstruct &);
  void Enter(const parser::OpenMPCriticalConstruct &);
  void Leave(const parser::OpenMPCriticalConstruct &);
  void Enter(const parser::OpenMPAtomicConstruct &);
  void Leave(const parser::OpenMPAtomicConstruct &);

  void Leave(const parser::OmpClauseList &);
  void Enter(const parser::OmpClause &);

  void Enter(const parser::DoConstruct &);
  void Leave(const parser::DoConstruct &);

  void Enter(const parser::OmpDirectiveSpecification &);
  void Leave(const parser::OmpDirectiveSpecification &);

  void Enter(const parser::OmpMetadirectiveDirective &);
  void Leave(const parser::OmpMetadirectiveDirective &);

  void Enter(const parser::OmpContextSelector &);
  void Leave(const parser::OmpContextSelector &);

#define GEN_FLANG_CLAUSE_CHECK_ENTER
#include "llvm/Frontend/OpenMP/OMP.inc"

private:
  bool CheckAllowedClause(llvmOmpClause clause);
  void CheckVariableListItem(const SymbolSourceMap &symbols);
  void CheckDirectiveSpelling(
      parser::CharBlock spelling, llvm::omp::Directive id);
  void AnalyzeObject(const parser::OmpObject &object);
  void AnalyzeObjects(const parser::OmpObjectList &objects);
  void CheckMultipleOccurrence(semantics::UnorderedSymbolSet &listVars,
      const std::list<parser::Name> &nameList, const parser::CharBlock &item,
      const std::string &clauseName);
  void CheckMultListItems();
  void CheckStructureComponent(
      const parser::OmpObjectList &objects, llvm::omp::Clause clauseId);
  bool HasInvalidWorksharingNesting(
      const parser::CharBlock &, const OmpDirectiveSet &);
  bool IsCloselyNestedRegion(const OmpDirectiveSet &set);
  bool IsNestedInDirective(llvm::omp::Directive directive);
  void HasInvalidTeamsNesting(
      const llvm::omp::Directive &dir, const parser::CharBlock &source);
  void HasInvalidDistributeNesting(const parser::OpenMPLoopConstruct &x);
  void HasInvalidLoopBinding(const parser::OpenMPLoopConstruct &x);
  // specific clause related
  void CheckAllowedMapTypes(
      parser::OmpMapType::Value, llvm::ArrayRef<parser::OmpMapType::Value>);

  const std::list<parser::OmpTraitProperty> &GetTraitPropertyList(
      const parser::OmpTraitSelector &);
  std::optional<llvm::omp::Clause> GetClauseFromProperty(
      const parser::OmpTraitProperty &);

  void CheckTraitSelectorList(const std::list<parser::OmpTraitSelector> &);
  void CheckTraitSetSelector(const parser::OmpTraitSetSelector &);
  void CheckTraitScore(const parser::OmpTraitScore &);
  bool VerifyTraitPropertyLists(
      const parser::OmpTraitSetSelector &, const parser::OmpTraitSelector &);
  void CheckTraitSelector(
      const parser::OmpTraitSetSelector &, const parser::OmpTraitSelector &);
  void CheckTraitADMO(
      const parser::OmpTraitSetSelector &, const parser::OmpTraitSelector &);
  void CheckTraitCondition(
      const parser::OmpTraitSetSelector &, const parser::OmpTraitSelector &);
  void CheckTraitDeviceNum(
      const parser::OmpTraitSetSelector &, const parser::OmpTraitSelector &);
  void CheckTraitRequires(
      const parser::OmpTraitSetSelector &, const parser::OmpTraitSelector &);
  void CheckTraitSimd(
      const parser::OmpTraitSetSelector &, const parser::OmpTraitSelector &);

  llvm::StringRef getClauseName(llvm::omp::Clause clause) override;
  llvm::StringRef getDirectiveName(llvm::omp::Directive directive) override;

  template < //
      typename LessTy, typename RangeTy,
      typename IterTy = decltype(std::declval<RangeTy>().begin())>
  std::optional<IterTy> FindDuplicate(RangeTy &&);

  void CheckDependList(const parser::DataRef &);
  void CheckDependArraySection(
      const common::Indirection<parser::ArrayElement> &, const parser::Name &);
  void CheckDoacross(const parser::OmpDoacross &doa);
  bool IsDataRefTypeParamInquiry(const parser::DataRef *dataRef);
  void CheckVarIsNotPartOfAnotherVar(const parser::CharBlock &source,
      const parser::OmpObject &obj, llvm::StringRef clause = "");
  void CheckVarIsNotPartOfAnotherVar(const parser::CharBlock &source,
      const parser::OmpObjectList &objList, llvm::StringRef clause = "");
  void CheckThreadprivateOrDeclareTargetVar(const parser::Designator &);
  void CheckThreadprivateOrDeclareTargetVar(const parser::Name &);
  void CheckThreadprivateOrDeclareTargetVar(const parser::OmpObject &);
  void CheckThreadprivateOrDeclareTargetVar(const parser::OmpObjectList &);
  void CheckSymbolName(
      const parser::CharBlock &source, const parser::OmpObject &object);
  void CheckSymbolNames(
      const parser::CharBlock &source, const parser::OmpObjectList &objList);
  void CheckIntentInPointer(SymbolSourceMap &, const llvm::omp::Clause);
  void CheckProcedurePointer(SymbolSourceMap &, const llvm::omp::Clause);
  void CheckCrayPointee(const parser::OmpObjectList &objectList,
      llvm::StringRef clause, bool suggestToUseCrayPointer = true);
  void GetSymbolsInObjectList(const parser::OmpObjectList &, SymbolSourceMap &);
  void CheckDefinableObjects(SymbolSourceMap &, const llvm::omp::Clause);
  void CheckCopyingPolymorphicAllocatable(
      SymbolSourceMap &, const llvm::omp::Clause);
  void CheckPrivateSymbolsInOuterCxt(
      SymbolSourceMap &, DirectivesClauseTriple &, const llvm::omp::Clause);
  const parser::Name GetLoopIndex(const parser::DoConstruct *x);
  void SetLoopInfo(const parser::OpenMPLoopConstruct &x);
  void CheckIsLoopIvPartOfClause(
      llvmOmpClause clause, const parser::OmpObjectList &ompObjectList);
  bool CheckTargetBlockOnlyTeams(const parser::Block &);
  void CheckWorkshareBlockStmts(const parser::Block &, parser::CharBlock);
  void CheckWorkdistributeBlockStmts(const parser::Block &, parser::CharBlock);

  void CheckIteratorRange(const parser::OmpIteratorSpecifier &x);
  void CheckIteratorModifier(const parser::OmpIterator &x);
  void CheckLoopItrVariableIsInt(const parser::OpenMPLoopConstruct &x);
  void CheckDoWhile(const parser::OpenMPLoopConstruct &x);
  void CheckAssociatedLoopConstraints(const parser::OpenMPLoopConstruct &x);
  template <typename T, typename D> bool IsOperatorValid(const T &, const D &);

  void CheckStorageOverlap(const evaluate::Expr<evaluate::SomeType> &,
      llvm::ArrayRef<evaluate::Expr<evaluate::SomeType>>, parser::CharBlock);
  void ErrorShouldBeVariable(const MaybeExpr &expr, parser::CharBlock source);
  void CheckAtomicType(SymbolRef sym, parser::CharBlock source,
      std::string_view name, bool checkTypeOnPointer = true);
  void CheckAtomicVariable(const evaluate::Expr<evaluate::SomeType> &,
      parser::CharBlock, bool checkTypeOnPointer = true);
  std::pair<const parser::ExecutionPartConstruct *,
      const parser::ExecutionPartConstruct *>
  CheckUpdateCapture(const parser::ExecutionPartConstruct *ec1,
      const parser::ExecutionPartConstruct *ec2, parser::CharBlock source);
  void CheckAtomicCaptureAssignment(const evaluate::Assignment &capture,
      const SomeExpr &atom, parser::CharBlock source);
  void CheckAtomicReadAssignment(
      const evaluate::Assignment &read, parser::CharBlock source);
  void CheckAtomicWriteAssignment(
      const evaluate::Assignment &write, parser::CharBlock source);
  std::optional<evaluate::Assignment> CheckAtomicUpdateAssignment(
      const evaluate::Assignment &update, parser::CharBlock source);
  std::pair<bool, bool> CheckAtomicUpdateAssignmentRhs(const SomeExpr &atom,
      const SomeExpr &rhs, parser::CharBlock source, bool suppressDiagnostics);
  void CheckAtomicConditionalUpdateAssignment(const SomeExpr &cond,
      parser::CharBlock condSource, const evaluate::Assignment &assign,
      parser::CharBlock assignSource);
  void CheckAtomicConditionalUpdateStmt(
      const AnalyzedCondStmt &update, parser::CharBlock source);
  void CheckAtomicUpdateOnly(const parser::OpenMPAtomicConstruct &x,
      const parser::Block &body, parser::CharBlock source);
  void CheckAtomicConditionalUpdate(const parser::OpenMPAtomicConstruct &x,
      const parser::Block &body, parser::CharBlock source);
  void CheckAtomicUpdateCapture(const parser::OpenMPAtomicConstruct &x,
      const parser::Block &body, parser::CharBlock source);
  void CheckAtomicConditionalUpdateCapture(
      const parser::OpenMPAtomicConstruct &x, const parser::Block &body,
      parser::CharBlock source);
  void CheckAtomicRead(const parser::OpenMPAtomicConstruct &x);
  void CheckAtomicWrite(const parser::OpenMPAtomicConstruct &x);
  void CheckAtomicUpdate(const parser::OpenMPAtomicConstruct &x);

  void CheckDistLinear(const parser::OpenMPLoopConstruct &x);
  void CheckSIMDNest(const parser::OpenMPConstruct &x);
  void CheckTargetNest(const parser::OpenMPConstruct &x);
  void CheckTargetUpdate();
  void CheckTaskgraph(const parser::OmpBlockConstruct &x);
  void CheckDependenceType(const parser::OmpDependenceType::Value &x);
  void CheckTaskDependenceType(const parser::OmpTaskDependenceType::Value &x);
  std::optional<llvm::omp::Directive> GetCancelType(
      llvm::omp::Directive cancelDir, const parser::CharBlock &cancelSource,
      const std::optional<parser::OmpClauseList> &maybeClauses);
  void CheckCancellationNest(
      const parser::CharBlock &source, llvm::omp::Directive type);
  void CheckAllNamesInAllocateStmt(const parser::CharBlock &source,
      const parser::OmpObjectList &ompObjectList,
      const parser::AllocateStmt &allocate);
  void CheckNameInAllocateStmt(const parser::CharBlock &source,
      const parser::Name &ompObject, const parser::AllocateStmt &allocate);
  std::int64_t GetOrdCollapseLevel(const parser::OpenMPLoopConstruct &x);
  void CheckReductionObjects(
      const parser::OmpObjectList &objects, llvm::omp::Clause clauseId);
  bool CheckReductionOperator(const parser::OmpReductionIdentifier &ident,
      parser::CharBlock source, llvm::omp::Clause clauseId);
  void CheckReductionObjectTypes(const parser::OmpObjectList &objects,
      const parser::OmpReductionIdentifier &ident);
  void CheckReductionModifier(const parser::OmpReductionModifier &);
  void CheckLastprivateModifier(const parser::OmpLastprivateModifier &);
  void CheckMasterNesting(const parser::OmpBlockConstruct &x);
  void ChecksOnOrderedAsBlock();
  void CheckBarrierNesting(const parser::OpenMPSimpleStandaloneConstruct &x);
  void CheckScan(const parser::OpenMPSimpleStandaloneConstruct &x);
  void ChecksOnOrderedAsStandalone();
  void CheckOrderedDependClause(std::optional<std::int64_t> orderedValue);
  void CheckReductionArraySection(
      const parser::OmpObjectList &ompObjectList, llvm::omp::Clause clauseId);
  void CheckArraySection(const parser::ArrayElement &arrayElement,
      const parser::Name &name, const llvm::omp::Clause clause);
  void CheckSharedBindingInOuterContext(
      const parser::OmpObjectList &ompObjectList);
  void CheckIfContiguous(const parser::OmpObject &object);
  const parser::Name *GetObjectName(const parser::OmpObject &object);
  void CheckPredefinedAllocatorRestriction(const parser::CharBlock &source,
      const parser::OmpObjectList &ompObjectList);
  void CheckPredefinedAllocatorRestriction(
      const parser::CharBlock &source, const parser::Name &name);
  bool isPredefinedAllocator{false};

  void CheckAllowedRequiresClause(llvmOmpClause clause);
  bool deviceConstructFound_{false};

  void CheckAlignValue(const parser::OmpClause &);

  void AddEndDirectiveClauses(const parser::OmpClauseList &clauses);

  void EnterDirectiveNest(const int index) { directiveNest_[index]++; }
  void ExitDirectiveNest(const int index) { directiveNest_[index]--; }
  int GetDirectiveNest(const int index) { return directiveNest_[index]; }
  inline void ErrIfAllocatableVariable(const parser::Variable &);
  inline void ErrIfLHSAndRHSSymbolsMatch(
      const parser::Variable &, const parser::Expr &);
  inline void ErrIfNonScalarAssignmentStmt(
      const parser::Variable &, const parser::Expr &);
  enum directiveNestType : int {
    SIMDNest,
    TargetBlockOnlyTeams,
    TargetNest,
    DeclarativeNest,
    ContextSelectorNest,
    MetadirectiveNest,
    LastType = MetadirectiveNest,
  };
  int directiveNest_[LastType + 1] = {0};

  parser::CharBlock visitedAtomicSource_;
  SymbolSourceMap deferredNonVariables_;

  using LoopConstruct = std::variant<const parser::DoConstruct *,
      const parser::OpenMPLoopConstruct *>;
  std::vector<LoopConstruct> loopStack_;
};

/// Find a duplicate entry in the range, and return an iterator to it.
/// If there are no duplicate entries, return nullopt.
template <typename LessTy, typename RangeTy, typename IterTy>
std::optional<IterTy> OmpStructureChecker::FindDuplicate(RangeTy &&range) {
  // Deal with iterators, since the actual elements may be rvalues (i.e.
  // have no addresses), for example with custom-constructed ranges that
  // are not simple c.begin()..c.end().
  std::set<IterTy, LessTy> uniq;
  for (auto it{range.begin()}, end{range.end()}; it != end; ++it) {
    if (!uniq.insert(it).second) {
      return it;
    }
  }
  return std::nullopt;
}

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_
