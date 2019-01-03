//===--- FillInMissingSwitchEnumCases.cpp -  ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/Edit/RefactoringFixits.h"
#include <unordered_map>

using namespace clang;

namespace {

struct CaseInfo {
  const SwitchCase *Case, *NextCase;
  unsigned Index;
};
typedef std::unordered_map<int64_t, CaseInfo> CoveredEnumCasesInfoType;

/// Return true if the ordering of the covered enum cases is similar to the
/// order of the enum case constants that are defined in the enum.
bool useCaseBasedOrdering(const ArrayRef<int64_t> &CoveredEnumCaseValues,
                          const CoveredEnumCasesInfoType &CoveredEnumCases) {
  if (CoveredEnumCaseValues.empty())
    return false;
  for (const auto &I : llvm::enumerate(CoveredEnumCaseValues)) {
    auto It = CoveredEnumCases.find(I.value());
    if (It == CoveredEnumCases.end())
      return false;
    const CaseInfo &Case = It->second;
    if (Case.Index != I.index())
      return false;
  }
  return true;
}

/// Determine if the inserted cases should be wrapped in braces using a simple
/// heuristic:
///   Wrap only if at least 90% of existing cases use braces.
bool useBraces(const SwitchStmt *S) {
  unsigned CaseCount = 0, CompoundCasesCount = 0;
  for (const SwitchCase *Case = S->getSwitchCaseList(); Case;
       Case = Case->getNextSwitchCase(), ++CaseCount) {
    if (!Case->getSubStmt())
      continue;
    if (isa<CompoundStmt>(Case->getSubStmt()))
      ++CompoundCasesCount;
  }
  return CaseCount && float(CompoundCasesCount) / float(CaseCount) >= 0.9;
}

} // end anonymous namespace

void edit::fillInMissingSwitchEnumCases(
    ASTContext &Context, const SwitchStmt *Switch, const EnumDecl *Enum,
    const DeclContext *SwitchContext,
    llvm::function_ref<void(const FixItHint &)> Consumer) {
  // Compute the number of cases in the switch.
  unsigned CaseCount = 0;
  for (const SwitchCase *Case = Switch->getSwitchCaseList(); Case;
       Case = Case->getNextSwitchCase())
    ++CaseCount;

  // Compute the set of enum values that are covered by the switch.
  CoveredEnumCasesInfoType CoveredEnumCases;
  const SwitchCase *DefaultCase = nullptr;
  const SwitchCase *FirstCoveredEnumCase = nullptr;
  const SwitchCase *NextCase = nullptr;
  unsigned CaseIndex = CaseCount;
  for (const SwitchCase *Case = Switch->getSwitchCaseList(); Case;
       NextCase = Case, Case = Case->getNextSwitchCase()) {
    // The cases in the switch are ordered back to front, so the index has
    // to be reversed.
    --CaseIndex;
    if (isa<DefaultStmt>(Case)) {
      DefaultCase = Case;
      continue;
    }
    const auto *CS = cast<CaseStmt>(Case);
    if (const auto *LHS = CS->getLHS()) {
      Expr::EvalResult Result;
      if (!LHS->EvaluateAsInt(Result, Context))
        continue;
      // Only allow constant that fix into 64 bits.
      if (Result.Val.getInt().getMinSignedBits() > 64)
        continue;
      CoveredEnumCases[Result.Val.getInt().getSExtValue()] =
          CaseInfo{Case, NextCase, CaseIndex};
      // The cases in the switch are ordered back to front, so the last
      //  case is actually the first enum case in the switch.
      FirstCoveredEnumCase = Case;
    }
  }

  // Wrap the inserted cases in braces using a simple heuristic:
  //   Wrap only if at least 90% of existing cases use braces.
  bool WrapInBraces = useBraces(Switch);
  auto CreateReplacementForMissingCaseGroup =
      [&](ArrayRef<const EnumConstantDecl *> UncoveredEnumCases,
          SourceLocation InsertionLoc = SourceLocation()) {
        if (UncoveredEnumCases.empty())
          return;
        std::string Result;
        llvm::raw_string_ostream OS(Result);
        for (const auto *EnumCase : UncoveredEnumCases) {
          OS << "case ";
          if (SwitchContext) {
            const auto *NS = NestedNameSpecifier::getRequiredQualification(
                Context, SwitchContext, Enum->getLexicalDeclContext());
            if (NS)
              NS->print(OS, Context.getPrintingPolicy());
          }
          if (Enum->isScoped())
            OS << Enum->getName() << "::";
          OS << EnumCase->getName() << ":";
          if (WrapInBraces)
            OS << " {";
          OS << "\n<#code#>\nbreak;\n";
          if (WrapInBraces)
            OS << "}\n";
        }

        if (InsertionLoc.isInvalid()) {
          // Insert the cases before the 'default' if it's the last case in the
          // switch.
          // Note: Switch cases are ordered back to front, so the last default
          // case would be the first case in the switch statement.
          if (DefaultCase && DefaultCase == Switch->getSwitchCaseList())
            InsertionLoc = DefaultCase->getBeginLoc();
          else
            InsertionLoc = Switch->getBody()->getEndLoc();
        }
        Consumer(FixItHint::CreateInsertion(
            Context.getSourceManager().getSpellingLoc(InsertionLoc), OS.str()));
      };

  // Determine which enum cases are uncovered.

  llvm::SmallVector<std::pair<const EnumConstantDecl *, int64_t>, 8> EnumCases;
  llvm::SmallVector<int64_t, 8> CoveredEnumCaseValues;
  for (const auto *EnumCase : Enum->enumerators()) {
    if (EnumCase->getInitVal().getMinSignedBits() > 64)
      continue;
    int64_t Value = EnumCase->getInitVal().getSExtValue();
    EnumCases.push_back(std::make_pair(EnumCase, Value));
    if (CoveredEnumCases.count(Value))
      CoveredEnumCaseValues.push_back(Value);
  }

  llvm::SmallVector<const EnumConstantDecl *, 8> UncoveredEnumCases;
  // Figure out if the ordering of the covered enum cases is similar to the
  // order of enum case values defined in the enum.
  if (useCaseBasedOrdering(CoveredEnumCaseValues, CoveredEnumCases)) {
    // Start inserting before the first covered case.
    SourceLocation InsertionLoc = FirstCoveredEnumCase->getBeginLoc();

    for (const auto &EnumCase : EnumCases) {
      if (!CoveredEnumCases.count(EnumCase.second)) {
        UncoveredEnumCases.push_back(EnumCase.first);
        continue;
      }
      // Create the insertion source replacement for this set of uncovered
      // cases.
      CreateReplacementForMissingCaseGroup(UncoveredEnumCases, InsertionLoc);
      UncoveredEnumCases.clear();
      // Find the insertion location for the next set of uncovered cases.
      auto It = CoveredEnumCases.find(EnumCase.second);
      assert(It != CoveredEnumCases.end() && "Missing enum case");
      const CaseInfo &Case = It->second;
      InsertionLoc = Case.NextCase ? Case.NextCase->getBeginLoc()
                                   : /*Insert before end*/ SourceLocation();
    }
    CreateReplacementForMissingCaseGroup(UncoveredEnumCases, InsertionLoc);
  } else {
    // Gather all of the uncovered enum cases.
    for (const auto &EnumCase : EnumCases) {
      if (!CoveredEnumCases.count(EnumCase.second))
        UncoveredEnumCases.push_back(EnumCase.first);
    }
    assert(!UncoveredEnumCases.empty() &&
           "Can't fill-in enum cases in a full switch");
    CreateReplacementForMissingCaseGroup(UncoveredEnumCases);
  }
}
