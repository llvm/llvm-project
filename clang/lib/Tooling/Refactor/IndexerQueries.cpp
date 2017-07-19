//===--- IndexerQueries.cpp - Indexer queries -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Core/RefactoringDiagnostic.h"
#include "clang/Tooling/Refactor/IndexerQuery.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLTraits.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::tooling::indexer;
using namespace clang::tooling::indexer::detail;
using namespace llvm::yaml;

const char *ASTProducerQuery::BaseUIDString = "ast.producer.query";
const char *DeclarationsQuery::BaseUIDString = "decl.query";
const char *ASTUnitForImplementationOfDeclarationQuery::NameUIDString =
    "file.for.impl.of.decl";

const char *DeclPredicateNodePredicate::NameUIDString = "decl.predicate";
const char *DeclPredicateNotPredicate::NameUIDString = "not.decl.predicate";

std::unique_ptr<DeclPredicateNode>
DeclPredicateNode::create(const DeclPredicate &Predicate) {
  return llvm::make_unique<DeclPredicateNodePredicate>(Predicate);
}

std::unique_ptr<DeclPredicateNode>
DeclPredicateNode::create(const BoolDeclPredicate &Predicate) {
  if (Predicate.IsInverted)
    return llvm::make_unique<DeclPredicateNotPredicate>(
        create(Predicate.Predicate));
  return create(Predicate.Predicate);
}

std::unique_ptr<ASTUnitForImplementationOfDeclarationQuery>
clang::tooling::indexer::fileThatShouldContainImplementationOf(const Decl *D) {
  return llvm::make_unique<ASTUnitForImplementationOfDeclarationQuery>(D);
}

bool ASTUnitForImplementationOfDeclarationQuery::verify(ASTContext &Context) {
  if (!D) {
    assert(false && "Query should be verified before persisting");
    return false;
  }
  // Check if we've got the filename.
  if (!Result.Filename.empty())
    return false;
  Context.getDiagnostics().Report(
      D->getLocation(), diag::err_ref_continuation_missing_implementation)
      << isa<ObjCContainerDecl>(D) << cast<NamedDecl>(D);
  return true;
}

bool DeclarationsQuery::verify(ASTContext &Context) {
  if (Input.empty()) {
    assert(false && "Query should be verified before persisting");
    return false;
  }
  if (!Output.empty()) {
    // At least one output declaration must be valid.
    for (const auto &Ref : Output) {
      if (!Ref.Decl.USR.empty())
        return false;
    }
  }
  // FIXME: This is too specific, the new refactoring engine at llvm.org should
  // generalize this.
  Context.getDiagnostics().Report(
      Input[0]->getLocation(),
      diag::err_implement_declared_methods_all_implemented);
  return true;
}

namespace {

struct QueryPredicateNode {
  std::string Name;
  std::vector<int> IntegerValues;
};

struct QueryYAMLNode {
  std::string Name;
  std::vector<QueryPredicateNode> PredicateResults;
  std::string FilenameResult;
};

} // end anonymous namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(QueryPredicateNode)
LLVM_YAML_IS_SEQUENCE_VECTOR(QueryYAMLNode)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<QueryPredicateNode> {
  static void mapping(IO &Yaml, QueryPredicateNode &Predicate) {
    Yaml.mapRequired("name", Predicate.Name);
    Yaml.mapRequired("intValues", Predicate.IntegerValues);
  }
};

template <> struct MappingTraits<QueryYAMLNode> {
  static void mapping(IO &Yaml, QueryYAMLNode &Query) {
    Yaml.mapRequired("name", Query.Name);
    Yaml.mapOptional("predicateResults", Query.PredicateResults);
    Yaml.mapOptional("filenameResult", Query.FilenameResult);
    // FIXME: Report an error if no results are provided at all.
  }
};

} // end namespace yaml
} // end namespace llvm

llvm::Error
IndexerQuery::loadResultsFromYAML(StringRef Source,
                                  ArrayRef<IndexerQuery *> Queries) {
  std::vector<QueryYAMLNode> QueryResults;
  Input YamlIn(Source);
  YamlIn >> QueryResults;
  if (YamlIn.error())
    return llvm::make_error<llvm::StringError>("Failed to parse query results",
                                               YamlIn.error());
  if (QueryResults.size() != Queries.size())
    return llvm::make_error<llvm::StringError>("Mismatch in query results size",
                                               llvm::errc::invalid_argument);
  for (const auto &QueryTuple : llvm::zip(Queries, QueryResults)) {
    IndexerQuery *Query = std::get<0>(QueryTuple);
    const QueryYAMLNode &Result = std::get<1>(QueryTuple);
    if ((Query->NameUID && Query->NameUID != Result.Name) &&
        (Query->BaseUID && Query->BaseUID != Result.Name))
      continue;
    if (auto *DQ = dyn_cast<DeclarationsQuery>(Query)) {
      const DeclPredicateNode &Predicate = DQ->getPredicateNode();
      DeclPredicate ActualPredicate("");
      bool IsNot = false;
      if (const auto *Not = dyn_cast<DeclPredicateNotPredicate>(&Predicate)) {
        ActualPredicate =
            cast<DeclPredicateNodePredicate>(Not->getChild()).getPredicate();
        IsNot = true;
      } else
        ActualPredicate =
            cast<DeclPredicateNodePredicate>(Predicate).getPredicate();
      for (const auto &PredicateResult : Result.PredicateResults) {
        if (PredicateResult.Name != ActualPredicate.Name)
          continue;
        std::vector<Indexed<PersistentDeclRef<Decl>>> Output;
        for (const auto &ResultTuple :
             zip(DQ->getInputs(), PredicateResult.IntegerValues)) {
          const Decl *D = std::get<0>(ResultTuple);
          int Result = std::get<1>(ResultTuple);
          bool Value = (IsNot ? !Result : !!Result);
          Output.push_back(Indexed<PersistentDeclRef<Decl>>(
              PersistentDeclRef<Decl>::create(Value ? D : nullptr),
              Value ? QueryBoolResult::Yes : QueryBoolResult::No));
        }
        DQ->setOutput(std::move(Output));
        break;
      }
    } else if (auto *AQ =
                   dyn_cast<ASTUnitForImplementationOfDeclarationQuery>(Query))
      AQ->setResult(Result.FilenameResult);
  }
  return llvm::Error::success();
}
