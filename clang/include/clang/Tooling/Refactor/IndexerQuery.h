//===--- IndexerQuery.h - A set of indexer query interfaces ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the base indexer queries that can be used with
// refactoring continuations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_INDEXER_QUERY_H
#define LLVM_CLANG_TOOLING_REFACTOR_INDEXER_QUERY_H

#include "clang/Tooling/Refactor/RefactoringOperationState.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace clang {
namespace tooling {
namespace indexer {

/// Represents an abstract indexer query.
class IndexerQuery {
public:
  const char *BaseUID;
  const char *NameUID;

  IndexerQuery(const char *BaseUID, const char *NameUID)
      : BaseUID(BaseUID), NameUID(NameUID) {}
  virtual ~IndexerQuery() {}

  virtual void invalidateTUSpecificState() = 0;

  /// Checks if this query was satisfied. Returns true if it wasn't and reports
  /// appropriate errors.
  virtual bool verify(ASTContext &) { return false; }

  // Mainly used for testing.
  static llvm::Error loadResultsFromYAML(StringRef Source,
                                         ArrayRef<IndexerQuery *> Queries);

  static bool classof(const IndexerQuery *) { return true; }
};

/// An abstract AST query that can produce an AST unit in which the refactoring
/// continuation will run.
class ASTProducerQuery : public IndexerQuery {
  static const char *BaseUIDString;

public:
  /// Deriving AST producer queries can redefine this type to generate custom
  /// results that are then passed into the refactoring continuations.
  using ResultTy = void;

  ASTProducerQuery(const char *NameUID)
      : IndexerQuery(BaseUIDString, NameUID) {}

  static bool classof(const IndexerQuery *Q) {
    return Q->BaseUID == BaseUIDString;
  }
};

/// A query that finds a file that contains/should contain the implementation of
/// some declaration.
class ASTUnitForImplementationOfDeclarationQuery final
    : public ASTProducerQuery {
  static const char *NameUIDString;

  const Decl *D;
  PersistentFileID Result;

public:
  ASTUnitForImplementationOfDeclarationQuery(const Decl *D)
      : ASTProducerQuery(NameUIDString), D(D), Result("") {}

  using ResultTy = FileID;

  const Decl *getDecl() const { return D; }

  void invalidateTUSpecificState() override { D = nullptr; }

  void setResult(PersistentFileID File) { Result = std::move(File); }

  bool verify(ASTContext &Context) override;

  const PersistentFileID &getResult() const { return Result; }

  static bool classof(const IndexerQuery *D) {
    return D->NameUID == NameUIDString;
  }
};

/// Returns an indexer query that will allow a refactoring continuation to run
/// in an AST unit that contains a file that should contain the implementation
/// of the given declaration \p D.
///
/// The continuation function will receive \c FileID that corresponds to the
/// implementation file. The indexer can decide which file should be used as an
/// implementation of a declaration based on a number of different heuristics.
/// It does not guarantee that the file will actually have any declarations that
/// correspond to the implementation of \p D yet, as the indexer may decide to
/// point to a file that it thinks will have the implementation declarations in
/// the future.
std::unique_ptr<ASTUnitForImplementationOfDeclarationQuery>
fileThatShouldContainImplementationOf(const Decl *D);

/// A declaration predicate operates.
struct DeclPredicate {
  const char *Name;

  DeclPredicate(const char *Name) : Name(Name) {}

  bool operator==(const DeclPredicate &P) const {
    return StringRef(Name) == P.Name;
  }
  bool operator!=(const DeclPredicate &P) const {
    return StringRef(Name) != P.Name;
  }
};

/// Represents a declaration predicate that will evaluate to either 'true' or
/// 'false' in an indexer query.
struct BoolDeclPredicate {
  DeclPredicate Predicate;
  bool IsInverted;

  BoolDeclPredicate(DeclPredicate Predicate, bool IsInverted = false)
      : Predicate(Predicate), IsInverted(IsInverted) {}

  BoolDeclPredicate operator!() const {
    return BoolDeclPredicate(Predicate, /*IsInverted=*/!IsInverted);
  }
};

namespace detail {

/// AST-like representation for decl predicates.
class DeclPredicateNode {
public:
  const char *NameUID;
  DeclPredicateNode(const char *NameUID) : NameUID(NameUID) {}

  virtual ~DeclPredicateNode() { }

  static std::unique_ptr<DeclPredicateNode>
  create(const DeclPredicate &Predicate);
  static std::unique_ptr<DeclPredicateNode>
  create(const BoolDeclPredicate &Predicate);

  static bool classof(const DeclPredicateNode *) { return true; }
};

class DeclPredicateNodePredicate : public DeclPredicateNode {
  static const char *NameUIDString;

  DeclPredicate Predicate;

public:
  DeclPredicateNodePredicate(const DeclPredicate &Predicate)
      : DeclPredicateNode(NameUIDString), Predicate(Predicate) {}

  const DeclPredicate &getPredicate() const { return Predicate; }

  static bool classof(const DeclPredicateNode *P) {
    return P->NameUID == NameUIDString;
  }
};

class DeclPredicateNotPredicate : public DeclPredicateNode {
  static const char *NameUIDString;

  std::unique_ptr<DeclPredicateNode> Child;

public:
  DeclPredicateNotPredicate(std::unique_ptr<DeclPredicateNode> Child)
      : DeclPredicateNode(NameUIDString), Child(std::move(Child)) {}

  const DeclPredicateNode &getChild() const { return *Child; }

  static bool classof(const DeclPredicateNode *P) {
    return P->NameUID == NameUIDString;
  }
};

} // end namespace detail

enum class QueryBoolResult {
  Unknown,
  Yes,
  No,
};

// FIXME: Check that 'T' is either a PersistentDeclRef<> or a Decl *.
template <typename T> struct Indexed {
  T Decl;
  // FIXME: Generalize better in the new refactoring engine.
  QueryBoolResult IsNotDefined;

  Indexed(T Decl, QueryBoolResult IsNotDefined = QueryBoolResult::Unknown)
      : Decl(Decl), IsNotDefined(IsNotDefined) {}

  Indexed(Indexed<T> &&Other) = default;
  Indexed &operator=(Indexed<T> &&Other) = default;
  Indexed(const Indexed<T> &Other) = default;
  Indexed &operator=(const Indexed<T> &Other) = default;

  /// True iff the declaration is not defined in the entire project.
  bool isNotDefined() const {
    // FIXME: This is hack. Need a better system in the new engine.
    return IsNotDefined == QueryBoolResult::Yes;
  }
};

/// Transforms one set of declarations into another using some predicate.
class DeclarationsQuery : public IndexerQuery {
  static const char *BaseUIDString;

  std::vector<const Decl *> Input;
  std::unique_ptr<detail::DeclPredicateNode> Predicate;

protected:
  std::vector<Indexed<PersistentDeclRef<Decl>>> Output;

public:
  DeclarationsQuery(std::vector<const Decl *> Input,
                    std::unique_ptr<detail::DeclPredicateNode> Predicate)
      : IndexerQuery(BaseUIDString, nullptr), Input(std::move(Input)),
        Predicate(std::move(Predicate)) {
    assert(!this->Input.empty() && "empty declarations list!");
  }

  ArrayRef<const Decl *> getInputs() const { return Input; }

  void invalidateTUSpecificState() override { Input.clear(); }

  bool verify(ASTContext &Context) override;

  void setOutput(std::vector<Indexed<PersistentDeclRef<Decl>>> Output) {
    this->Output = Output;
  }

  const detail::DeclPredicateNode &getPredicateNode() const {
    return *Predicate;
  }

  static bool classof(const IndexerQuery *Q) {
    return Q->BaseUID == BaseUIDString;
  }
};

/// The \c DeclEntity class acts as a proxy for the entity that represents a
/// declaration in the indexer. It defines a set of declaration predicates that
/// can be used in indexer queries.
struct DeclEntity {
  /// The indexer will evaluate this predicate to 'true' when a certain
  /// declaration has a corresponding definition.
  BoolDeclPredicate isDefined() const {
    return BoolDeclPredicate("decl.isDefined");
  }
};

template <typename T>
class ManyToManyDeclarationsQuery final
    : public std::enable_if<std::is_base_of<Decl, T>::value,
                            DeclarationsQuery>::type {
public:
  ManyToManyDeclarationsQuery(
      ArrayRef<const T *> Input,
      std::unique_ptr<detail::DeclPredicateNode> Predicate)
      : DeclarationsQuery(std::vector<const Decl *>(Input.begin(), Input.end()),
                          std::move(Predicate)) {}

  std::vector<Indexed<PersistentDeclRef<T>>> getOutput() const {
    std::vector<Indexed<PersistentDeclRef<T>>> Results;
    for (const auto &Ref : DeclarationsQuery::Output)
      Results.push_back(Indexed<PersistentDeclRef<T>>(
          PersistentDeclRef<T>(Ref.Decl.USR), Ref.IsNotDefined));
    return Results;
  }
};

/// Returns an indexer query that will pass a filtered list of declarations to
/// a refactoring continuation.
///
/// The filtering is done based on predicates that are available on the \c
/// DeclEntity types. For example, you can use the following invocation to
/// find a set of declarations that are defined in the entire project:
///
/// \code
/// filter({ MyDeclA, MyDeclB }, [] (const DeclEntity &D) { return D.isDefined()
/// })
/// \endcode
template <typename T>
std::unique_ptr<ManyToManyDeclarationsQuery<T>>
filter(ArrayRef<const T *> Declarations,
       BoolDeclPredicate (*Fn)(const DeclEntity &),
       typename std::enable_if<std::is_base_of<Decl, T>::value>::type * =
           nullptr) {
  return llvm::make_unique<ManyToManyDeclarationsQuery<T>>(
      Declarations, detail::DeclPredicateNode::create(Fn(DeclEntity())));
}

} // end namespace indexer
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_INDEXER_QUERY_H
