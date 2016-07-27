//===--- CloneDetection.cpp - Finds code clones in an AST -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  This file implements classes for searching and anlyzing source code clones.
///
//===----------------------------------------------------------------------===//

#include "clang/Analysis/CloneDetection.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/StringRef.h"

using namespace clang;

StmtSequence::StmtSequence(const CompoundStmt *Stmt, ASTContext &Context,
                           unsigned StartIndex, unsigned EndIndex)
    : S(Stmt), Context(&Context), StartIndex(StartIndex), EndIndex(EndIndex) {
  assert(Stmt && "Stmt must not be a nullptr");
  assert(StartIndex < EndIndex && "Given array should not be empty");
  assert(EndIndex <= Stmt->size() && "Given array too big for this Stmt");
}

StmtSequence::StmtSequence(const Stmt *Stmt, ASTContext &Context)
    : S(Stmt), Context(&Context), StartIndex(0), EndIndex(0) {}

StmtSequence::StmtSequence()
    : S(nullptr), Context(nullptr), StartIndex(0), EndIndex(0) {}

bool StmtSequence::contains(const StmtSequence &Other) const {
  // If both sequences reside in different translation units, they can never
  // contain each other.
  if (Context != Other.Context)
    return false;

  const SourceManager &SM = Context->getSourceManager();

  // Otherwise check if the start and end locations of the current sequence
  // surround the other sequence.
  bool StartIsInBounds =
      SM.isBeforeInTranslationUnit(getStartLoc(), Other.getStartLoc()) ||
      getStartLoc() == Other.getStartLoc();
  if (!StartIsInBounds)
    return false;

  bool EndIsInBounds =
      SM.isBeforeInTranslationUnit(Other.getEndLoc(), getEndLoc()) ||
      Other.getEndLoc() == getEndLoc();
  return EndIsInBounds;
}

StmtSequence::iterator StmtSequence::begin() const {
  if (!holdsSequence()) {
    return &S;
  }
  auto CS = cast<CompoundStmt>(S);
  return CS->body_begin() + StartIndex;
}

StmtSequence::iterator StmtSequence::end() const {
  if (!holdsSequence()) {
    return &S + 1;
  }
  auto CS = cast<CompoundStmt>(S);
  return CS->body_begin() + EndIndex;
}

SourceLocation StmtSequence::getStartLoc() const {
  return front()->getLocStart();
}

SourceLocation StmtSequence::getEndLoc() const { return back()->getLocEnd(); }

namespace {
/// Generates CloneSignatures for a set of statements and stores the results in
/// a CloneDetector object.
class CloneSignatureGenerator {

  CloneDetector &CD;
  ASTContext &Context;

  /// \brief Generates CloneSignatures for all statements in the given statement
  /// tree and stores them in the CloneDetector.
  ///
  /// \param S The root of the given statement tree.
  /// \return The CloneSignature of the root statement.
  CloneDetector::CloneSignature generateSignatures(const Stmt *S) {
    // Create an empty signature that will be filled in this method.
    CloneDetector::CloneSignature Signature;

    // The only relevant data for now is the class of the statement.
    // TODO: Collect statement class specific data.
    Signature.Data.push_back(S->getStmtClass());

    // Storage for the signatures of the direct child statements. This is only
    // needed if the current statement is a CompoundStmt.
    std::vector<CloneDetector::CloneSignature> ChildSignatures;
    const CompoundStmt *CS = dyn_cast<const CompoundStmt>(S);

    // The signature of a statement includes the signatures of its children.
    // Therefore we create the signatures for every child and add them to the
    // current signature.
    for (const Stmt *Child : S->children()) {
      // Some statements like 'if' can have nullptr children that we will skip.
      if (!Child)
        continue;

      // Recursive call to create the signature of the child statement. This
      // will also create and store all clone groups in this child statement.
      auto ChildSignature = generateSignatures(Child);

      // Add the collected data to the signature of the current statement.
      Signature.add(ChildSignature);

      // If the current statement is a CompoundStatement, we need to store the
      // signature for the generation of the sub-sequences.
      if (CS)
        ChildSignatures.push_back(ChildSignature);
    }

    // If the current statement is a CompoundStmt, we also need to create the
    // clone groups from the sub-sequences inside the children.
    if (CS)
      handleSubSequences(CS, ChildSignatures);

    // Save the signature for the current statement in the CloneDetector object.
    CD.add(StmtSequence(S, Context), Signature);

    return Signature;
  }

  /// \brief Adds all possible sub-sequences in the child array of the given
  ///        CompoundStmt to the CloneDetector.
  /// \param CS The given CompoundStmt.
  /// \param ChildSignatures A list of calculated signatures for each child in
  ///                        the given CompoundStmt.
  void handleSubSequences(
      const CompoundStmt *CS,
      const std::vector<CloneDetector::CloneSignature> &ChildSignatures) {

    // FIXME: This function has quadratic runtime right now. Check if skipping
    // this function for too long CompoundStmts is an option.

    // The length of the sub-sequence. We don't need to handle sequences with
    // the length 1 as they are already handled in CollectData().
    for (unsigned Length = 2; Length <= CS->size(); ++Length) {
      // The start index in the body of the CompoundStmt. We increase the
      // position until the end of the sub-sequence reaches the end of the
      // CompoundStmt body.
      for (unsigned Pos = 0; Pos <= CS->size() - Length; ++Pos) {
        // Create an empty signature and add the signatures of all selected
        // child statements to it.
        CloneDetector::CloneSignature SubSignature;

        for (unsigned i = Pos; i < Pos + Length; ++i) {
          SubSignature.add(ChildSignatures[i]);
        }

        // Save the signature together with the information about what children
        // sequence we selected.
        CD.add(StmtSequence(CS, Context, Pos, Pos + Length), SubSignature);
      }
    }
  }

public:
  explicit CloneSignatureGenerator(CloneDetector &CD, ASTContext &Context)
      : CD(CD), Context(Context) {}

  /// \brief Generates signatures for all statements in the given function body.
  void consumeCodeBody(const Stmt *S) { generateSignatures(S); }
};
} // end anonymous namespace

void CloneDetector::analyzeCodeBody(const Decl *D) {
  assert(D);
  assert(D->hasBody());
  CloneSignatureGenerator Generator(*this, D->getASTContext());
  Generator.consumeCodeBody(D->getBody());
}

void CloneDetector::add(const StmtSequence &S,
                        const CloneSignature &Signature) {
  // StringMap only works with StringRefs, so we create one for our data vector.
  auto &Data = Signature.Data;
  StringRef DataRef = StringRef(reinterpret_cast<const char *>(Data.data()),
                                Data.size() * sizeof(unsigned));

  // Search with the help of the signature if we already have encountered a
  // clone of the given StmtSequence.
  auto I = CloneGroupIndexes.find(DataRef);
  if (I == CloneGroupIndexes.end()) {
    // We haven't found an existing clone group, so we create a new clone group
    // for this StmtSequence and store the index of it in our search map.
    CloneGroupIndexes[DataRef] = CloneGroups.size();
    CloneGroups.emplace_back(S, Signature.Complexity);
    return;
  }

  // We have found an existing clone group and can expand it with the given
  // StmtSequence.
  CloneGroups[I->getValue()].Sequences.push_back(S);
}

namespace {
/// \brief Returns true if and only if \p Stmt contains at least one other
/// sequence in the \p Group.
bool containsAnyInGroup(StmtSequence &Stmt,
                            CloneDetector::CloneGroup &Group) {
  for (StmtSequence &GroupStmt : Group.Sequences) {
    if (Stmt.contains(GroupStmt))
      return true;
  }
  return false;
}

/// \brief Returns true if and only if all sequences in \p OtherGroup are
/// contained by a sequence in \p Group.
bool containsGroup(CloneDetector::CloneGroup &Group,
                   CloneDetector::CloneGroup &OtherGroup) {
  // We have less sequences in the current group than we have in the other,
  // so we will never fulfill the requirement for returning true. This is only
  // possible because we know that a sequence in Group can contain at most
  // one sequence in OtherGroup.
  if (Group.Sequences.size() < OtherGroup.Sequences.size())
    return false;

  for (StmtSequence &Stmt : Group.Sequences) {
    if (!containsAnyInGroup(Stmt, OtherGroup))
      return false;
  }
  return true;
}
} // end anonymous namespace

void CloneDetector::findClones(std::vector<CloneGroup> &Result,
                               unsigned MinGroupComplexity) {
  // Add every valid clone group that fulfills the complexity requirement.
  for (const CloneGroup &Group : CloneGroups) {
    if (Group.isValid() && Group.Complexity >= MinGroupComplexity) {
      Result.push_back(Group);
    }
  }

  std::vector<unsigned> IndexesToRemove;

  // Compare every group in the result with the rest. If one groups contains
  // another group, we only need to return the bigger group.
  // Note: This doesn't scale well, so if possible avoid calling any heavy
  // function from this loop to minimize the performance impact.
  for (unsigned i = 0; i < Result.size(); ++i) {
    for (unsigned j = 0; j < Result.size(); ++j) {
      // Don't compare a group with itself.
      if (i == j)
        continue;

      if (containsGroup(Result[j], Result[i])) {
        IndexesToRemove.push_back(i);
        break;
      }
    }
  }

  // Erasing a list of indexes from the vector should be done with decreasing
  // indexes. As IndexesToRemove is constructed with increasing values, we just
  // reverse iterate over it to get the desired order.
  for (auto I = IndexesToRemove.rbegin(); I != IndexesToRemove.rend(); ++I) {
    Result.erase(Result.begin() + *I);
  }
}
