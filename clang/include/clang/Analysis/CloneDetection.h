//===--- CloneDetection.h - Finds code clones in an AST ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// /file
/// This file defines classes for searching and anlyzing source code clones.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_CLONEDETECTION_H
#define LLVM_CLANG_AST_CLONEDETECTION_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringMap.h"

#include <vector>

namespace clang {

class Stmt;
class Decl;
class VarDecl;
class ASTContext;
class CompoundStmt;

/// \brief Identifies a list of statements.
///
/// Can either identify a single arbitrary Stmt object, a continuous sequence of
/// child statements inside a CompoundStmt or no statements at all.
class StmtSequence {
  /// If this object identifies a sequence of statements inside a CompoundStmt,
  /// S points to this CompoundStmt. If this object only identifies a single
  /// Stmt, then S is a pointer to this Stmt.
  const Stmt *S;

  /// The related ASTContext for S.
  ASTContext *Context;

  /// If EndIndex is non-zero, then S is a CompoundStmt and this StmtSequence
  /// instance is representing the CompoundStmt children inside the array
  /// [StartIndex, EndIndex).
  unsigned StartIndex;
  unsigned EndIndex;

public:
  /// \brief Constructs a StmtSequence holding multiple statements.
  ///
  /// The resulting StmtSequence identifies a continuous sequence of statements
  /// in the body of the given CompoundStmt. Which statements of the body should
  /// be identified needs to be specified by providing a start and end index
  /// that describe a non-empty sub-array in the body of the given CompoundStmt.
  ///
  /// \param Stmt A CompoundStmt that contains all statements in its body.
  /// \param Context The ASTContext for the given CompoundStmt.
  /// \param StartIndex The inclusive start index in the children array of
  ///                   \p Stmt
  /// \param EndIndex The exclusive end index in the children array of \p Stmt.
  StmtSequence(const CompoundStmt *Stmt, ASTContext &Context,
               unsigned StartIndex, unsigned EndIndex);

  /// \brief Constructs a StmtSequence holding a single statement.
  ///
  /// \param Stmt An arbitrary Stmt.
  /// \param Context The ASTContext for the given Stmt.
  StmtSequence(const Stmt *Stmt, ASTContext &Context);

  /// \brief Constructs an empty StmtSequence.
  StmtSequence();

  typedef const Stmt *const *iterator;

  /// Returns an iterator pointing to the first statement in this sequence.
  iterator begin() const;

  /// Returns an iterator pointing behind the last statement in this sequence.
  iterator end() const;

  /// Returns the first statement in this sequence.
  ///
  /// This method should only be called on a non-empty StmtSequence object.
  const Stmt *front() const {
    assert(!empty());
    return begin()[0];
  }

  /// Returns the last statement in this sequence.
  ///
  /// This method should only be called on a non-empty StmtSequence object.
  const Stmt *back() const {
    assert(!empty());
    return begin()[size() - 1];
  }

  /// Returns the number of statements this object holds.
  unsigned size() const {
    if (holdsSequence())
      return EndIndex - StartIndex;
    if (S == nullptr)
      return 0;
    return 1;
  }

  /// Returns true if and only if this StmtSequence contains no statements.
  bool empty() const { return size() == 0; }

  /// Returns the related ASTContext for the stored Stmts.
  ASTContext &getASTContext() const {
    assert(Context);
    return *Context;
  }

  /// Returns true if this objects holds a list of statements.
  bool holdsSequence() const { return EndIndex != 0; }

  /// Returns the start sourcelocation of the first statement in this sequence.
  ///
  /// This method should only be called on a non-empty StmtSequence object.
  SourceLocation getStartLoc() const;

  /// Returns the end sourcelocation of the last statement in this sequence.
  ///
  /// This method should only be called on a non-empty StmtSequence object.
  SourceLocation getEndLoc() const;

  /// Returns the source range of the whole sequence - from the beginning
  /// of the first statement to the end of the last statement.
  SourceRange getSourceRange() const;

  bool operator==(const StmtSequence &Other) const {
    return std::tie(S, StartIndex, EndIndex) ==
           std::tie(Other.S, Other.StartIndex, Other.EndIndex);
  }

  bool operator!=(const StmtSequence &Other) const {
    return std::tie(S, StartIndex, EndIndex) !=
           std::tie(Other.S, Other.StartIndex, Other.EndIndex);
  }

  /// Returns true if and only if this sequence covers a source range that
  /// contains the source range of the given sequence \p Other.
  ///
  /// This method should only be called on a non-empty StmtSequence object
  /// and passed a non-empty StmtSequence object.
  bool contains(const StmtSequence &Other) const;
};

/// \brief Searches for clones in source code.
///
/// First, this class needs a translation unit which is passed via
/// \p analyzeTranslationUnit . It will then generate and store search data
/// for all statements inside the given translation unit.
/// Afterwards the generated data can be used to find code clones by calling
/// \p findClones .
///
/// This class only searches for clones in exectuable source code
/// (e.g. function bodies). Other clones (e.g. cloned comments or declarations)
/// are not supported.
class CloneDetector {
public:
  typedef unsigned DataPiece;

  /// Holds the data about a StmtSequence that is needed during the search for
  /// code clones.
  struct CloneSignature {
    /// \brief The hash code of the StmtSequence.
    ///
    /// The initial clone groups that are formed during the search for clones
    /// consist only of Sequences that share the same hash code. This makes this
    /// value the central part of this heuristic that is needed to find clones
    /// in a performant way. For this to work, the type of this variable
    /// always needs to be small and fast to compare.
    ///
    /// Also, StmtSequences that are clones of each others have to share
    /// the same hash code. StmtSequences that are not clones of each other
    /// shouldn't share the same hash code, but if they do, it will only
    /// degrade the performance of the hash search but doesn't influence
    /// the correctness of the result.
    size_t Hash;

    /// \brief The complexity of the StmtSequence.
    ///
    /// This value gives an approximation on how many direct or indirect child
    /// statements are contained in the related StmtSequence. In general, the
    /// greater this value, the greater the amount of statements. However, this
    /// is only an approximation and the actual amount of statements can be
    /// higher or lower than this value. Statements that are generated by the
    /// compiler (e.g. macro expansions) for example barely influence the
    /// complexity value.
    ///
    /// The main purpose of this value is to filter clones that are too small
    /// and therefore probably not interesting enough for the user.
    unsigned Complexity;

    /// \brief Creates an empty CloneSignature without any data.
    CloneSignature() : Complexity(1) {}

    CloneSignature(llvm::hash_code Hash, unsigned Complexity)
        : Hash(Hash), Complexity(Complexity) {}
  };

  /// Holds group of StmtSequences that are clones of each other and the
  /// complexity value (see CloneSignature::Complexity) that all stored
  /// StmtSequences have in common.
  struct CloneGroup {
    std::vector<StmtSequence> Sequences;
    CloneSignature Signature;

    CloneGroup() {}

    CloneGroup(const StmtSequence &Seq, CloneSignature Signature)
        : Signature(Signature) {
      Sequences.push_back(Seq);
    }

    /// \brief Returns false if and only if this group should be skipped when
    ///        searching for clones.
    bool isValid() const {
      // A clone group with only one member makes no sense, so we skip them.
      return Sequences.size() > 1;
    }
  };

  /// \brief Generates and stores search data for all statements in the body of
  ///        the given Decl.
  void analyzeCodeBody(const Decl *D);

  /// \brief Stores the CloneSignature to allow future querying.
  void add(const StmtSequence &S, const CloneSignature &Signature);

  /// \brief Searches the provided statements for clones.
  ///
  /// \param Result Output parameter that is filled with a list of found
  ///               clone groups. Each group contains multiple StmtSequences
  ///               that were identified to be clones of each other.
  /// \param MinGroupComplexity Only return clones which have at least this
  ///                           complexity value.
  /// \param CheckPatterns Returns only clone groups in which the referenced
  ///                      variables follow the same pattern.
  void findClones(std::vector<CloneGroup> &Result, unsigned MinGroupComplexity,
                  bool CheckPatterns = true);

  /// \brief Describes two clones that reference their variables in a different
  ///        pattern which could indicate a programming error.
  struct SuspiciousClonePair {
    /// \brief Utility class holding the relevant information about a single
    ///        clone in this pair.
    struct SuspiciousCloneInfo {
      /// The variable which referencing in this clone was against the pattern.
      const VarDecl *Variable;
      /// Where the variable was referenced.
      const Stmt *Mention;
      /// The variable that should have been referenced to follow the pattern.
      /// If Suggestion is a nullptr then it's not possible to fix the pattern
      /// by referencing a different variable in this clone.
      const VarDecl *Suggestion;
      SuspiciousCloneInfo(const VarDecl *Variable, const Stmt *Mention,
                          const VarDecl *Suggestion)
          : Variable(Variable), Mention(Mention), Suggestion(Suggestion) {}
      SuspiciousCloneInfo() {}
    };
    /// The first clone in the pair which always has a suggested variable.
    SuspiciousCloneInfo FirstCloneInfo;
    /// This other clone in the pair which can have a suggested variable.
    SuspiciousCloneInfo SecondCloneInfo;
  };

  /// \brief Searches the provided statements for pairs of clones that don't
  ///        follow the same pattern when referencing variables.
  /// \param Result Output parameter that will contain the clone pairs.
  /// \param MinGroupComplexity Only clone pairs in which the clones have at
  ///                           least this complexity value.
  void findSuspiciousClones(std::vector<SuspiciousClonePair> &Result,
                            unsigned MinGroupComplexity);

private:
  /// Stores all encountered StmtSequences alongside their CloneSignature.
  std::vector<std::pair<CloneSignature, StmtSequence>> Sequences;
};

} // end namespace clang

#endif // LLVM_CLANG_AST_CLONEDETECTION_H
