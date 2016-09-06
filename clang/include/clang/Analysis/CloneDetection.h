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
#include "llvm/ADT/StringMap.h"

#include <vector>

namespace clang {

class Stmt;
class Decl;
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
    /// \brief Holds all relevant data of a StmtSequence.
    ///
    /// If this variable is equal for two different StmtSequences, then they can
    /// be considered clones of each other.
    std::vector<DataPiece> Data;

    /// \brief The complexity of the StmtSequence.
    ///
    /// This scalar value serves as a simple way of filtering clones that are
    /// too small to be reported. A greater value indicates that the related
    /// StmtSequence is probably more interesting to the user.
    unsigned Complexity;

    /// \brief Creates an empty CloneSignature without any data.
    CloneSignature() : Complexity(1) {}

    CloneSignature(const std::vector<unsigned> &Data, unsigned Complexity)
        : Data(Data), Complexity(Complexity) {}

    /// \brief Adds the data from the given CloneSignature to this one.
    void add(const CloneSignature &Other) {
      Data.insert(Data.end(), Other.Data.begin(), Other.Data.end());
      Complexity += Other.Complexity;
    }
  };

  /// Holds group of StmtSequences that are clones of each other and the
  /// complexity value (see CloneSignature::Complexity) that all stored
  /// StmtSequences have in common.
  struct CloneGroup {
    std::vector<StmtSequence> Sequences;
    unsigned Complexity;

    CloneGroup(const StmtSequence &Seq, unsigned Complexity)
        : Complexity(Complexity) {
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
  void findClones(std::vector<CloneGroup> &Result, unsigned MinGroupComplexity);

private:
  /// Stores all found clone groups including invalid groups with only a single
  /// statement.
  std::vector<CloneGroup> CloneGroups;
  /// Maps search data to its related index in the \p CloneGroups vector.
  llvm::StringMap<std::size_t> CloneGroupIndexes;
};

} // end namespace clang

#endif // LLVM_CLANG_AST_CLONEDETECTION_H
