//===- ASTStructuralEquivalence.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the StructuralEquivalenceContext class which checks for
//  structural equivalence between types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTSTRUCTURALEQUIVALENCE_H
#define LLVM_CLANG_AST_ASTSTRUCTURALEQUIVALENCE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include <deque>
#include <utility>

namespace clang {

class ASTContext;
class Decl;
class DiagnosticBuilder;
class QualType;
class RecordDecl;
class SourceLocation;

struct StructuralEquivalenceContext {
  /// AST contexts for which we are checking structural equivalence.
  ASTContext &FromCtx, &ToCtx;

  /// The set of "tentative" equivalences between two canonical
  /// declarations, mapping from a declaration in the first context to the
  /// declaration in the second context that we believe to be equivalent.
  llvm::DenseMap<Decl *, Decl *> TentativeEquivalences;

  /// Queue of declarations in the first context whose equivalence
  /// with a declaration in the second context still needs to be verified.
  std::deque<Decl *> DeclsToCheck;

  /// Declaration (from, to) pairs that are known not to be equivalent
  /// (which we have already complained about).
  llvm::DenseSet<std::pair<Decl *, Decl *>> &NonEquivalentDecls;

  /// Whether we're being strict about the spelling of types when
  /// unifying two types.
  bool StrictTypeSpelling;

  /// Whether warn or error on tag type mismatches.
  bool ErrorOnTagTypeMismatch;

  /// Whether to complain about failures.
  bool Complain;

  /// \c true if the last diagnostic came from ToCtx.
  bool LastDiagFromC2 = false;

  StructuralEquivalenceContext(
      ASTContext &FromCtx, ASTContext &ToCtx,
      llvm::DenseSet<std::pair<Decl *, Decl *>> &NonEquivalentDecls,
      bool StrictTypeSpelling = false, bool Complain = true,
      bool ErrorOnTagTypeMismatch = false)
      : FromCtx(FromCtx), ToCtx(ToCtx), NonEquivalentDecls(NonEquivalentDecls),
        StrictTypeSpelling(StrictTypeSpelling),
        ErrorOnTagTypeMismatch(ErrorOnTagTypeMismatch), Complain(Complain) {}

  DiagnosticBuilder Diag1(SourceLocation Loc, unsigned DiagID);
  DiagnosticBuilder Diag2(SourceLocation Loc, unsigned DiagID);

  /// Determine whether the two declarations are structurally
  /// equivalent.
  bool IsStructurallyEquivalent(Decl *D1, Decl *D2);

  /// Determine whether the two types are structurally equivalent.
  bool IsStructurallyEquivalent(QualType T1, QualType T2);

  /// Find the index of the given anonymous struct/union within its
  /// context.
  ///
  /// \returns Returns the index of this anonymous struct/union in its context,
  /// including the next assigned index (if none of them match). Returns an
  /// empty option if the context is not a record, i.e.. if the anonymous
  /// struct/union is at namespace or block scope.
  ///
  /// FIXME: This is needed by ASTImporter and ASTStructureEquivalence. It
  /// probably makes more sense in some other common place then here.
  static llvm::Optional<unsigned>
  findUntaggedStructOrUnionIndex(RecordDecl *Anon);

private:
  /// Finish checking all of the structural equivalences.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool Finish();
};

} // namespace clang

#endif // LLVM_CLANG_AST_ASTSTRUCTURALEQUIVALENCE_H
