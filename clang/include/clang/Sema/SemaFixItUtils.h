//===--- SemaFixItUtils.h - Sema FixIts -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines helper classes for generation of Sema FixItHints.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SEMA_SEMAFIXITUTILS_H
#define LLVM_CLANG_SEMA_SEMAFIXITUTILS_H

#include "clang/AST/Expr.h"

namespace clang {

enum OverloadFixItKind {
  OFIK_Undefined = 0,
  OFIK_Dereference,
  OFIK_TakeAddress,
  OFIK_RemoveDereference,
  OFIK_RemoveTakeAddress
};

class Sema;

/// The class facilities generation and storage of conversion FixIts. Hints for
/// new conversions are added using TryToFixConversion method. The default type
/// conversion checker can be reset.
struct ConversionFixItGenerator {
  /// Performs a simple check to see if From type can be converted to To type.
  static bool compareTypesSimple(CanQualType From,
                                 CanQualType To,
                                 Sema &S,
                                 SourceLocation Loc,
                                 ExprValueKind FromVK);

  /// The list of Hints generated so far.
  std::vector<FixItHint> Hints;

  /// The number of Conversions fixed. This can be different from the size
  /// of the Hints vector since we allow multiple FixIts per conversion.
  unsigned NumConversionsFixed;

  /// The type of fix applied. If multiple conversions are fixed, corresponds
  /// to the kid of the very first conversion.
  OverloadFixItKind Kind;

  typedef bool (*TypeComparisonFuncTy) (const CanQualType FromTy,
                                        const CanQualType ToTy,
                                        Sema &S,
                                        SourceLocation Loc,
                                        ExprValueKind FromVK);
  /// The type comparison function used to decide if expression FromExpr of
  /// type FromTy can be converted to ToTy. For example, one could check if
  /// an implicit conversion exists. Returns true if comparison exists.
  TypeComparisonFuncTy CompareTypes;

  ConversionFixItGenerator(TypeComparisonFuncTy Foo): NumConversionsFixed(0),
                                                      Kind(OFIK_Undefined),
                                                      CompareTypes(Foo) {}

  ConversionFixItGenerator(): NumConversionsFixed(0),
                              Kind(OFIK_Undefined),
                              CompareTypes(compareTypesSimple) {}

  /// Resets the default conversion checker method.
  void setConversionChecker(TypeComparisonFuncTy Foo) {
    CompareTypes = Foo;
  }

  /// If possible, generates and stores a fix for the given conversion.
  bool tryToFixConversion(const Expr *FromExpr,
                          const QualType FromQTy, const QualType ToQTy,
                          Sema &S);

  void clear() {
    Hints.clear();
    NumConversionsFixed = 0;
  }

  bool isNull() {
    return (NumConversionsFixed == 0);
  }
};

/* TO_UPSTREAM(BoundsSafety) ON */
class BoundsSafetyFixItUtils {
public:
  /// Try to find the SourceLocation where a bounds-safety attribute could
  /// be inserted on a pointer. Note this method does not check if there is an
  /// attribute already present. Clients should handle this themselves.
  ///
  ///
  /// \param TL - TypeLoc that the attribute could be added to
  /// \param S - Sema instance
  ///
  /// \return a tuple of the SourceLocation where insertion could be performed
  /// and a boolean that is true iff a space should be inserted after the
  /// inserted attribute. If the returned SourceLocation is invalid no insertion
  /// point could be found.
  static std::tuple<SourceLocation, bool>
  FindPointerAttrInsertPoint(const TypeLoc TL, Sema &S);

  /// Try to create a FixItHint that adds the provided bounds-safety attribute
  /// as a new attribute to the variable declaration. Note this method does
  /// not check for existing attributes. Clients should have this themselves.
  ///
  /// \param VD - Variable Declaration to suggest FixIt for. This Variable
  /// should have a pointer type.
  /// \param Attribute - The string representation of the Attribute to add.
  /// \param S - Sema instance
  ///
  /// \return A FixIt hint that adds the supplied Attribute to the type
  /// specifier on the variable declaration. If creating the FixIt fails the
  /// returned FixIt will be invalid.
  static FixItHint
  CreateAnnotatePointerDeclFixIt(const VarDecl *VD,
                                 const llvm::StringRef Attribute, Sema &S);

  /// Try to create a FixItHint that adds the provided bounds-safety attribute
  /// as a new attribute to the field declaration. Note this method does
  /// not check for existing attributes. Clients should have this themselves.
  ///
  /// \param VD - Field Declaration to suggest FixIt for. This field
  /// should have a pointer type.
  /// \param Attribute - The string representation of the Attribute to add.
  /// \param S - Sema instance
  ///
  /// \return A FixIt hint that adds the supplied Attribute to the type
  /// specifier on the field declaration. If creating the FixIt fails the
  /// returned FixIt will be invalid.
  static FixItHint
  CreateAnnotatePointerDeclFixIt(const FieldDecl *FD,
                                 const llvm::StringRef Attribute, Sema &S);

  /// Try to create a FixItHint that adds the provided bounds-safety attribute
  /// as a new attribute to the variable declaration and all of its previous
  /// declarations. Note only global variables may have previous declarations.
  /// Note this method does not check for existing attributes.
  /// Clients should have this themselves.
  ///
  /// \param VD - Variable Declaration to suggest FixIt for. This Variable
  /// should have a pointer type.
  /// \param Attribute - The string representation of the Attribute to add.
  /// \param S - Sema instance
  /// \param FixIts - A SmallVector of (FixIts hint, VarDecl) tuples. Every
  /// valid FixHit that can be created will be added to this SmallVector. Each
  /// FixIt hint adds the supplied Attribute to the type specifier on each of
  /// the variable declarations.
  static void CreateAnnotateAllPointerDeclsFixIts(
      const VarDecl *VD, const llvm::StringRef Attribute, Sema &S,
      llvm::SmallVectorImpl<std::tuple<FixItHint, const DeclaratorDecl *>>
          &FixIts);

private:
  static FixItHint CreateAnnotateVarDeclOrFieldDeclFixIt(
      const DeclaratorDecl *VD, const llvm::StringRef Attribute, Sema &S);
};
/* TO_UPSTREAM(BoundsSafety) OFF */

} // endof namespace clang
#endif
