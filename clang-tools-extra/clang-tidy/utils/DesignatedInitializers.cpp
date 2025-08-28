//===--- DesignatedInitializers.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utilities for designated initializers.
///
//===----------------------------------------------------------------------===//

#include "DesignatedInitializers.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"

namespace clang::tidy::utils {

namespace {

/// Returns true if Name is reserved, like _Foo or __Vector_base.
static inline bool isReservedName(llvm::StringRef Name) {
  // This doesn't catch all cases, but the most common.
  return Name.size() >= 2 && Name[0] == '_' &&
         (isUppercase(Name[1]) || Name[1] == '_');
}

// Helper class to iterate over the designator names of an aggregate type.
//
// For an array type, yields [0], [1], [2]...
// For aggregate classes, yields null for each base, then .field1, .field2,
// ...
class AggregateDesignatorNames {
public:
  AggregateDesignatorNames(QualType T) {
    if (!T.isNull()) {
      T = T.getCanonicalType();
      if (T->isArrayType()) {
        IsArray = true;
        Valid = true;
        return;
      }
      if (const RecordDecl *RD = T->getAsRecordDecl()) {
        Valid = true;
        FieldsIt = RD->field_begin();
        FieldsEnd = RD->field_end();
        if (const auto *CRD = llvm::dyn_cast<CXXRecordDecl>(RD)) {
          BasesIt = CRD->bases_begin();
          BasesEnd = CRD->bases_end();
          Valid = CRD->isAggregate();
        }
        OneField = Valid && BasesIt == BasesEnd && FieldsIt != FieldsEnd &&
                   std::next(FieldsIt) == FieldsEnd;
      }
    }
  }
  // Returns false if the type was not an aggregate.
  operator bool() const { return Valid; }
  // Advance to the next element in the aggregate.
  void next() {
    if (IsArray)
      ++Index;
    else if (BasesIt != BasesEnd)
      ++BasesIt;
    else if (FieldsIt != FieldsEnd)
      ++FieldsIt;
  }
  // Print the designator to Out.
  // Returns false if we could not produce a designator for this element.
  bool append(std::string &Out, bool ForSubobject) {
    if (IsArray) {
      Out.push_back('[');
      Out.append(std::to_string(Index));
      Out.push_back(']');
      return true;
    }
    if (BasesIt != BasesEnd)
      return false; // Bases can't be designated. Should we make one up?
    if (FieldsIt != FieldsEnd) {
      llvm::StringRef FieldName;
      if (const IdentifierInfo *II = FieldsIt->getIdentifier())
        FieldName = II->getName();

      // For certain objects, their subobjects may be named directly.
      if (ForSubobject &&
          (FieldsIt->isAnonymousStructOrUnion() ||
           // std::array<int,3> x = {1,2,3}. Designators not strictly valid!
           (OneField && isReservedName(FieldName))))
        return true;

      if (!FieldName.empty() && !isReservedName(FieldName)) {
        Out.push_back('.');
        Out.append(FieldName.begin(), FieldName.end());
        return true;
      }
      return false;
    }
    return false;
  }

private:
  bool Valid = false;
  bool IsArray = false;
  bool OneField = false; // e.g. std::array { T __elements[N]; }
  unsigned Index = 0;
  CXXRecordDecl::base_class_const_iterator BasesIt;
  CXXRecordDecl::base_class_const_iterator BasesEnd;
  RecordDecl::field_iterator FieldsIt;
  RecordDecl::field_iterator FieldsEnd;
};

// Collect designator labels describing the elements of an init list.
//
// This function contributes the designators of some (sub)object, which is
// represented by the semantic InitListExpr Sem.
// This includes any nested subobjects, but *only* if they are part of the
// same original syntactic init list (due to brace elision). In other words,
// it may descend into subobjects but not written init-lists.
//
// For example: struct Outer { Inner a,b; }; struct Inner { int x, y; }
//              Outer o{{1, 2}, 3};
// This function will be called with Sem = { {1, 2}, {3, ImplicitValue} }
// It should generate designators '.a:' and '.b.x:'.
// '.a:' is produced directly without recursing into the written sublist.
// (The written sublist will have a separate collectDesignators() call later).
// Recursion with Prefix='.b' and Sem = {3, ImplicitValue} produces '.b.x:'.
void collectDesignators(const InitListExpr *Sem,
                        llvm::DenseMap<SourceLocation, std::string> &Out,
                        const llvm::DenseSet<SourceLocation> &NestedBraces,
                        std::string &Prefix) {
  if (!Sem || Sem->isTransparent())
    return;
  assert(Sem->isSemanticForm());

  // The elements of the semantic form all correspond to direct subobjects of
  // the aggregate type. `Fields` iterates over these subobject names.
  AggregateDesignatorNames Fields(Sem->getType());
  if (!Fields)
    return;
  for (const Expr *Init : Sem->inits()) {
    auto Next = llvm::make_scope_exit([&, Size(Prefix.size())] {
      Fields.next();       // Always advance to the next subobject name.
      Prefix.resize(Size); // Erase any designator we appended.
    });
    // Skip for a broken initializer or if it is a "hole" in a subobject that
    // was not explicitly initialized.
    if (!Init || llvm::isa<ImplicitValueInitExpr>(Init))
      continue;

    const auto *BraceElidedSubobject = llvm::dyn_cast<InitListExpr>(Init);
    if (BraceElidedSubobject &&
        NestedBraces.contains(BraceElidedSubobject->getLBraceLoc()))
      BraceElidedSubobject = nullptr; // there were braces!

    if (!Fields.append(Prefix, BraceElidedSubobject != nullptr))
      continue; // no designator available for this subobject
    if (BraceElidedSubobject) {
      // If the braces were elided, this aggregate subobject is initialized
      // inline in the same syntactic list.
      // Descend into the semantic list describing the subobject.
      // (NestedBraces are still correct, they're from the same syntactic
      // list).
      collectDesignators(BraceElidedSubobject, Out, NestedBraces, Prefix);
      continue;
    }
    Out.try_emplace(Init->getBeginLoc(), Prefix);
  }
}

} // namespace

llvm::DenseMap<SourceLocation, std::string>
getUnwrittenDesignators(const InitListExpr *Syn) {
  assert(Syn->isSyntacticForm());

  // collectDesignators needs to know which InitListExprs in the semantic tree
  // were actually written, but InitListExpr::isExplicit() lies.
  // Instead, record where braces of sub-init-lists occur in the syntactic form.
  llvm::DenseSet<SourceLocation> NestedBraces;
  for (const Expr *Init : Syn->inits())
    if (auto *Nested = llvm::dyn_cast<InitListExpr>(Init))
      NestedBraces.insert(Nested->getLBraceLoc());

  // Traverse the semantic form to find the designators.
  // We use their SourceLocation to correlate with the syntactic form later.
  llvm::DenseMap<SourceLocation, std::string> Designators;
  std::string EmptyPrefix;
  collectDesignators(Syn->isSemanticForm() ? Syn : Syn->getSemanticForm(),
                     Designators, NestedBraces, EmptyPrefix);
  return Designators;
}

} // namespace clang::tidy::utils
