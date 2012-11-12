//===--- TypoCorrection.h - Class for typo correction results ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TypoCorrection class, which stores the results of
// Sema's typo correction (Sema::CorrectTypo).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_TYPOCORRECTION_H
#define LLVM_CLANG_SEMA_TYPOCORRECTION_H

#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

/// @brief Simple class containing the result of Sema::CorrectTypo
class TypoCorrection {
public:
  // "Distance" for unusable corrections
  static const unsigned InvalidDistance = ~0U;
  // The largest distance still considered valid (larger edit distances are
  // mapped to InvalidDistance by getEditDistance).
  static const unsigned MaximumDistance = 10000U;

  // Relative weightings of the "edit distance" components. The higher the
  // weight, the more of a penalty to fitness the component will give (higher
  // weights mean greater contribution to the total edit distance, with the
  // best correction candidates having the lowest edit distance).
  static const unsigned CharDistanceWeight = 100U;
  static const unsigned QualifierDistanceWeight = 110U;
  static const unsigned CallbackDistanceWeight = 150U;

  TypoCorrection(const DeclarationName &Name, NamedDecl *NameDecl,
                 NestedNameSpecifier *NNS=0, unsigned CharDistance=0,
                 unsigned QualifierDistance=0)
      : CorrectionName(Name), CorrectionNameSpec(NNS),
      CharDistance(CharDistance), QualifierDistance(QualifierDistance),
      CallbackDistance(0) {
    if (NameDecl)
      CorrectionDecls.push_back(NameDecl);
  }

  TypoCorrection(NamedDecl *Name, NestedNameSpecifier *NNS=0,
                 unsigned CharDistance=0)
      : CorrectionName(Name->getDeclName()), CorrectionNameSpec(NNS),
      CharDistance(CharDistance), QualifierDistance(0), CallbackDistance(0) {
    if (Name)
      CorrectionDecls.push_back(Name);
  }

  TypoCorrection(DeclarationName Name, NestedNameSpecifier *NNS=0,
                 unsigned CharDistance=0)
      : CorrectionName(Name), CorrectionNameSpec(NNS),
      CharDistance(CharDistance), QualifierDistance(0), CallbackDistance(0) {}

  TypoCorrection()
      : CorrectionNameSpec(0), CharDistance(0), QualifierDistance(0),
      CallbackDistance(0) {}

  /// \brief Gets the DeclarationName of the typo correction
  DeclarationName getCorrection() const { return CorrectionName; }
  IdentifierInfo* getCorrectionAsIdentifierInfo() const {
    return CorrectionName.getAsIdentifierInfo();
  }

  /// \brief Gets the NestedNameSpecifier needed to use the typo correction
  NestedNameSpecifier* getCorrectionSpecifier() const {
    return CorrectionNameSpec;
  }
  void setCorrectionSpecifier(NestedNameSpecifier* NNS) {
    CorrectionNameSpec = NNS;
  }

  void setQualifierDistance(unsigned ED) {
    QualifierDistance = ED;
  }

  void setCallbackDistance(unsigned ED) {
    CallbackDistance = ED;
  }

  // Convert the given weighted edit distance to a roughly equivalent number of
  // single-character edits (typically for comparison to the length of the
  // string being edited).
  static unsigned NormalizeEditDistance(unsigned ED) {
    if (ED > MaximumDistance)
      return InvalidDistance;
    return (ED + CharDistanceWeight / 2) / CharDistanceWeight;
  }

  /// \brief Gets the "edit distance" of the typo correction from the typo.
  /// If Normalized is true, scale the distance down by the CharDistanceWeight
  /// to return the edit distance in terms of single-character edits.
  unsigned getEditDistance(bool Normalized = true) const {
    if (CharDistance > MaximumDistance || QualifierDistance > MaximumDistance ||
        CallbackDistance > MaximumDistance)
      return InvalidDistance;
    unsigned ED =
        CharDistance * CharDistanceWeight +
        QualifierDistance * QualifierDistanceWeight +
        CallbackDistance * CallbackDistanceWeight;
    if (ED > MaximumDistance)
      return InvalidDistance;
    // Half the CharDistanceWeight is added to ED to simulate rounding since
    // integer division truncates the value (i.e. round-to-nearest-int instead
    // of round-to-zero).
    return Normalized ? NormalizeEditDistance(ED) : ED;
  }

  /// \brief Gets the pointer to the declaration of the typo correction
  NamedDecl* getCorrectionDecl() const {
    return hasCorrectionDecl() ? *(CorrectionDecls.begin()) : 0;
  }
  template <class DeclClass>
  DeclClass *getCorrectionDeclAs() const {
    return dyn_cast_or_null<DeclClass>(getCorrectionDecl());
  }
  
  /// \brief Clears the list of NamedDecls before adding the new one.
  void setCorrectionDecl(NamedDecl *CDecl) {
    CorrectionDecls.clear();
    addCorrectionDecl(CDecl);
  }

  /// \brief Add the given NamedDecl to the list of NamedDecls that are the
  /// declarations associated with the DeclarationName of this TypoCorrection
  void addCorrectionDecl(NamedDecl *CDecl);

  std::string getAsString(const LangOptions &LO) const;
  std::string getQuoted(const LangOptions &LO) const {
    return "'" + getAsString(LO) + "'";
  }

  /// \brief Returns whether this TypoCorrection has a non-empty DeclarationName
  operator bool() const { return bool(CorrectionName); }

  /// \brief Mark this TypoCorrection as being a keyword.
  /// Since addCorrectionDeclsand setCorrectionDecl don't allow NULL to be
  /// added to the list of the correction's NamedDecl pointers, NULL is added
  /// as the only element in the list to mark this TypoCorrection as a keyword.
  void makeKeyword() {
    CorrectionDecls.clear();
    CorrectionDecls.push_back(0);
  }

  // Check if this TypoCorrection is a keyword by checking if the first
  // item in CorrectionDecls is NULL.
  bool isKeyword() const {
    return !CorrectionDecls.empty() &&
        CorrectionDecls.front() == 0;
  }

  // Check if this TypoCorrection is the given keyword.
  template<std::size_t StrLen>
  bool isKeyword(const char (&Str)[StrLen]) const {
    return isKeyword() && getCorrectionAsIdentifierInfo()->isStr(Str);
  }

  // Returns true if the correction either is a keyword or has a known decl.
  bool isResolved() const { return !CorrectionDecls.empty(); }

  bool isOverloaded() const {
    return CorrectionDecls.size() > 1;
  }

  void setCorrectionRange(CXXScopeSpec* SS,
                          const DeclarationNameInfo &TypoName) {
    CorrectionRange.setBegin(CorrectionNameSpec && SS ? SS->getBeginLoc()
                                                      : TypoName.getLoc());
    CorrectionRange.setEnd(TypoName.getLoc());
  }

  SourceRange getCorrectionRange() const {
    return CorrectionRange;
  }

  typedef llvm::SmallVector<NamedDecl*, 1>::iterator decl_iterator;
  decl_iterator begin() {
    return isKeyword() ? CorrectionDecls.end() : CorrectionDecls.begin();
  }
  decl_iterator end() { return CorrectionDecls.end(); }
  typedef llvm::SmallVector<NamedDecl*, 1>::const_iterator const_decl_iterator;
  const_decl_iterator begin() const {
    return isKeyword() ? CorrectionDecls.end() : CorrectionDecls.begin();
  }
  const_decl_iterator end() const { return CorrectionDecls.end(); }

private:
  bool hasCorrectionDecl() const {
    return (!isKeyword() && !CorrectionDecls.empty());
  }

  // Results.
  DeclarationName CorrectionName;
  NestedNameSpecifier *CorrectionNameSpec;
  llvm::SmallVector<NamedDecl*, 1> CorrectionDecls;
  unsigned CharDistance;
  unsigned QualifierDistance;
  unsigned CallbackDistance;
  SourceRange CorrectionRange;
};

/// @brief Base class for callback objects used by Sema::CorrectTypo to check
/// the validity of a potential typo correction.
class CorrectionCandidateCallback {
 public:
  static const unsigned InvalidDistance = TypoCorrection::InvalidDistance;

  CorrectionCandidateCallback()
      : WantTypeSpecifiers(true), WantExpressionKeywords(true),
        WantCXXNamedCasts(true), WantRemainingKeywords(true),
        WantObjCSuper(false),
        IsObjCIvarLookup(false) {}

  virtual ~CorrectionCandidateCallback() {}

  /// \brief Simple predicate used by the default RankCandidate to
  /// determine whether to return an edit distance of 0 or InvalidDistance.
  /// This can be overrided by validators that only need to determine if a
  /// candidate is viable, without ranking potentially viable candidates.
  /// Only ValidateCandidate or RankCandidate need to be overriden by a
  /// callback wishing to check the viability of correction candidates.
  virtual bool ValidateCandidate(const TypoCorrection &candidate) {
    return true;
  }

  /// \brief Method used by Sema::CorrectTypo to assign an "edit distance" rank
  /// to a candidate (where a lower value represents a better candidate), or
  /// returning InvalidDistance if the candidate is not at all viable. For
  /// validation callbacks that only need to determine if a candidate is viable,
  /// the default RankCandidate returns either 0 or InvalidDistance depending
  /// whether ValidateCandidate returns true or false.
  virtual unsigned RankCandidate(const TypoCorrection &candidate) {
    return ValidateCandidate(candidate) ? 0 : InvalidDistance;
  }

  // Flags for context-dependent keywords.
  // TODO: Expand these to apply to non-keywords or possibly remove them.
  bool WantTypeSpecifiers;
  bool WantExpressionKeywords;
  bool WantCXXNamedCasts;
  bool WantRemainingKeywords;
  bool WantObjCSuper;
  // Temporary hack for the one case where a CorrectTypoContext enum is used
  // when looking up results.
  bool IsObjCIvarLookup;
};

/// @brief Simple template class for restricting typo correction candidates
/// to ones having a single Decl* of the given type.
template <class C>
class DeclFilterCCC : public CorrectionCandidateCallback {
 public:
  virtual bool ValidateCandidate(const TypoCorrection &candidate) {
    return candidate.getCorrectionDeclAs<C>();
  }
};

}

#endif
