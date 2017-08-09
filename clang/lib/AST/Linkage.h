//===----- Linkage.h - Linkage calculation-related utilities ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides AST-internal utilities for linkage and visibility
// calculation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_AST_LINKAGE_H
#define LLVM_CLANG_LIB_AST_LINKAGE_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

namespace clang {
enum : unsigned {
  IgnoreExplicitVisibilityBit = 2,
  IgnoreAllVisibilityBit = 4
};

/// Kinds of LV computation.  The linkage side of the computation is
/// always the same, but different things can change how visibility is
/// computed.
enum LVComputationKind {
  /// Do an LV computation for, ultimately, a type.
  /// Visibility may be restricted by type visibility settings and
  /// the visibility of template arguments.
  LVForType = NamedDecl::VisibilityForType,

  /// Do an LV computation for, ultimately, a non-type declaration.
  /// Visibility may be restricted by value visibility settings and
  /// the visibility of template arguments.
  LVForValue = NamedDecl::VisibilityForValue,

  /// Do an LV computation for, ultimately, a type that already has
  /// some sort of explicit visibility.  Visibility may only be
  /// restricted by the visibility of template arguments.
  LVForExplicitType = (LVForType | IgnoreExplicitVisibilityBit),

  /// Do an LV computation for, ultimately, a non-type declaration
  /// that already has some sort of explicit visibility.  Visibility
  /// may only be restricted by the visibility of template arguments.
  LVForExplicitValue = (LVForValue | IgnoreExplicitVisibilityBit),

  /// Do an LV computation when we only care about the linkage.
  LVForLinkageOnly =
      LVForValue | IgnoreExplicitVisibilityBit | IgnoreAllVisibilityBit
};
} // namespace clang

namespace llvm {
template <> struct DenseMapInfo<clang::LVComputationKind> {
  static inline clang::LVComputationKind getEmptyKey() {
    return static_cast<clang::LVComputationKind>(-1);
  }
  static inline clang::LVComputationKind getTombstoneKey() {
    return static_cast<clang::LVComputationKind>(-2);
  }
  static unsigned getHashValue(const clang::LVComputationKind &Val) {
    return Val;
  }
  static bool isEqual(const clang::LVComputationKind &LHS,
                      const clang::LVComputationKind &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

namespace clang {
class LinkageComputer {
  // We have a cache for repeated linkage/visibility computations. This saves us
  // from exponential behavior in heavily templated code, such as:
  //
  // template <typename T, typename V> struct {};
  // using A = int;
  // using B = Foo<A, A>;
  // using C = Foo<B, B>;
  // using D = Foo<C, C>;
  using QueryType = std::pair<const NamedDecl *, LVComputationKind>;
  llvm::SmallDenseMap<QueryType, LinkageInfo, 8> CachedLinkageInfo;
  llvm::Optional<LinkageInfo> lookup(const NamedDecl *ND,
                                     LVComputationKind Kind) const {
    auto Iter = CachedLinkageInfo.find(std::make_pair(ND, Kind));
    if (Iter == CachedLinkageInfo.end())
      return None;
    return Iter->second;
  }

  void cache(const NamedDecl *ND, LVComputationKind Kind, LinkageInfo Info) {
    CachedLinkageInfo[std::make_pair(ND, Kind)] = Info;
  }

  LinkageInfo getLVForTemplateArgumentList(ArrayRef<TemplateArgument> Args,
                                           LVComputationKind computation);

  LinkageInfo getLVForTemplateArgumentList(const TemplateArgumentList &TArgs,
                                           LVComputationKind computation);

  void mergeTemplateLV(LinkageInfo &LV, const FunctionDecl *fn,
                       const FunctionTemplateSpecializationInfo *specInfo,
                       LVComputationKind computation);

  void mergeTemplateLV(LinkageInfo &LV,
                       const ClassTemplateSpecializationDecl *spec,
                       LVComputationKind computation);

  void mergeTemplateLV(LinkageInfo &LV,
                       const VarTemplateSpecializationDecl *spec,
                       LVComputationKind computation);

  LinkageInfo getLVForNamespaceScopeDecl(const NamedDecl *D,
                                         LVComputationKind computation);

  LinkageInfo getLVForClassMember(const NamedDecl *D,
                                  LVComputationKind computation);

  LinkageInfo getLVForClosure(const DeclContext *DC, Decl *ContextDecl,
                              LVComputationKind computation);

  LinkageInfo getLVForLocalDecl(const NamedDecl *D,
                                LVComputationKind computation);

  LinkageInfo getLVForType(const Type &T, LVComputationKind computation);

  LinkageInfo getLVForTemplateParameterList(const TemplateParameterList *Params,
                                            LVComputationKind computation);

public:
  LinkageInfo computeLVForDecl(const NamedDecl *D,
                               LVComputationKind computation);

  LinkageInfo getLVForDecl(const NamedDecl *D, LVComputationKind computation);

  LinkageInfo computeTypeLinkageInfo(const Type *T);
  LinkageInfo computeTypeLinkageInfo(QualType T) {
    return computeTypeLinkageInfo(T.getTypePtr());
  }

  LinkageInfo getDeclLinkageAndVisibility(const NamedDecl *D);

  LinkageInfo getTypeLinkageAndVisibility(const Type *T);
  LinkageInfo getTypeLinkageAndVisibility(QualType T) {
    return getTypeLinkageAndVisibility(T.getTypePtr());
  }
};
} // namespace clang

#endif
