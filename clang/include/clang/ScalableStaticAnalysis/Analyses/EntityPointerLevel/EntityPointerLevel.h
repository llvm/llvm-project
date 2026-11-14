//===- EntityPointerLevel.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_ENTITYPOINTERLEVEL_ENTITYPOINTERLEVEL_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_ENTITYPOINTERLEVEL_ENTITYPOINTERLEVEL_H

#include "clang/AST/Expr.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "llvm/ADT/SmallVector.h"
#include <set>

namespace clang::ssaf {
class TUSummaryExtractor;

// Intermediate data structure that can be converted to a `EntityPointerLevel`,
// which abstracts away all the additional information carried in the NamedDecl.
struct DeclPointerLevel {
  const NamedDecl *Decl;
  unsigned PointerLevel;
  bool IsReturn;
};

using DeclPointerLevelVec = llvm::SmallVector<DeclPointerLevel, 2>;

/// An EntityPointerLevel is associated with a level of the declared
/// pointer/array type of an entity.  In the fully-expanded spelling of the
/// declared type, a EntityPointerLevel is associated with a '*' (or a '[]`) in
/// that declaration.
///
/// For example, for 'int *p[10];', there are two EntityPointerLevels.
/// One is associated with 'int *[10]' of 'p' and the other is associated with
/// 'int *' of 'p'.
///
/// An EntityPointerLevel can be identified by an EntityId and an unsigned
/// integer indicating the pointer level: '(EntityId, PointerLevel)'.
/// An EntityPointerLevel 'P' is valid iff 'P.EntityId' has a pointer type with
/// at least 'P.PointerLevel' levels (This implies 'P.PointerLevel > 0').
///
/// For the same example 'int *p[10];', the EntityPointerLevels below are valid:
/// - '(p, 2)' is associated with the 'int *' part of the declared type of 'p';
/// - '(p, 1)' is associated with the 'int *[10]' part of the declared type of
///   'p'.
class EntityPointerLevel {
  EntityId Entity;
  unsigned PointerLevel;

  friend class EntityPointerLevelTranslator;
  // For unittests:
  friend EntityPointerLevel buildEntityPointerLevel(EntityId, unsigned);

  explicit EntityPointerLevel(std::pair<EntityId, unsigned> Pair)
      : Entity(Pair.first), PointerLevel(Pair.second) {}

public:
  EntityId getEntity() const { return Entity; }
  unsigned getPointerLevel() const { return PointerLevel; }

  bool operator==(const EntityPointerLevel &Other) const {
    return std::tie(Entity, PointerLevel) ==
           std::tie(Other.Entity, Other.PointerLevel);
  }

  bool operator!=(const EntityPointerLevel &Other) const {
    return !(*this == Other);
  }

  bool operator<(const EntityPointerLevel &Other) const {
    return std::tie(Entity, PointerLevel) <
           std::tie(Other.Entity, Other.PointerLevel);
  }

  /// Compares `EntityPointerLevel`s; additionally, partially compares
  /// `EntityPointerLevel` with `EntityId`.
  struct Comparator {
    using is_transparent = void;
    bool operator()(const EntityPointerLevel &L,
                    const EntityPointerLevel &R) const {
      return L < R;
    }
    bool operator()(const EntityId &L, const EntityPointerLevel &R) const {
      return L < R.getEntity();
    }
    bool operator()(const EntityPointerLevel &L, const EntityId &R) const {
      return L.getEntity() < R;
    }
  };
};

using EntityPointerLevelSet =
    std::set<EntityPointerLevel, EntityPointerLevel::Comparator>;

/// Translate a pointer/array type expression 'E' to a (set of)
/// EntityPointerLevel(s) associated with the declared type of the base address
/// of `E`. If the base address of `E` is not associated with an entity, the
/// translation result is an empty set.
///
/// \param E the pointer expression to be translated
/// \param Ctx the AST context of `E`
/// \param Extractor the TUSummaryExtractor used to convert NamedDecls to
/// EntityIds.
llvm::Expected<EntityPointerLevelSet>
translateEntityPointerLevel(const Expr *E, ASTContext &Ctx,
                            TUSummaryExtractor &Extractor);

/// Same as \c translateEntityPointerLevel, except it returns raw
/// `(NamedDecl *, pointer level, is-return)` tuples (a.k.a. DeclPointerLevels)
/// instead of assembling an `EntityPointerLevelSet` directly.
llvm::Expected<DeclPointerLevelVec>
translateDeclPointerLevel(const Expr *E, ASTContext &Ctx,
                          TUSummaryExtractor &Extractor);

/// Assemble `DeclPointerLevels` into an `EntityPointerLevelSet`.
Expected<EntityPointerLevelSet>
toEntityPointerLevels(const DeclPointerLevelVec &DPLs, ASTContext &Ctx,
                      TUSummaryExtractor &Extractor);

/// Convert a single `DeclPointerLevel` to an `EntityPointerLevel`.
Expected<EntityPointerLevel>
toEntityPointerLevel(const DeclPointerLevel &DPL, ASTContext &Ctx,
                     TUSummaryExtractor &Extractor);

/// Creates a `EntityPointerLevel` from a pair of an EntityId and a pointer
/// level:
EntityPointerLevel buildEntityPointerLevel(EntityId, unsigned);

/// Create an DeclPointerLevel (DPL) from a NamedDecl of a pointer/array type.
///
/// \param ND the NamedDecl of a pointer/array type.
/// \param IsFunRet true iff the created DPL is associated with the return type
/// of a function entity.
DeclPointerLevel createDeclPointerLevel(const NamedDecl *ND,
                                        bool IsFunRet = false);

/// Create an EntityPointerLevel (EPL) from a NamedDecl of a pointer/array type.
///
/// \param ND the NamedDecl of a pointer/array type.
/// \param AddEntity the callback provided by the caller to convert EntityNames
/// to EntityIds.
/// \param IsFunRet true iff the created EPL is associated with the return type
/// of a function entity.
llvm::Expected<EntityPointerLevel>
createEntityPointerLevel(const NamedDecl *ND, TUSummaryExtractor &Extractor,
                         bool IsFunRet = false);

/// \return an exhaustive vector of unique DeclPointerLevels, sorted in
/// ascending order of pointer levels. All elements are identical to \p DPL
/// except with a pointer level greater than or equal to that of \p DPL. The
/// pointer level of each element is bounded by the type of \p DPL's NamedDecl.
DeclPointerLevelVec
elaborateHigherDeclPointerLevels(const DeclPointerLevel &DPL);
} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_ENTITYPOINTERLEVEL_ENTITYPOINTERLEVEL_H
