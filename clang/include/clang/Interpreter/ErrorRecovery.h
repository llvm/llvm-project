//===--- ErrorRecovery.h - Errory Recovery Impl --------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_ERROR_RECOVERY_H
#define LLVM_CLANG_INTERPRETER_ERROR_RECOVERY_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"

namespace clang {
class ASTContext;
class Sema;

/// Index of a per PTU State.
using PTUID = unsigned;

// A snapshot of the handful of DefinitionData bits/bitfields that can
// change after a class (or class template specialization -- the two are
// handled identically here since a specialization is non-dependent by the
// time it's instantiated) is otherwise "done" being defined.
//
// These fields only move when the compiler generates implicit special
// members.
//
// This footprint exists purely to undo the effects of implicit special
// member generation.
struct DefinitionDataFootprint {
#define FIELD(Name, Width, Merge) unsigned Name : Width;
#include "clang/AST/CXXRecordDeclDefinitionBits.def"

  bool operator==(const DefinitionDataFootprint &O) const {
#define FIELD(Name, Width, Merge)                                              \
  if (Name != O.Name)                                                          \
    return false;
#include "clang/AST/CXXRecordDeclDefinitionBits.def"
    return true;
  }
  bool operator!=(const DefinitionDataFootprint &O) const {
    return !(*this == O);
  }
};

// Stores the state of a class template specialization. Tracks its
// specialization kind, point of instantiation, source location, and lexical
// declaration context so the state can be compared and restored.
struct SpecializationFootprint {
  unsigned SpecializationKind : 3;
  SourceLocation PointOfInstantiation;
  SourceLocation Location;
  const DeclContext *LexicalDC;

  bool operator==(const SpecializationFootprint &O) const {
    return SpecializationKind == O.SpecializationKind &&
           PointOfInstantiation == O.PointOfInstantiation &&
           Location == O.Location && LexicalDC == O.LexicalDC;
  }
  bool operator!=(const SpecializationFootprint &O) const {
    return !(*this == O);
  }

  void update(const ClassTemplateSpecializationDecl &Live) {
    SpecializationKind = Live.getSpecializationKind();
    PointOfInstantiation = Live.getPointOfInstantiation();
    Location = Live.getLocation();
    LexicalDC = Live.getLexicalDeclContext();
  }

  void restore(ClassTemplateSpecializationDecl &Live) const {
    Live.setSpecializationKind(
        static_cast<TemplateSpecializationKind>(SpecializationKind));
    Live.setPointOfInstantiation(PointOfInstantiation);
    Live.setLocation(Location);
    Live.setLexicalDeclContext(const_cast<DeclContext *>(LexicalDC));
  }
};

// Stores the state of a variable template specialization. Tracks its
// specialization kind and point of instantiation so the state can be
// compared and restored.
struct VarSpecializationFootprint {
  unsigned SpecializationKind : 3;
  SourceLocation PointOfInstantiation;

  bool operator==(const VarSpecializationFootprint &O) const {
    return SpecializationKind == O.SpecializationKind &&
           PointOfInstantiation == O.PointOfInstantiation;
  }
  bool operator!=(const VarSpecializationFootprint &O) const {
    return !(*this == O);
  }

  void update(const VarTemplateSpecializationDecl &Live) {
    SpecializationKind = Live.getSpecializationKind();
    PointOfInstantiation = Live.getPointOfInstantiation();
  }

  void restore(VarTemplateSpecializationDecl &Live) const {
    Live.setSpecializationKind(
        static_cast<TemplateSpecializationKind>(SpecializationKind));
    Live.setPointOfInstantiation(PointOfInstantiation);
  }
};

// Stores the state of a function template specialization. Tracks its
// specialization kind, point of instantiation, source location, lexical
// declaration context, and constexpr state so the state can be compared
// and restored.
struct FunctionSpecializationFootprint {
  unsigned SpecializationKind : 3;
  SourceLocation PointOfInstantiation;
  SourceLocation Location;
  const DeclContext *LexicalDC;
  ConstexprSpecKind ConstexprKind;

  bool operator==(const FunctionSpecializationFootprint &O) const {
    return SpecializationKind == O.SpecializationKind &&
           PointOfInstantiation == O.PointOfInstantiation &&
           Location == O.Location && LexicalDC == O.LexicalDC &&
           ConstexprKind == O.ConstexprKind;
  }
  bool operator!=(const FunctionSpecializationFootprint &O) const {
    return !(*this == O);
  }

  void update(const FunctionDecl &Live) {
    SpecializationKind = Live.getTemplateSpecializationKind();
    PointOfInstantiation = Live.getPointOfInstantiation();
    Location = Live.getLocation();
    LexicalDC = Live.getLexicalDeclContext();
    ConstexprKind = Live.getConstexprKind();
  }

  void restore(FunctionDecl &Live) const {
    Live.setTemplateSpecializationKind(
        static_cast<TemplateSpecializationKind>(SpecializationKind),
        PointOfInstantiation);
    Live.setLocation(Location);
    Live.setLexicalDeclContext(const_cast<DeclContext *>(LexicalDC));
    Live.setConstexprKind(ConstexprKind);
  }
};

// Stores the state of an ordinary class member created from a class template
// instantiation. Tracks its specialization kind and point of instantiation
// so the state can be compared and restored.
struct MemberSpecializationFootprint {
  unsigned SpecializationKind : 3;
  SourceLocation PointOfInstantiation;

  bool operator==(const MemberSpecializationFootprint &O) const {
    return SpecializationKind == O.SpecializationKind &&
           PointOfInstantiation == O.PointOfInstantiation;
  }
  bool operator!=(const MemberSpecializationFootprint &O) const {
    return !(*this == O);
  }

  template <typename OwnerT> void update(const OwnerT &Live) {
    const MemberSpecializationInfo *MSI = Live.getMemberSpecializationInfo();
    SpecializationKind = MSI->getTemplateSpecializationKind();
    PointOfInstantiation = MSI->getPointOfInstantiation();
  }

  template <typename OwnerT> void restore(OwnerT &Live) const {
    MemberSpecializationInfo *MSI = Live.getMemberSpecializationInfo();
    MSI->setTemplateSpecializationKind(
        static_cast<TemplateSpecializationKind>(SpecializationKind));
    MSI->setPointOfInstantiation(PointOfInstantiation);
  }
};

class PTUMutationActions;
class PTUCheckpointLedger;

class DeclStateReverter {
private:
  struct TemplateCommonAccess : RedeclarableTemplateDecl {
    using RedeclarableTemplateDecl::Common;
  };

  static const void *rawCommonPtr(const RedeclarableTemplateDecl &RT) {
    return static_cast<const TemplateCommonAccess &>(RT).Common;
  }

  // Common is `mutable`, so this is legal to call through a const
  // reference too -- no const_cast needed at any call site.
  static void clearCommonPtr(const RedeclarableTemplateDecl &RT) {
    static_cast<const TemplateCommonAccess &>(RT).Common = nullptr;
  }

  struct ClassTemplateCommonAccess : ClassTemplateDecl {
    using ClassTemplateDecl::getCommonPtr;
  };

  struct ClassTemplateSpecAccess : ClassTemplateDecl {
    using ClassTemplateDecl::getSpecializations;
  };
  struct FunctionTemplateSpecAccess : FunctionTemplateDecl {
    using FunctionTemplateDecl::getSpecializations;
  };
  struct VarTemplateSpecAccess : VarTemplateDecl {
    using VarTemplateDecl::getSpecializations;
  };

  static bool canonInjectedTSTValid(const ClassTemplateDecl &CTD) {
    auto *Ptr =
        static_cast<const ClassTemplateCommonAccess &>(CTD).getCommonPtr();
    return !Ptr->CanonInjectedTST.isNull();
  }

  static void clearCanonInjectedTST(ClassTemplateDecl &CTD) {
    auto *Ptr = static_cast<ClassTemplateCommonAccess &>(CTD).getCommonPtr();
    Ptr->CanonInjectedTST = CanQualType();
  }

  // DeclContext::FirstDecl/LastDecl -- protected.
  struct DeclContextLinkAccess : DeclContext {
    using DeclContext::FirstDecl;
    using DeclContext::LastDecl;
  };

  static void clearDeclContextChain(DeclContext &DC) {
    auto &Access = static_cast<DeclContextLinkAccess &>(DC);
    Access.FirstDecl = nullptr;
    Access.LastDecl = nullptr;
  }

  struct TagDeclDefinitionAccess : TagDecl {
    using TagDecl::setBeingDefined;
  };

  static void clearBeingDefined(TagDecl &TD) {
    static_cast<TagDeclDefinitionAccess &>(TD).setBeingDefined(false);
  }

  struct DeclLexicalLinkAccess : Decl {
    using Decl::NextInContextAndBits;
  };

  static void clearNextInContext(Decl &D) {
    static_cast<DeclLexicalLinkAccess &>(D).NextInContextAndBits.setPointer(
        nullptr);
  }

  template <typename decl_type>
  struct RedeclLinkAccess : Redeclarable<decl_type> {
    using Redeclarable<decl_type>::RedeclLink;
  };

  template <typename decl_type>
  static void setLatestRedecl(Redeclarable<decl_type> &D, decl_type *Latest) {
    static_cast<RedeclLinkAccess<decl_type> &>(D).RedeclLink.setLatest(Latest);
  }

protected:
  static DefinitionDataFootprint *
  createDefinitionDataFootprint(const ASTContext &Ctx, const CXXRecordDecl &RD);

  static bool compareDefinitionDataFootprint(const DefinitionDataFootprint &FP,
                                             const CXXRecordDecl &RD);

  static void restoreDefinitionDataFootprint(const DefinitionDataFootprint &FP,
                                             CXXRecordDecl &RD);

  static SpecializationFootprint *
  createSpecializationFootprint(const ASTContext &Ctx,
                                const ClassTemplateSpecializationDecl &Spec) {
    auto *FP =
        new (Ctx, alignof(SpecializationFootprint)) SpecializationFootprint();
    FP->update(Spec);
    return FP;
  }
  static bool
  compareSpecializationFootprint(const SpecializationFootprint &FP,
                                 const ClassTemplateSpecializationDecl &Spec) {
    SpecializationFootprint Live;
    Live.update(Spec);
    return FP == Live;
  }
  static void
  restoreSpecializationFootprint(const SpecializationFootprint &FP,
                                 ClassTemplateSpecializationDecl &Spec) {
    FP.restore(Spec);
  }

  static VarSpecializationFootprint *
  createVarSpecializationFootprint(const ASTContext &Ctx,
                                   const VarTemplateSpecializationDecl &Spec) {
    auto *FP = new (Ctx, alignof(VarSpecializationFootprint))
        VarSpecializationFootprint();
    FP->update(Spec);
    return FP;
  }
  static bool
  compareVarSpecializationFootprint(const VarSpecializationFootprint &FP,
                                    const VarTemplateSpecializationDecl &Spec) {
    VarSpecializationFootprint Live;
    Live.update(Spec);
    return FP == Live;
  }
  static void
  restoreVarSpecializationFootprint(const VarSpecializationFootprint &FP,
                                    VarTemplateSpecializationDecl &Spec) {
    FP.restore(Spec);
  }

  static FunctionSpecializationFootprint *
  createFunctionSpecializationFootprint(const ASTContext &Ctx,
                                        const FunctionDecl &FD) {
    auto *FP = new (Ctx, alignof(FunctionSpecializationFootprint))
        FunctionSpecializationFootprint();
    FP->update(FD);
    return FP;
  }
  static bool compareFunctionSpecializationFootprint(
      const FunctionSpecializationFootprint &FP, const FunctionDecl &FD) {
    FunctionSpecializationFootprint Live;
    Live.update(FD);
    return FP == Live;
  }
  static void restoreFunctionSpecializationFootprint(
      const FunctionSpecializationFootprint &FP, FunctionDecl &FD) {
    FP.restore(FD);
  }

  template <typename OwnerT>
  static MemberSpecializationFootprint *
  createMemberSpecializationFootprint(const ASTContext &Ctx, const OwnerT &D) {
    auto *FP = new (Ctx, alignof(MemberSpecializationFootprint))
        MemberSpecializationFootprint();
    FP->update(D);
    return FP;
  }
  template <typename OwnerT>
  static bool
  compareMemberSpecializationFootprint(const MemberSpecializationFootprint &FP,
                                       const OwnerT &D) {
    MemberSpecializationFootprint Live;
    Live.update(D);
    return FP == Live;
  }
  template <typename OwnerT>
  static void
  restoreMemberSpecializationFootprint(const MemberSpecializationFootprint &FP,
                                       OwnerT &D) {
    FP.restore(D);
  }

  static void restoreDefinitionAndRevertDC(CXXRecordDecl &RD);

  static void revertDefinitionArrival(Decl &D);

  // True if Common could still be created later -- i.e. nobody has called
  // getCommonPtr() anywhere in this template's redecl chain yet.
  static bool isCommonPtrValid(const RedeclarableTemplateDecl &RT) {
    return !rawCommonPtr(RT);
  }

  // True if CanonInjectedTST could still be cached later. Only valid to
  // call once Common itself is confirmed to exist (see needToTrackCommonPtr).
  static bool isTemplateCanonInjectedTSTValid(const ClassTemplateDecl *CTD) {
    return !canonInjectedTSTValid(*CTD);
  }

  // Write-once fields, fixed null prior state (Common/CanonInjectedTST
  // never existed before whichever PTU created them), so revert is a
  // direct reset, not a snapshot restore.
  static void resetTemplateCommonBase(RedeclarableTemplateDecl &RT) {
    clearCommonPtr(RT);
  }

  static void resetCanonInjectedTST(ClassTemplateDecl &CTD) {
    clearCanonInjectedTST(CTD);
  }

  // Specializations are appended in commit order, and rollback is strictly
  // LIFO (only the most recently committed PTU is ever rolled back). So all
  // specializations added by a given PTU form a trailing run at the back of
  // the FoldingSetVector. Popping entries while the back entry belongs to
  // this PTU is therefore sufficient.
  //
  // FoldingSetVector also has no arbitrary-position erase – only
  // pop_back()/clear() – which is another reason to walk from the back
  // rather than filter the entries in place.
  static void removeSpecializations(PTUCheckpointLedger &Ledger,
                                    const RedeclarableTemplateDecl *TD,
                                    PTUID ID);

  // static const Type *getRawTypeForDecl(const TypeDecl *TD) {
  //   return TD->TypeForDecl;
  // }

  // static bool needToTrackTypeForDecl(const TypeDecl *TD) {
  //   const Type *T = getRawTypeForDecl(TD);
  //   return !T || T->isCanonicalUnqualified();
  // }

  // static void resetTypeForDecl(TypeDecl *TD) { TD->TypeForDecl = nullptr; }

private:
  PTUCheckpointLedger &PTUSlabCheckpoints;

public:
  friend PTUMutationActions;

  explicit DeclStateReverter(PTUCheckpointLedger &Ledger)
      : PTUSlabCheckpoints(Ledger) {}

  static bool isExtensibleContainer(const Decl *D) {
    return isa<NamespaceDecl>(D) || isa<CXXRecordDecl>(D);
  }

  static bool hasNoPreviousDecl(Decl *D) {
    return D->getPreviousDecl() == nullptr;
  }

  void detachDefData(const Decl *D);

  static void detachCommonBase(const RedeclarableTemplateDecl *RT);

  /// Remove D from its semantic context's lookup map, reinstating the
  /// previous declaration if D had replaced one in-place (which is what
  /// StoredDeclsList::HandleRedeclaration does on a redeclaration --
  /// erasing the slot outright would lose the older decl entirely; that
  /// is the ReopenNs failure).
  void detachFromDCLookup(Decl *D, PTUID ID);

  // Walk D's redecl chain looking for the newest decl that predates this
  // PTU. Returns nullptr if the entire chain was created this PTU.
  template <typename DeclT> DeclT *findSurvivor(DeclT *D, PTUID ID) const;

  template <typename decl_type>
  void patchRedeclLink(Redeclarable<decl_type> *D, decl_type *Survivor) {
    setLatestRedecl<decl_type>(*D->getFirstDecl(), Survivor);
  }

  /// Point the canonical decl's "most recent" link back at the newest
  /// redeclaration that predates this PTU.
  void detachFromRedeclChain(const Decl *D, PTUID ID);

  void repairLexicalChain(DeclContext &DC, PTUID ID);

private:
  NamedDecl *tryDetachRedeclChain(Decl *D, PTUID ID);
};

template <typename DataT> struct Snapshot {
  PTUID ID;
  DataT *Data;
};

template <typename DataT> class StateAwareChain {
  llvm::SmallVector<Snapshot<DataT>, 4> History;

public:
  bool empty() const { return History.empty(); }

  const DataT *mostRecent() const {
    return History.empty() ? nullptr : History.back().Data;
  }

  std::optional<PTUID> mostRecentID() const {
    return History.empty() ? std::nullopt
                           : std::optional<PTUID>(History.back().ID);
  }

  std::optional<PTUID> oldestID() const {
    return History.empty() ? std::nullopt
                           : std::optional<PTUID>(History.front().ID);
  }

  void commit(PTUID ID, DataT *Fresh) {
    if (!History.empty() && *History.back().Data == *Fresh)
      return;
    assert(History.back().ID != ID);
    History.push_back(Snapshot<DataT>{ID, Fresh});
  }

  // const DataT *getPrevious(PTUID ID) const {
  //   for (auto It = History.rbegin(); It != History.rend(); ++It)
  //     if (It->ID < ID)
  //       return It->Data;
  //   return nullptr;
  // }

  /// removal: drop every entry with ID >= \p ID.
  void removeFrom(PTUID ID) {
    while (!History.empty() && History.back().ID >= ID)
      History.pop_back(); // no delete.
  }
};

using RecordDeclDefinitionDataChain = StateAwareChain<DefinitionDataFootprint>;
using SpecializationChain = StateAwareChain<SpecializationFootprint>;
using VarSpecializationChain = StateAwareChain<VarSpecializationFootprint>;
using FunctionSpecializationChain =
    StateAwareChain<FunctionSpecializationFootprint>;
using MemberSpecializationChain =
    StateAwareChain<MemberSpecializationFootprint>;

/// Maps addresses to the PTU that allocated them, by keeping the slab
/// checkpoint taken before each PTU began.
///
/// PTU IDs are the vector index, which is what makes the lookup a binary
/// search: checkpoints are monotonically increasing, so "is this address
/// after checkpoint N" is true for all N up to the allocating PTU and
/// false after.
class PTUCheckpointLedger {
  const ASTContext &Ctx;
  llvm::SmallVector<llvm::SlabCheckPoint, 16> CheckpointBeforePTU;

  bool isAfter(const void *Ptr, PTUID ID) const {
    return Ctx.getAllocator().isAfterCheckpoint(Ptr, CheckpointBeforePTU[ID]);
  }

public:
  explicit PTUCheckpointLedger(const ASTContext &Ctx) : Ctx(Ctx) {}

  /// Called once, right before parsing PTU \p ID begins.
  void recordCheckpoint(PTUID ID, llvm::SlabCheckPoint CP) {
    assert(ID == CheckpointBeforePTU.size() &&
           "PTUs must be recorded in order");
    CheckpointBeforePTU.push_back(CP);
  }

  /// Was \p Ptr allocated during PTU \p ID or later?
  bool isFromThisPTU(const void *Ptr, PTUID ID) const {
    assert(ID < CheckpointBeforePTU.size());
    return isAfter(Ptr, ID);
  }

  /// The PTU that allocated \p Ptr, or nullopt if it predates the oldest
  /// recorded checkpoint. The newest ID whose checkpoint \p Ptr is after.
  std::optional<PTUID> attribute(const void *Ptr) const {
    size_t Lo = 0, Hi = CheckpointBeforePTU.size();
    std::optional<PTUID> Result;
    while (Lo < Hi) {
      size_t Mid = Lo + (Hi - Lo) / 2;
      if (isAfter(Ptr, Mid)) {
        Result = static_cast<PTUID>(Mid);
        Lo = Mid + 1;
      } else {
        Hi = Mid;
      }
    }
    return Result;
  }

  /// Drop checkpoints from \p ID onward, after rolling those PTUs back.
  void undoFrom(PTUID ID) {
    if (ID < CheckpointBeforePTU.size())
      CheckpointBeforePTU.resize(ID);
  }
};

template <typename ValueT> struct FieldMutation {
  PTUID ID;
  ValueT OldValue;
};

template <typename OwnerT, typename ValueT> class FieldMutationChain {
  llvm::DenseMap<const OwnerT *, llvm::SmallVector<FieldMutation<ValueT>, 2>>
      Log;

public:
  void noteMutation(PTUID ID, const OwnerT *Owner, ValueT OldValue) {
    auto &Entries = Log[Owner];
    if (!Entries.empty() && Entries.back().ID == ID)
      return;
    Entries.push_back(FieldMutation<ValueT>{ID, OldValue});
  }

  std::optional<ValueT> mostRecent(const OwnerT *Owner) const {
    auto It = Log.find(Owner);
    if (It == Log.end() || It->second.empty())
      return std::nullopt;
    return It->second.back().OldValue;
  }

  std::optional<PTUID> mostRecentID(const OwnerT *Owner) const {
    auto It = Log.find(Owner);
    if (It == Log.end())
      return std::nullopt;
    auto &History = It->second;
    return History.empty() ? std::nullopt
                           : std::optional<PTUID>(History.back().ID);
  }

  std::optional<PTUID> oldestID(const OwnerT *Owner) const {
    auto It = Log.find(Owner);
    if (It == Log.end())
      return std::nullopt;
    auto &History = It->second;
    return History.empty() ? std::nullopt
                           : std::optional<PTUID>(History.front().ID);
  }

  // template <typename FnT> void forEachOwnerSince(PTUID ID, FnT &&Fn) const {
  //   for (auto &Entry : Log)
  //     if (!Entry.second.empty() && Entry.second.back().ID >= ID)
  //       Fn(Entry.first);
  // }

  // void removeFrom(PTUID ID) {
  //   llvm::SmallVector<OwnerT *> ToErase;
  //   for (auto &Entry : Log) {
  //     auto &Entries = Entry.second;
  //     while (!Entries.empty() && Entries.back().ID >= ID)
  //       Entries.pop_back();
  //     if (Entries.empty())
  //       ToErase.push_back(Entry.first);
  //   }
  //   for (const OwnerT *O : ToErase)
  //     Log.erase(O);
  // }

  /// Same trim-then-erase-if-empty as removeFrom(ID), scoped to one owner
  /// instead of sweeping the whole Log -- for callers that already know
  /// exactly which owners this PTU touched (see PTUStateInfo's
  /// TouchedFunctionTypeOwners/TouchedTagdeclOwners) and don't need to
  /// rediscover that by walking every owner ever logged.
  void removeFromOwner(const OwnerT *Owner, PTUID ID) {
    auto It = Log.find(Owner);
    if (It == Log.end())
      return;
    auto &Entries = It->second;
    while (!Entries.empty() && Entries.back().ID >= ID)
      Entries.pop_back();
    if (Entries.empty())
      Log.erase(It);
  }

  void forget(const OwnerT *Owner) { Log.erase(Owner); }
};

using FunctionExceptionSpecChain = FieldMutationChain<FunctionDecl, QualType>;
using TypeForDeclChain = FieldMutationChain<TagDecl, const Type *>;

struct MutationRecord {
  enum class DeclShape : uint16_t {
    None = 0,
    // Base shapes -- what the decl fundamentally IS. Exactly one is set.
    Class = 1 << 0,    // CXXRecordDecl/TagDecl: DefinitionData, TypeForDecl
    Function = 1 << 1, // FunctionDecl: exception spec, deduced return, body
    Var = 1 << 2,      // VarDecl: cached constant-eval result
    Enum = 1 << 3,     // EnumDecl: TypeForDecl
    Template = 1 << 4, // RedeclarableTemplateDecl: spec list grows
    Typedef = 1 << 5,  // TypedefDecl/TypeAliasDecl: TypeForDecl.
  };

  enum class MutationKind : uint32_t {
    // ---- Common ----
    DefinitionInstantiate = 1 << 0, // Class, Function, Var
    SpecInfo = 1 << 1,              // Class, Function, Var  (Spec)
    MemberSpecInfo = 1 << 2,        // Class, Function, Var, Enum (Member)
    TypeForDecl = 1 << 3,           // Class, Enum

    // ---- Shape-specific ----
    DefinitionData = 1 << 8,       // Class only
    ExceptionSpec = 1 << 9,        // Function only
    DeducedReturnType = 1 << 10,   // Function only
    EvaluatedValue = 1 << 11,      // Var only
    SpecializationAdded = 1 << 12, // Template only

    TemplateCommon = 1 << 4,   // Template CommonBase
    CanonInjectedTST = 1 << 5, // Template CommonBase Type
    None = 1 << 24,
  };

  DeclShape S = DeclShape::None;
  uint32_t MutationType = 0; // what this decl can ever have

  void add(MutationKind K) { MutationType |= uint32_t(K); }
  void add(uint32_t K) { MutationType |= uint32_t(K); }
  bool has(MutationKind K) const { return MutationType & uint32_t(K); }
  void clear(MutationKind K) { MutationType &= ~uint32_t(K); }
};

using DeclShape = MutationRecord::DeclShape;
using MutationType = MutationRecord::MutationKind;

struct PTUStateInfo {
  PTUID ID;
  const TranslationUnitDecl *ThisPTU; // current info

  llvm::MapVector<const Decl *, MutationRecord> Mutations;

  llvm::SmallPtrSet<const Decl *, 4> ImplicitDecls;

  /// here touched info mean other this belongs to other PTUs;
  llvm::SmallPtrSet<const DeclContext *, 4> TouchedDC;

  /// Exactly the decls this PTU's own commit() pushed a footprint-chain
  /// entry for (DefinitionData/Specialization/MemberSpecialization, via
  /// chainFor()/memberSpecChainFor()).
  llvm::SmallPtrSet<const Decl *, 8> TouchedFootprintDecls;

  llvm::SmallPtrSet<const FunctionDecl *, 4> TouchedFunctionTypeOwners;
  llvm::SmallPtrSet<const TagDecl *, 4> TouchedTagdeclOwners;

  explicit PTUStateInfo(PTUID ID) : ID(ID) {}

  template <typename KindT>
  void noteMutated(const Decl *D, DeclShape S, KindT K) {
    if (!D->isDefinedOutsideFunctionOrMethod())
      return;
    auto [It, Inserted] = Mutations.try_emplace(D);
    if (Inserted) {
      It->second.S = S;
    }

    assert(It->second.S == S && "same decl noted under two different shapes");

    It->second.add(K);
  }

  void verifyMutations(PTUMutationActions &Actions);

  // False until this PTU is actually committed.
  // PTUMutationActions uses this to decide what "undo" should do:
  // - If this PTU was never committed, nothing was added to the chain,
  //   so we just restore the last PTU that was committed.
  // - If this PTU was already committed (for example, by using %undo),
  //   first remove the entries created by this PTU, then restore the
  //   state from before this PTU was added.
  bool Commited = false;
};

//===----------------------------------------------------------------------===//
// SweepTracker -- Tracks declarations whose mutations cannot be reported
// directly by an ASTMutationListener.
//
// Each Decl is tracked with the hidden mutation kinds that are still possible.
// At commit time, sweep() checks the tracked kinds to find mutations made by
// the current PTU. Once a mutation kind is confirmed or can no longer happen,
// it is removed from tracking.
//
// During restore, a mutation kind can be tracked again so its mutation sites
// can be restored when needed.
//===----------------------------------------------------------------------===//
class SweepTracker {
  llvm::DenseMap<const Decl *, uint32_t> Active;

public:
  void track(const Decl *D, uint32_t HiddenBits) {
    if (HiddenBits)
      Active[D] |= HiddenBits;
  }

  bool isTrackedFor(const Decl *D, uint32_t Flag) const {
    auto It = Active.find(D);
    return It != Active.end() && (It->second & Flag);
  }

  void settle(const Decl *D, uint32_t Flag) {
    auto It = Active.find(D);
    if (It == Active.end())
      return;
    It->second &= ~Flag;
    if (!It->second)
      Active.erase(It);
  }

  // Call at commit/restore time. OnConfirmed is invoked as
  // (const Decl *D, DeclShape S, uint32_t ConfirmedKinds) for every decl
  // that had something newly confirmed this sweep.
  template <typename OnConfirmedFn>
  void sweep(PTUMutationActions &Act, OnConfirmedFn OnConfirmed);
};

//===----------------------------------------------------------------------===//
// DeclLinkedState / LinkedDeclNodeGenerator
//
// These nodes keep the footprint-chain state associated with a Decl.
// Each Decl kind has its own node type so the node only contains state that
// is valid for that kind. DeclLinkedState then uses the node type to identify
// which kind of Decl it belongs to.
//
// The generator owns these nodes and their chains and reuses them through
// per-type free lists. This is intentional: nodes need to be released and
// reused, so they use normal heap allocation instead of ASTContext's
// bump allocator.
//
// The goal is to keep the linked state small, type-safe, and reusable without
// adding fields for states that a particular Decl can never have.
//===----------------------------------------------------------------------===//
struct CXXClassDeclNode {
  PTUID OriginID;
  RecordDeclDefinitionDataChain *DefData = nullptr;
  llvm::PointerUnion<SpecializationChain *, MemberSpecializationChain *> Spec;
};

struct VarDeclNode {
  PTUID OriginID;
  llvm::PointerUnion<VarSpecializationChain *, MemberSpecializationChain *>
      Spec;
};

struct FunctionDeclNode {
  PTUID OriginID;
  llvm::PointerUnion<FunctionSpecializationChain *, MemberSpecializationChain *>
      Spec;
};

struct EnumDeclNode {
  PTUID OriginID;
  MemberSpecializationChain *MemberSpec =
      nullptr; // the only thing an enum can ever have
};

using DeclLinkedState = llvm::PointerUnion<CXXClassDeclNode *, VarDeclNode *,
                                           FunctionDeclNode *, EnumDeclNode *>;

/// Pool for reusing node/chain objects without requiring any intrusive
/// free-list field on T itself -- freed pointers are tracked in a side
/// vector instead, so this works for the four node types above and the
/// five StateAwareChain-based chain types alike. Released objects are kept
/// for reuse, and the pool is bounded by Capacity so unused objects do not
/// cause unbounded memory growth.
template <typename T> class ObjectPool {
  llvm::SmallVector<T *, 8> Free;
  unsigned Capacity;

public:
  explicit ObjectPool(unsigned Capacity = 64) : Capacity(Capacity) {}

  ~ObjectPool() {
    for (T *C : Free)
      delete C;
  }

  T *acquire() {
    if (!Free.empty()) {
      T *C = Free.pop_back_val();
      *C = T();
      return C;
    }
    return new T();
  }
  /// Reclaimed for real once Free is at capacity, rather than growing
  /// without bound.
  void release(T *C) {
    if (Free.size() >= Capacity) {
      delete C;
      return;
    }
    Free.push_back(C);
  }
};

class LinkedDeclNodeGenerator {
  ObjectPool<CXXClassDeclNode> RecordNodes;
  ObjectPool<VarDeclNode> VarNodes;
  ObjectPool<FunctionDeclNode> FunctionNodes;
  ObjectPool<EnumDeclNode> EnumNodes;

  ObjectPool<RecordDeclDefinitionDataChain> DefDataChains;
  ObjectPool<SpecializationChain> ClassSpecChains;
  ObjectPool<VarSpecializationChain> VarSpecChains;
  ObjectPool<FunctionSpecializationChain> FunctionSpecChains;
  ObjectPool<MemberSpecializationChain> MemberSpecChains;

public:
  CXXClassDeclNode *acquireRecordNode() { return RecordNodes.acquire(); }
  VarDeclNode *acquireVarNode() { return VarNodes.acquire(); }
  FunctionDeclNode *acquireFunctionNode() { return FunctionNodes.acquire(); }
  EnumDeclNode *acquireEnumNode() { return EnumNodes.acquire(); }

  RecordDeclDefinitionDataChain *acquireDefDataChain() {
    return DefDataChains.acquire();
  }
  SpecializationChain *acquireClassSpecChain() {
    return ClassSpecChains.acquire();
  }
  VarSpecializationChain *acquireVarSpecChain() {
    return VarSpecChains.acquire();
  }
  FunctionSpecializationChain *acquireFunctionSpecChain() {
    return FunctionSpecChains.acquire();
  }
  MemberSpecializationChain *acquireMemberSpecChain() {
    return MemberSpecChains.acquire();
  }

  void release(CXXClassDeclNode *N) { RecordNodes.release(N); }
  void release(VarDeclNode *N) { VarNodes.release(N); }
  void release(FunctionDeclNode *N) { FunctionNodes.release(N); }
  void release(EnumDeclNode *N) { EnumNodes.release(N); }
  void release(RecordDeclDefinitionDataChain *C) { DefDataChains.release(C); }
  void release(SpecializationChain *C) { ClassSpecChains.release(C); }
  void release(VarSpecializationChain *C) { VarSpecChains.release(C); }
  void release(FunctionSpecializationChain *C) {
    FunctionSpecChains.release(C);
  }
  void release(MemberSpecializationChain *C) { MemberSpecChains.release(C); }
};

enum class LangMode : uint8_t {
  C,   // no DefinitionData, no templates, no implicit special members
  CXX, // full set
};

// Overall flow:
// - Track changes made by other PTUs so they can be applied to the relevant
//   chains/mutations when those PTUs are committed.
// - On rollback of a committed PTU, remove the entries created by that PTU
//   and restore the state that existed before it.
// - On rollback of a PTU that was never committed, restore the chain to its
//   current mostRecent() state, since this PTU never added anything to it.
// - For the current PTU, collect newly added declarations that are
//   modifiable/visible to other PTUs. Local-only declarations are not exposed.
//
// In short, this keeps the shared declaration state in sync across PTUs while
// keeping declarations that are local to the current PTU private.
class IncrementalStateTracker {
private:
  ASTContext &Ctx;
  // Only needed so remove stale
  // Sema::SpecialMemberCache entries for decls this PTU created.
  Sema &SemaRef;
  PTUCheckpointLedger PTUSlabCheckpoints;
  LangMode Mode;
  PTUID NextID = 0;

  mutable std::vector<PTUStateInfo> PTUStack;

  // Stores all declaration footprint state in one map instead of maintaining
  // separate maps for each declaration/specialization kind. The generator
  // creates the appropriate node and chain for each Decl.
  mutable LinkedDeclNodeGenerator Generator;
  mutable llvm::DenseMap<const Decl *, DeclLinkedState> LinkedDecls;

  // Tracks hidden mutations for all supported Decl kinds in one shared tracker,
  // so they can be restored correctly during rollback.
  SweepTracker HiddenMutationTracker;

  FunctionExceptionSpecChain FunctionTypeMutations;
  TypeForDeclChain TagDeclTypes;

  friend class PTUMutationActions;

  void noteFunctionTypeMutation(PTUID ID, const FunctionDecl *Owner,
                                QualType OldValue) {
    FunctionTypeMutations.noteMutation(ID, Owner, OldValue);
    current().TouchedFunctionTypeOwners.insert(Owner);
  }
  void noteTagDeclTypeMutation(PTUID ID, const TagDecl *Owner,
                               const Type *OldValue) {
    TagDeclTypes.noteMutation(ID, Owner, OldValue);
    current().TouchedTagdeclOwners.insert(Owner);
  }

  TypeForDeclChain &getTagDeclTypeInfo() { return TagDeclTypes; }
  FunctionExceptionSpecChain &getFunctionTypeMutations() {
    return FunctionTypeMutations;
  }

  /// Get-or-create the CXXClassDeclNode backing Canon, acquiring one from
  /// the generator on first use. The only place a CXXClassDeclNode is
  /// created for this map.
  template <typename NodeT, typename AcquireFn>
  NodeT &nodeFor(const Decl *D, AcquireFn Acquire) {
    current().TouchedFootprintDecls.insert(D);
    DeclLinkedState &Info = LinkedDecls[D];
    if (Info.isNull())
      Info = (Generator.*Acquire)();
    return *cast<NodeT *>(Info);
  }

  template <typename NodeT> NodeT *lookupNode(const Decl *D) const {
    auto It = LinkedDecls.find(D);
    if (It == LinkedDecls.end() || It->second.isNull())
      return nullptr;
    return dyn_cast<NodeT *>(It->second);
  }

  CXXClassDeclNode &recordNodeFor(const CXXRecordDecl *RD) {
    return nodeFor<CXXClassDeclNode>(
        RD, &LinkedDeclNodeGenerator::acquireRecordNode);
  }
  VarDeclNode &varNodeFor(const VarDecl *VD) {
    return nodeFor<VarDeclNode>(VD, &LinkedDeclNodeGenerator::acquireVarNode);
  }
  FunctionDeclNode &functionNodeFor(const FunctionDecl *FD) {
    return nodeFor<FunctionDeclNode>(
        FD, &LinkedDeclNodeGenerator::acquireFunctionNode);
  }
  EnumDeclNode &enumNodeFor(const EnumDecl *ED) {
    return nodeFor<EnumDeclNode>(ED, &LinkedDeclNodeGenerator::acquireEnumNode);
  }

  template <typename WantT, typename NodeT, typename AcquireFn>
  WantT &specChain(NodeT &Node, AcquireFn Acquire) {
    if (auto *C = Node.Spec.template dyn_cast<WantT *>())
      return *C;
    assert(Node.Spec.isNull() && "node already holds the other Spec kind");
    auto *C = (Generator.*Acquire)();
    Node.Spec = C;
    return *C;
  }

  RecordDeclDefinitionDataChain &chainFor(const CXXRecordDecl *RD) {
    CXXClassDeclNode &Node = recordNodeFor(RD);
    if (!Node.DefData)
      Node.DefData = Generator.acquireDefDataChain();
    return *Node.DefData;
  }

  SpecializationChain &chainFor(const ClassTemplateSpecializationDecl *S) {
    return specChain<SpecializationChain>(
        recordNodeFor(cast<CXXRecordDecl>(S)),
        &LinkedDeclNodeGenerator::acquireClassSpecChain);
  }

  VarSpecializationChain &chainFor(const VarTemplateSpecializationDecl *S) {
    return specChain<VarSpecializationChain>(
        varNodeFor(cast<VarDecl>(S)),
        &LinkedDeclNodeGenerator::acquireVarSpecChain);
  }

  FunctionSpecializationChain &chainFor(const FunctionDecl *S) {
    return specChain<FunctionSpecializationChain>(
        functionNodeFor(cast<FunctionDecl>(S)),
        &LinkedDeclNodeGenerator::acquireFunctionSpecChain);
  }

  MemberSpecializationChain &memberSpecChainFor(const FunctionDecl *FD) {
    FunctionDeclNode &Node = functionNodeFor(FD);
    auto *Chain = Node.Spec.dyn_cast<MemberSpecializationChain *>();
    if (!Chain) {
      Chain = Generator.acquireMemberSpecChain();
      Node.Spec = Chain;
    }
    return *Chain;
  }
  MemberSpecializationChain &memberSpecChainFor(const VarDecl *VD) {
    VarDeclNode &Node = varNodeFor(VD);
    auto *Chain = Node.Spec.dyn_cast<MemberSpecializationChain *>();
    if (!Chain) {
      Chain = Generator.acquireMemberSpecChain();
      Node.Spec = Chain;
    }
    return *Chain;
  }
  MemberSpecializationChain &memberSpecChainFor(const CXXRecordDecl *RD) {
    CXXClassDeclNode &Node = recordNodeFor(RD);
    auto *Chain = Node.Spec.dyn_cast<MemberSpecializationChain *>();
    if (!Chain) {
      Chain = Generator.acquireMemberSpecChain();
      Node.Spec = Chain;
    }
    return *Chain;
  }
  MemberSpecializationChain &memberSpecChainFor(const EnumDecl *ED) {
    EnumDeclNode &Node = enumNodeFor(ED);
    if (!Node.MemberSpec)
      Node.MemberSpec = Generator.acquireMemberSpecChain();
    return *Node.MemberSpec;
  }

  const RecordDeclDefinitionDataChain *
  getChainFor(const CXXRecordDecl *RD) const {
    CXXClassDeclNode *N = lookupNode<CXXClassDeclNode>(RD);
    return N ? N->DefData : nullptr;
  }
  const SpecializationChain *
  getChainFor(const ClassTemplateSpecializationDecl *S) const {
    CXXClassDeclNode *N = lookupNode<CXXClassDeclNode>(S);
    return N ? N->Spec.dyn_cast<SpecializationChain *>() : nullptr;
  }

  const VarSpecializationChain *
  getChainFor(const VarTemplateSpecializationDecl *Spec) const {
    VarDeclNode *N = lookupNode<VarDeclNode>(Spec);
    return N ? N->Spec.dyn_cast<VarSpecializationChain *>() : nullptr;
  }

  const FunctionSpecializationChain *
  getChainFor(const FunctionDecl *Spec) const {
    FunctionDeclNode *N = lookupNode<FunctionDeclNode>(Spec);
    return N ? N->Spec.dyn_cast<FunctionSpecializationChain *>() : nullptr;
  }

  const MemberSpecializationChain *
  getMemberSpecChainFor(const FunctionDecl *FD) const {
    FunctionDeclNode *N = lookupNode<FunctionDeclNode>(FD);
    return N ? N->Spec.dyn_cast<MemberSpecializationChain *>() : nullptr;
  }

  const MemberSpecializationChain *
  getMemberSpecChainFor(const VarDecl *VD) const {
    VarDeclNode *N = lookupNode<VarDeclNode>(VD);
    return N ? N->Spec.dyn_cast<MemberSpecializationChain *>() : nullptr;
  }

  const MemberSpecializationChain *
  getMemberSpecChainFor(const CXXRecordDecl *RD) const {
    CXXClassDeclNode *N = lookupNode<CXXClassDeclNode>(RD);
    return N ? N->Spec.dyn_cast<MemberSpecializationChain *>() : nullptr;
  }

  const MemberSpecializationChain *
  getMemberSpecChainFor(const EnumDecl *ED) const {
    EnumDeclNode *N = lookupNode<EnumDeclNode>(ED);
    return N ? N->MemberSpec : nullptr;
  }

public:
  IncrementalStateTracker(ASTContext &Ctx, Sema &S, LangMode M)
      : Ctx(Ctx), SemaRef(S), PTUSlabCheckpoints(Ctx), Mode(M) {}

  LangMode getMode() const { return Mode; }

  PTUStateInfo &current() const {
    assert(!PTUStack.empty() && "no PTU has been started");
    return PTUStack.back();
  }

  PTUID currentID() const { return current().ID; }
  bool hasOpenPTU() const { return !PTUStack.empty(); }

  void beginPTU(llvm::SlabCheckPoint CP) {
    PTUID ID = NextID++;
    PTUSlabCheckpoints.recordCheckpoint(ID, CP);
    PTUStack.emplace_back(ID);
  }

  bool isFromThisPTU(const void *Ptr, PTUID ID) const {
    return PTUSlabCheckpoints.isFromThisPTU(Ptr, ID);
  }
  bool isFromCurrentPTU(const void *Ptr) const {
    return isFromThisPTU(Ptr, currentID());
  }

  SweepTracker &getHiddenMutationTracker() { return HiddenMutationTracker; }
  PTUCheckpointLedger &getPTUSlabCheckpoints() { return PTUSlabCheckpoints; }
  Sema &getSema() const { return SemaRef; }

  void undoLastEntries();

private:
  void popCurrentPTU() {
    PTUID ID = currentID();
    PTUStack.pop_back();
    PTUSlabCheckpoints.undoFrom(ID);
    NextID = ID;
  }

  template <typename ChainT>
  static bool removeChainField(ChainT *&Field, PTUID ID,
                               LinkedDeclNodeGenerator &Gen);

  static bool removeRecordSpec(
      llvm::PointerUnion<SpecializationChain *, MemberSpecializationChain *>
          &Spec,
      PTUID ID, LinkedDeclNodeGenerator &Gen);
  static bool removeVarSpec(
      llvm::PointerUnion<VarSpecializationChain *, MemberSpecializationChain *>
          &Spec,
      PTUID ID, LinkedDeclNodeGenerator &Gen);
  static bool
  removeFunctionSpec(llvm::PointerUnion<FunctionSpecializationChain *,
                                        MemberSpecializationChain *> &Spec,
                     PTUID ID, LinkedDeclNodeGenerator &Gen);
  void removeLinkedDecl(const Decl *D, PTUID ID);
  // just only remove entries from current();
};

class PTUMutationActions {
private:
  IncrementalStateTracker &Tracker;
  SweepTracker &HiddenMutationTracker;
  DeclStateReverter Reverter;

public:
  explicit PTUMutationActions(IncrementalStateTracker &Tracker)
      : Tracker(Tracker),
        HiddenMutationTracker(Tracker.getHiddenMutationTracker()),
        Reverter(Tracker.getPTUSlabCheckpoints()) {}

  template <typename DeclStatePolicyT>
  void walkDecls(const DeclContext *DC, DeclStatePolicyT &Proxy);

  uint32_t DeclNeedingTracking(DeclShape S, const Decl *D);

  uint32_t verifyMutationFor(const Decl *D, DeclShape S, uint32_t FlaggedKinds);

private:
  void commitMembers(PTUID ID, const DeclContext *Members);

  void commitClass(PTUID ID, const CXXRecordDecl *RD, MutationRecord &Rec,
                   bool IsNew);
  void commitFunction(PTUID ID, const FunctionDecl *FD, MutationRecord &Rec,
                      bool IsNew);
  void commitVar(PTUID ID, const VarDecl *VD, MutationRecord &Rec, bool IsNew);

  void commitEnum(PTUID ID, const EnumDecl *ED, MutationRecord &Rec,
                  bool IsNew);
  void commitTemplate(PTUID ID, const RedeclarableTemplateDecl *ED,
                      MutationRecord &Rec, bool IsNew);
  void commitTypedef(PTUID ID, const TypedefNameDecl *ED, MutationRecord &Rec,
                     bool IsNew);

  void restoreClass(PTUID ID, const CXXRecordDecl *RD, MutationRecord &Rec);

  void restoreFunction(PTUID ID, const FunctionDecl *FD, MutationRecord &Rec);

  void restoreTemplate(PTUID ID, const RedeclarableTemplateDecl *TD,
                       MutationRecord &Rec);

  void restoreVar(PTUID ID, const VarDecl *VD, MutationRecord &Rec);

  void restoreEnum(PTUID ID, const EnumDecl *ED, MutationRecord &Rec);

  void restoreTypedef(PTUID ID, const TypedefNameDecl *TD, MutationRecord &Rec);

public:
  void restoreDecl(PTUID ID, const Decl *D, MutationRecord &Rec);
  void commitDecl(PTUID ID, const Decl *D, MutationRecord &Rec,
                  bool IsNew = false);
  // respect the LIFO so only current inside map not commited can be commited
  // not randon PTUID
  // global map info shouldn't be commited before only added here. not note*
  // time.
  void commit(TranslationUnitDecl *MostRecentTU);
  void restore(TranslationUnitDecl *MostRecentTU);
};

class PTUMutationRecorder : public ASTMutationListener {
private:
  IncrementalStateTracker &Tracker;

  template <typename TemplateT, typename SpecT>
  void noteTemplateDeclMutation(const TemplateT *TD, const SpecT *Spec,
                                DeclShape S);

  void noteExceptionSpecMutation(const FunctionDecl *FD,
                                 MutationType K = MutationType::ExceptionSpec);

  void noteDefinitionInstantiated(const Decl *D);

public:
  PTUMutationRecorder(IncrementalStateTracker &Tracker) : Tracker(Tracker) {}

  /// A new TagDecl definition was completed.
  void CompletedTagDefinition(const TagDecl *D) override;

  /// A new declaration with name has been added to a DeclContext.
  void AddedVisibleDecl(const DeclContext *DC, const Decl *D) override;

  /// An implicit member was added after the definition was completed.
  void AddedCXXImplicitMember(const CXXRecordDecl *RD, const Decl *D) override;

  /// A template specialization (or partial one) was added to the
  /// template declaration.
  void AddedCXXTemplateSpecialization(
      const ClassTemplateDecl *TD,
      const ClassTemplateSpecializationDecl *D) override;

  /// A template specialization (or partial one) was added to the
  /// template declaration.
  void AddedCXXTemplateSpecialization(
      const VarTemplateDecl *TD,
      const VarTemplateSpecializationDecl *D) override;

  /// A template specialization (or partial one) was added to the
  /// template declaration.
  void AddedCXXTemplateSpecialization(const FunctionTemplateDecl *TD,
                                      const FunctionDecl *D) override;

  /// A function's exception specification has been evaluated or
  /// instantiated.
  void ResolvedExceptionSpec(const FunctionDecl *FD) override;

  /// A function's return type has been deduced.
  void DeducedReturnType(const FunctionDecl *FD, QualType ReturnType) override;

  /// A virtual destructor's operator delete has been resolved.
  void ResolvedOperatorDelete(const CXXDestructorDecl *DD,
                              const FunctionDecl *Delete,
                              Expr *ThisArg) override {}

  /// A virtual destructor's operator global delete has been resolved.
  void ResolvedOperatorGlobDelete(const CXXDestructorDecl *DD,
                                  const FunctionDecl *GlobDelete) override {}

  /// A virtual destructor's operator array delete has been resolved.
  void ResolvedOperatorArrayDelete(const CXXDestructorDecl *DD,
                                   const FunctionDecl *ArrayDelete) override {}

  /// A virtual destructor's operator global array delete has been resolved.
  void ResolvedOperatorGlobArrayDelete(
      const CXXDestructorDecl *DD,
      const FunctionDecl *GlobArrayDelete) override {}

  /// An implicit member got a definition.
  void CompletedImplicitDefinition(const FunctionDecl *D) override {
    noteDefinitionInstantiated(D);
  }

  /// The instantiation of a templated function or variable was
  /// requested. In particular, the point of instantiation and template
  /// specialization kind of \p D may have changed.
  void InstantiationRequested(const ValueDecl *D) override;

  /// A templated variable's definition was implicitly instantiated.
  void VariableDefinitionInstantiated(const VarDecl *D) override {
    noteDefinitionInstantiated(D);
  }

  /// A function template's definition was instantiated.
  void FunctionDefinitionInstantiated(const FunctionDecl *D) override {
    noteDefinitionInstantiated(D);
  }

  /// A default argument was instantiated.
  void DefaultArgumentInstantiated(const ParmVarDecl *D) override {}

  /// A default member initializer was instantiated.
  void DefaultMemberInitializerInstantiated(const FieldDecl *D) override {}

  /// A new objc category class was added for an interface.
  void AddedObjCCategoryToInterface(const ObjCCategoryDecl *CatD,
                                    const ObjCInterfaceDecl *IFD) override {}

  /// A declaration is marked used which was not previously marked used.
  ///
  /// \param D the declaration marked used
  void DeclarationMarkedUsed(const Decl *D) override {}

  /// A declaration is marked as OpenMP threadprivate which was not
  /// previously marked as threadprivate.
  ///
  /// \param D the declaration marked OpenMP threadprivate.
  void DeclarationMarkedOpenMPThreadPrivate(const Decl *D) override {}

  /// A declaration is marked as OpenMP groupprivate which was not
  /// previously marked as groupprivate.
  ///
  /// \param D the declaration marked OpenMP groupprivate.
  void DeclarationMarkedOpenMPGroupPrivate(const Decl *D) override {}

  /// A declaration is marked as OpenMP declaretarget which was not
  /// previously marked as declaretarget.
  ///
  /// \param D the declaration marked OpenMP declaretarget.
  /// \param Attr the added attribute.
  void DeclarationMarkedOpenMPDeclareTarget(const Decl *D,
                                            const Attr *Attr) override {}

  /// A declaration is marked as a variable with OpenMP allocator.
  ///
  /// \param D the declaration marked as a variable with OpenMP allocator.
  void DeclarationMarkedOpenMPAllocate(const Decl *D, const Attr *A) override {}

  /// A declaration is marked as an OpenMP indirect call target.
  ///
  /// \param D the declaration marked as an indirect call target.
  void DeclarationMarkedOpenMPIndirectCall(const Decl *D) override {}

  /// A definition has been made visible by being redefined locally.
  ///
  /// \param D The definition that was previously not visible.
  /// \param M The containing module in which the definition was made visible,
  ///        if any.
  void RedefinedHiddenDefinition(const NamedDecl *D, Module *M) override {}

  /// An attribute was added to a RecordDecl
  ///
  /// \param Attr The attribute that was added to the Record
  ///
  /// \param Record The RecordDecl that got a new attribute
  void AddedAttributeToRecord(const Attr *Attr,
                              const RecordDecl *Record) override {}

  /// An mangling number was added to a Decl
  ///
  /// \param D The decl that got a mangling number
  ///
  /// \param Number The mangling number that was added to the Decl
  void AddedManglingNumber(const Decl *D, unsigned Number) override {}

  /// An static local number was added to a Decl
  ///
  /// \param D The decl that got a static local number
  ///
  /// \param Number The static local number that was added to the Decl
  void AddedStaticLocalNumbers(const Decl *D, unsigned Number) override {}

  /// An anonymous namespace was added the translation unit decl
  ///
  /// \param TU The translation unit decl that got a new anonymous namespace
  ///
  /// \param AnonNamespace The anonymous namespace that was added
  void AddedAnonymousNamespace(const TranslationUnitDecl *TU,
                               NamespaceDecl *AnonNamespace) override {}
};
} // end namespace clang
#endif // LLVM_CLANG_INTERPRETER_ERROR_RECOVERY_H
