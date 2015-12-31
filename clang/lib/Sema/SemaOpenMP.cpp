//===--- SemaOpenMP.cpp - Semantic Analysis for OpenMP constructs ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements semantic analysis for OpenMP directives and
/// clauses.
///
//===----------------------------------------------------------------------===//

#include "TreeTransform.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Stack of data-sharing attributes for variables
//===----------------------------------------------------------------------===//

namespace {
/// \brief Default data sharing attributes, which can be applied to directive.
enum DefaultDataSharingAttributes {
  DSA_unspecified = 0, /// \brief Data sharing attribute not specified.
  DSA_none = 1 << 0,   /// \brief Default data sharing attribute 'none'.
  DSA_shared = 1 << 1  /// \brief Default data sharing attribute 'shared'.
};

template <class T> struct MatchesAny {
  explicit MatchesAny(ArrayRef<T> Arr) : Arr(std::move(Arr)) {}
  bool operator()(T Kind) {
    for (auto KindEl : Arr)
      if (KindEl == Kind)
        return true;
    return false;
  }

private:
  ArrayRef<T> Arr;
};
struct MatchesAlways {
  MatchesAlways() {}
  template <class T> bool operator()(T) { return true; }
};

typedef MatchesAny<OpenMPClauseKind> MatchesAnyClause;
typedef MatchesAny<OpenMPDirectiveKind> MatchesAnyDirective;

/// \brief Stack for tracking declarations used in OpenMP directives and
/// clauses and their data-sharing attributes.
class DSAStackTy {
public:
  struct DSAVarData {
    OpenMPDirectiveKind DKind;
    OpenMPClauseKind CKind;
    DeclRefExpr *RefExpr;
    SourceLocation ImplicitDSALoc;
    DSAVarData()
        : DKind(OMPD_unknown), CKind(OMPC_unknown), RefExpr(nullptr),
          ImplicitDSALoc() {}
  };

public:
  struct MapInfo {
    Expr *RefExpr;
  };

private:
  struct DSAInfo {
    OpenMPClauseKind Attributes;
    DeclRefExpr *RefExpr;
  };
  typedef llvm::SmallDenseMap<VarDecl *, DSAInfo, 64> DeclSAMapTy;
  typedef llvm::SmallDenseMap<VarDecl *, DeclRefExpr *, 64> AlignedMapTy;
  typedef llvm::DenseMap<VarDecl *, unsigned> LoopControlVariablesMapTy;
  typedef llvm::SmallDenseMap<VarDecl *, MapInfo, 64> MappedDeclsTy;
  typedef llvm::StringMap<std::pair<OMPCriticalDirective *, llvm::APSInt>>
      CriticalsWithHintsTy;

  struct SharingMapTy {
    DeclSAMapTy SharingMap;
    AlignedMapTy AlignedMap;
    MappedDeclsTy MappedDecls;
    LoopControlVariablesMapTy LCVMap;
    DefaultDataSharingAttributes DefaultAttr;
    SourceLocation DefaultAttrLoc;
    OpenMPDirectiveKind Directive;
    DeclarationNameInfo DirectiveName;
    Scope *CurScope;
    SourceLocation ConstructLoc;
    /// \brief first argument (Expr *) contains optional argument of the
    /// 'ordered' clause, the second one is true if the regions has 'ordered'
    /// clause, false otherwise.
    llvm::PointerIntPair<Expr *, 1, bool> OrderedRegion;
    bool NowaitRegion;
    bool CancelRegion;
    unsigned AssociatedLoops;
    SourceLocation InnerTeamsRegionLoc;
    SharingMapTy(OpenMPDirectiveKind DKind, DeclarationNameInfo Name,
                 Scope *CurScope, SourceLocation Loc)
        : SharingMap(), AlignedMap(), LCVMap(), DefaultAttr(DSA_unspecified),
          Directive(DKind), DirectiveName(std::move(Name)), CurScope(CurScope),
          ConstructLoc(Loc), OrderedRegion(), NowaitRegion(false),
          CancelRegion(false), AssociatedLoops(1), InnerTeamsRegionLoc() {}
    SharingMapTy()
        : SharingMap(), AlignedMap(), LCVMap(), DefaultAttr(DSA_unspecified),
          Directive(OMPD_unknown), DirectiveName(), CurScope(nullptr),
          ConstructLoc(), OrderedRegion(), NowaitRegion(false),
          CancelRegion(false), AssociatedLoops(1), InnerTeamsRegionLoc() {}
  };

  typedef SmallVector<SharingMapTy, 64> StackTy;

  /// \brief Stack of used declaration and their data-sharing attributes.
  StackTy Stack;
  /// \brief true, if check for DSA must be from parent directive, false, if
  /// from current directive.
  OpenMPClauseKind ClauseKindMode;
  Sema &SemaRef;
  bool ForceCapturing;
  CriticalsWithHintsTy Criticals;

  typedef SmallVector<SharingMapTy, 8>::reverse_iterator reverse_iterator;

  DSAVarData getDSA(StackTy::reverse_iterator Iter, VarDecl *D);

  /// \brief Checks if the variable is a local for OpenMP region.
  bool isOpenMPLocal(VarDecl *D, StackTy::reverse_iterator Iter);

public:
  explicit DSAStackTy(Sema &S)
      : Stack(1), ClauseKindMode(OMPC_unknown), SemaRef(S),
        ForceCapturing(false) {}

  bool isClauseParsingMode() const { return ClauseKindMode != OMPC_unknown; }
  void setClauseParsingMode(OpenMPClauseKind K) { ClauseKindMode = K; }

  bool isForceVarCapturing() const { return ForceCapturing; }
  void setForceVarCapturing(bool V) { ForceCapturing = V; }

  void push(OpenMPDirectiveKind DKind, const DeclarationNameInfo &DirName,
            Scope *CurScope, SourceLocation Loc) {
    Stack.push_back(SharingMapTy(DKind, DirName, CurScope, Loc));
    Stack.back().DefaultAttrLoc = Loc;
  }

  void pop() {
    assert(Stack.size() > 1 && "Data-sharing attributes stack is empty!");
    Stack.pop_back();
  }

  void addCriticalWithHint(OMPCriticalDirective *D, llvm::APSInt Hint) {
    Criticals[D->getDirectiveName().getAsString()] = std::make_pair(D, Hint);
  }
  const std::pair<OMPCriticalDirective *, llvm::APSInt>
  getCriticalWithHint(const DeclarationNameInfo &Name) const {
    auto I = Criticals.find(Name.getAsString());
    if (I != Criticals.end())
      return I->second;
    return std::make_pair(nullptr, llvm::APSInt());
  }
  /// \brief If 'aligned' declaration for given variable \a D was not seen yet,
  /// add it and return NULL; otherwise return previous occurrence's expression
  /// for diagnostics.
  DeclRefExpr *addUniqueAligned(VarDecl *D, DeclRefExpr *NewDE);

  /// \brief Register specified variable as loop control variable.
  void addLoopControlVariable(VarDecl *D);
  /// \brief Check if the specified variable is a loop control variable for
  /// current region.
  /// \return The index of the loop control variable in the list of associated
  /// for-loops (from outer to inner).
  unsigned isLoopControlVariable(VarDecl *D);
  /// \brief Check if the specified variable is a loop control variable for
  /// parent region.
  /// \return The index of the loop control variable in the list of associated
  /// for-loops (from outer to inner).
  unsigned isParentLoopControlVariable(VarDecl *D);
  /// \brief Get the loop control variable for the I-th loop (or nullptr) in
  /// parent directive.
  VarDecl *getParentLoopControlVariable(unsigned I);

  /// \brief Adds explicit data sharing attribute to the specified declaration.
  void addDSA(VarDecl *D, DeclRefExpr *E, OpenMPClauseKind A);

  /// \brief Returns data sharing attributes from top of the stack for the
  /// specified declaration.
  DSAVarData getTopDSA(VarDecl *D, bool FromParent);
  /// \brief Returns data-sharing attributes for the specified declaration.
  DSAVarData getImplicitDSA(VarDecl *D, bool FromParent);
  /// \brief Checks if the specified variables has data-sharing attributes which
  /// match specified \a CPred predicate in any directive which matches \a DPred
  /// predicate.
  template <class ClausesPredicate, class DirectivesPredicate>
  DSAVarData hasDSA(VarDecl *D, ClausesPredicate CPred,
                    DirectivesPredicate DPred, bool FromParent);
  /// \brief Checks if the specified variables has data-sharing attributes which
  /// match specified \a CPred predicate in any innermost directive which
  /// matches \a DPred predicate.
  template <class ClausesPredicate, class DirectivesPredicate>
  DSAVarData hasInnermostDSA(VarDecl *D, ClausesPredicate CPred,
                             DirectivesPredicate DPred,
                             bool FromParent);
  /// \brief Checks if the specified variables has explicit data-sharing
  /// attributes which match specified \a CPred predicate at the specified
  /// OpenMP region.
  bool hasExplicitDSA(VarDecl *D,
                      const llvm::function_ref<bool(OpenMPClauseKind)> &CPred,
                      unsigned Level);

  /// \brief Returns true if the directive at level \Level matches in the
  /// specified \a DPred predicate.
  bool hasExplicitDirective(
      const llvm::function_ref<bool(OpenMPDirectiveKind)> &DPred,
      unsigned Level);

  /// \brief Finds a directive which matches specified \a DPred predicate.
  template <class NamedDirectivesPredicate>
  bool hasDirective(NamedDirectivesPredicate DPred, bool FromParent);

  /// \brief Returns currently analyzed directive.
  OpenMPDirectiveKind getCurrentDirective() const {
    return Stack.back().Directive;
  }
  /// \brief Returns parent directive.
  OpenMPDirectiveKind getParentDirective() const {
    if (Stack.size() > 2)
      return Stack[Stack.size() - 2].Directive;
    return OMPD_unknown;
  }
  /// \brief Return the directive associated with the provided scope.
  OpenMPDirectiveKind getDirectiveForScope(const Scope *S) const;

  /// \brief Set default data sharing attribute to none.
  void setDefaultDSANone(SourceLocation Loc) {
    Stack.back().DefaultAttr = DSA_none;
    Stack.back().DefaultAttrLoc = Loc;
  }
  /// \brief Set default data sharing attribute to shared.
  void setDefaultDSAShared(SourceLocation Loc) {
    Stack.back().DefaultAttr = DSA_shared;
    Stack.back().DefaultAttrLoc = Loc;
  }

  DefaultDataSharingAttributes getDefaultDSA() const {
    return Stack.back().DefaultAttr;
  }
  SourceLocation getDefaultDSALocation() const {
    return Stack.back().DefaultAttrLoc;
  }

  /// \brief Checks if the specified variable is a threadprivate.
  bool isThreadPrivate(VarDecl *D) {
    DSAVarData DVar = getTopDSA(D, false);
    return isOpenMPThreadPrivate(DVar.CKind);
  }

  /// \brief Marks current region as ordered (it has an 'ordered' clause).
  void setOrderedRegion(bool IsOrdered, Expr *Param) {
    Stack.back().OrderedRegion.setInt(IsOrdered);
    Stack.back().OrderedRegion.setPointer(Param);
  }
  /// \brief Returns true, if parent region is ordered (has associated
  /// 'ordered' clause), false - otherwise.
  bool isParentOrderedRegion() const {
    if (Stack.size() > 2)
      return Stack[Stack.size() - 2].OrderedRegion.getInt();
    return false;
  }
  /// \brief Returns optional parameter for the ordered region.
  Expr *getParentOrderedRegionParam() const {
    if (Stack.size() > 2)
      return Stack[Stack.size() - 2].OrderedRegion.getPointer();
    return nullptr;
  }
  /// \brief Marks current region as nowait (it has a 'nowait' clause).
  void setNowaitRegion(bool IsNowait = true) {
    Stack.back().NowaitRegion = IsNowait;
  }
  /// \brief Returns true, if parent region is nowait (has associated
  /// 'nowait' clause), false - otherwise.
  bool isParentNowaitRegion() const {
    if (Stack.size() > 2)
      return Stack[Stack.size() - 2].NowaitRegion;
    return false;
  }
  /// \brief Marks parent region as cancel region.
  void setParentCancelRegion(bool Cancel = true) {
    if (Stack.size() > 2)
      Stack[Stack.size() - 2].CancelRegion =
          Stack[Stack.size() - 2].CancelRegion || Cancel;
  }
  /// \brief Return true if current region has inner cancel construct.
  bool isCancelRegion() const {
    return Stack.back().CancelRegion;
  }

  /// \brief Set collapse value for the region.
  void setAssociatedLoops(unsigned Val) { Stack.back().AssociatedLoops = Val; }
  /// \brief Return collapse value for region.
  unsigned getAssociatedLoops() const { return Stack.back().AssociatedLoops; }

  /// \brief Marks current target region as one with closely nested teams
  /// region.
  void setParentTeamsRegionLoc(SourceLocation TeamsRegionLoc) {
    if (Stack.size() > 2)
      Stack[Stack.size() - 2].InnerTeamsRegionLoc = TeamsRegionLoc;
  }
  /// \brief Returns true, if current region has closely nested teams region.
  bool hasInnerTeamsRegion() const {
    return getInnerTeamsRegionLoc().isValid();
  }
  /// \brief Returns location of the nested teams region (if any).
  SourceLocation getInnerTeamsRegionLoc() const {
    if (Stack.size() > 1)
      return Stack.back().InnerTeamsRegionLoc;
    return SourceLocation();
  }

  Scope *getCurScope() const { return Stack.back().CurScope; }
  Scope *getCurScope() { return Stack.back().CurScope; }
  SourceLocation getConstructLoc() { return Stack.back().ConstructLoc; }

  MapInfo getMapInfoForVar(VarDecl *VD) {
    MapInfo VarMI = {0};
    for (auto Cnt = Stack.size() - 1; Cnt > 0; --Cnt) {
      if (Stack[Cnt].MappedDecls.count(VD)) {
        VarMI = Stack[Cnt].MappedDecls[VD];
        break;
      }
    }
    return VarMI;
  }

  void addMapInfoForVar(VarDecl *VD, MapInfo MI) {
    if (Stack.size() > 1) {
      Stack.back().MappedDecls[VD] = MI;
    }
  }

  MapInfo IsMappedInCurrentRegion(VarDecl *VD) {
    assert(Stack.size() > 1 && "Target level is 0");
    MapInfo VarMI = {0};
    if (Stack.size() > 1 && Stack.back().MappedDecls.count(VD)) {
      VarMI = Stack.back().MappedDecls[VD];
    }
    return VarMI;
  }
};
bool isParallelOrTaskRegion(OpenMPDirectiveKind DKind) {
  return isOpenMPParallelDirective(DKind) || DKind == OMPD_task ||
         isOpenMPTeamsDirective(DKind) || DKind == OMPD_unknown ||
         isOpenMPTaskLoopDirective(DKind);
}
} // namespace

DSAStackTy::DSAVarData DSAStackTy::getDSA(StackTy::reverse_iterator Iter,
                                          VarDecl *D) {
  D = D->getCanonicalDecl();
  DSAVarData DVar;
  if (Iter == std::prev(Stack.rend())) {
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a region but not in construct]
    //  File-scope or namespace-scope variables referenced in called routines
    //  in the region are shared unless they appear in a threadprivate
    //  directive.
    if (!D->isFunctionOrMethodVarDecl() && !isa<ParmVarDecl>(D))
      DVar.CKind = OMPC_shared;

    // OpenMP [2.9.1.2, Data-sharing Attribute Rules for Variables Referenced
    // in a region but not in construct]
    //  Variables with static storage duration that are declared in called
    //  routines in the region are shared.
    if (D->hasGlobalStorage())
      DVar.CKind = OMPC_shared;

    return DVar;
  }

  DVar.DKind = Iter->Directive;
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.1]
  // Variables with automatic storage duration that are declared in a scope
  // inside the construct are private.
  if (isOpenMPLocal(D, Iter) && D->isLocalVarDecl() &&
      (D->getStorageClass() == SC_Auto || D->getStorageClass() == SC_None)) {
    DVar.CKind = OMPC_private;
    return DVar;
  }

  // Explicitly specified attributes and local variables with predetermined
  // attributes.
  if (Iter->SharingMap.count(D)) {
    DVar.RefExpr = Iter->SharingMap[D].RefExpr;
    DVar.CKind = Iter->SharingMap[D].Attributes;
    DVar.ImplicitDSALoc = Iter->DefaultAttrLoc;
    return DVar;
  }

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, implicitly determined, p.1]
  //  In a parallel or task construct, the data-sharing attributes of these
  //  variables are determined by the default clause, if present.
  switch (Iter->DefaultAttr) {
  case DSA_shared:
    DVar.CKind = OMPC_shared;
    DVar.ImplicitDSALoc = Iter->DefaultAttrLoc;
    return DVar;
  case DSA_none:
    return DVar;
  case DSA_unspecified:
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, implicitly determined, p.2]
    //  In a parallel construct, if no default clause is present, these
    //  variables are shared.
    DVar.ImplicitDSALoc = Iter->DefaultAttrLoc;
    if (isOpenMPParallelDirective(DVar.DKind) ||
        isOpenMPTeamsDirective(DVar.DKind)) {
      DVar.CKind = OMPC_shared;
      return DVar;
    }

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, implicitly determined, p.4]
    //  In a task construct, if no default clause is present, a variable that in
    //  the enclosing context is determined to be shared by all implicit tasks
    //  bound to the current team is shared.
    if (DVar.DKind == OMPD_task) {
      DSAVarData DVarTemp;
      for (StackTy::reverse_iterator I = std::next(Iter), EE = Stack.rend();
           I != EE; ++I) {
        // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables
        // Referenced
        // in a Construct, implicitly determined, p.6]
        //  In a task construct, if no default clause is present, a variable
        //  whose data-sharing attribute is not determined by the rules above is
        //  firstprivate.
        DVarTemp = getDSA(I, D);
        if (DVarTemp.CKind != OMPC_shared) {
          DVar.RefExpr = nullptr;
          DVar.DKind = OMPD_task;
          DVar.CKind = OMPC_firstprivate;
          return DVar;
        }
        if (isParallelOrTaskRegion(I->Directive))
          break;
      }
      DVar.DKind = OMPD_task;
      DVar.CKind =
          (DVarTemp.CKind == OMPC_unknown) ? OMPC_firstprivate : OMPC_shared;
      return DVar;
    }
  }
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, implicitly determined, p.3]
  //  For constructs other than task, if no default clause is present, these
  //  variables inherit their data-sharing attributes from the enclosing
  //  context.
  return getDSA(std::next(Iter), D);
}

DeclRefExpr *DSAStackTy::addUniqueAligned(VarDecl *D, DeclRefExpr *NewDE) {
  assert(Stack.size() > 1 && "Data sharing attributes stack is empty");
  D = D->getCanonicalDecl();
  auto It = Stack.back().AlignedMap.find(D);
  if (It == Stack.back().AlignedMap.end()) {
    assert(NewDE && "Unexpected nullptr expr to be added into aligned map");
    Stack.back().AlignedMap[D] = NewDE;
    return nullptr;
  } else {
    assert(It->second && "Unexpected nullptr expr in the aligned map");
    return It->second;
  }
  return nullptr;
}

void DSAStackTy::addLoopControlVariable(VarDecl *D) {
  assert(Stack.size() > 1 && "Data-sharing attributes stack is empty");
  D = D->getCanonicalDecl();
  Stack.back().LCVMap.insert(std::make_pair(D, Stack.back().LCVMap.size() + 1));
}

unsigned DSAStackTy::isLoopControlVariable(VarDecl *D) {
  assert(Stack.size() > 1 && "Data-sharing attributes stack is empty");
  D = D->getCanonicalDecl();
  return Stack.back().LCVMap.count(D) > 0 ? Stack.back().LCVMap[D] : 0;
}

unsigned DSAStackTy::isParentLoopControlVariable(VarDecl *D) {
  assert(Stack.size() > 2 && "Data-sharing attributes stack is empty");
  D = D->getCanonicalDecl();
  return Stack[Stack.size() - 2].LCVMap.count(D) > 0
             ? Stack[Stack.size() - 2].LCVMap[D]
             : 0;
}

VarDecl *DSAStackTy::getParentLoopControlVariable(unsigned I) {
  assert(Stack.size() > 2 && "Data-sharing attributes stack is empty");
  if (Stack[Stack.size() - 2].LCVMap.size() < I)
    return nullptr;
  for (auto &Pair : Stack[Stack.size() - 2].LCVMap) {
    if (Pair.second == I)
      return Pair.first;
  }
  return nullptr;
}

void DSAStackTy::addDSA(VarDecl *D, DeclRefExpr *E, OpenMPClauseKind A) {
  D = D->getCanonicalDecl();
  if (A == OMPC_threadprivate) {
    Stack[0].SharingMap[D].Attributes = A;
    Stack[0].SharingMap[D].RefExpr = E;
  } else {
    assert(Stack.size() > 1 && "Data-sharing attributes stack is empty");
    Stack.back().SharingMap[D].Attributes = A;
    Stack.back().SharingMap[D].RefExpr = E;
  }
}

bool DSAStackTy::isOpenMPLocal(VarDecl *D, StackTy::reverse_iterator Iter) {
  D = D->getCanonicalDecl();
  if (Stack.size() > 2) {
    reverse_iterator I = Iter, E = std::prev(Stack.rend());
    Scope *TopScope = nullptr;
    while (I != E && !isParallelOrTaskRegion(I->Directive)) {
      ++I;
    }
    if (I == E)
      return false;
    TopScope = I->CurScope ? I->CurScope->getParent() : nullptr;
    Scope *CurScope = getCurScope();
    while (CurScope != TopScope && !CurScope->isDeclScope(D)) {
      CurScope = CurScope->getParent();
    }
    return CurScope != TopScope;
  }
  return false;
}

/// \brief Build a variable declaration for OpenMP loop iteration variable.
static VarDecl *buildVarDecl(Sema &SemaRef, SourceLocation Loc, QualType Type,
                             StringRef Name, const AttrVec *Attrs = nullptr) {
  DeclContext *DC = SemaRef.CurContext;
  IdentifierInfo *II = &SemaRef.PP.getIdentifierTable().get(Name);
  TypeSourceInfo *TInfo = SemaRef.Context.getTrivialTypeSourceInfo(Type, Loc);
  VarDecl *Decl =
      VarDecl::Create(SemaRef.Context, DC, Loc, Loc, II, Type, TInfo, SC_None);
  if (Attrs) {
    for (specific_attr_iterator<AlignedAttr> I(Attrs->begin()), E(Attrs->end());
         I != E; ++I)
      Decl->addAttr(*I);
  }
  Decl->setImplicit();
  return Decl;
}

static DeclRefExpr *buildDeclRefExpr(Sema &S, VarDecl *D, QualType Ty,
                                     SourceLocation Loc,
                                     bool RefersToCapture = false) {
  D->setReferenced();
  D->markUsed(S.Context);
  return DeclRefExpr::Create(S.getASTContext(), NestedNameSpecifierLoc(),
                             SourceLocation(), D, RefersToCapture, Loc, Ty,
                             VK_LValue);
}

DSAStackTy::DSAVarData DSAStackTy::getTopDSA(VarDecl *D, bool FromParent) {
  D = D->getCanonicalDecl();
  DSAVarData DVar;

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.1]
  //  Variables appearing in threadprivate directives are threadprivate.
  if ((D->getTLSKind() != VarDecl::TLS_None &&
       !(D->hasAttr<OMPThreadPrivateDeclAttr>() &&
         SemaRef.getLangOpts().OpenMPUseTLS &&
         SemaRef.getASTContext().getTargetInfo().isTLSSupported())) ||
      (D->getStorageClass() == SC_Register && D->hasAttr<AsmLabelAttr>() &&
       !D->isLocalVarDecl())) {
    addDSA(D, buildDeclRefExpr(SemaRef, D, D->getType().getNonReferenceType(),
                               D->getLocation()),
           OMPC_threadprivate);
  }
  if (Stack[0].SharingMap.count(D)) {
    DVar.RefExpr = Stack[0].SharingMap[D].RefExpr;
    DVar.CKind = OMPC_threadprivate;
    return DVar;
  }

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.4]
  //  Static data members are shared.
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.7]
  //  Variables with static storage duration that are declared in a scope
  //  inside the construct are shared.
  if (D->isStaticDataMember()) {
    DSAVarData DVarTemp =
        hasDSA(D, isOpenMPPrivate, MatchesAlways(), FromParent);
    if (DVarTemp.CKind != OMPC_unknown && DVarTemp.RefExpr)
      return DVar;

    DVar.CKind = OMPC_shared;
    return DVar;
  }

  QualType Type = D->getType().getNonReferenceType().getCanonicalType();
  bool IsConstant = Type.isConstant(SemaRef.getASTContext());
  Type = SemaRef.getASTContext().getBaseElementType(Type);
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.6]
  //  Variables with const qualified type having no mutable member are
  //  shared.
  CXXRecordDecl *RD =
      SemaRef.getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : nullptr;
  if (auto *CTSD = dyn_cast_or_null<ClassTemplateSpecializationDecl>(RD))
    if (auto *CTD = CTSD->getSpecializedTemplate())
      RD = CTD->getTemplatedDecl();
  if (IsConstant &&
      !(SemaRef.getLangOpts().CPlusPlus && RD && RD->hasMutableFields())) {
    // Variables with const-qualified type having no mutable member may be
    // listed in a firstprivate clause, even if they are static data members.
    DSAVarData DVarTemp = hasDSA(D, MatchesAnyClause(OMPC_firstprivate),
                                 MatchesAlways(), FromParent);
    if (DVarTemp.CKind == OMPC_firstprivate && DVarTemp.RefExpr)
      return DVar;

    DVar.CKind = OMPC_shared;
    return DVar;
  }

  // Explicitly specified attributes and local variables with predetermined
  // attributes.
  auto StartI = std::next(Stack.rbegin());
  auto EndI = std::prev(Stack.rend());
  if (FromParent && StartI != EndI) {
    StartI = std::next(StartI);
  }
  auto I = std::prev(StartI);
  if (I->SharingMap.count(D)) {
    DVar.RefExpr = I->SharingMap[D].RefExpr;
    DVar.CKind = I->SharingMap[D].Attributes;
    DVar.ImplicitDSALoc = I->DefaultAttrLoc;
  }

  return DVar;
}

DSAStackTy::DSAVarData DSAStackTy::getImplicitDSA(VarDecl *D, bool FromParent) {
  D = D->getCanonicalDecl();
  auto StartI = Stack.rbegin();
  auto EndI = std::prev(Stack.rend());
  if (FromParent && StartI != EndI) {
    StartI = std::next(StartI);
  }
  return getDSA(StartI, D);
}

template <class ClausesPredicate, class DirectivesPredicate>
DSAStackTy::DSAVarData DSAStackTy::hasDSA(VarDecl *D, ClausesPredicate CPred,
                                          DirectivesPredicate DPred,
                                          bool FromParent) {
  D = D->getCanonicalDecl();
  auto StartI = std::next(Stack.rbegin());
  auto EndI = std::prev(Stack.rend());
  if (FromParent && StartI != EndI) {
    StartI = std::next(StartI);
  }
  for (auto I = StartI, EE = EndI; I != EE; ++I) {
    if (!DPred(I->Directive) && !isParallelOrTaskRegion(I->Directive))
      continue;
    DSAVarData DVar = getDSA(I, D);
    if (CPred(DVar.CKind))
      return DVar;
  }
  return DSAVarData();
}

template <class ClausesPredicate, class DirectivesPredicate>
DSAStackTy::DSAVarData
DSAStackTy::hasInnermostDSA(VarDecl *D, ClausesPredicate CPred,
                            DirectivesPredicate DPred, bool FromParent) {
  D = D->getCanonicalDecl();
  auto StartI = std::next(Stack.rbegin());
  auto EndI = std::prev(Stack.rend());
  if (FromParent && StartI != EndI) {
    StartI = std::next(StartI);
  }
  for (auto I = StartI, EE = EndI; I != EE; ++I) {
    if (!DPred(I->Directive))
      break;
    DSAVarData DVar = getDSA(I, D);
    if (CPred(DVar.CKind))
      return DVar;
    return DSAVarData();
  }
  return DSAVarData();
}

bool DSAStackTy::hasExplicitDSA(
    VarDecl *D, const llvm::function_ref<bool(OpenMPClauseKind)> &CPred,
    unsigned Level) {
  if (CPred(ClauseKindMode))
    return true;
  if (isClauseParsingMode())
    ++Level;
  D = D->getCanonicalDecl();
  auto StartI = Stack.rbegin();
  auto EndI = std::prev(Stack.rend());
  if (std::distance(StartI, EndI) <= (int)Level)
    return false;
  std::advance(StartI, Level);
  return (StartI->SharingMap.count(D) > 0) && StartI->SharingMap[D].RefExpr &&
         CPred(StartI->SharingMap[D].Attributes);
}

bool DSAStackTy::hasExplicitDirective(
    const llvm::function_ref<bool(OpenMPDirectiveKind)> &DPred,
    unsigned Level) {
  if (isClauseParsingMode())
    ++Level;
  auto StartI = Stack.rbegin();
  auto EndI = std::prev(Stack.rend());
  if (std::distance(StartI, EndI) <= (int)Level)
    return false;
  std::advance(StartI, Level);
  return DPred(StartI->Directive);
}

template <class NamedDirectivesPredicate>
bool DSAStackTy::hasDirective(NamedDirectivesPredicate DPred, bool FromParent) {
  auto StartI = std::next(Stack.rbegin());
  auto EndI = std::prev(Stack.rend());
  if (FromParent && StartI != EndI) {
    StartI = std::next(StartI);
  }
  for (auto I = StartI, EE = EndI; I != EE; ++I) {
    if (DPred(I->Directive, I->DirectiveName, I->ConstructLoc))
      return true;
  }
  return false;
}

OpenMPDirectiveKind DSAStackTy::getDirectiveForScope(const Scope *S) const {
  for (auto I = Stack.rbegin(), EE = Stack.rend(); I != EE; ++I)
    if (I->CurScope == S)
      return I->Directive;
  return OMPD_unknown;
}

void Sema::InitDataSharingAttributesStack() {
  VarDataSharingAttributesStack = new DSAStackTy(*this);
}

#define DSAStack static_cast<DSAStackTy *>(VarDataSharingAttributesStack)

bool Sema::IsOpenMPCapturedByRef(VarDecl *VD,
                                 const CapturedRegionScopeInfo *RSI) {
  assert(LangOpts.OpenMP && "OpenMP is not allowed");

  auto &Ctx = getASTContext();
  bool IsByRef = true;

  // Find the directive that is associated with the provided scope.
  auto DKind = DSAStack->getDirectiveForScope(RSI->TheScope);
  auto Ty = VD->getType();

  if (isOpenMPTargetDirective(DKind)) {
    // This table summarizes how a given variable should be passed to the device
    // given its type and the clauses where it appears. This table is based on
    // the description in OpenMP 4.5 [2.10.4, target Construct] and
    // OpenMP 4.5 [2.15.5, Data-mapping Attribute Rules and Clauses].
    //
    // =========================================================================
    // | type |  defaultmap   | pvt | first | is_device_ptr |    map   | res.  |
    // |      |(tofrom:scalar)|     |  pvt  |               |          |       |
    // =========================================================================
    // | scl  |               |     |       |       -       |          | bycopy|
    // | scl  |               |  -  |   x   |       -       |     -    | bycopy|
    // | scl  |               |  x  |   -   |       -       |     -    | null  |
    // | scl  |       x       |     |       |       -       |          | byref |
    // | scl  |       x       |  -  |   x   |       -       |     -    | bycopy|
    // | scl  |       x       |  x  |   -   |       -       |     -    | null  |
    // | scl  |               |  -  |   -   |       -       |     x    | byref |
    // | scl  |       x       |  -  |   -   |       -       |     x    | byref |
    //
    // | agg  |      n.a.     |     |       |       -       |          | byref |
    // | agg  |      n.a.     |  -  |   x   |       -       |     -    | byref |
    // | agg  |      n.a.     |  x  |   -   |       -       |     -    | null  |
    // | agg  |      n.a.     |  -  |   -   |       -       |     x    | byref |
    // | agg  |      n.a.     |  -  |   -   |       -       |    x[]   | byref |
    //
    // | ptr  |      n.a.     |     |       |       -       |          | bycopy|
    // | ptr  |      n.a.     |  -  |   x   |       -       |     -    | bycopy|
    // | ptr  |      n.a.     |  x  |   -   |       -       |     -    | null  |
    // | ptr  |      n.a.     |  -  |   -   |       -       |     x    | byref |
    // | ptr  |      n.a.     |  -  |   -   |       -       |    x[]   | bycopy|
    // | ptr  |      n.a.     |  -  |   -   |       x       |          | bycopy|
    // | ptr  |      n.a.     |  -  |   -   |       x       |     x    | bycopy|
    // | ptr  |      n.a.     |  -  |   -   |       x       |    x[]   | bycopy|
    // =========================================================================
    // Legend:
    //  scl - scalar
    //  ptr - pointer
    //  agg - aggregate
    //  x - applies
    //  - - invalid in this combination
    //  [] - mapped with an array section
    //  byref - should be mapped by reference
    //  byval - should be mapped by value
    //  null - initialize a local variable to null on the device
    //
    // Observations:
    //  - All scalar declarations that show up in a map clause have to be passed
    //    by reference, because they may have been mapped in the enclosing data
    //    environment.
    //  - If the scalar value does not fit the size of uintptr, it has to be
    //    passed by reference, regardless the result in the table above.
    //  - For pointers mapped by value that have either an implicit map or an
    //    array section, the runtime library may pass the NULL value to the
    //    device instead of the value passed to it by the compiler.

    // FIXME: Right now, only implicit maps are implemented. Properly mapping
    // values requires having the map, private, and firstprivate clauses SEMA
    // and parsing in place, which we don't yet.

    if (Ty->isReferenceType())
      Ty = Ty->castAs<ReferenceType>()->getPointeeType();
    IsByRef = !Ty->isScalarType();
  }

  // When passing data by value, we need to make sure it fits the uintptr size
  // and alignment, because the runtime library only deals with uintptr types.
  // If it does not fit the uintptr size, we need to pass the data by reference
  // instead.
  if (!IsByRef &&
      (Ctx.getTypeSizeInChars(Ty) >
           Ctx.getTypeSizeInChars(Ctx.getUIntPtrType()) ||
       Ctx.getDeclAlign(VD) > Ctx.getTypeAlignInChars(Ctx.getUIntPtrType())))
    IsByRef = true;

  return IsByRef;
}

bool Sema::IsOpenMPCapturedVar(VarDecl *VD) {
  assert(LangOpts.OpenMP && "OpenMP is not allowed");
  VD = VD->getCanonicalDecl();

  // If we are attempting to capture a global variable in a directive with
  // 'target' we return true so that this global is also mapped to the device.
  //
  // FIXME: If the declaration is enclosed in a 'declare target' directive,
  // then it should not be captured. Therefore, an extra check has to be
  // inserted here once support for 'declare target' is added.
  //
  if (!VD->hasLocalStorage()) {
    if (DSAStack->getCurrentDirective() == OMPD_target &&
        !DSAStack->isClauseParsingMode()) {
      return true;
    }
    if (DSAStack->getCurScope() &&
        DSAStack->hasDirective(
            [](OpenMPDirectiveKind K, const DeclarationNameInfo &DNI,
               SourceLocation Loc) -> bool {
              return isOpenMPTargetDirective(K);
            },
            false)) {
      return true;
    }
  }

  if (DSAStack->getCurrentDirective() != OMPD_unknown &&
      (!DSAStack->isClauseParsingMode() ||
       DSAStack->getParentDirective() != OMPD_unknown)) {
    if (DSAStack->isLoopControlVariable(VD) ||
        (VD->hasLocalStorage() &&
         isParallelOrTaskRegion(DSAStack->getCurrentDirective())) ||
        DSAStack->isForceVarCapturing())
      return true;
    auto DVarPrivate = DSAStack->getTopDSA(VD, DSAStack->isClauseParsingMode());
    if (DVarPrivate.CKind != OMPC_unknown && isOpenMPPrivate(DVarPrivate.CKind))
      return true;
    DVarPrivate = DSAStack->hasDSA(VD, isOpenMPPrivate, MatchesAlways(),
                                   DSAStack->isClauseParsingMode());
    return DVarPrivate.CKind != OMPC_unknown;
  }
  return false;
}

bool Sema::isOpenMPPrivateVar(VarDecl *VD, unsigned Level) {
  assert(LangOpts.OpenMP && "OpenMP is not allowed");
  return DSAStack->hasExplicitDSA(
      VD, [](OpenMPClauseKind K) -> bool { return K == OMPC_private; }, Level);
}

bool Sema::isOpenMPTargetCapturedVar(VarDecl *VD, unsigned Level) {
  assert(LangOpts.OpenMP && "OpenMP is not allowed");
  // Return true if the current level is no longer enclosed in a target region.

  return !VD->hasLocalStorage() &&
         DSAStack->hasExplicitDirective(isOpenMPTargetDirective, Level);
}

void Sema::DestroyDataSharingAttributesStack() { delete DSAStack; }

void Sema::StartOpenMPDSABlock(OpenMPDirectiveKind DKind,
                               const DeclarationNameInfo &DirName,
                               Scope *CurScope, SourceLocation Loc) {
  DSAStack->push(DKind, DirName, CurScope, Loc);
  PushExpressionEvaluationContext(PotentiallyEvaluated);
}

void Sema::StartOpenMPClause(OpenMPClauseKind K) {
  DSAStack->setClauseParsingMode(K);
}

void Sema::EndOpenMPClause() {
  DSAStack->setClauseParsingMode(/*K=*/OMPC_unknown);
}

void Sema::EndOpenMPDSABlock(Stmt *CurDirective) {
  // OpenMP [2.14.3.5, Restrictions, C/C++, p.1]
  //  A variable of class type (or array thereof) that appears in a lastprivate
  //  clause requires an accessible, unambiguous default constructor for the
  //  class type, unless the list item is also specified in a firstprivate
  //  clause.
  if (auto D = dyn_cast_or_null<OMPExecutableDirective>(CurDirective)) {
    for (auto *C : D->clauses()) {
      if (auto *Clause = dyn_cast<OMPLastprivateClause>(C)) {
        SmallVector<Expr *, 8> PrivateCopies;
        for (auto *DE : Clause->varlists()) {
          if (DE->isValueDependent() || DE->isTypeDependent()) {
            PrivateCopies.push_back(nullptr);
            continue;
          }
          auto *VD = cast<VarDecl>(cast<DeclRefExpr>(DE)->getDecl());
          QualType Type = VD->getType().getNonReferenceType();
          auto DVar = DSAStack->getTopDSA(VD, false);
          if (DVar.CKind == OMPC_lastprivate) {
            // Generate helper private variable and initialize it with the
            // default value. The address of the original variable is replaced
            // by the address of the new private variable in CodeGen. This new
            // variable is not added to IdResolver, so the code in the OpenMP
            // region uses original variable for proper diagnostics.
            auto *VDPrivate = buildVarDecl(
                *this, DE->getExprLoc(), Type.getUnqualifiedType(),
                VD->getName(), VD->hasAttrs() ? &VD->getAttrs() : nullptr);
            ActOnUninitializedDecl(VDPrivate, /*TypeMayContainAuto=*/false);
            if (VDPrivate->isInvalidDecl())
              continue;
            PrivateCopies.push_back(buildDeclRefExpr(
                *this, VDPrivate, DE->getType(), DE->getExprLoc()));
          } else {
            // The variable is also a firstprivate, so initialization sequence
            // for private copy is generated already.
            PrivateCopies.push_back(nullptr);
          }
        }
        // Set initializers to private copies if no errors were found.
        if (PrivateCopies.size() == Clause->varlist_size()) {
          Clause->setPrivateCopies(PrivateCopies);
        }
      }
    }
  }

  DSAStack->pop();
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();
}

static bool FinishOpenMPLinearClause(OMPLinearClause &Clause, DeclRefExpr *IV,
                                     Expr *NumIterations, Sema &SemaRef,
                                     Scope *S);

namespace {

class VarDeclFilterCCC : public CorrectionCandidateCallback {
private:
  Sema &SemaRef;

public:
  explicit VarDeclFilterCCC(Sema &S) : SemaRef(S) {}
  bool ValidateCandidate(const TypoCorrection &Candidate) override {
    NamedDecl *ND = Candidate.getCorrectionDecl();
    if (VarDecl *VD = dyn_cast_or_null<VarDecl>(ND)) {
      return VD->hasGlobalStorage() &&
             SemaRef.isDeclInScope(ND, SemaRef.getCurLexicalContext(),
                                   SemaRef.getCurScope());
    }
    return false;
  }
};
} // namespace

ExprResult Sema::ActOnOpenMPIdExpression(Scope *CurScope,
                                         CXXScopeSpec &ScopeSpec,
                                         const DeclarationNameInfo &Id) {
  LookupResult Lookup(*this, Id, LookupOrdinaryName);
  LookupParsedName(Lookup, CurScope, &ScopeSpec, true);

  if (Lookup.isAmbiguous())
    return ExprError();

  VarDecl *VD;
  if (!Lookup.isSingleResult()) {
    if (TypoCorrection Corrected = CorrectTypo(
            Id, LookupOrdinaryName, CurScope, nullptr,
            llvm::make_unique<VarDeclFilterCCC>(*this), CTK_ErrorRecovery)) {
      diagnoseTypo(Corrected,
                   PDiag(Lookup.empty()
                             ? diag::err_undeclared_var_use_suggest
                             : diag::err_omp_expected_var_arg_suggest)
                       << Id.getName());
      VD = Corrected.getCorrectionDeclAs<VarDecl>();
    } else {
      Diag(Id.getLoc(), Lookup.empty() ? diag::err_undeclared_var_use
                                       : diag::err_omp_expected_var_arg)
          << Id.getName();
      return ExprError();
    }
  } else {
    if (!(VD = Lookup.getAsSingle<VarDecl>())) {
      Diag(Id.getLoc(), diag::err_omp_expected_var_arg) << Id.getName();
      Diag(Lookup.getFoundDecl()->getLocation(), diag::note_declared_at);
      return ExprError();
    }
  }
  Lookup.suppressDiagnostics();

  // OpenMP [2.9.2, Syntax, C/C++]
  //   Variables must be file-scope, namespace-scope, or static block-scope.
  if (!VD->hasGlobalStorage()) {
    Diag(Id.getLoc(), diag::err_omp_global_var_arg)
        << getOpenMPDirectiveName(OMPD_threadprivate) << !VD->isStaticLocal();
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
    return ExprError();
  }

  VarDecl *CanonicalVD = VD->getCanonicalDecl();
  NamedDecl *ND = cast<NamedDecl>(CanonicalVD);
  // OpenMP [2.9.2, Restrictions, C/C++, p.2]
  //   A threadprivate directive for file-scope variables must appear outside
  //   any definition or declaration.
  if (CanonicalVD->getDeclContext()->isTranslationUnit() &&
      !getCurLexicalContext()->isTranslationUnit()) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
        << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
    return ExprError();
  }
  // OpenMP [2.9.2, Restrictions, C/C++, p.3]
  //   A threadprivate directive for static class member variables must appear
  //   in the class definition, in the same scope in which the member
  //   variables are declared.
  if (CanonicalVD->isStaticDataMember() &&
      !CanonicalVD->getDeclContext()->Equals(getCurLexicalContext())) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
        << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
    return ExprError();
  }
  // OpenMP [2.9.2, Restrictions, C/C++, p.4]
  //   A threadprivate directive for namespace-scope variables must appear
  //   outside any definition or declaration other than the namespace
  //   definition itself.
  if (CanonicalVD->getDeclContext()->isNamespace() &&
      (!getCurLexicalContext()->isFileContext() ||
       !getCurLexicalContext()->Encloses(CanonicalVD->getDeclContext()))) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
        << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
    return ExprError();
  }
  // OpenMP [2.9.2, Restrictions, C/C++, p.6]
  //   A threadprivate directive for static block-scope variables must appear
  //   in the scope of the variable and not in a nested scope.
  if (CanonicalVD->isStaticLocal() && CurScope &&
      !isDeclInScope(ND, getCurLexicalContext(), CurScope)) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
        << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
    return ExprError();
  }

  // OpenMP [2.9.2, Restrictions, C/C++, p.2-6]
  //   A threadprivate directive must lexically precede all references to any
  //   of the variables in its list.
  if (VD->isUsed() && !DSAStack->isThreadPrivate(VD)) {
    Diag(Id.getLoc(), diag::err_omp_var_used)
        << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    return ExprError();
  }

  QualType ExprType = VD->getType().getNonReferenceType();
  ExprResult DE = buildDeclRefExpr(*this, VD, ExprType, Id.getLoc());
  return DE;
}

Sema::DeclGroupPtrTy
Sema::ActOnOpenMPThreadprivateDirective(SourceLocation Loc,
                                        ArrayRef<Expr *> VarList) {
  if (OMPThreadPrivateDecl *D = CheckOMPThreadPrivateDecl(Loc, VarList)) {
    CurContext->addDecl(D);
    return DeclGroupPtrTy::make(DeclGroupRef(D));
  }
  return DeclGroupPtrTy();
}

namespace {
class LocalVarRefChecker : public ConstStmtVisitor<LocalVarRefChecker, bool> {
  Sema &SemaRef;

public:
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    if (auto VD = dyn_cast<VarDecl>(E->getDecl())) {
      if (VD->hasLocalStorage()) {
        SemaRef.Diag(E->getLocStart(),
                     diag::err_omp_local_var_in_threadprivate_init)
            << E->getSourceRange();
        SemaRef.Diag(VD->getLocation(), diag::note_defined_here)
            << VD << VD->getSourceRange();
        return true;
      }
    }
    return false;
  }
  bool VisitStmt(const Stmt *S) {
    for (auto Child : S->children()) {
      if (Child && Visit(Child))
        return true;
    }
    return false;
  }
  explicit LocalVarRefChecker(Sema &SemaRef) : SemaRef(SemaRef) {}
};
} // namespace

OMPThreadPrivateDecl *
Sema::CheckOMPThreadPrivateDecl(SourceLocation Loc, ArrayRef<Expr *> VarList) {
  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    DeclRefExpr *DE = cast<DeclRefExpr>(RefExpr);
    VarDecl *VD = cast<VarDecl>(DE->getDecl());
    SourceLocation ILoc = DE->getExprLoc();

    QualType QType = VD->getType();
    if (QType->isDependentType() || QType->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      continue;
    }

    // OpenMP [2.9.2, Restrictions, C/C++, p.10]
    //   A threadprivate variable must not have an incomplete type.
    if (RequireCompleteType(ILoc, VD->getType(),
                            diag::err_omp_threadprivate_incomplete_type)) {
      continue;
    }

    // OpenMP [2.9.2, Restrictions, C/C++, p.10]
    //   A threadprivate variable must not have a reference type.
    if (VD->getType()->isReferenceType()) {
      Diag(ILoc, diag::err_omp_ref_type_arg)
          << getOpenMPDirectiveName(OMPD_threadprivate) << VD->getType();
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // Check if this is a TLS variable. If TLS is not being supported, produce
    // the corresponding diagnostic.
    if ((VD->getTLSKind() != VarDecl::TLS_None &&
         !(VD->hasAttr<OMPThreadPrivateDeclAttr>() &&
           getLangOpts().OpenMPUseTLS &&
           getASTContext().getTargetInfo().isTLSSupported())) ||
        (VD->getStorageClass() == SC_Register && VD->hasAttr<AsmLabelAttr>() &&
         !VD->isLocalVarDecl())) {
      Diag(ILoc, diag::err_omp_var_thread_local)
          << VD << ((VD->getTLSKind() != VarDecl::TLS_None) ? 0 : 1);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // Check if initial value of threadprivate variable reference variable with
    // local storage (it is not supported by runtime).
    if (auto Init = VD->getAnyInitializer()) {
      LocalVarRefChecker Checker(*this);
      if (Checker.Visit(Init))
        continue;
    }

    Vars.push_back(RefExpr);
    DSAStack->addDSA(VD, DE, OMPC_threadprivate);
    VD->addAttr(OMPThreadPrivateDeclAttr::CreateImplicit(
        Context, SourceRange(Loc, Loc)));
    if (auto *ML = Context.getASTMutationListener())
      ML->DeclarationMarkedOpenMPThreadPrivate(VD);
  }
  OMPThreadPrivateDecl *D = nullptr;
  if (!Vars.empty()) {
    D = OMPThreadPrivateDecl::Create(Context, getCurLexicalContext(), Loc,
                                     Vars);
    D->setAccess(AS_public);
  }
  return D;
}

static void ReportOriginalDSA(Sema &SemaRef, DSAStackTy *Stack,
                              const VarDecl *VD, DSAStackTy::DSAVarData DVar,
                              bool IsLoopIterVar = false) {
  if (DVar.RefExpr) {
    SemaRef.Diag(DVar.RefExpr->getExprLoc(), diag::note_omp_explicit_dsa)
        << getOpenMPClauseName(DVar.CKind);
    return;
  }
  enum {
    PDSA_StaticMemberShared,
    PDSA_StaticLocalVarShared,
    PDSA_LoopIterVarPrivate,
    PDSA_LoopIterVarLinear,
    PDSA_LoopIterVarLastprivate,
    PDSA_ConstVarShared,
    PDSA_GlobalVarShared,
    PDSA_TaskVarFirstprivate,
    PDSA_LocalVarPrivate,
    PDSA_Implicit
  } Reason = PDSA_Implicit;
  bool ReportHint = false;
  auto ReportLoc = VD->getLocation();
  if (IsLoopIterVar) {
    if (DVar.CKind == OMPC_private)
      Reason = PDSA_LoopIterVarPrivate;
    else if (DVar.CKind == OMPC_lastprivate)
      Reason = PDSA_LoopIterVarLastprivate;
    else
      Reason = PDSA_LoopIterVarLinear;
  } else if (DVar.DKind == OMPD_task && DVar.CKind == OMPC_firstprivate) {
    Reason = PDSA_TaskVarFirstprivate;
    ReportLoc = DVar.ImplicitDSALoc;
  } else if (VD->isStaticLocal())
    Reason = PDSA_StaticLocalVarShared;
  else if (VD->isStaticDataMember())
    Reason = PDSA_StaticMemberShared;
  else if (VD->isFileVarDecl())
    Reason = PDSA_GlobalVarShared;
  else if (VD->getType().isConstant(SemaRef.getASTContext()))
    Reason = PDSA_ConstVarShared;
  else if (VD->isLocalVarDecl() && DVar.CKind == OMPC_private) {
    ReportHint = true;
    Reason = PDSA_LocalVarPrivate;
  }
  if (Reason != PDSA_Implicit) {
    SemaRef.Diag(ReportLoc, diag::note_omp_predetermined_dsa)
        << Reason << ReportHint
        << getOpenMPDirectiveName(Stack->getCurrentDirective());
  } else if (DVar.ImplicitDSALoc.isValid()) {
    SemaRef.Diag(DVar.ImplicitDSALoc, diag::note_omp_implicit_dsa)
        << getOpenMPClauseName(DVar.CKind);
  }
}

namespace {
class DSAAttrChecker : public StmtVisitor<DSAAttrChecker, void> {
  DSAStackTy *Stack;
  Sema &SemaRef;
  bool ErrorFound;
  CapturedStmt *CS;
  llvm::SmallVector<Expr *, 8> ImplicitFirstprivate;
  llvm::DenseMap<VarDecl *, Expr *> VarsWithInheritedDSA;

public:
  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
      // Skip internally declared variables.
      if (VD->isLocalVarDecl() && !CS->capturesVariable(VD))
        return;

      auto DVar = Stack->getTopDSA(VD, false);
      // Check if the variable has explicit DSA set and stop analysis if it so.
      if (DVar.RefExpr) return;

      auto ELoc = E->getExprLoc();
      auto DKind = Stack->getCurrentDirective();
      // The default(none) clause requires that each variable that is referenced
      // in the construct, and does not have a predetermined data-sharing
      // attribute, must have its data-sharing attribute explicitly determined
      // by being listed in a data-sharing attribute clause.
      if (DVar.CKind == OMPC_unknown && Stack->getDefaultDSA() == DSA_none &&
          isParallelOrTaskRegion(DKind) &&
          VarsWithInheritedDSA.count(VD) == 0) {
        VarsWithInheritedDSA[VD] = E;
        return;
      }

      // OpenMP [2.9.3.6, Restrictions, p.2]
      //  A list item that appears in a reduction clause of the innermost
      //  enclosing worksharing or parallel construct may not be accessed in an
      //  explicit task.
      DVar = Stack->hasInnermostDSA(VD, MatchesAnyClause(OMPC_reduction),
                                    [](OpenMPDirectiveKind K) -> bool {
                                      return isOpenMPParallelDirective(K) ||
                                             isOpenMPWorksharingDirective(K) ||
                                             isOpenMPTeamsDirective(K);
                                    },
                                    false);
      if (DKind == OMPD_task && DVar.CKind == OMPC_reduction) {
        ErrorFound = true;
        SemaRef.Diag(ELoc, diag::err_omp_reduction_in_task);
        ReportOriginalDSA(SemaRef, Stack, VD, DVar);
        return;
      }

      // Define implicit data-sharing attributes for task.
      DVar = Stack->getImplicitDSA(VD, false);
      if (DKind == OMPD_task && DVar.CKind != OMPC_shared)
        ImplicitFirstprivate.push_back(E);
    }
  }
  void VisitOMPExecutableDirective(OMPExecutableDirective *S) {
    for (auto *C : S->clauses()) {
      // Skip analysis of arguments of implicitly defined firstprivate clause
      // for task directives.
      if (C && (!isa<OMPFirstprivateClause>(C) || C->getLocStart().isValid()))
        for (auto *CC : C->children()) {
          if (CC)
            Visit(CC);
        }
    }
  }
  void VisitStmt(Stmt *S) {
    for (auto *C : S->children()) {
      if (C && !isa<OMPExecutableDirective>(C))
        Visit(C);
    }
  }

  bool isErrorFound() { return ErrorFound; }
  ArrayRef<Expr *> getImplicitFirstprivate() { return ImplicitFirstprivate; }
  llvm::DenseMap<VarDecl *, Expr *> &getVarsWithInheritedDSA() {
    return VarsWithInheritedDSA;
  }

  DSAAttrChecker(DSAStackTy *S, Sema &SemaRef, CapturedStmt *CS)
      : Stack(S), SemaRef(SemaRef), ErrorFound(false), CS(CS) {}
};
} // namespace

void Sema::ActOnOpenMPRegionStart(OpenMPDirectiveKind DKind, Scope *CurScope) {
  switch (DKind) {
  case OMPD_parallel: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1);
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_simd: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_for: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_for_simd: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_sections: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_section: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_single: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_master: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_critical: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_parallel_for: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1);
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_parallel_for_simd: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1);
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_parallel_sections: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1);
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_task: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1);
    QualType Args[] = {Context.VoidPtrTy.withConst().withRestrict()};
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.Variadic = true;
    QualType CopyFnType = Context.getFunctionType(Context.VoidTy, Args, EPI);
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32Ty),
        std::make_pair(".part_id.", KmpInt32Ty),
        std::make_pair(".privates.",
                       Context.VoidPtrTy.withConst().withRestrict()),
        std::make_pair(
            ".copy_fn.",
            Context.getPointerType(CopyFnType).withConst().withRestrict()),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    // Mark this captured region as inlined, because we don't use outlined
    // function directly.
    getCurCapturedRegion()->TheCapturedDecl->addAttr(
        AlwaysInlineAttr::CreateImplicit(
            Context, AlwaysInlineAttr::Keyword_forceinline, SourceRange()));
    break;
  }
  case OMPD_ordered: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_atomic: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_target_data:
  case OMPD_target: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_teams: {
    QualType KmpInt32Ty = Context.getIntTypeForBitwidth(32, 1);
    QualType KmpInt32PtrTy =
        Context.getPointerType(KmpInt32Ty).withConst().withRestrict();
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(".global_tid.", KmpInt32PtrTy),
        std::make_pair(".bound_tid.", KmpInt32PtrTy),
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_taskgroup: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_taskloop: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_taskloop_simd: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_distribute: {
    Sema::CapturedParamNameType Params[] = {
        std::make_pair(StringRef(), QualType()) // __context with shared vars
    };
    ActOnCapturedRegionStart(DSAStack->getConstructLoc(), CurScope, CR_OpenMP,
                             Params);
    break;
  }
  case OMPD_threadprivate:
  case OMPD_taskyield:
  case OMPD_barrier:
  case OMPD_taskwait:
  case OMPD_cancellation_point:
  case OMPD_cancel:
  case OMPD_flush:
    llvm_unreachable("OpenMP Directive is not allowed");
  case OMPD_unknown:
    llvm_unreachable("Unknown OpenMP directive");
  }
}

StmtResult Sema::ActOnOpenMPRegionEnd(StmtResult S,
                                      ArrayRef<OMPClause *> Clauses) {
  if (!S.isUsable()) {
    ActOnCapturedRegionError();
    return StmtError();
  }

  OMPOrderedClause *OC = nullptr;
  OMPScheduleClause *SC = nullptr;
  SmallVector<OMPLinearClause *, 4> LCs;
  // This is required for proper codegen.
  for (auto *Clause : Clauses) {
    if (isOpenMPPrivate(Clause->getClauseKind()) ||
        Clause->getClauseKind() == OMPC_copyprivate ||
        (getLangOpts().OpenMPUseTLS &&
         getASTContext().getTargetInfo().isTLSSupported() &&
         Clause->getClauseKind() == OMPC_copyin)) {
      DSAStack->setForceVarCapturing(Clause->getClauseKind() == OMPC_copyin);
      // Mark all variables in private list clauses as used in inner region.
      for (auto *VarRef : Clause->children()) {
        if (auto *E = cast_or_null<Expr>(VarRef)) {
          MarkDeclarationsReferencedInExpr(E);
        }
      }
      DSAStack->setForceVarCapturing(/*V=*/false);
    } else if (isParallelOrTaskRegion(DSAStack->getCurrentDirective()) &&
               Clause->getClauseKind() == OMPC_schedule) {
      // Mark all variables in private list clauses as used in inner region.
      // Required for proper codegen of combined directives.
      // TODO: add processing for other clauses.
      if (auto *E = cast_or_null<Expr>(
              cast<OMPScheduleClause>(Clause)->getHelperChunkSize()))
        MarkDeclarationsReferencedInExpr(E);
    }
    if (Clause->getClauseKind() == OMPC_schedule)
      SC = cast<OMPScheduleClause>(Clause);
    else if (Clause->getClauseKind() == OMPC_ordered)
      OC = cast<OMPOrderedClause>(Clause);
    else if (Clause->getClauseKind() == OMPC_linear)
      LCs.push_back(cast<OMPLinearClause>(Clause));
  }
  bool ErrorFound = false;
  // OpenMP, 2.7.1 Loop Construct, Restrictions
  // The nonmonotonic modifier cannot be specified if an ordered clause is
  // specified.
  if (SC &&
      (SC->getFirstScheduleModifier() == OMPC_SCHEDULE_MODIFIER_nonmonotonic ||
       SC->getSecondScheduleModifier() ==
           OMPC_SCHEDULE_MODIFIER_nonmonotonic) &&
      OC) {
    Diag(SC->getFirstScheduleModifier() == OMPC_SCHEDULE_MODIFIER_nonmonotonic
             ? SC->getFirstScheduleModifierLoc()
             : SC->getSecondScheduleModifierLoc(),
         diag::err_omp_schedule_nonmonotonic_ordered)
        << SourceRange(OC->getLocStart(), OC->getLocEnd());
    ErrorFound = true;
  }
  if (!LCs.empty() && OC && OC->getNumForLoops()) {
    for (auto *C : LCs) {
      Diag(C->getLocStart(), diag::err_omp_linear_ordered)
          << SourceRange(OC->getLocStart(), OC->getLocEnd());
    }
    ErrorFound = true;
  }
  if (isOpenMPWorksharingDirective(DSAStack->getCurrentDirective()) &&
      isOpenMPSimdDirective(DSAStack->getCurrentDirective()) && OC &&
      OC->getNumForLoops()) {
    Diag(OC->getLocStart(), diag::err_omp_ordered_simd)
        << getOpenMPDirectiveName(DSAStack->getCurrentDirective());
    ErrorFound = true;
  }
  if (ErrorFound) {
    ActOnCapturedRegionError();
    return StmtError();
  }
  return ActOnCapturedRegionEnd(S.get());
}

static bool CheckNestingOfRegions(Sema &SemaRef, DSAStackTy *Stack,
                                  OpenMPDirectiveKind CurrentRegion,
                                  const DeclarationNameInfo &CurrentName,
                                  OpenMPDirectiveKind CancelRegion,
                                  SourceLocation StartLoc) {
  // Allowed nesting of constructs
  // +------------------+-----------------+------------------------------------+
  // | Parent directive | Child directive | Closely (!), No-Closely(+), Both(*)|
  // +------------------+-----------------+------------------------------------+
  // | parallel         | parallel        | *                                  |
  // | parallel         | for             | *                                  |
  // | parallel         | for simd        | *                                  |
  // | parallel         | master          | *                                  |
  // | parallel         | critical        | *                                  |
  // | parallel         | simd            | *                                  |
  // | parallel         | sections        | *                                  |
  // | parallel         | section         | +                                  |
  // | parallel         | single          | *                                  |
  // | parallel         | parallel for    | *                                  |
  // | parallel         |parallel for simd| *                                  |
  // | parallel         |parallel sections| *                                  |
  // | parallel         | task            | *                                  |
  // | parallel         | taskyield       | *                                  |
  // | parallel         | barrier         | *                                  |
  // | parallel         | taskwait        | *                                  |
  // | parallel         | taskgroup       | *                                  |
  // | parallel         | flush           | *                                  |
  // | parallel         | ordered         | +                                  |
  // | parallel         | atomic          | *                                  |
  // | parallel         | target          | *                                  |
  // | parallel         | teams           | +                                  |
  // | parallel         | cancellation    |                                    |
  // |                  | point           | !                                  |
  // | parallel         | cancel          | !                                  |
  // | parallel         | taskloop        | *                                  |
  // | parallel         | taskloop simd   | *                                  |
  // | parallel         | distribute      |                                    |  
  // +------------------+-----------------+------------------------------------+
  // | for              | parallel        | *                                  |
  // | for              | for             | +                                  |
  // | for              | for simd        | +                                  |
  // | for              | master          | +                                  |
  // | for              | critical        | *                                  |
  // | for              | simd            | *                                  |
  // | for              | sections        | +                                  |
  // | for              | section         | +                                  |
  // | for              | single          | +                                  |
  // | for              | parallel for    | *                                  |
  // | for              |parallel for simd| *                                  |
  // | for              |parallel sections| *                                  |
  // | for              | task            | *                                  |
  // | for              | taskyield       | *                                  |
  // | for              | barrier         | +                                  |
  // | for              | taskwait        | *                                  |
  // | for              | taskgroup       | *                                  |
  // | for              | flush           | *                                  |
  // | for              | ordered         | * (if construct is ordered)        |
  // | for              | atomic          | *                                  |
  // | for              | target          | *                                  |
  // | for              | teams           | +                                  |
  // | for              | cancellation    |                                    |
  // |                  | point           | !                                  |
  // | for              | cancel          | !                                  |
  // | for              | taskloop        | *                                  |
  // | for              | taskloop simd   | *                                  |
  // | for              | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | master           | parallel        | *                                  |
  // | master           | for             | +                                  |
  // | master           | for simd        | +                                  |
  // | master           | master          | *                                  |
  // | master           | critical        | *                                  |
  // | master           | simd            | *                                  |
  // | master           | sections        | +                                  |
  // | master           | section         | +                                  |
  // | master           | single          | +                                  |
  // | master           | parallel for    | *                                  |
  // | master           |parallel for simd| *                                  |
  // | master           |parallel sections| *                                  |
  // | master           | task            | *                                  |
  // | master           | taskyield       | *                                  |
  // | master           | barrier         | +                                  |
  // | master           | taskwait        | *                                  |
  // | master           | taskgroup       | *                                  |
  // | master           | flush           | *                                  |
  // | master           | ordered         | +                                  |
  // | master           | atomic          | *                                  |
  // | master           | target          | *                                  |
  // | master           | teams           | +                                  |
  // | master           | cancellation    |                                    |
  // |                  | point           |                                    |
  // | master           | cancel          |                                    |
  // | master           | taskloop        | *                                  |
  // | master           | taskloop simd   | *                                  |
  // | master           | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | critical         | parallel        | *                                  |
  // | critical         | for             | +                                  |
  // | critical         | for simd        | +                                  |
  // | critical         | master          | *                                  |
  // | critical         | critical        | * (should have different names)    |
  // | critical         | simd            | *                                  |
  // | critical         | sections        | +                                  |
  // | critical         | section         | +                                  |
  // | critical         | single          | +                                  |
  // | critical         | parallel for    | *                                  |
  // | critical         |parallel for simd| *                                  |
  // | critical         |parallel sections| *                                  |
  // | critical         | task            | *                                  |
  // | critical         | taskyield       | *                                  |
  // | critical         | barrier         | +                                  |
  // | critical         | taskwait        | *                                  |
  // | critical         | taskgroup       | *                                  |
  // | critical         | ordered         | +                                  |
  // | critical         | atomic          | *                                  |
  // | critical         | target          | *                                  |
  // | critical         | teams           | +                                  |
  // | critical         | cancellation    |                                    |
  // |                  | point           |                                    |
  // | critical         | cancel          |                                    |
  // | critical         | taskloop        | *                                  |
  // | critical         | taskloop simd   | *                                  |
  // | critical         | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | simd             | parallel        |                                    |
  // | simd             | for             |                                    |
  // | simd             | for simd        |                                    |
  // | simd             | master          |                                    |
  // | simd             | critical        |                                    |
  // | simd             | simd            |                                    |
  // | simd             | sections        |                                    |
  // | simd             | section         |                                    |
  // | simd             | single          |                                    |
  // | simd             | parallel for    |                                    |
  // | simd             |parallel for simd|                                    |
  // | simd             |parallel sections|                                    |
  // | simd             | task            |                                    |
  // | simd             | taskyield       |                                    |
  // | simd             | barrier         |                                    |
  // | simd             | taskwait        |                                    |
  // | simd             | taskgroup       |                                    |
  // | simd             | flush           |                                    |
  // | simd             | ordered         | + (with simd clause)               |
  // | simd             | atomic          |                                    |
  // | simd             | target          |                                    |
  // | simd             | teams           |                                    |
  // | simd             | cancellation    |                                    |
  // |                  | point           |                                    |
  // | simd             | cancel          |                                    |
  // | simd             | taskloop        |                                    |
  // | simd             | taskloop simd   |                                    |
  // | simd             | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | for simd         | parallel        |                                    |
  // | for simd         | for             |                                    |
  // | for simd         | for simd        |                                    |
  // | for simd         | master          |                                    |
  // | for simd         | critical        |                                    |
  // | for simd         | simd            |                                    |
  // | for simd         | sections        |                                    |
  // | for simd         | section         |                                    |
  // | for simd         | single          |                                    |
  // | for simd         | parallel for    |                                    |
  // | for simd         |parallel for simd|                                    |
  // | for simd         |parallel sections|                                    |
  // | for simd         | task            |                                    |
  // | for simd         | taskyield       |                                    |
  // | for simd         | barrier         |                                    |
  // | for simd         | taskwait        |                                    |
  // | for simd         | taskgroup       |                                    |
  // | for simd         | flush           |                                    |
  // | for simd         | ordered         | + (with simd clause)               |
  // | for simd         | atomic          |                                    |
  // | for simd         | target          |                                    |
  // | for simd         | teams           |                                    |
  // | for simd         | cancellation    |                                    |
  // |                  | point           |                                    |
  // | for simd         | cancel          |                                    |
  // | for simd         | taskloop        |                                    |
  // | for simd         | taskloop simd   |                                    |
  // | for simd         | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | parallel for simd| parallel        |                                    |
  // | parallel for simd| for             |                                    |
  // | parallel for simd| for simd        |                                    |
  // | parallel for simd| master          |                                    |
  // | parallel for simd| critical        |                                    |
  // | parallel for simd| simd            |                                    |
  // | parallel for simd| sections        |                                    |
  // | parallel for simd| section         |                                    |
  // | parallel for simd| single          |                                    |
  // | parallel for simd| parallel for    |                                    |
  // | parallel for simd|parallel for simd|                                    |
  // | parallel for simd|parallel sections|                                    |
  // | parallel for simd| task            |                                    |
  // | parallel for simd| taskyield       |                                    |
  // | parallel for simd| barrier         |                                    |
  // | parallel for simd| taskwait        |                                    |
  // | parallel for simd| taskgroup       |                                    |
  // | parallel for simd| flush           |                                    |
  // | parallel for simd| ordered         | + (with simd clause)               |
  // | parallel for simd| atomic          |                                    |
  // | parallel for simd| target          |                                    |
  // | parallel for simd| teams           |                                    |
  // | parallel for simd| cancellation    |                                    |
  // |                  | point           |                                    |
  // | parallel for simd| cancel          |                                    |
  // | parallel for simd| taskloop        |                                    |
  // | parallel for simd| taskloop simd   |                                    |
  // | parallel for simd| distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | sections         | parallel        | *                                  |
  // | sections         | for             | +                                  |
  // | sections         | for simd        | +                                  |
  // | sections         | master          | +                                  |
  // | sections         | critical        | *                                  |
  // | sections         | simd            | *                                  |
  // | sections         | sections        | +                                  |
  // | sections         | section         | *                                  |
  // | sections         | single          | +                                  |
  // | sections         | parallel for    | *                                  |
  // | sections         |parallel for simd| *                                  |
  // | sections         |parallel sections| *                                  |
  // | sections         | task            | *                                  |
  // | sections         | taskyield       | *                                  |
  // | sections         | barrier         | +                                  |
  // | sections         | taskwait        | *                                  |
  // | sections         | taskgroup       | *                                  |
  // | sections         | flush           | *                                  |
  // | sections         | ordered         | +                                  |
  // | sections         | atomic          | *                                  |
  // | sections         | target          | *                                  |
  // | sections         | teams           | +                                  |
  // | sections         | cancellation    |                                    |
  // |                  | point           | !                                  |
  // | sections         | cancel          | !                                  |
  // | sections         | taskloop        | *                                  |
  // | sections         | taskloop simd   | *                                  |
  // | sections         | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | section          | parallel        | *                                  |
  // | section          | for             | +                                  |
  // | section          | for simd        | +                                  |
  // | section          | master          | +                                  |
  // | section          | critical        | *                                  |
  // | section          | simd            | *                                  |
  // | section          | sections        | +                                  |
  // | section          | section         | +                                  |
  // | section          | single          | +                                  |
  // | section          | parallel for    | *                                  |
  // | section          |parallel for simd| *                                  |
  // | section          |parallel sections| *                                  |
  // | section          | task            | *                                  |
  // | section          | taskyield       | *                                  |
  // | section          | barrier         | +                                  |
  // | section          | taskwait        | *                                  |
  // | section          | taskgroup       | *                                  |
  // | section          | flush           | *                                  |
  // | section          | ordered         | +                                  |
  // | section          | atomic          | *                                  |
  // | section          | target          | *                                  |
  // | section          | teams           | +                                  |
  // | section          | cancellation    |                                    |
  // |                  | point           | !                                  |
  // | section          | cancel          | !                                  |
  // | section          | taskloop        | *                                  |
  // | section          | taskloop simd   | *                                  |
  // | section          | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | single           | parallel        | *                                  |
  // | single           | for             | +                                  |
  // | single           | for simd        | +                                  |
  // | single           | master          | +                                  |
  // | single           | critical        | *                                  |
  // | single           | simd            | *                                  |
  // | single           | sections        | +                                  |
  // | single           | section         | +                                  |
  // | single           | single          | +                                  |
  // | single           | parallel for    | *                                  |
  // | single           |parallel for simd| *                                  |
  // | single           |parallel sections| *                                  |
  // | single           | task            | *                                  |
  // | single           | taskyield       | *                                  |
  // | single           | barrier         | +                                  |
  // | single           | taskwait        | *                                  |
  // | single           | taskgroup       | *                                  |
  // | single           | flush           | *                                  |
  // | single           | ordered         | +                                  |
  // | single           | atomic          | *                                  |
  // | single           | target          | *                                  |
  // | single           | teams           | +                                  |
  // | single           | cancellation    |                                    |
  // |                  | point           |                                    |
  // | single           | cancel          |                                    |
  // | single           | taskloop        | *                                  |
  // | single           | taskloop simd   | *                                  |
  // | single           | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | parallel for     | parallel        | *                                  |
  // | parallel for     | for             | +                                  |
  // | parallel for     | for simd        | +                                  |
  // | parallel for     | master          | +                                  |
  // | parallel for     | critical        | *                                  |
  // | parallel for     | simd            | *                                  |
  // | parallel for     | sections        | +                                  |
  // | parallel for     | section         | +                                  |
  // | parallel for     | single          | +                                  |
  // | parallel for     | parallel for    | *                                  |
  // | parallel for     |parallel for simd| *                                  |
  // | parallel for     |parallel sections| *                                  |
  // | parallel for     | task            | *                                  |
  // | parallel for     | taskyield       | *                                  |
  // | parallel for     | barrier         | +                                  |
  // | parallel for     | taskwait        | *                                  |
  // | parallel for     | taskgroup       | *                                  |
  // | parallel for     | flush           | *                                  |
  // | parallel for     | ordered         | * (if construct is ordered)        |
  // | parallel for     | atomic          | *                                  |
  // | parallel for     | target          | *                                  |
  // | parallel for     | teams           | +                                  |
  // | parallel for     | cancellation    |                                    |
  // |                  | point           | !                                  |
  // | parallel for     | cancel          | !                                  |
  // | parallel for     | taskloop        | *                                  |
  // | parallel for     | taskloop simd   | *                                  |
  // | parallel for     | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | parallel sections| parallel        | *                                  |
  // | parallel sections| for             | +                                  |
  // | parallel sections| for simd        | +                                  |
  // | parallel sections| master          | +                                  |
  // | parallel sections| critical        | +                                  |
  // | parallel sections| simd            | *                                  |
  // | parallel sections| sections        | +                                  |
  // | parallel sections| section         | *                                  |
  // | parallel sections| single          | +                                  |
  // | parallel sections| parallel for    | *                                  |
  // | parallel sections|parallel for simd| *                                  |
  // | parallel sections|parallel sections| *                                  |
  // | parallel sections| task            | *                                  |
  // | parallel sections| taskyield       | *                                  |
  // | parallel sections| barrier         | +                                  |
  // | parallel sections| taskwait        | *                                  |
  // | parallel sections| taskgroup       | *                                  |
  // | parallel sections| flush           | *                                  |
  // | parallel sections| ordered         | +                                  |
  // | parallel sections| atomic          | *                                  |
  // | parallel sections| target          | *                                  |
  // | parallel sections| teams           | +                                  |
  // | parallel sections| cancellation    |                                    |
  // |                  | point           | !                                  |
  // | parallel sections| cancel          | !                                  |
  // | parallel sections| taskloop        | *                                  |
  // | parallel sections| taskloop simd   | *                                  |
  // | parallel sections| distribute      |                                    | 
  // +------------------+-----------------+------------------------------------+
  // | task             | parallel        | *                                  |
  // | task             | for             | +                                  |
  // | task             | for simd        | +                                  |
  // | task             | master          | +                                  |
  // | task             | critical        | *                                  |
  // | task             | simd            | *                                  |
  // | task             | sections        | +                                  |
  // | task             | section         | +                                  |
  // | task             | single          | +                                  |
  // | task             | parallel for    | *                                  |
  // | task             |parallel for simd| *                                  |
  // | task             |parallel sections| *                                  |
  // | task             | task            | *                                  |
  // | task             | taskyield       | *                                  |
  // | task             | barrier         | +                                  |
  // | task             | taskwait        | *                                  |
  // | task             | taskgroup       | *                                  |
  // | task             | flush           | *                                  |
  // | task             | ordered         | +                                  |
  // | task             | atomic          | *                                  |
  // | task             | target          | *                                  |
  // | task             | teams           | +                                  |
  // | task             | cancellation    |                                    |
  // |                  | point           | !                                  |
  // | task             | cancel          | !                                  |
  // | task             | taskloop        | *                                  |
  // | task             | taskloop simd   | *                                  |
  // | task             | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | ordered          | parallel        | *                                  |
  // | ordered          | for             | +                                  |
  // | ordered          | for simd        | +                                  |
  // | ordered          | master          | *                                  |
  // | ordered          | critical        | *                                  |
  // | ordered          | simd            | *                                  |
  // | ordered          | sections        | +                                  |
  // | ordered          | section         | +                                  |
  // | ordered          | single          | +                                  |
  // | ordered          | parallel for    | *                                  |
  // | ordered          |parallel for simd| *                                  |
  // | ordered          |parallel sections| *                                  |
  // | ordered          | task            | *                                  |
  // | ordered          | taskyield       | *                                  |
  // | ordered          | barrier         | +                                  |
  // | ordered          | taskwait        | *                                  |
  // | ordered          | taskgroup       | *                                  |
  // | ordered          | flush           | *                                  |
  // | ordered          | ordered         | +                                  |
  // | ordered          | atomic          | *                                  |
  // | ordered          | target          | *                                  |
  // | ordered          | teams           | +                                  |
  // | ordered          | cancellation    |                                    |
  // |                  | point           |                                    |
  // | ordered          | cancel          |                                    |
  // | ordered          | taskloop        | *                                  |
  // | ordered          | taskloop simd   | *                                  |
  // | ordered          | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | atomic           | parallel        |                                    |
  // | atomic           | for             |                                    |
  // | atomic           | for simd        |                                    |
  // | atomic           | master          |                                    |
  // | atomic           | critical        |                                    |
  // | atomic           | simd            |                                    |
  // | atomic           | sections        |                                    |
  // | atomic           | section         |                                    |
  // | atomic           | single          |                                    |
  // | atomic           | parallel for    |                                    |
  // | atomic           |parallel for simd|                                    |
  // | atomic           |parallel sections|                                    |
  // | atomic           | task            |                                    |
  // | atomic           | taskyield       |                                    |
  // | atomic           | barrier         |                                    |
  // | atomic           | taskwait        |                                    |
  // | atomic           | taskgroup       |                                    |
  // | atomic           | flush           |                                    |
  // | atomic           | ordered         |                                    |
  // | atomic           | atomic          |                                    |
  // | atomic           | target          |                                    |
  // | atomic           | teams           |                                    |
  // | atomic           | cancellation    |                                    |
  // |                  | point           |                                    |
  // | atomic           | cancel          |                                    |
  // | atomic           | taskloop        |                                    |
  // | atomic           | taskloop simd   |                                    |
  // | atomic           | distribute      |                                    | 
  // +------------------+-----------------+------------------------------------+
  // | target           | parallel        | *                                  |
  // | target           | for             | *                                  |
  // | target           | for simd        | *                                  |
  // | target           | master          | *                                  |
  // | target           | critical        | *                                  |
  // | target           | simd            | *                                  |
  // | target           | sections        | *                                  |
  // | target           | section         | *                                  |
  // | target           | single          | *                                  |
  // | target           | parallel for    | *                                  |
  // | target           |parallel for simd| *                                  |
  // | target           |parallel sections| *                                  |
  // | target           | task            | *                                  |
  // | target           | taskyield       | *                                  |
  // | target           | barrier         | *                                  |
  // | target           | taskwait        | *                                  |
  // | target           | taskgroup       | *                                  |
  // | target           | flush           | *                                  |
  // | target           | ordered         | *                                  |
  // | target           | atomic          | *                                  |
  // | target           | target          | *                                  |
  // | target           | teams           | *                                  |
  // | target           | cancellation    |                                    |
  // |                  | point           |                                    |
  // | target           | cancel          |                                    |
  // | target           | taskloop        | *                                  |
  // | target           | taskloop simd   | *                                  |
  // | target           | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | teams            | parallel        | *                                  |
  // | teams            | for             | +                                  |
  // | teams            | for simd        | +                                  |
  // | teams            | master          | +                                  |
  // | teams            | critical        | +                                  |
  // | teams            | simd            | +                                  |
  // | teams            | sections        | +                                  |
  // | teams            | section         | +                                  |
  // | teams            | single          | +                                  |
  // | teams            | parallel for    | *                                  |
  // | teams            |parallel for simd| *                                  |
  // | teams            |parallel sections| *                                  |
  // | teams            | task            | +                                  |
  // | teams            | taskyield       | +                                  |
  // | teams            | barrier         | +                                  |
  // | teams            | taskwait        | +                                  |
  // | teams            | taskgroup       | +                                  |
  // | teams            | flush           | +                                  |
  // | teams            | ordered         | +                                  |
  // | teams            | atomic          | +                                  |
  // | teams            | target          | +                                  |
  // | teams            | teams           | +                                  |
  // | teams            | cancellation    |                                    |
  // |                  | point           |                                    |
  // | teams            | cancel          |                                    |
  // | teams            | taskloop        | +                                  |
  // | teams            | taskloop simd   | +                                  |
  // | teams            | distribute      | !                                  |
  // +------------------+-----------------+------------------------------------+
  // | taskloop         | parallel        | *                                  |
  // | taskloop         | for             | +                                  |
  // | taskloop         | for simd        | +                                  |
  // | taskloop         | master          | +                                  |
  // | taskloop         | critical        | *                                  |
  // | taskloop         | simd            | *                                  |
  // | taskloop         | sections        | +                                  |
  // | taskloop         | section         | +                                  |
  // | taskloop         | single          | +                                  |
  // | taskloop         | parallel for    | *                                  |
  // | taskloop         |parallel for simd| *                                  |
  // | taskloop         |parallel sections| *                                  |
  // | taskloop         | task            | *                                  |
  // | taskloop         | taskyield       | *                                  |
  // | taskloop         | barrier         | +                                  |
  // | taskloop         | taskwait        | *                                  |
  // | taskloop         | taskgroup       | *                                  |
  // | taskloop         | flush           | *                                  |
  // | taskloop         | ordered         | +                                  |
  // | taskloop         | atomic          | *                                  |
  // | taskloop         | target          | *                                  |
  // | taskloop         | teams           | +                                  |
  // | taskloop         | cancellation    |                                    |
  // |                  | point           |                                    |
  // | taskloop         | cancel          |                                    |
  // | taskloop         | taskloop        | *                                  |
  // | taskloop         | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | taskloop simd    | parallel        |                                    |
  // | taskloop simd    | for             |                                    |
  // | taskloop simd    | for simd        |                                    |
  // | taskloop simd    | master          |                                    |
  // | taskloop simd    | critical        |                                    |
  // | taskloop simd    | simd            |                                    |
  // | taskloop simd    | sections        |                                    |
  // | taskloop simd    | section         |                                    |
  // | taskloop simd    | single          |                                    |
  // | taskloop simd    | parallel for    |                                    |
  // | taskloop simd    |parallel for simd|                                    |
  // | taskloop simd    |parallel sections|                                    |
  // | taskloop simd    | task            |                                    |
  // | taskloop simd    | taskyield       |                                    |
  // | taskloop simd    | barrier         |                                    |
  // | taskloop simd    | taskwait        |                                    |
  // | taskloop simd    | taskgroup       |                                    |
  // | taskloop simd    | flush           |                                    |
  // | taskloop simd    | ordered         | + (with simd clause)               |
  // | taskloop simd    | atomic          |                                    |
  // | taskloop simd    | target          |                                    |
  // | taskloop simd    | teams           |                                    |
  // | taskloop simd    | cancellation    |                                    |
  // |                  | point           |                                    |
  // | taskloop simd    | cancel          |                                    |
  // | taskloop simd    | taskloop        |                                    |
  // | taskloop simd    | taskloop simd   |                                    |
  // | taskloop simd    | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  // | distribute       | parallel        | *                                  |
  // | distribute       | for             | *                                  |
  // | distribute       | for simd        | *                                  |
  // | distribute       | master          | *                                  |
  // | distribute       | critical        | *                                  |
  // | distribute       | simd            | *                                  |
  // | distribute       | sections        | *                                  |
  // | distribute       | section         | *                                  |
  // | distribute       | single          | *                                  |
  // | distribute       | parallel for    | *                                  |
  // | distribute       |parallel for simd| *                                  |
  // | distribute       |parallel sections| *                                  |
  // | distribute       | task            | *                                  |
  // | distribute       | taskyield       | *                                  |
  // | distribute       | barrier         | *                                  |
  // | distribute       | taskwait        | *                                  |
  // | distribute       | taskgroup       | *                                  |
  // | distribute       | flush           | *                                  |
  // | distribute       | ordered         | +                                  |
  // | distribute       | atomic          | *                                  |
  // | distribute       | target          |                                    |
  // | distribute       | teams           |                                    |
  // | distribute       | cancellation    | +                                  |
  // |                  | point           |                                    |
  // | distribute       | cancel          | +                                  |
  // | distribute       | taskloop        | *                                  |
  // | distribute       | taskloop simd   | *                                  |
  // | distribute       | distribute      |                                    |
  // +------------------+-----------------+------------------------------------+
  if (Stack->getCurScope()) {
    auto ParentRegion = Stack->getParentDirective();
    bool NestingProhibited = false;
    bool CloseNesting = true;
    enum {
      NoRecommend,
      ShouldBeInParallelRegion,
      ShouldBeInOrderedRegion,
      ShouldBeInTargetRegion,
      ShouldBeInTeamsRegion
    } Recommend = NoRecommend;
    if (isOpenMPSimdDirective(ParentRegion) && CurrentRegion != OMPD_ordered) {
      // OpenMP [2.16, Nesting of Regions]
      // OpenMP constructs may not be nested inside a simd region.
      // OpenMP [2.8.1,simd Construct, Restrictions]
      // An ordered construct with the simd clause is the only OpenMP construct
      // that can appear in the simd region.
      SemaRef.Diag(StartLoc, diag::err_omp_prohibited_region_simd);
      return true;
    }
    if (ParentRegion == OMPD_atomic) {
      // OpenMP [2.16, Nesting of Regions]
      // OpenMP constructs may not be nested inside an atomic region.
      SemaRef.Diag(StartLoc, diag::err_omp_prohibited_region_atomic);
      return true;
    }
    if (CurrentRegion == OMPD_section) {
      // OpenMP [2.7.2, sections Construct, Restrictions]
      // Orphaned section directives are prohibited. That is, the section
      // directives must appear within the sections construct and must not be
      // encountered elsewhere in the sections region.
      if (ParentRegion != OMPD_sections &&
          ParentRegion != OMPD_parallel_sections) {
        SemaRef.Diag(StartLoc, diag::err_omp_orphaned_section_directive)
            << (ParentRegion != OMPD_unknown)
            << getOpenMPDirectiveName(ParentRegion);
        return true;
      }
      return false;
    }
    // Allow some constructs to be orphaned (they could be used in functions,
    // called from OpenMP regions with the required preconditions).
    if (ParentRegion == OMPD_unknown)
      return false;
    if (CurrentRegion == OMPD_cancellation_point ||
        CurrentRegion == OMPD_cancel) {
      // OpenMP [2.16, Nesting of Regions]
      // A cancellation point construct for which construct-type-clause is
      // taskgroup must be nested inside a task construct. A cancellation
      // point construct for which construct-type-clause is not taskgroup must
      // be closely nested inside an OpenMP construct that matches the type
      // specified in construct-type-clause.
      // A cancel construct for which construct-type-clause is taskgroup must be
      // nested inside a task construct. A cancel construct for which
      // construct-type-clause is not taskgroup must be closely nested inside an
      // OpenMP construct that matches the type specified in
      // construct-type-clause.
      NestingProhibited =
          !((CancelRegion == OMPD_parallel && ParentRegion == OMPD_parallel) ||
            (CancelRegion == OMPD_for &&
             (ParentRegion == OMPD_for || ParentRegion == OMPD_parallel_for)) ||
            (CancelRegion == OMPD_taskgroup && ParentRegion == OMPD_task) ||
            (CancelRegion == OMPD_sections &&
             (ParentRegion == OMPD_section || ParentRegion == OMPD_sections ||
              ParentRegion == OMPD_parallel_sections)));
    } else if (CurrentRegion == OMPD_master) {
      // OpenMP [2.16, Nesting of Regions]
      // A master region may not be closely nested inside a worksharing,
      // atomic, or explicit task region.
      NestingProhibited = isOpenMPWorksharingDirective(ParentRegion) ||
                          ParentRegion == OMPD_task ||
                          isOpenMPTaskLoopDirective(ParentRegion);
    } else if (CurrentRegion == OMPD_critical && CurrentName.getName()) {
      // OpenMP [2.16, Nesting of Regions]
      // A critical region may not be nested (closely or otherwise) inside a
      // critical region with the same name. Note that this restriction is not
      // sufficient to prevent deadlock.
      SourceLocation PreviousCriticalLoc;
      bool DeadLock =
          Stack->hasDirective([CurrentName, &PreviousCriticalLoc](
                                  OpenMPDirectiveKind K,
                                  const DeclarationNameInfo &DNI,
                                  SourceLocation Loc)
                                  ->bool {
                                if (K == OMPD_critical &&
                                    DNI.getName() == CurrentName.getName()) {
                                  PreviousCriticalLoc = Loc;
                                  return true;
                                } else
                                  return false;
                              },
                              false /* skip top directive */);
      if (DeadLock) {
        SemaRef.Diag(StartLoc,
                     diag::err_omp_prohibited_region_critical_same_name)
            << CurrentName.getName();
        if (PreviousCriticalLoc.isValid())
          SemaRef.Diag(PreviousCriticalLoc,
                       diag::note_omp_previous_critical_region);
        return true;
      }
    } else if (CurrentRegion == OMPD_barrier) {
      // OpenMP [2.16, Nesting of Regions]
      // A barrier region may not be closely nested inside a worksharing,
      // explicit task, critical, ordered, atomic, or master region.
      NestingProhibited =
          isOpenMPWorksharingDirective(ParentRegion) ||
          ParentRegion == OMPD_task || ParentRegion == OMPD_master ||
          ParentRegion == OMPD_critical || ParentRegion == OMPD_ordered ||
          isOpenMPTaskLoopDirective(ParentRegion);
    } else if (isOpenMPWorksharingDirective(CurrentRegion) &&
               !isOpenMPParallelDirective(CurrentRegion)) {
      // OpenMP [2.16, Nesting of Regions]
      // A worksharing region may not be closely nested inside a worksharing,
      // explicit task, critical, ordered, atomic, or master region.
      NestingProhibited =
          isOpenMPWorksharingDirective(ParentRegion) ||
          ParentRegion == OMPD_task || ParentRegion == OMPD_master ||
          ParentRegion == OMPD_critical || ParentRegion == OMPD_ordered ||
          isOpenMPTaskLoopDirective(ParentRegion);
      Recommend = ShouldBeInParallelRegion;
    } else if (CurrentRegion == OMPD_ordered) {
      // OpenMP [2.16, Nesting of Regions]
      // An ordered region may not be closely nested inside a critical,
      // atomic, or explicit task region.
      // An ordered region must be closely nested inside a loop region (or
      // parallel loop region) with an ordered clause.
      // OpenMP [2.8.1,simd Construct, Restrictions]
      // An ordered construct with the simd clause is the only OpenMP construct
      // that can appear in the simd region.
      NestingProhibited = ParentRegion == OMPD_critical ||
                          ParentRegion == OMPD_task ||
                          isOpenMPTaskLoopDirective(ParentRegion) ||
                          !(isOpenMPSimdDirective(ParentRegion) ||
                            Stack->isParentOrderedRegion());
      Recommend = ShouldBeInOrderedRegion;
    } else if (isOpenMPTeamsDirective(CurrentRegion)) {
      // OpenMP [2.16, Nesting of Regions]
      // If specified, a teams construct must be contained within a target
      // construct.
      NestingProhibited = ParentRegion != OMPD_target;
      Recommend = ShouldBeInTargetRegion;
      Stack->setParentTeamsRegionLoc(Stack->getConstructLoc());
    }
    if (!NestingProhibited && isOpenMPTeamsDirective(ParentRegion)) {
      // OpenMP [2.16, Nesting of Regions]
      // distribute, parallel, parallel sections, parallel workshare, and the
      // parallel loop and parallel loop SIMD constructs are the only OpenMP
      // constructs that can be closely nested in the teams region.
      NestingProhibited = !isOpenMPParallelDirective(CurrentRegion) &&
                          !isOpenMPDistributeDirective(CurrentRegion);
      Recommend = ShouldBeInParallelRegion;
    }
    if (!NestingProhibited && isOpenMPDistributeDirective(CurrentRegion)) {
      // OpenMP 4.5 [2.17 Nesting of Regions]
      // The region associated with the distribute construct must be strictly
      // nested inside a teams region
      NestingProhibited = !isOpenMPTeamsDirective(ParentRegion);
      Recommend = ShouldBeInTeamsRegion;
    }
    if (NestingProhibited) {
      SemaRef.Diag(StartLoc, diag::err_omp_prohibited_region)
          << CloseNesting << getOpenMPDirectiveName(ParentRegion) << Recommend
          << getOpenMPDirectiveName(CurrentRegion);
      return true;
    }
  }
  return false;
}

static bool checkIfClauses(Sema &S, OpenMPDirectiveKind Kind,
                           ArrayRef<OMPClause *> Clauses,
                           ArrayRef<OpenMPDirectiveKind> AllowedNameModifiers) {
  bool ErrorFound = false;
  unsigned NamedModifiersNumber = 0;
  SmallVector<const OMPIfClause *, OMPC_unknown + 1> FoundNameModifiers(
      OMPD_unknown + 1);
  SmallVector<SourceLocation, 4> NameModifierLoc;
  for (const auto *C : Clauses) {
    if (const auto *IC = dyn_cast_or_null<OMPIfClause>(C)) {
      // At most one if clause without a directive-name-modifier can appear on
      // the directive.
      OpenMPDirectiveKind CurNM = IC->getNameModifier();
      if (FoundNameModifiers[CurNM]) {
        S.Diag(C->getLocStart(), diag::err_omp_more_one_clause)
            << getOpenMPDirectiveName(Kind) << getOpenMPClauseName(OMPC_if)
            << (CurNM != OMPD_unknown) << getOpenMPDirectiveName(CurNM);
        ErrorFound = true;
      } else if (CurNM != OMPD_unknown) {
        NameModifierLoc.push_back(IC->getNameModifierLoc());
        ++NamedModifiersNumber;
      }
      FoundNameModifiers[CurNM] = IC;
      if (CurNM == OMPD_unknown)
        continue;
      // Check if the specified name modifier is allowed for the current
      // directive.
      // At most one if clause with the particular directive-name-modifier can
      // appear on the directive.
      bool MatchFound = false;
      for (auto NM : AllowedNameModifiers) {
        if (CurNM == NM) {
          MatchFound = true;
          break;
        }
      }
      if (!MatchFound) {
        S.Diag(IC->getNameModifierLoc(),
               diag::err_omp_wrong_if_directive_name_modifier)
            << getOpenMPDirectiveName(CurNM) << getOpenMPDirectiveName(Kind);
        ErrorFound = true;
      }
    }
  }
  // If any if clause on the directive includes a directive-name-modifier then
  // all if clauses on the directive must include a directive-name-modifier.
  if (FoundNameModifiers[OMPD_unknown] && NamedModifiersNumber > 0) {
    if (NamedModifiersNumber == AllowedNameModifiers.size()) {
      S.Diag(FoundNameModifiers[OMPD_unknown]->getLocStart(),
             diag::err_omp_no_more_if_clause);
    } else {
      std::string Values;
      std::string Sep(", ");
      unsigned AllowedCnt = 0;
      unsigned TotalAllowedNum =
          AllowedNameModifiers.size() - NamedModifiersNumber;
      for (unsigned Cnt = 0, End = AllowedNameModifiers.size(); Cnt < End;
           ++Cnt) {
        OpenMPDirectiveKind NM = AllowedNameModifiers[Cnt];
        if (!FoundNameModifiers[NM]) {
          Values += "'";
          Values += getOpenMPDirectiveName(NM);
          Values += "'";
          if (AllowedCnt + 2 == TotalAllowedNum)
            Values += " or ";
          else if (AllowedCnt + 1 != TotalAllowedNum)
            Values += Sep;
          ++AllowedCnt;
        }
      }
      S.Diag(FoundNameModifiers[OMPD_unknown]->getCondition()->getLocStart(),
             diag::err_omp_unnamed_if_clause)
          << (TotalAllowedNum > 1) << Values;
    }
    for (auto Loc : NameModifierLoc) {
      S.Diag(Loc, diag::note_omp_previous_named_if_clause);
    }
    ErrorFound = true;
  }
  return ErrorFound;
}

StmtResult Sema::ActOnOpenMPExecutableDirective(
    OpenMPDirectiveKind Kind, const DeclarationNameInfo &DirName,
    OpenMPDirectiveKind CancelRegion, ArrayRef<OMPClause *> Clauses,
    Stmt *AStmt, SourceLocation StartLoc, SourceLocation EndLoc) {
  StmtResult Res = StmtError();
  if (CheckNestingOfRegions(*this, DSAStack, Kind, DirName, CancelRegion,
                            StartLoc))
    return StmtError();

  llvm::SmallVector<OMPClause *, 8> ClausesWithImplicit;
  llvm::DenseMap<VarDecl *, Expr *> VarsWithInheritedDSA;
  bool ErrorFound = false;
  ClausesWithImplicit.append(Clauses.begin(), Clauses.end());
  if (AStmt) {
    assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

    // Check default data sharing attributes for referenced variables.
    DSAAttrChecker DSAChecker(DSAStack, *this, cast<CapturedStmt>(AStmt));
    DSAChecker.Visit(cast<CapturedStmt>(AStmt)->getCapturedStmt());
    if (DSAChecker.isErrorFound())
      return StmtError();
    // Generate list of implicitly defined firstprivate variables.
    VarsWithInheritedDSA = DSAChecker.getVarsWithInheritedDSA();

    if (!DSAChecker.getImplicitFirstprivate().empty()) {
      if (OMPClause *Implicit = ActOnOpenMPFirstprivateClause(
              DSAChecker.getImplicitFirstprivate(), SourceLocation(),
              SourceLocation(), SourceLocation())) {
        ClausesWithImplicit.push_back(Implicit);
        ErrorFound = cast<OMPFirstprivateClause>(Implicit)->varlist_size() !=
                     DSAChecker.getImplicitFirstprivate().size();
      } else
        ErrorFound = true;
    }
  }

  llvm::SmallVector<OpenMPDirectiveKind, 4> AllowedNameModifiers;
  switch (Kind) {
  case OMPD_parallel:
    Res = ActOnOpenMPParallelDirective(ClausesWithImplicit, AStmt, StartLoc,
                                       EndLoc);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_simd:
    Res = ActOnOpenMPSimdDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc,
                                   VarsWithInheritedDSA);
    break;
  case OMPD_for:
    Res = ActOnOpenMPForDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc,
                                  VarsWithInheritedDSA);
    break;
  case OMPD_for_simd:
    Res = ActOnOpenMPForSimdDirective(ClausesWithImplicit, AStmt, StartLoc,
                                      EndLoc, VarsWithInheritedDSA);
    break;
  case OMPD_sections:
    Res = ActOnOpenMPSectionsDirective(ClausesWithImplicit, AStmt, StartLoc,
                                       EndLoc);
    break;
  case OMPD_section:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp section' directive");
    Res = ActOnOpenMPSectionDirective(AStmt, StartLoc, EndLoc);
    break;
  case OMPD_single:
    Res = ActOnOpenMPSingleDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    break;
  case OMPD_master:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp master' directive");
    Res = ActOnOpenMPMasterDirective(AStmt, StartLoc, EndLoc);
    break;
  case OMPD_critical:
    Res = ActOnOpenMPCriticalDirective(DirName, ClausesWithImplicit, AStmt,
                                       StartLoc, EndLoc);
    break;
  case OMPD_parallel_for:
    Res = ActOnOpenMPParallelForDirective(ClausesWithImplicit, AStmt, StartLoc,
                                          EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_parallel_for_simd:
    Res = ActOnOpenMPParallelForSimdDirective(
        ClausesWithImplicit, AStmt, StartLoc, EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_parallel_sections:
    Res = ActOnOpenMPParallelSectionsDirective(ClausesWithImplicit, AStmt,
                                               StartLoc, EndLoc);
    AllowedNameModifiers.push_back(OMPD_parallel);
    break;
  case OMPD_task:
    Res =
        ActOnOpenMPTaskDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    AllowedNameModifiers.push_back(OMPD_task);
    break;
  case OMPD_taskyield:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp taskyield' directive");
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp taskyield' directive");
    Res = ActOnOpenMPTaskyieldDirective(StartLoc, EndLoc);
    break;
  case OMPD_barrier:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp barrier' directive");
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp barrier' directive");
    Res = ActOnOpenMPBarrierDirective(StartLoc, EndLoc);
    break;
  case OMPD_taskwait:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp taskwait' directive");
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp taskwait' directive");
    Res = ActOnOpenMPTaskwaitDirective(StartLoc, EndLoc);
    break;
  case OMPD_taskgroup:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp taskgroup' directive");
    Res = ActOnOpenMPTaskgroupDirective(AStmt, StartLoc, EndLoc);
    break;
  case OMPD_flush:
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp flush' directive");
    Res = ActOnOpenMPFlushDirective(ClausesWithImplicit, StartLoc, EndLoc);
    break;
  case OMPD_ordered:
    Res = ActOnOpenMPOrderedDirective(ClausesWithImplicit, AStmt, StartLoc,
                                      EndLoc);
    break;
  case OMPD_atomic:
    Res = ActOnOpenMPAtomicDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    break;
  case OMPD_teams:
    Res =
        ActOnOpenMPTeamsDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_target:
    Res = ActOnOpenMPTargetDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    AllowedNameModifiers.push_back(OMPD_target);
    break;
  case OMPD_cancellation_point:
    assert(ClausesWithImplicit.empty() &&
           "No clauses are allowed for 'omp cancellation point' directive");
    assert(AStmt == nullptr && "No associated statement allowed for 'omp "
                               "cancellation point' directive");
    Res = ActOnOpenMPCancellationPointDirective(StartLoc, EndLoc, CancelRegion);
    break;
  case OMPD_cancel:
    assert(AStmt == nullptr &&
           "No associated statement allowed for 'omp cancel' directive");
    Res = ActOnOpenMPCancelDirective(ClausesWithImplicit, StartLoc, EndLoc,
                                     CancelRegion);
    AllowedNameModifiers.push_back(OMPD_cancel);
    break;
  case OMPD_target_data:
    Res = ActOnOpenMPTargetDataDirective(ClausesWithImplicit, AStmt, StartLoc,
                                         EndLoc);
    AllowedNameModifiers.push_back(OMPD_target_data);
    break;
  case OMPD_taskloop:
    Res = ActOnOpenMPTaskLoopDirective(ClausesWithImplicit, AStmt, StartLoc,
                                       EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_taskloop);
    break;
  case OMPD_taskloop_simd:
    Res = ActOnOpenMPTaskLoopSimdDirective(ClausesWithImplicit, AStmt, StartLoc,
                                           EndLoc, VarsWithInheritedDSA);
    AllowedNameModifiers.push_back(OMPD_taskloop);
    break;
  case OMPD_distribute:
    Res = ActOnOpenMPDistributeDirective(ClausesWithImplicit, AStmt, StartLoc,
                                         EndLoc, VarsWithInheritedDSA);
    break;
  case OMPD_threadprivate:
    llvm_unreachable("OpenMP Directive is not allowed");
  case OMPD_unknown:
    llvm_unreachable("Unknown OpenMP directive");
  }

  for (auto P : VarsWithInheritedDSA) {
    Diag(P.second->getExprLoc(), diag::err_omp_no_dsa_for_variable)
        << P.first << P.second->getSourceRange();
  }
  ErrorFound = !VarsWithInheritedDSA.empty() || ErrorFound;

  if (!AllowedNameModifiers.empty())
    ErrorFound = checkIfClauses(*this, Kind, Clauses, AllowedNameModifiers) ||
                 ErrorFound;

  if (ErrorFound)
    return StmtError();
  return Res;
}

StmtResult Sema::ActOnOpenMPParallelDirective(ArrayRef<OMPClause *> Clauses,
                                              Stmt *AStmt,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  CapturedStmt *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  getCurFunction()->setHasBranchProtectedScope();

  return OMPParallelDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                      DSAStack->isCancelRegion());
}

namespace {
/// \brief Helper class for checking canonical form of the OpenMP loops and
/// extracting iteration space of each loop in the loop nest, that will be used
/// for IR generation.
class OpenMPIterationSpaceChecker {
  /// \brief Reference to Sema.
  Sema &SemaRef;
  /// \brief A location for diagnostics (when there is no some better location).
  SourceLocation DefaultLoc;
  /// \brief A location for diagnostics (when increment is not compatible).
  SourceLocation ConditionLoc;
  /// \brief A source location for referring to loop init later.
  SourceRange InitSrcRange;
  /// \brief A source location for referring to condition later.
  SourceRange ConditionSrcRange;
  /// \brief A source location for referring to increment later.
  SourceRange IncrementSrcRange;
  /// \brief Loop variable.
  VarDecl *Var;
  /// \brief Reference to loop variable.
  DeclRefExpr *VarRef;
  /// \brief Lower bound (initializer for the var).
  Expr *LB;
  /// \brief Upper bound.
  Expr *UB;
  /// \brief Loop step (increment).
  Expr *Step;
  /// \brief This flag is true when condition is one of:
  ///   Var <  UB
  ///   Var <= UB
  ///   UB  >  Var
  ///   UB  >= Var
  bool TestIsLessOp;
  /// \brief This flag is true when condition is strict ( < or > ).
  bool TestIsStrictOp;
  /// \brief This flag is true when step is subtracted on each iteration.
  bool SubtractStep;

public:
  OpenMPIterationSpaceChecker(Sema &SemaRef, SourceLocation DefaultLoc)
      : SemaRef(SemaRef), DefaultLoc(DefaultLoc), ConditionLoc(DefaultLoc),
        InitSrcRange(SourceRange()), ConditionSrcRange(SourceRange()),
        IncrementSrcRange(SourceRange()), Var(nullptr), VarRef(nullptr),
        LB(nullptr), UB(nullptr), Step(nullptr), TestIsLessOp(false),
        TestIsStrictOp(false), SubtractStep(false) {}
  /// \brief Check init-expr for canonical loop form and save loop counter
  /// variable - #Var and its initialization value - #LB.
  bool CheckInit(Stmt *S, bool EmitDiags = true);
  /// \brief Check test-expr for canonical form, save upper-bound (#UB), flags
  /// for less/greater and for strict/non-strict comparison.
  bool CheckCond(Expr *S);
  /// \brief Check incr-expr for canonical loop form and return true if it
  /// does not conform, otherwise save loop step (#Step).
  bool CheckInc(Expr *S);
  /// \brief Return the loop counter variable.
  VarDecl *GetLoopVar() const { return Var; }
  /// \brief Return the reference expression to loop counter variable.
  DeclRefExpr *GetLoopVarRefExpr() const { return VarRef; }
  /// \brief Source range of the loop init.
  SourceRange GetInitSrcRange() const { return InitSrcRange; }
  /// \brief Source range of the loop condition.
  SourceRange GetConditionSrcRange() const { return ConditionSrcRange; }
  /// \brief Source range of the loop increment.
  SourceRange GetIncrementSrcRange() const { return IncrementSrcRange; }
  /// \brief True if the step should be subtracted.
  bool ShouldSubtractStep() const { return SubtractStep; }
  /// \brief Build the expression to calculate the number of iterations.
  Expr *BuildNumIterations(Scope *S, const bool LimitedType) const;
  /// \brief Build the precondition expression for the loops.
  Expr *BuildPreCond(Scope *S, Expr *Cond) const;
  /// \brief Build reference expression to the counter be used for codegen.
  Expr *BuildCounterVar() const;
  /// \brief Build reference expression to the private counter be used for
  /// codegen.
  Expr *BuildPrivateCounterVar() const;
  /// \brief Build initization of the counter be used for codegen.
  Expr *BuildCounterInit() const;
  /// \brief Build step of the counter be used for codegen.
  Expr *BuildCounterStep() const;
  /// \brief Return true if any expression is dependent.
  bool Dependent() const;

private:
  /// \brief Check the right-hand side of an assignment in the increment
  /// expression.
  bool CheckIncRHS(Expr *RHS);
  /// \brief Helper to set loop counter variable and its initializer.
  bool SetVarAndLB(VarDecl *NewVar, DeclRefExpr *NewVarRefExpr, Expr *NewLB);
  /// \brief Helper to set upper bound.
  bool SetUB(Expr *NewUB, bool LessOp, bool StrictOp, SourceRange SR,
             SourceLocation SL);
  /// \brief Helper to set loop increment.
  bool SetStep(Expr *NewStep, bool Subtract);
};

bool OpenMPIterationSpaceChecker::Dependent() const {
  if (!Var) {
    assert(!LB && !UB && !Step);
    return false;
  }
  return Var->getType()->isDependentType() || (LB && LB->isValueDependent()) ||
         (UB && UB->isValueDependent()) || (Step && Step->isValueDependent());
}

template <typename T>
static T *getExprAsWritten(T *E) {
  if (auto *ExprTemp = dyn_cast<ExprWithCleanups>(E))
    E = ExprTemp->getSubExpr();

  if (auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
    E = MTE->GetTemporaryExpr();

  while (auto *Binder = dyn_cast<CXXBindTemporaryExpr>(E))
    E = Binder->getSubExpr();

  if (auto *ICE = dyn_cast<ImplicitCastExpr>(E))
    E = ICE->getSubExprAsWritten();
  return E->IgnoreParens();
}

bool OpenMPIterationSpaceChecker::SetVarAndLB(VarDecl *NewVar,
                                              DeclRefExpr *NewVarRefExpr,
                                              Expr *NewLB) {
  // State consistency checking to ensure correct usage.
  assert(Var == nullptr && LB == nullptr && VarRef == nullptr &&
         UB == nullptr && Step == nullptr && !TestIsLessOp && !TestIsStrictOp);
  if (!NewVar || !NewLB)
    return true;
  Var = NewVar;
  VarRef = NewVarRefExpr;
  if (auto *CE = dyn_cast_or_null<CXXConstructExpr>(NewLB))
    if (const CXXConstructorDecl *Ctor = CE->getConstructor())
      if ((Ctor->isCopyOrMoveConstructor() ||
           Ctor->isConvertingConstructor(/*AllowExplicit=*/false)) &&
          CE->getNumArgs() > 0 && CE->getArg(0) != nullptr)
        NewLB = CE->getArg(0)->IgnoreParenImpCasts();
  LB = NewLB;
  return false;
}

bool OpenMPIterationSpaceChecker::SetUB(Expr *NewUB, bool LessOp, bool StrictOp,
                                        SourceRange SR, SourceLocation SL) {
  // State consistency checking to ensure correct usage.
  assert(Var != nullptr && LB != nullptr && UB == nullptr && Step == nullptr &&
         !TestIsLessOp && !TestIsStrictOp);
  if (!NewUB)
    return true;
  UB = NewUB;
  TestIsLessOp = LessOp;
  TestIsStrictOp = StrictOp;
  ConditionSrcRange = SR;
  ConditionLoc = SL;
  return false;
}

bool OpenMPIterationSpaceChecker::SetStep(Expr *NewStep, bool Subtract) {
  // State consistency checking to ensure correct usage.
  assert(Var != nullptr && LB != nullptr && Step == nullptr);
  if (!NewStep)
    return true;
  if (!NewStep->isValueDependent()) {
    // Check that the step is integer expression.
    SourceLocation StepLoc = NewStep->getLocStart();
    ExprResult Val =
        SemaRef.PerformOpenMPImplicitIntegerConversion(StepLoc, NewStep);
    if (Val.isInvalid())
      return true;
    NewStep = Val.get();

    // OpenMP [2.6, Canonical Loop Form, Restrictions]
    //  If test-expr is of form var relational-op b and relational-op is < or
    //  <= then incr-expr must cause var to increase on each iteration of the
    //  loop. If test-expr is of form var relational-op b and relational-op is
    //  > or >= then incr-expr must cause var to decrease on each iteration of
    //  the loop.
    //  If test-expr is of form b relational-op var and relational-op is < or
    //  <= then incr-expr must cause var to decrease on each iteration of the
    //  loop. If test-expr is of form b relational-op var and relational-op is
    //  > or >= then incr-expr must cause var to increase on each iteration of
    //  the loop.
    llvm::APSInt Result;
    bool IsConstant = NewStep->isIntegerConstantExpr(Result, SemaRef.Context);
    bool IsUnsigned = !NewStep->getType()->hasSignedIntegerRepresentation();
    bool IsConstNeg =
        IsConstant && Result.isSigned() && (Subtract != Result.isNegative());
    bool IsConstPos =
        IsConstant && Result.isSigned() && (Subtract == Result.isNegative());
    bool IsConstZero = IsConstant && !Result.getBoolValue();
    if (UB && (IsConstZero ||
               (TestIsLessOp ? (IsConstNeg || (IsUnsigned && Subtract))
                             : (IsConstPos || (IsUnsigned && !Subtract))))) {
      SemaRef.Diag(NewStep->getExprLoc(),
                   diag::err_omp_loop_incr_not_compatible)
          << Var << TestIsLessOp << NewStep->getSourceRange();
      SemaRef.Diag(ConditionLoc,
                   diag::note_omp_loop_cond_requres_compatible_incr)
          << TestIsLessOp << ConditionSrcRange;
      return true;
    }
    if (TestIsLessOp == Subtract) {
      NewStep = SemaRef.CreateBuiltinUnaryOp(NewStep->getExprLoc(), UO_Minus,
                                             NewStep).get();
      Subtract = !Subtract;
    }
  }

  Step = NewStep;
  SubtractStep = Subtract;
  return false;
}

bool OpenMPIterationSpaceChecker::CheckInit(Stmt *S, bool EmitDiags) {
  // Check init-expr for canonical loop form and save loop counter
  // variable - #Var and its initialization value - #LB.
  // OpenMP [2.6] Canonical loop form. init-expr may be one of the following:
  //   var = lb
  //   integer-type var = lb
  //   random-access-iterator-type var = lb
  //   pointer-type var = lb
  //
  if (!S) {
    if (EmitDiags) {
      SemaRef.Diag(DefaultLoc, diag::err_omp_loop_not_canonical_init);
    }
    return true;
  }
  InitSrcRange = S->getSourceRange();
  if (Expr *E = dyn_cast<Expr>(S))
    S = E->IgnoreParens();
  if (auto BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->getOpcode() == BO_Assign)
      if (auto DRE = dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParens()))
        return SetVarAndLB(dyn_cast<VarDecl>(DRE->getDecl()), DRE,
                           BO->getRHS());
  } else if (auto DS = dyn_cast<DeclStmt>(S)) {
    if (DS->isSingleDecl()) {
      if (auto Var = dyn_cast_or_null<VarDecl>(DS->getSingleDecl())) {
        if (Var->hasInit() && !Var->getType()->isReferenceType()) {
          // Accept non-canonical init form here but emit ext. warning.
          if (Var->getInitStyle() != VarDecl::CInit && EmitDiags)
            SemaRef.Diag(S->getLocStart(),
                         diag::ext_omp_loop_not_canonical_init)
                << S->getSourceRange();
          return SetVarAndLB(Var, nullptr, Var->getInit());
        }
      }
    }
  } else if (auto CE = dyn_cast<CXXOperatorCallExpr>(S))
    if (CE->getOperator() == OO_Equal)
      if (auto DRE = dyn_cast<DeclRefExpr>(CE->getArg(0)))
        return SetVarAndLB(dyn_cast<VarDecl>(DRE->getDecl()), DRE,
                           CE->getArg(1));

  if (EmitDiags) {
    SemaRef.Diag(S->getLocStart(), diag::err_omp_loop_not_canonical_init)
        << S->getSourceRange();
  }
  return true;
}

/// \brief Ignore parenthesizes, implicit casts, copy constructor and return the
/// variable (which may be the loop variable) if possible.
static const VarDecl *GetInitVarDecl(const Expr *E) {
  if (!E)
    return nullptr;
  E = getExprAsWritten(E);
  if (auto *CE = dyn_cast_or_null<CXXConstructExpr>(E))
    if (const CXXConstructorDecl *Ctor = CE->getConstructor())
      if ((Ctor->isCopyOrMoveConstructor() ||
           Ctor->isConvertingConstructor(/*AllowExplicit=*/false)) &&
          CE->getNumArgs() > 0 && CE->getArg(0) != nullptr)
        E = CE->getArg(0)->IgnoreParenImpCasts();
  auto DRE = dyn_cast_or_null<DeclRefExpr>(E);
  if (!DRE)
    return nullptr;
  return dyn_cast<VarDecl>(DRE->getDecl());
}

bool OpenMPIterationSpaceChecker::CheckCond(Expr *S) {
  // Check test-expr for canonical form, save upper-bound UB, flags for
  // less/greater and for strict/non-strict comparison.
  // OpenMP [2.6] Canonical loop form. Test-expr may be one of the following:
  //   var relational-op b
  //   b relational-op var
  //
  if (!S) {
    SemaRef.Diag(DefaultLoc, diag::err_omp_loop_not_canonical_cond) << Var;
    return true;
  }
  S = getExprAsWritten(S);
  SourceLocation CondLoc = S->getLocStart();
  if (auto BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->isRelationalOp()) {
      if (GetInitVarDecl(BO->getLHS()) == Var)
        return SetUB(BO->getRHS(),
                     (BO->getOpcode() == BO_LT || BO->getOpcode() == BO_LE),
                     (BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT),
                     BO->getSourceRange(), BO->getOperatorLoc());
      if (GetInitVarDecl(BO->getRHS()) == Var)
        return SetUB(BO->getLHS(),
                     (BO->getOpcode() == BO_GT || BO->getOpcode() == BO_GE),
                     (BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT),
                     BO->getSourceRange(), BO->getOperatorLoc());
    }
  } else if (auto CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (CE->getNumArgs() == 2) {
      auto Op = CE->getOperator();
      switch (Op) {
      case OO_Greater:
      case OO_GreaterEqual:
      case OO_Less:
      case OO_LessEqual:
        if (GetInitVarDecl(CE->getArg(0)) == Var)
          return SetUB(CE->getArg(1), Op == OO_Less || Op == OO_LessEqual,
                       Op == OO_Less || Op == OO_Greater, CE->getSourceRange(),
                       CE->getOperatorLoc());
        if (GetInitVarDecl(CE->getArg(1)) == Var)
          return SetUB(CE->getArg(0), Op == OO_Greater || Op == OO_GreaterEqual,
                       Op == OO_Less || Op == OO_Greater, CE->getSourceRange(),
                       CE->getOperatorLoc());
        break;
      default:
        break;
      }
    }
  }
  SemaRef.Diag(CondLoc, diag::err_omp_loop_not_canonical_cond)
      << S->getSourceRange() << Var;
  return true;
}

bool OpenMPIterationSpaceChecker::CheckIncRHS(Expr *RHS) {
  // RHS of canonical loop form increment can be:
  //   var + incr
  //   incr + var
  //   var - incr
  //
  RHS = RHS->IgnoreParenImpCasts();
  if (auto BO = dyn_cast<BinaryOperator>(RHS)) {
    if (BO->isAdditiveOp()) {
      bool IsAdd = BO->getOpcode() == BO_Add;
      if (GetInitVarDecl(BO->getLHS()) == Var)
        return SetStep(BO->getRHS(), !IsAdd);
      if (IsAdd && GetInitVarDecl(BO->getRHS()) == Var)
        return SetStep(BO->getLHS(), false);
    }
  } else if (auto CE = dyn_cast<CXXOperatorCallExpr>(RHS)) {
    bool IsAdd = CE->getOperator() == OO_Plus;
    if ((IsAdd || CE->getOperator() == OO_Minus) && CE->getNumArgs() == 2) {
      if (GetInitVarDecl(CE->getArg(0)) == Var)
        return SetStep(CE->getArg(1), !IsAdd);
      if (IsAdd && GetInitVarDecl(CE->getArg(1)) == Var)
        return SetStep(CE->getArg(0), false);
    }
  }
  SemaRef.Diag(RHS->getLocStart(), diag::err_omp_loop_not_canonical_incr)
      << RHS->getSourceRange() << Var;
  return true;
}

bool OpenMPIterationSpaceChecker::CheckInc(Expr *S) {
  // Check incr-expr for canonical loop form and return true if it
  // does not conform.
  // OpenMP [2.6] Canonical loop form. Test-expr may be one of the following:
  //   ++var
  //   var++
  //   --var
  //   var--
  //   var += incr
  //   var -= incr
  //   var = var + incr
  //   var = incr + var
  //   var = var - incr
  //
  if (!S) {
    SemaRef.Diag(DefaultLoc, diag::err_omp_loop_not_canonical_incr) << Var;
    return true;
  }
  IncrementSrcRange = S->getSourceRange();
  S = S->IgnoreParens();
  if (auto UO = dyn_cast<UnaryOperator>(S)) {
    if (UO->isIncrementDecrementOp() && GetInitVarDecl(UO->getSubExpr()) == Var)
      return SetStep(
          SemaRef.ActOnIntegerConstant(UO->getLocStart(),
                                       (UO->isDecrementOp() ? -1 : 1)).get(),
          false);
  } else if (auto BO = dyn_cast<BinaryOperator>(S)) {
    switch (BO->getOpcode()) {
    case BO_AddAssign:
    case BO_SubAssign:
      if (GetInitVarDecl(BO->getLHS()) == Var)
        return SetStep(BO->getRHS(), BO->getOpcode() == BO_SubAssign);
      break;
    case BO_Assign:
      if (GetInitVarDecl(BO->getLHS()) == Var)
        return CheckIncRHS(BO->getRHS());
      break;
    default:
      break;
    }
  } else if (auto CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    switch (CE->getOperator()) {
    case OO_PlusPlus:
    case OO_MinusMinus:
      if (GetInitVarDecl(CE->getArg(0)) == Var)
        return SetStep(
            SemaRef.ActOnIntegerConstant(
                        CE->getLocStart(),
                        ((CE->getOperator() == OO_MinusMinus) ? -1 : 1)).get(),
            false);
      break;
    case OO_PlusEqual:
    case OO_MinusEqual:
      if (GetInitVarDecl(CE->getArg(0)) == Var)
        return SetStep(CE->getArg(1), CE->getOperator() == OO_MinusEqual);
      break;
    case OO_Equal:
      if (GetInitVarDecl(CE->getArg(0)) == Var)
        return CheckIncRHS(CE->getArg(1));
      break;
    default:
      break;
    }
  }
  SemaRef.Diag(S->getLocStart(), diag::err_omp_loop_not_canonical_incr)
      << S->getSourceRange() << Var;
  return true;
}

namespace {
// Transform variables declared in GNU statement expressions to new ones to
// avoid crash on codegen.
class TransformToNewDefs : public TreeTransform<TransformToNewDefs> {
  typedef TreeTransform<TransformToNewDefs> BaseTransform;

public:
  TransformToNewDefs(Sema &SemaRef) : BaseTransform(SemaRef) {}

  Decl *TransformDefinition(SourceLocation Loc, Decl *D) {
    if (auto *VD = cast<VarDecl>(D))
      if (!isa<ParmVarDecl>(D) && !isa<VarTemplateSpecializationDecl>(D) &&
          !isa<ImplicitParamDecl>(D)) {
        auto *NewVD = VarDecl::Create(
            SemaRef.Context, VD->getDeclContext(), VD->getLocStart(),
            VD->getLocation(), VD->getIdentifier(), VD->getType(),
            VD->getTypeSourceInfo(), VD->getStorageClass());
        NewVD->setTSCSpec(VD->getTSCSpec());
        NewVD->setInit(VD->getInit());
        NewVD->setInitStyle(VD->getInitStyle());
        NewVD->setExceptionVariable(VD->isExceptionVariable());
        NewVD->setNRVOVariable(VD->isNRVOVariable());
        NewVD->setCXXForRangeDecl(VD->isInExternCXXContext());
        NewVD->setConstexpr(VD->isConstexpr());
        NewVD->setInitCapture(VD->isInitCapture());
        NewVD->setPreviousDeclInSameBlockScope(
            VD->isPreviousDeclInSameBlockScope());
        VD->getDeclContext()->addHiddenDecl(NewVD);
        if (VD->hasAttrs())
          NewVD->setAttrs(VD->getAttrs());
        transformedLocalDecl(VD, NewVD);
        return NewVD;
      }
    return BaseTransform::TransformDefinition(Loc, D);
  }

  ExprResult TransformDeclRefExpr(DeclRefExpr *E) {
    if (auto *NewD = TransformDecl(E->getExprLoc(), E->getDecl()))
      if (E->getDecl() != NewD) {
        NewD->setReferenced();
        NewD->markUsed(SemaRef.Context);
        return DeclRefExpr::Create(
            SemaRef.Context, E->getQualifierLoc(), E->getTemplateKeywordLoc(),
            cast<ValueDecl>(NewD), E->refersToEnclosingVariableOrCapture(),
            E->getNameInfo(), E->getType(), E->getValueKind());
      }
    return BaseTransform::TransformDeclRefExpr(E);
  }
};
}

/// \brief Build the expression to calculate the number of iterations.
Expr *
OpenMPIterationSpaceChecker::BuildNumIterations(Scope *S,
                                                const bool LimitedType) const {
  TransformToNewDefs Transform(SemaRef);
  ExprResult Diff;
  auto VarType = Var->getType().getNonReferenceType();
  if (VarType->isIntegerType() || VarType->isPointerType() ||
      SemaRef.getLangOpts().CPlusPlus) {
    // Upper - Lower
    auto *UBExpr = TestIsLessOp ? UB : LB;
    auto *LBExpr = TestIsLessOp ? LB : UB;
    Expr *Upper = Transform.TransformExpr(UBExpr).get();
    Expr *Lower = Transform.TransformExpr(LBExpr).get();
    if (!Upper || !Lower)
      return nullptr;
    Upper = SemaRef.PerformImplicitConversion(Upper, UBExpr->getType(),
                                                    Sema::AA_Converting,
                                                    /*AllowExplicit=*/true)
                      .get();
    Lower = SemaRef.PerformImplicitConversion(Lower, LBExpr->getType(),
                                              Sema::AA_Converting,
                                              /*AllowExplicit=*/true)
                .get();
    if (!Upper || !Lower)
      return nullptr;

    Diff = SemaRef.BuildBinOp(S, DefaultLoc, BO_Sub, Upper, Lower);

    if (!Diff.isUsable() && VarType->getAsCXXRecordDecl()) {
      // BuildBinOp already emitted error, this one is to point user to upper
      // and lower bound, and to tell what is passed to 'operator-'.
      SemaRef.Diag(Upper->getLocStart(), diag::err_omp_loop_diff_cxx)
          << Upper->getSourceRange() << Lower->getSourceRange();
      return nullptr;
    }
  }

  if (!Diff.isUsable())
    return nullptr;

  // Upper - Lower [- 1]
  if (TestIsStrictOp)
    Diff = SemaRef.BuildBinOp(
        S, DefaultLoc, BO_Sub, Diff.get(),
        SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
  if (!Diff.isUsable())
    return nullptr;

  // Upper - Lower [- 1] + Step
  auto NewStep = Transform.TransformExpr(Step->IgnoreImplicit());
  if (NewStep.isInvalid())
    return nullptr;
  NewStep = SemaRef.PerformImplicitConversion(
      NewStep.get(), Step->IgnoreImplicit()->getType(), Sema::AA_Converting,
      /*AllowExplicit=*/true);
  if (NewStep.isInvalid())
    return nullptr;
  Diff = SemaRef.BuildBinOp(S, DefaultLoc, BO_Add, Diff.get(), NewStep.get());
  if (!Diff.isUsable())
    return nullptr;

  // Parentheses (for dumping/debugging purposes only).
  Diff = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Diff.get());
  if (!Diff.isUsable())
    return nullptr;

  // (Upper - Lower [- 1] + Step) / Step
  NewStep = Transform.TransformExpr(Step->IgnoreImplicit());
  if (NewStep.isInvalid())
    return nullptr;
  NewStep = SemaRef.PerformImplicitConversion(
      NewStep.get(), Step->IgnoreImplicit()->getType(), Sema::AA_Converting,
      /*AllowExplicit=*/true);
  if (NewStep.isInvalid())
    return nullptr;
  Diff = SemaRef.BuildBinOp(S, DefaultLoc, BO_Div, Diff.get(), NewStep.get());
  if (!Diff.isUsable())
    return nullptr;

  // OpenMP runtime requires 32-bit or 64-bit loop variables.
  QualType Type = Diff.get()->getType();
  auto &C = SemaRef.Context;
  bool UseVarType = VarType->hasIntegerRepresentation() &&
                    C.getTypeSize(Type) > C.getTypeSize(VarType);
  if (!Type->isIntegerType() || UseVarType) {
    unsigned NewSize =
        UseVarType ? C.getTypeSize(VarType) : C.getTypeSize(Type);
    bool IsSigned = UseVarType ? VarType->hasSignedIntegerRepresentation()
                               : Type->hasSignedIntegerRepresentation();
    Type = C.getIntTypeForBitwidth(NewSize, IsSigned);
    Diff = SemaRef.PerformImplicitConversion(
        Diff.get(), Type, Sema::AA_Converting, /*AllowExplicit=*/true);
    if (!Diff.isUsable())
      return nullptr;
  }
  if (LimitedType) {
    unsigned NewSize = (C.getTypeSize(Type) > 32) ? 64 : 32;
    if (NewSize != C.getTypeSize(Type)) {
      if (NewSize < C.getTypeSize(Type)) {
        assert(NewSize == 64 && "incorrect loop var size");
        SemaRef.Diag(DefaultLoc, diag::warn_omp_loop_64_bit_var)
            << InitSrcRange << ConditionSrcRange;
      }
      QualType NewType = C.getIntTypeForBitwidth(
          NewSize, Type->hasSignedIntegerRepresentation() ||
                       C.getTypeSize(Type) < NewSize);
      Diff = SemaRef.PerformImplicitConversion(Diff.get(), NewType,
                                               Sema::AA_Converting, true);
      if (!Diff.isUsable())
        return nullptr;
    }
  }

  return Diff.get();
}

Expr *OpenMPIterationSpaceChecker::BuildPreCond(Scope *S, Expr *Cond) const {
  // Try to build LB <op> UB, where <op> is <, >, <=, or >=.
  bool Suppress = SemaRef.getDiagnostics().getSuppressAllDiagnostics();
  SemaRef.getDiagnostics().setSuppressAllDiagnostics(/*Val=*/true);
  TransformToNewDefs Transform(SemaRef);

  auto NewLB = Transform.TransformExpr(LB);
  auto NewUB = Transform.TransformExpr(UB);
  if (NewLB.isInvalid() || NewUB.isInvalid())
    return Cond;
  NewLB = SemaRef.PerformImplicitConversion(NewLB.get(), LB->getType(),
                                            Sema::AA_Converting,
                                            /*AllowExplicit=*/true);
  NewUB = SemaRef.PerformImplicitConversion(NewUB.get(), UB->getType(),
                                            Sema::AA_Converting,
                                            /*AllowExplicit=*/true);
  if (NewLB.isInvalid() || NewUB.isInvalid())
    return Cond;
  auto CondExpr = SemaRef.BuildBinOp(
      S, DefaultLoc, TestIsLessOp ? (TestIsStrictOp ? BO_LT : BO_LE)
                                  : (TestIsStrictOp ? BO_GT : BO_GE),
      NewLB.get(), NewUB.get());
  if (CondExpr.isUsable()) {
    CondExpr = SemaRef.PerformImplicitConversion(
        CondExpr.get(), SemaRef.Context.BoolTy, /*Action=*/Sema::AA_Casting,
        /*AllowExplicit=*/true);
  }
  SemaRef.getDiagnostics().setSuppressAllDiagnostics(Suppress);
  // Otherwise use original loop conditon and evaluate it in runtime.
  return CondExpr.isUsable() ? CondExpr.get() : Cond;
}

/// \brief Build reference expression to the counter be used for codegen.
Expr *OpenMPIterationSpaceChecker::BuildCounterVar() const {
  return buildDeclRefExpr(SemaRef, Var, Var->getType().getNonReferenceType(),
                          DefaultLoc);
}

Expr *OpenMPIterationSpaceChecker::BuildPrivateCounterVar() const {
  if (Var && !Var->isInvalidDecl()) {
    auto Type = Var->getType().getNonReferenceType();
    auto *PrivateVar =
        buildVarDecl(SemaRef, DefaultLoc, Type, Var->getName(),
                     Var->hasAttrs() ? &Var->getAttrs() : nullptr);
    if (PrivateVar->isInvalidDecl())
      return nullptr;
    return buildDeclRefExpr(SemaRef, PrivateVar, Type, DefaultLoc);
  }
  return nullptr;
}

/// \brief Build initization of the counter be used for codegen.
Expr *OpenMPIterationSpaceChecker::BuildCounterInit() const { return LB; }

/// \brief Build step of the counter be used for codegen.
Expr *OpenMPIterationSpaceChecker::BuildCounterStep() const { return Step; }

/// \brief Iteration space of a single for loop.
struct LoopIterationSpace {
  /// \brief Condition of the loop.
  Expr *PreCond;
  /// \brief This expression calculates the number of iterations in the loop.
  /// It is always possible to calculate it before starting the loop.
  Expr *NumIterations;
  /// \brief The loop counter variable.
  Expr *CounterVar;
  /// \brief Private loop counter variable.
  Expr *PrivateCounterVar;
  /// \brief This is initializer for the initial value of #CounterVar.
  Expr *CounterInit;
  /// \brief This is step for the #CounterVar used to generate its update:
  /// #CounterVar = #CounterInit + #CounterStep * CurrentIteration.
  Expr *CounterStep;
  /// \brief Should step be subtracted?
  bool Subtract;
  /// \brief Source range of the loop init.
  SourceRange InitSrcRange;
  /// \brief Source range of the loop condition.
  SourceRange CondSrcRange;
  /// \brief Source range of the loop increment.
  SourceRange IncSrcRange;
};

} // namespace

void Sema::ActOnOpenMPLoopInitialization(SourceLocation ForLoc, Stmt *Init) {
  assert(getLangOpts().OpenMP && "OpenMP is not active.");
  assert(Init && "Expected loop in canonical form.");
  unsigned AssociatedLoops = DSAStack->getAssociatedLoops();
  if (AssociatedLoops > 0 &&
      isOpenMPLoopDirective(DSAStack->getCurrentDirective())) {
    OpenMPIterationSpaceChecker ISC(*this, ForLoc);
    if (!ISC.CheckInit(Init, /*EmitDiags=*/false))
      DSAStack->addLoopControlVariable(ISC.GetLoopVar());
    DSAStack->setAssociatedLoops(AssociatedLoops - 1);
  }
}

/// \brief Called on a for stmt to check and extract its iteration space
/// for further processing (such as collapsing).
static bool CheckOpenMPIterationSpace(
    OpenMPDirectiveKind DKind, Stmt *S, Sema &SemaRef, DSAStackTy &DSA,
    unsigned CurrentNestedLoopCount, unsigned NestedLoopCount,
    Expr *CollapseLoopCountExpr, Expr *OrderedLoopCountExpr,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA,
    LoopIterationSpace &ResultIterSpace) {
  // OpenMP [2.6, Canonical Loop Form]
  //   for (init-expr; test-expr; incr-expr) structured-block
  auto For = dyn_cast_or_null<ForStmt>(S);
  if (!For) {
    SemaRef.Diag(S->getLocStart(), diag::err_omp_not_for)
        << (CollapseLoopCountExpr != nullptr || OrderedLoopCountExpr != nullptr)
        << getOpenMPDirectiveName(DKind) << NestedLoopCount
        << (CurrentNestedLoopCount > 0) << CurrentNestedLoopCount;
    if (NestedLoopCount > 1) {
      if (CollapseLoopCountExpr && OrderedLoopCountExpr)
        SemaRef.Diag(DSA.getConstructLoc(),
                     diag::note_omp_collapse_ordered_expr)
            << 2 << CollapseLoopCountExpr->getSourceRange()
            << OrderedLoopCountExpr->getSourceRange();
      else if (CollapseLoopCountExpr)
        SemaRef.Diag(CollapseLoopCountExpr->getExprLoc(),
                     diag::note_omp_collapse_ordered_expr)
            << 0 << CollapseLoopCountExpr->getSourceRange();
      else
        SemaRef.Diag(OrderedLoopCountExpr->getExprLoc(),
                     diag::note_omp_collapse_ordered_expr)
            << 1 << OrderedLoopCountExpr->getSourceRange();
    }
    return true;
  }
  assert(For->getBody());

  OpenMPIterationSpaceChecker ISC(SemaRef, For->getForLoc());

  // Check init.
  auto Init = For->getInit();
  if (ISC.CheckInit(Init)) {
    return true;
  }

  bool HasErrors = false;

  // Check loop variable's type.
  auto Var = ISC.GetLoopVar();

  // OpenMP [2.6, Canonical Loop Form]
  // Var is one of the following:
  //   A variable of signed or unsigned integer type.
  //   For C++, a variable of a random access iterator type.
  //   For C, a variable of a pointer type.
  auto VarType = Var->getType().getNonReferenceType();
  if (!VarType->isDependentType() && !VarType->isIntegerType() &&
      !VarType->isPointerType() &&
      !(SemaRef.getLangOpts().CPlusPlus && VarType->isOverloadableType())) {
    SemaRef.Diag(Init->getLocStart(), diag::err_omp_loop_variable_type)
        << SemaRef.getLangOpts().CPlusPlus;
    HasErrors = true;
  }

  // OpenMP, 2.14.1.1 Data-sharing Attribute Rules for Variables Referenced in a
  // Construct
  // The loop iteration variable(s) in the associated for-loop(s) of a for or
  // parallel for construct is (are) private.
  // The loop iteration variable in the associated for-loop of a simd construct
  // with just one associated for-loop is linear with a constant-linear-step
  // that is the increment of the associated for-loop.
  // Exclude loop var from the list of variables with implicitly defined data
  // sharing attributes.
  VarsWithImplicitDSA.erase(Var);

  // OpenMP [2.14.1.1, Data-sharing Attribute Rules for Variables Referenced in
  // a Construct, C/C++].
  // The loop iteration variable in the associated for-loop of a simd construct
  // with just one associated for-loop may be listed in a linear clause with a
  // constant-linear-step that is the increment of the associated for-loop.
  // The loop iteration variable(s) in the associated for-loop(s) of a for or
  // parallel for construct may be listed in a private or lastprivate clause.
  DSAStackTy::DSAVarData DVar = DSA.getTopDSA(Var, false);
  auto LoopVarRefExpr = ISC.GetLoopVarRefExpr();
  // If LoopVarRefExpr is nullptr it means the corresponding loop variable is
  // declared in the loop and it is predetermined as a private.
  auto PredeterminedCKind =
      isOpenMPSimdDirective(DKind)
          ? ((NestedLoopCount == 1) ? OMPC_linear : OMPC_lastprivate)
          : OMPC_private;
  if (((isOpenMPSimdDirective(DKind) && DVar.CKind != OMPC_unknown &&
        DVar.CKind != PredeterminedCKind) ||
       ((isOpenMPWorksharingDirective(DKind) || DKind == OMPD_taskloop ||
         isOpenMPDistributeDirective(DKind)) &&
        !isOpenMPSimdDirective(DKind) && DVar.CKind != OMPC_unknown &&
        DVar.CKind != OMPC_private && DVar.CKind != OMPC_lastprivate)) &&
      (DVar.CKind != OMPC_private || DVar.RefExpr != nullptr)) {
    SemaRef.Diag(Init->getLocStart(), diag::err_omp_loop_var_dsa)
        << getOpenMPClauseName(DVar.CKind) << getOpenMPDirectiveName(DKind)
        << getOpenMPClauseName(PredeterminedCKind);
    if (DVar.RefExpr == nullptr)
      DVar.CKind = PredeterminedCKind;
    ReportOriginalDSA(SemaRef, &DSA, Var, DVar, /*IsLoopIterVar=*/true);
    HasErrors = true;
  } else if (LoopVarRefExpr != nullptr) {
    // Make the loop iteration variable private (for worksharing constructs),
    // linear (for simd directives with the only one associated loop) or
    // lastprivate (for simd directives with several collapsed or ordered
    // loops).
    if (DVar.CKind == OMPC_unknown)
      DVar = DSA.hasDSA(Var, isOpenMPPrivate, MatchesAlways(),
                        /*FromParent=*/false);
    DSA.addDSA(Var, LoopVarRefExpr, PredeterminedCKind);
  }

  assert(isOpenMPLoopDirective(DKind) && "DSA for non-loop vars");

  // Check test-expr.
  HasErrors |= ISC.CheckCond(For->getCond());

  // Check incr-expr.
  HasErrors |= ISC.CheckInc(For->getInc());

  if (ISC.Dependent() || SemaRef.CurContext->isDependentContext() || HasErrors)
    return HasErrors;

  // Build the loop's iteration space representation.
  ResultIterSpace.PreCond = ISC.BuildPreCond(DSA.getCurScope(), For->getCond());
  ResultIterSpace.NumIterations = ISC.BuildNumIterations(
      DSA.getCurScope(), (isOpenMPWorksharingDirective(DKind) ||
                          isOpenMPTaskLoopDirective(DKind) ||
                          isOpenMPDistributeDirective(DKind)));
  ResultIterSpace.CounterVar = ISC.BuildCounterVar();
  ResultIterSpace.PrivateCounterVar = ISC.BuildPrivateCounterVar();
  ResultIterSpace.CounterInit = ISC.BuildCounterInit();
  ResultIterSpace.CounterStep = ISC.BuildCounterStep();
  ResultIterSpace.InitSrcRange = ISC.GetInitSrcRange();
  ResultIterSpace.CondSrcRange = ISC.GetConditionSrcRange();
  ResultIterSpace.IncSrcRange = ISC.GetIncrementSrcRange();
  ResultIterSpace.Subtract = ISC.ShouldSubtractStep();

  HasErrors |= (ResultIterSpace.PreCond == nullptr ||
                ResultIterSpace.NumIterations == nullptr ||
                ResultIterSpace.CounterVar == nullptr ||
                ResultIterSpace.PrivateCounterVar == nullptr ||
                ResultIterSpace.CounterInit == nullptr ||
                ResultIterSpace.CounterStep == nullptr);

  return HasErrors;
}

/// \brief Build 'VarRef = Start.
static ExprResult BuildCounterInit(Sema &SemaRef, Scope *S, SourceLocation Loc,
                                   ExprResult VarRef, ExprResult Start) {
  TransformToNewDefs Transform(SemaRef);
  // Build 'VarRef = Start.
  auto NewStart = Transform.TransformExpr(Start.get()->IgnoreImplicit());
  if (NewStart.isInvalid())
    return ExprError();
  NewStart = SemaRef.PerformImplicitConversion(
      NewStart.get(), Start.get()->IgnoreImplicit()->getType(),
      Sema::AA_Converting,
      /*AllowExplicit=*/true);
  if (NewStart.isInvalid())
    return ExprError();
  NewStart = SemaRef.PerformImplicitConversion(
      NewStart.get(), VarRef.get()->getType(), Sema::AA_Converting,
      /*AllowExplicit=*/true);
  if (!NewStart.isUsable())
    return ExprError();

  auto Init =
      SemaRef.BuildBinOp(S, Loc, BO_Assign, VarRef.get(), NewStart.get());
  return Init;
}

/// \brief Build 'VarRef = Start + Iter * Step'.
static ExprResult BuildCounterUpdate(Sema &SemaRef, Scope *S,
                                     SourceLocation Loc, ExprResult VarRef,
                                     ExprResult Start, ExprResult Iter,
                                     ExprResult Step, bool Subtract) {
  // Add parentheses (for debugging purposes only).
  Iter = SemaRef.ActOnParenExpr(Loc, Loc, Iter.get());
  if (!VarRef.isUsable() || !Start.isUsable() || !Iter.isUsable() ||
      !Step.isUsable())
    return ExprError();

  TransformToNewDefs Transform(SemaRef);
  auto NewStep = Transform.TransformExpr(Step.get()->IgnoreImplicit());
  if (NewStep.isInvalid())
    return ExprError();
  NewStep = SemaRef.PerformImplicitConversion(
      NewStep.get(), Step.get()->IgnoreImplicit()->getType(),
      Sema::AA_Converting,
      /*AllowExplicit=*/true);
  if (NewStep.isInvalid())
    return ExprError();
  ExprResult Update =
      SemaRef.BuildBinOp(S, Loc, BO_Mul, Iter.get(), NewStep.get());
  if (!Update.isUsable())
    return ExprError();

  // Build 'VarRef = Start + Iter * Step'.
  auto NewStart = Transform.TransformExpr(Start.get()->IgnoreImplicit());
  if (NewStart.isInvalid())
    return ExprError();
  NewStart = SemaRef.PerformImplicitConversion(
      NewStart.get(), Start.get()->IgnoreImplicit()->getType(),
      Sema::AA_Converting,
      /*AllowExplicit=*/true);
  if (NewStart.isInvalid())
    return ExprError();
  Update = SemaRef.BuildBinOp(S, Loc, (Subtract ? BO_Sub : BO_Add),
                              NewStart.get(), Update.get());
  if (!Update.isUsable())
    return ExprError();

  Update = SemaRef.PerformImplicitConversion(
      Update.get(), VarRef.get()->getType(), Sema::AA_Converting, true);
  if (!Update.isUsable())
    return ExprError();

  Update = SemaRef.BuildBinOp(S, Loc, BO_Assign, VarRef.get(), Update.get());
  return Update;
}

/// \brief Convert integer expression \a E to make it have at least \a Bits
/// bits.
static ExprResult WidenIterationCount(unsigned Bits, Expr *E,
                                      Sema &SemaRef) {
  if (E == nullptr)
    return ExprError();
  auto &C = SemaRef.Context;
  QualType OldType = E->getType();
  unsigned HasBits = C.getTypeSize(OldType);
  if (HasBits >= Bits)
    return ExprResult(E);
  // OK to convert to signed, because new type has more bits than old.
  QualType NewType = C.getIntTypeForBitwidth(Bits, /* Signed */ true);
  return SemaRef.PerformImplicitConversion(E, NewType, Sema::AA_Converting,
                                           true);
}

/// \brief Check if the given expression \a E is a constant integer that fits
/// into \a Bits bits.
static bool FitsInto(unsigned Bits, bool Signed, Expr *E, Sema &SemaRef) {
  if (E == nullptr)
    return false;
  llvm::APSInt Result;
  if (E->isIntegerConstantExpr(Result, SemaRef.Context))
    return Signed ? Result.isSignedIntN(Bits) : Result.isIntN(Bits);
  return false;
}

/// \brief Called on a for stmt to check itself and nested loops (if any).
/// \return Returns 0 if one of the collapsed stmts is not canonical for loop,
/// number of collapsed loops otherwise.
static unsigned
CheckOpenMPLoop(OpenMPDirectiveKind DKind, Expr *CollapseLoopCountExpr,
                Expr *OrderedLoopCountExpr, Stmt *AStmt, Sema &SemaRef,
                DSAStackTy &DSA,
                llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA,
                OMPLoopDirective::HelperExprs &Built) {
  unsigned NestedLoopCount = 1;
  if (CollapseLoopCountExpr) {
    // Found 'collapse' clause - calculate collapse number.
    llvm::APSInt Result;
    if (CollapseLoopCountExpr->EvaluateAsInt(Result, SemaRef.getASTContext()))
      NestedLoopCount = Result.getLimitedValue();
  }
  if (OrderedLoopCountExpr) {
    // Found 'ordered' clause - calculate collapse number.
    llvm::APSInt Result;
    if (OrderedLoopCountExpr->EvaluateAsInt(Result, SemaRef.getASTContext())) {
      if (Result.getLimitedValue() < NestedLoopCount) {
        SemaRef.Diag(OrderedLoopCountExpr->getExprLoc(),
                     diag::err_omp_wrong_ordered_loop_count)
            << OrderedLoopCountExpr->getSourceRange();
        SemaRef.Diag(CollapseLoopCountExpr->getExprLoc(),
                     diag::note_collapse_loop_count)
            << CollapseLoopCountExpr->getSourceRange();
      }
      NestedLoopCount = Result.getLimitedValue();
    }
  }
  // This is helper routine for loop directives (e.g., 'for', 'simd',
  // 'for simd', etc.).
  SmallVector<LoopIterationSpace, 4> IterSpaces;
  IterSpaces.resize(NestedLoopCount);
  Stmt *CurStmt = AStmt->IgnoreContainers(/* IgnoreCaptured */ true);
  for (unsigned Cnt = 0; Cnt < NestedLoopCount; ++Cnt) {
    if (CheckOpenMPIterationSpace(DKind, CurStmt, SemaRef, DSA, Cnt,
                                  NestedLoopCount, CollapseLoopCountExpr,
                                  OrderedLoopCountExpr, VarsWithImplicitDSA,
                                  IterSpaces[Cnt]))
      return 0;
    // Move on to the next nested for loop, or to the loop body.
    // OpenMP [2.8.1, simd construct, Restrictions]
    // All loops associated with the construct must be perfectly nested; that
    // is, there must be no intervening code nor any OpenMP directive between
    // any two loops.
    CurStmt = cast<ForStmt>(CurStmt)->getBody()->IgnoreContainers();
  }

  Built.clear(/* size */ NestedLoopCount);

  if (SemaRef.CurContext->isDependentContext())
    return NestedLoopCount;

  // An example of what is generated for the following code:
  //
  //   #pragma omp simd collapse(2) ordered(2)
  //   for (i = 0; i < NI; ++i)
  //     for (k = 0; k < NK; ++k)
  //       for (j = J0; j < NJ; j+=2) {
  //         <loop body>
  //       }
  //
  // We generate the code below.
  // Note: the loop body may be outlined in CodeGen.
  // Note: some counters may be C++ classes, operator- is used to find number of
  // iterations and operator+= to calculate counter value.
  // Note: decltype(NumIterations) must be integer type (in 'omp for', only i32
  // or i64 is currently supported).
  //
  //   #define NumIterations (NI * ((NJ - J0 - 1 + 2) / 2))
  //   for (int[32|64]_t IV = 0; IV < NumIterations; ++IV ) {
  //     .local.i = IV / ((NJ - J0 - 1 + 2) / 2);
  //     .local.j = J0 + (IV % ((NJ - J0 - 1 + 2) / 2)) * 2;
  //     // similar updates for vars in clauses (e.g. 'linear')
  //     <loop body (using local i and j)>
  //   }
  //   i = NI; // assign final values of counters
  //   j = NJ;
  //

  // Last iteration number is (I1 * I2 * ... In) - 1, where I1, I2 ... In are
  // the iteration counts of the collapsed for loops.
  // Precondition tests if there is at least one iteration (all conditions are
  // true).
  auto PreCond = ExprResult(IterSpaces[0].PreCond);
  auto N0 = IterSpaces[0].NumIterations;
  ExprResult LastIteration32 = WidenIterationCount(
      32 /* Bits */, SemaRef.PerformImplicitConversion(
                                N0->IgnoreImpCasts(), N0->getType(),
                                Sema::AA_Converting, /*AllowExplicit=*/true)
                         .get(),
      SemaRef);
  ExprResult LastIteration64 = WidenIterationCount(
      64 /* Bits */, SemaRef.PerformImplicitConversion(
                                N0->IgnoreImpCasts(), N0->getType(),
                                Sema::AA_Converting, /*AllowExplicit=*/true)
                         .get(),
      SemaRef);

  if (!LastIteration32.isUsable() || !LastIteration64.isUsable())
    return NestedLoopCount;

  auto &C = SemaRef.Context;
  bool AllCountsNeedLessThan32Bits = C.getTypeSize(N0->getType()) < 32;

  Scope *CurScope = DSA.getCurScope();
  for (unsigned Cnt = 1; Cnt < NestedLoopCount; ++Cnt) {
    if (PreCond.isUsable()) {
      PreCond = SemaRef.BuildBinOp(CurScope, SourceLocation(), BO_LAnd,
                                   PreCond.get(), IterSpaces[Cnt].PreCond);
    }
    auto N = IterSpaces[Cnt].NumIterations;
    AllCountsNeedLessThan32Bits &= C.getTypeSize(N->getType()) < 32;
    if (LastIteration32.isUsable())
      LastIteration32 = SemaRef.BuildBinOp(
          CurScope, SourceLocation(), BO_Mul, LastIteration32.get(),
          SemaRef.PerformImplicitConversion(N->IgnoreImpCasts(), N->getType(),
                                            Sema::AA_Converting,
                                            /*AllowExplicit=*/true)
              .get());
    if (LastIteration64.isUsable())
      LastIteration64 = SemaRef.BuildBinOp(
          CurScope, SourceLocation(), BO_Mul, LastIteration64.get(),
          SemaRef.PerformImplicitConversion(N->IgnoreImpCasts(), N->getType(),
                                            Sema::AA_Converting,
                                            /*AllowExplicit=*/true)
              .get());
  }

  // Choose either the 32-bit or 64-bit version.
  ExprResult LastIteration = LastIteration64;
  if (LastIteration32.isUsable() &&
      C.getTypeSize(LastIteration32.get()->getType()) == 32 &&
      (AllCountsNeedLessThan32Bits || NestedLoopCount == 1 ||
       FitsInto(
           32 /* Bits */,
           LastIteration32.get()->getType()->hasSignedIntegerRepresentation(),
           LastIteration64.get(), SemaRef)))
    LastIteration = LastIteration32;

  if (!LastIteration.isUsable())
    return 0;

  // Save the number of iterations.
  ExprResult NumIterations = LastIteration;
  {
    LastIteration = SemaRef.BuildBinOp(
        CurScope, SourceLocation(), BO_Sub, LastIteration.get(),
        SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
    if (!LastIteration.isUsable())
      return 0;
  }

  // Calculate the last iteration number beforehand instead of doing this on
  // each iteration. Do not do this if the number of iterations may be kfold-ed.
  llvm::APSInt Result;
  bool IsConstant =
      LastIteration.get()->isIntegerConstantExpr(Result, SemaRef.Context);
  ExprResult CalcLastIteration;
  if (!IsConstant) {
    SourceLocation SaveLoc;
    VarDecl *SaveVar =
        buildVarDecl(SemaRef, SaveLoc, LastIteration.get()->getType(),
                     ".omp.last.iteration");
    ExprResult SaveRef = buildDeclRefExpr(
        SemaRef, SaveVar, LastIteration.get()->getType(), SaveLoc);
    CalcLastIteration = SemaRef.BuildBinOp(CurScope, SaveLoc, BO_Assign,
                                           SaveRef.get(), LastIteration.get());
    LastIteration = SaveRef;

    // Prepare SaveRef + 1.
    NumIterations = SemaRef.BuildBinOp(
        CurScope, SaveLoc, BO_Add, SaveRef.get(),
        SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
    if (!NumIterations.isUsable())
      return 0;
  }

  SourceLocation InitLoc = IterSpaces[0].InitSrcRange.getBegin();

  QualType VType = LastIteration.get()->getType();
  // Build variables passed into runtime, nesessary for worksharing directives.
  ExprResult LB, UB, IL, ST, EUB;
  if (isOpenMPWorksharingDirective(DKind) || isOpenMPTaskLoopDirective(DKind) ||
      isOpenMPDistributeDirective(DKind)) {
    // Lower bound variable, initialized with zero.
    VarDecl *LBDecl = buildVarDecl(SemaRef, InitLoc, VType, ".omp.lb");
    LB = buildDeclRefExpr(SemaRef, LBDecl, VType, InitLoc);
    SemaRef.AddInitializerToDecl(
        LBDecl, SemaRef.ActOnIntegerConstant(InitLoc, 0).get(),
        /*DirectInit*/ false, /*TypeMayContainAuto*/ false);

    // Upper bound variable, initialized with last iteration number.
    VarDecl *UBDecl = buildVarDecl(SemaRef, InitLoc, VType, ".omp.ub");
    UB = buildDeclRefExpr(SemaRef, UBDecl, VType, InitLoc);
    SemaRef.AddInitializerToDecl(UBDecl, LastIteration.get(),
                                 /*DirectInit*/ false,
                                 /*TypeMayContainAuto*/ false);

    // A 32-bit variable-flag where runtime returns 1 for the last iteration.
    // This will be used to implement clause 'lastprivate'.
    QualType Int32Ty = SemaRef.Context.getIntTypeForBitwidth(32, true);
    VarDecl *ILDecl = buildVarDecl(SemaRef, InitLoc, Int32Ty, ".omp.is_last");
    IL = buildDeclRefExpr(SemaRef, ILDecl, Int32Ty, InitLoc);
    SemaRef.AddInitializerToDecl(
        ILDecl, SemaRef.ActOnIntegerConstant(InitLoc, 0).get(),
        /*DirectInit*/ false, /*TypeMayContainAuto*/ false);

    // Stride variable returned by runtime (we initialize it to 1 by default).
    VarDecl *STDecl = buildVarDecl(SemaRef, InitLoc, VType, ".omp.stride");
    ST = buildDeclRefExpr(SemaRef, STDecl, VType, InitLoc);
    SemaRef.AddInitializerToDecl(
        STDecl, SemaRef.ActOnIntegerConstant(InitLoc, 1).get(),
        /*DirectInit*/ false, /*TypeMayContainAuto*/ false);

    // Build expression: UB = min(UB, LastIteration)
    // It is nesessary for CodeGen of directives with static scheduling.
    ExprResult IsUBGreater = SemaRef.BuildBinOp(CurScope, InitLoc, BO_GT,
                                                UB.get(), LastIteration.get());
    ExprResult CondOp = SemaRef.ActOnConditionalOp(
        InitLoc, InitLoc, IsUBGreater.get(), LastIteration.get(), UB.get());
    EUB = SemaRef.BuildBinOp(CurScope, InitLoc, BO_Assign, UB.get(),
                             CondOp.get());
    EUB = SemaRef.ActOnFinishFullExpr(EUB.get());
  }

  // Build the iteration variable and its initialization before loop.
  ExprResult IV;
  ExprResult Init;
  {
    VarDecl *IVDecl = buildVarDecl(SemaRef, InitLoc, VType, ".omp.iv");
    IV = buildDeclRefExpr(SemaRef, IVDecl, VType, InitLoc);
    Expr *RHS = (isOpenMPWorksharingDirective(DKind) ||
                 isOpenMPTaskLoopDirective(DKind) ||
                 isOpenMPDistributeDirective(DKind))
                    ? LB.get()
                    : SemaRef.ActOnIntegerConstant(SourceLocation(), 0).get();
    Init = SemaRef.BuildBinOp(CurScope, InitLoc, BO_Assign, IV.get(), RHS);
    Init = SemaRef.ActOnFinishFullExpr(Init.get());
  }

  // Loop condition (IV < NumIterations) or (IV <= UB) for worksharing loops.
  SourceLocation CondLoc;
  ExprResult Cond =
      (isOpenMPWorksharingDirective(DKind) ||
       isOpenMPTaskLoopDirective(DKind) || isOpenMPDistributeDirective(DKind))
          ? SemaRef.BuildBinOp(CurScope, CondLoc, BO_LE, IV.get(), UB.get())
          : SemaRef.BuildBinOp(CurScope, CondLoc, BO_LT, IV.get(),
                               NumIterations.get());

  // Loop increment (IV = IV + 1)
  SourceLocation IncLoc;
  ExprResult Inc =
      SemaRef.BuildBinOp(CurScope, IncLoc, BO_Add, IV.get(),
                         SemaRef.ActOnIntegerConstant(IncLoc, 1).get());
  if (!Inc.isUsable())
    return 0;
  Inc = SemaRef.BuildBinOp(CurScope, IncLoc, BO_Assign, IV.get(), Inc.get());
  Inc = SemaRef.ActOnFinishFullExpr(Inc.get());
  if (!Inc.isUsable())
    return 0;

  // Increments for worksharing loops (LB = LB + ST; UB = UB + ST).
  // Used for directives with static scheduling.
  ExprResult NextLB, NextUB;
  if (isOpenMPWorksharingDirective(DKind) || isOpenMPTaskLoopDirective(DKind) ||
      isOpenMPDistributeDirective(DKind)) {
    // LB + ST
    NextLB = SemaRef.BuildBinOp(CurScope, IncLoc, BO_Add, LB.get(), ST.get());
    if (!NextLB.isUsable())
      return 0;
    // LB = LB + ST
    NextLB =
        SemaRef.BuildBinOp(CurScope, IncLoc, BO_Assign, LB.get(), NextLB.get());
    NextLB = SemaRef.ActOnFinishFullExpr(NextLB.get());
    if (!NextLB.isUsable())
      return 0;
    // UB + ST
    NextUB = SemaRef.BuildBinOp(CurScope, IncLoc, BO_Add, UB.get(), ST.get());
    if (!NextUB.isUsable())
      return 0;
    // UB = UB + ST
    NextUB =
        SemaRef.BuildBinOp(CurScope, IncLoc, BO_Assign, UB.get(), NextUB.get());
    NextUB = SemaRef.ActOnFinishFullExpr(NextUB.get());
    if (!NextUB.isUsable())
      return 0;
  }

  // Build updates and final values of the loop counters.
  bool HasErrors = false;
  Built.Counters.resize(NestedLoopCount);
  Built.Inits.resize(NestedLoopCount);
  Built.Updates.resize(NestedLoopCount);
  Built.Finals.resize(NestedLoopCount);
  {
    ExprResult Div;
    // Go from inner nested loop to outer.
    for (int Cnt = NestedLoopCount - 1; Cnt >= 0; --Cnt) {
      LoopIterationSpace &IS = IterSpaces[Cnt];
      SourceLocation UpdLoc = IS.IncSrcRange.getBegin();
      // Build: Iter = (IV / Div) % IS.NumIters
      // where Div is product of previous iterations' IS.NumIters.
      ExprResult Iter;
      if (Div.isUsable()) {
        Iter =
            SemaRef.BuildBinOp(CurScope, UpdLoc, BO_Div, IV.get(), Div.get());
      } else {
        Iter = IV;
        assert((Cnt == (int)NestedLoopCount - 1) &&
               "unusable div expected on first iteration only");
      }

      if (Cnt != 0 && Iter.isUsable())
        Iter = SemaRef.BuildBinOp(CurScope, UpdLoc, BO_Rem, Iter.get(),
                                  IS.NumIterations);
      if (!Iter.isUsable()) {
        HasErrors = true;
        break;
      }

      // Build update: IS.CounterVar(Private) = IS.Start + Iter * IS.Step
      auto *CounterVar = buildDeclRefExpr(
          SemaRef, cast<VarDecl>(cast<DeclRefExpr>(IS.CounterVar)->getDecl()),
          IS.CounterVar->getType(), IS.CounterVar->getExprLoc(),
          /*RefersToCapture=*/true);
      ExprResult Init = BuildCounterInit(SemaRef, CurScope, UpdLoc, CounterVar,
                                         IS.CounterInit);
      if (!Init.isUsable()) {
        HasErrors = true;
        break;
      }
      ExprResult Update =
          BuildCounterUpdate(SemaRef, CurScope, UpdLoc, CounterVar,
                             IS.CounterInit, Iter, IS.CounterStep, IS.Subtract);
      if (!Update.isUsable()) {
        HasErrors = true;
        break;
      }

      // Build final: IS.CounterVar = IS.Start + IS.NumIters * IS.Step
      ExprResult Final = BuildCounterUpdate(
          SemaRef, CurScope, UpdLoc, CounterVar, IS.CounterInit,
          IS.NumIterations, IS.CounterStep, IS.Subtract);
      if (!Final.isUsable()) {
        HasErrors = true;
        break;
      }

      // Build Div for the next iteration: Div <- Div * IS.NumIters
      if (Cnt != 0) {
        if (Div.isUnset())
          Div = IS.NumIterations;
        else
          Div = SemaRef.BuildBinOp(CurScope, UpdLoc, BO_Mul, Div.get(),
                                   IS.NumIterations);

        // Add parentheses (for debugging purposes only).
        if (Div.isUsable())
          Div = SemaRef.ActOnParenExpr(UpdLoc, UpdLoc, Div.get());
        if (!Div.isUsable()) {
          HasErrors = true;
          break;
        }
      }
      if (!Update.isUsable() || !Final.isUsable()) {
        HasErrors = true;
        break;
      }
      // Save results
      Built.Counters[Cnt] = IS.CounterVar;
      Built.PrivateCounters[Cnt] = IS.PrivateCounterVar;
      Built.Inits[Cnt] = Init.get();
      Built.Updates[Cnt] = Update.get();
      Built.Finals[Cnt] = Final.get();
    }
  }

  if (HasErrors)
    return 0;

  // Save results
  Built.IterationVarRef = IV.get();
  Built.LastIteration = LastIteration.get();
  Built.NumIterations = NumIterations.get();
  Built.CalcLastIteration =
      SemaRef.ActOnFinishFullExpr(CalcLastIteration.get()).get();
  Built.PreCond = PreCond.get();
  Built.Cond = Cond.get();
  Built.Init = Init.get();
  Built.Inc = Inc.get();
  Built.LB = LB.get();
  Built.UB = UB.get();
  Built.IL = IL.get();
  Built.ST = ST.get();
  Built.EUB = EUB.get();
  Built.NLB = NextLB.get();
  Built.NUB = NextUB.get();

  return NestedLoopCount;
}

static Expr *getCollapseNumberExpr(ArrayRef<OMPClause *> Clauses) {
  auto CollapseClauses =
      OMPExecutableDirective::getClausesOfKind<OMPCollapseClause>(Clauses);
  if (CollapseClauses.begin() != CollapseClauses.end())
    return (*CollapseClauses.begin())->getNumForLoops();
  return nullptr;
}

static Expr *getOrderedNumberExpr(ArrayRef<OMPClause *> Clauses) {
  auto OrderedClauses =
      OMPExecutableDirective::getClausesOfKind<OMPOrderedClause>(Clauses);
  if (OrderedClauses.begin() != OrderedClauses.end())
    return (*OrderedClauses.begin())->getNumForLoops();
  return nullptr;
}

static bool checkSimdlenSafelenValues(Sema &S, const Expr *Simdlen,
                                      const Expr *Safelen) {
  llvm::APSInt SimdlenRes, SafelenRes;
  if (Simdlen->isValueDependent() || Simdlen->isTypeDependent() ||
      Simdlen->isInstantiationDependent() ||
      Simdlen->containsUnexpandedParameterPack())
    return false;
  if (Safelen->isValueDependent() || Safelen->isTypeDependent() ||
      Safelen->isInstantiationDependent() ||
      Safelen->containsUnexpandedParameterPack())
    return false;
  Simdlen->EvaluateAsInt(SimdlenRes, S.Context);
  Safelen->EvaluateAsInt(SafelenRes, S.Context);
  // OpenMP 4.1 [2.8.1, simd Construct, Restrictions]
  // If both simdlen and safelen clauses are specified, the value of the simdlen
  // parameter must be less than or equal to the value of the safelen parameter.
  if (SimdlenRes > SafelenRes) {
    S.Diag(Simdlen->getExprLoc(), diag::err_omp_wrong_simdlen_safelen_values)
        << Simdlen->getSourceRange() << Safelen->getSourceRange();
    return true;
  }
  return false;
}

StmtResult Sema::ActOnOpenMPSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = CheckOpenMPLoop(
      OMPD_simd, getCollapseNumberExpr(Clauses), getOrderedNumberExpr(Clauses),
      AStmt, *this, *DSAStack, VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp simd loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (auto C : Clauses) {
      if (auto LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope))
          return StmtError();
    }
  }

  // OpenMP 4.1 [2.8.1, simd Construct, Restrictions]
  // If both simdlen and safelen clauses are specified, the value of the simdlen
  // parameter must be less than or equal to the value of the safelen parameter.
  OMPSafelenClause *Safelen = nullptr;
  OMPSimdlenClause *Simdlen = nullptr;
  for (auto *Clause : Clauses) {
    if (Clause->getClauseKind() == OMPC_safelen)
      Safelen = cast<OMPSafelenClause>(Clause);
    else if (Clause->getClauseKind() == OMPC_simdlen)
      Simdlen = cast<OMPSimdlenClause>(Clause);
    if (Safelen && Simdlen)
      break;
  }
  if (Simdlen && Safelen &&
      checkSimdlenSafelenValues(*this, Simdlen->getSimdlen(),
                                Safelen->getSafelen()))
    return StmtError();

  getCurFunction()->setHasBranchProtectedScope();
  return OMPSimdDirective::Create(Context, StartLoc, EndLoc, NestedLoopCount,
                                  Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPForDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount = CheckOpenMPLoop(
      OMPD_for, getCollapseNumberExpr(Clauses), getOrderedNumberExpr(Clauses),
      AStmt, *this, *DSAStack, VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (auto C : Clauses) {
      if (auto LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope))
          return StmtError();
    }
  }

  getCurFunction()->setHasBranchProtectedScope();
  return OMPForDirective::Create(Context, StartLoc, EndLoc, NestedLoopCount,
                                 Clauses, AStmt, B, DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPForSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      CheckOpenMPLoop(OMPD_for_simd, getCollapseNumberExpr(Clauses),
                      getOrderedNumberExpr(Clauses), AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for simd loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (auto C : Clauses) {
      if (auto LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope))
          return StmtError();
    }
  }

  // OpenMP 4.1 [2.8.1, simd Construct, Restrictions]
  // If both simdlen and safelen clauses are specified, the value of the simdlen
  // parameter must be less than or equal to the value of the safelen parameter.
  OMPSafelenClause *Safelen = nullptr;
  OMPSimdlenClause *Simdlen = nullptr;
  for (auto *Clause : Clauses) {
    if (Clause->getClauseKind() == OMPC_safelen)
      Safelen = cast<OMPSafelenClause>(Clause);
    else if (Clause->getClauseKind() == OMPC_simdlen)
      Simdlen = cast<OMPSimdlenClause>(Clause);
    if (Safelen && Simdlen)
      break;
  }
  if (Simdlen && Safelen &&
      checkSimdlenSafelenValues(*this, Simdlen->getSimdlen(),
                                Safelen->getSafelen()))
    return StmtError();

  getCurFunction()->setHasBranchProtectedScope();
  return OMPForSimdDirective::Create(Context, StartLoc, EndLoc, NestedLoopCount,
                                     Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPSectionsDirective(ArrayRef<OMPClause *> Clauses,
                                              Stmt *AStmt,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  auto BaseStmt = AStmt;
  while (CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(BaseStmt))
    BaseStmt = CS->getCapturedStmt();
  if (auto C = dyn_cast_or_null<CompoundStmt>(BaseStmt)) {
    auto S = C->children();
    if (S.begin() == S.end())
      return StmtError();
    // All associated statements must be '#pragma omp section' except for
    // the first one.
    for (Stmt *SectionStmt : llvm::make_range(std::next(S.begin()), S.end())) {
      if (!SectionStmt || !isa<OMPSectionDirective>(SectionStmt)) {
        if (SectionStmt)
          Diag(SectionStmt->getLocStart(),
               diag::err_omp_sections_substmt_not_section);
        return StmtError();
      }
      cast<OMPSectionDirective>(SectionStmt)
          ->setHasCancel(DSAStack->isCancelRegion());
    }
  } else {
    Diag(AStmt->getLocStart(), diag::err_omp_sections_not_compound_stmt);
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();

  return OMPSectionsDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                      DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPSectionDirective(Stmt *AStmt,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

  getCurFunction()->setHasBranchProtectedScope();
  DSAStack->setParentCancelRegion(DSAStack->isCancelRegion());

  return OMPSectionDirective::Create(Context, StartLoc, EndLoc, AStmt,
                                     DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPSingleDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

  getCurFunction()->setHasBranchProtectedScope();

  // OpenMP [2.7.3, single Construct, Restrictions]
  // The copyprivate clause must not be used with the nowait clause.
  OMPClause *Nowait = nullptr;
  OMPClause *Copyprivate = nullptr;
  for (auto *Clause : Clauses) {
    if (Clause->getClauseKind() == OMPC_nowait)
      Nowait = Clause;
    else if (Clause->getClauseKind() == OMPC_copyprivate)
      Copyprivate = Clause;
    if (Copyprivate && Nowait) {
      Diag(Copyprivate->getLocStart(),
           diag::err_omp_single_copyprivate_with_nowait);
      Diag(Nowait->getLocStart(), diag::note_omp_nowait_clause_here);
      return StmtError();
    }
  }

  return OMPSingleDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPMasterDirective(Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

  getCurFunction()->setHasBranchProtectedScope();

  return OMPMasterDirective::Create(Context, StartLoc, EndLoc, AStmt);
}

StmtResult Sema::ActOnOpenMPCriticalDirective(
    const DeclarationNameInfo &DirName, ArrayRef<OMPClause *> Clauses,
    Stmt *AStmt, SourceLocation StartLoc, SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

  bool ErrorFound = false;
  llvm::APSInt Hint;
  SourceLocation HintLoc;
  bool DependentHint = false;
  for (auto *C : Clauses) {
    if (C->getClauseKind() == OMPC_hint) {
      if (!DirName.getName()) {
        Diag(C->getLocStart(), diag::err_omp_hint_clause_no_name);
        ErrorFound = true;
      }
      Expr *E = cast<OMPHintClause>(C)->getHint();
      if (E->isTypeDependent() || E->isValueDependent() ||
          E->isInstantiationDependent())
        DependentHint = true;
      else {
        Hint = E->EvaluateKnownConstInt(Context);
        HintLoc = C->getLocStart();
      }
    }
  }
  if (ErrorFound)
    return StmtError();
  auto Pair = DSAStack->getCriticalWithHint(DirName);
  if (Pair.first && DirName.getName() && !DependentHint) {
    if (llvm::APSInt::compareValues(Hint, Pair.second) != 0) {
      Diag(StartLoc, diag::err_omp_critical_with_hint);
      if (HintLoc.isValid()) {
        Diag(HintLoc, diag::note_omp_critical_hint_here)
            << 0 << Hint.toString(/*Radix=*/10, /*Signed=*/false);
      } else
        Diag(StartLoc, diag::note_omp_critical_no_hint) << 0;
      if (auto *C = Pair.first->getSingleClause<OMPHintClause>()) {
        Diag(C->getLocStart(), diag::note_omp_critical_hint_here)
            << 1
            << C->getHint()->EvaluateKnownConstInt(Context).toString(
                   /*Radix=*/10, /*Signed=*/false);
      } else
        Diag(Pair.first->getLocStart(), diag::note_omp_critical_no_hint) << 1;
    }
  }

  getCurFunction()->setHasBranchProtectedScope();

  auto *Dir = OMPCriticalDirective::Create(Context, DirName, StartLoc, EndLoc,
                                           Clauses, AStmt);
  if (!Pair.first && DirName.getName() && !DependentHint)
    DSAStack->addCriticalWithHint(Dir, Hint);
  return Dir;
}

StmtResult Sema::ActOnOpenMPParallelForDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  CapturedStmt *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  OMPLoopDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      CheckOpenMPLoop(OMPD_parallel_for, getCollapseNumberExpr(Clauses),
                      getOrderedNumberExpr(Clauses), AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp parallel for loop exprs were not built");

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (auto C : Clauses) {
      if (auto LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope))
          return StmtError();
    }
  }

  getCurFunction()->setHasBranchProtectedScope();
  return OMPParallelForDirective::Create(Context, StartLoc, EndLoc,
                                         NestedLoopCount, Clauses, AStmt, B,
                                         DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPParallelForSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  CapturedStmt *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  OMPLoopDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      CheckOpenMPLoop(OMPD_parallel_for_simd, getCollapseNumberExpr(Clauses),
                      getOrderedNumberExpr(Clauses), AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  if (!CurContext->isDependentContext()) {
    // Finalize the clauses that need pre-built expressions for CodeGen.
    for (auto C : Clauses) {
      if (auto LC = dyn_cast<OMPLinearClause>(C))
        if (FinishOpenMPLinearClause(*LC, cast<DeclRefExpr>(B.IterationVarRef),
                                     B.NumIterations, *this, CurScope))
          return StmtError();
    }
  }

  // OpenMP 4.1 [2.8.1, simd Construct, Restrictions]
  // If both simdlen and safelen clauses are specified, the value of the simdlen
  // parameter must be less than or equal to the value of the safelen parameter.
  OMPSafelenClause *Safelen = nullptr;
  OMPSimdlenClause *Simdlen = nullptr;
  for (auto *Clause : Clauses) {
    if (Clause->getClauseKind() == OMPC_safelen)
      Safelen = cast<OMPSafelenClause>(Clause);
    else if (Clause->getClauseKind() == OMPC_simdlen)
      Simdlen = cast<OMPSimdlenClause>(Clause);
    if (Safelen && Simdlen)
      break;
  }
  if (Simdlen && Safelen &&
      checkSimdlenSafelenValues(*this, Simdlen->getSimdlen(),
                                Safelen->getSafelen()))
    return StmtError();

  getCurFunction()->setHasBranchProtectedScope();
  return OMPParallelForSimdDirective::Create(
      Context, StartLoc, EndLoc, NestedLoopCount, Clauses, AStmt, B);
}

StmtResult
Sema::ActOnOpenMPParallelSectionsDirective(ArrayRef<OMPClause *> Clauses,
                                           Stmt *AStmt, SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  auto BaseStmt = AStmt;
  while (CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(BaseStmt))
    BaseStmt = CS->getCapturedStmt();
  if (auto C = dyn_cast_or_null<CompoundStmt>(BaseStmt)) {
    auto S = C->children();
    if (S.begin() == S.end())
      return StmtError();
    // All associated statements must be '#pragma omp section' except for
    // the first one.
    for (Stmt *SectionStmt : llvm::make_range(std::next(S.begin()), S.end())) {
      if (!SectionStmt || !isa<OMPSectionDirective>(SectionStmt)) {
        if (SectionStmt)
          Diag(SectionStmt->getLocStart(),
               diag::err_omp_parallel_sections_substmt_not_section);
        return StmtError();
      }
      cast<OMPSectionDirective>(SectionStmt)
          ->setHasCancel(DSAStack->isCancelRegion());
    }
  } else {
    Diag(AStmt->getLocStart(),
         diag::err_omp_parallel_sections_not_compound_stmt);
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();

  return OMPParallelSectionsDirective::Create(
      Context, StartLoc, EndLoc, Clauses, AStmt, DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPTaskDirective(ArrayRef<OMPClause *> Clauses,
                                          Stmt *AStmt, SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  CapturedStmt *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  getCurFunction()->setHasBranchProtectedScope();

  return OMPTaskDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                  DSAStack->isCancelRegion());
}

StmtResult Sema::ActOnOpenMPTaskyieldDirective(SourceLocation StartLoc,
                                               SourceLocation EndLoc) {
  return OMPTaskyieldDirective::Create(Context, StartLoc, EndLoc);
}

StmtResult Sema::ActOnOpenMPBarrierDirective(SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  return OMPBarrierDirective::Create(Context, StartLoc, EndLoc);
}

StmtResult Sema::ActOnOpenMPTaskwaitDirective(SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  return OMPTaskwaitDirective::Create(Context, StartLoc, EndLoc);
}

StmtResult Sema::ActOnOpenMPTaskgroupDirective(Stmt *AStmt,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

  getCurFunction()->setHasBranchProtectedScope();

  return OMPTaskgroupDirective::Create(Context, StartLoc, EndLoc, AStmt);
}

StmtResult Sema::ActOnOpenMPFlushDirective(ArrayRef<OMPClause *> Clauses,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  assert(Clauses.size() <= 1 && "Extra clauses in flush directive");
  return OMPFlushDirective::Create(Context, StartLoc, EndLoc, Clauses);
}

StmtResult Sema::ActOnOpenMPOrderedDirective(ArrayRef<OMPClause *> Clauses,
                                             Stmt *AStmt,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  OMPClause *DependFound = nullptr;
  OMPClause *DependSourceClause = nullptr;
  OMPClause *DependSinkClause = nullptr;
  bool ErrorFound = false;
  OMPThreadsClause *TC = nullptr;
  OMPSIMDClause *SC = nullptr;
  for (auto *C : Clauses) {
    if (auto *DC = dyn_cast<OMPDependClause>(C)) {
      DependFound = C;
      if (DC->getDependencyKind() == OMPC_DEPEND_source) {
        if (DependSourceClause) {
          Diag(C->getLocStart(), diag::err_omp_more_one_clause)
              << getOpenMPDirectiveName(OMPD_ordered)
              << getOpenMPClauseName(OMPC_depend) << 2;
          ErrorFound = true;
        } else
          DependSourceClause = C;
        if (DependSinkClause) {
          Diag(C->getLocStart(), diag::err_omp_depend_sink_source_not_allowed)
              << 0;
          ErrorFound = true;
        }
      } else if (DC->getDependencyKind() == OMPC_DEPEND_sink) {
        if (DependSourceClause) {
          Diag(C->getLocStart(), diag::err_omp_depend_sink_source_not_allowed)
              << 1;
          ErrorFound = true;
        }
        DependSinkClause = C;
      }
    } else if (C->getClauseKind() == OMPC_threads)
      TC = cast<OMPThreadsClause>(C);
    else if (C->getClauseKind() == OMPC_simd)
      SC = cast<OMPSIMDClause>(C);
  }
  if (!ErrorFound && !SC &&
      isOpenMPSimdDirective(DSAStack->getParentDirective())) {
    // OpenMP [2.8.1,simd Construct, Restrictions]
    // An ordered construct with the simd clause is the only OpenMP construct
    // that can appear in the simd region.
    Diag(StartLoc, diag::err_omp_prohibited_region_simd);
    ErrorFound = true;
  } else if (DependFound && (TC || SC)) {
    Diag(DependFound->getLocStart(), diag::err_omp_depend_clause_thread_simd)
        << getOpenMPClauseName(TC ? TC->getClauseKind() : SC->getClauseKind());
    ErrorFound = true;
  } else if (DependFound && !DSAStack->getParentOrderedRegionParam()) {
    Diag(DependFound->getLocStart(),
         diag::err_omp_ordered_directive_without_param);
    ErrorFound = true;
  } else if (TC || Clauses.empty()) {
    if (auto *Param = DSAStack->getParentOrderedRegionParam()) {
      SourceLocation ErrLoc = TC ? TC->getLocStart() : StartLoc;
      Diag(ErrLoc, diag::err_omp_ordered_directive_with_param)
          << (TC != nullptr);
      Diag(Param->getLocStart(), diag::note_omp_ordered_param);
      ErrorFound = true;
    }
  }
  if ((!AStmt && !DependFound) || ErrorFound)
    return StmtError();

  if (AStmt) {
    assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

    getCurFunction()->setHasBranchProtectedScope();
  }

  return OMPOrderedDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

namespace {
/// \brief Helper class for checking expression in 'omp atomic [update]'
/// construct.
class OpenMPAtomicUpdateChecker {
  /// \brief Error results for atomic update expressions.
  enum ExprAnalysisErrorCode {
    /// \brief A statement is not an expression statement.
    NotAnExpression,
    /// \brief Expression is not builtin binary or unary operation.
    NotABinaryOrUnaryExpression,
    /// \brief Unary operation is not post-/pre- increment/decrement operation.
    NotAnUnaryIncDecExpression,
    /// \brief An expression is not of scalar type.
    NotAScalarType,
    /// \brief A binary operation is not an assignment operation.
    NotAnAssignmentOp,
    /// \brief RHS part of the binary operation is not a binary expression.
    NotABinaryExpression,
    /// \brief RHS part is not additive/multiplicative/shift/biwise binary
    /// expression.
    NotABinaryOperator,
    /// \brief RHS binary operation does not have reference to the updated LHS
    /// part.
    NotAnUpdateExpression,
    /// \brief No errors is found.
    NoError
  };
  /// \brief Reference to Sema.
  Sema &SemaRef;
  /// \brief A location for note diagnostics (when error is found).
  SourceLocation NoteLoc;
  /// \brief 'x' lvalue part of the source atomic expression.
  Expr *X;
  /// \brief 'expr' rvalue part of the source atomic expression.
  Expr *E;
  /// \brief Helper expression of the form
  /// 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  Expr *UpdateExpr;
  /// \brief Is 'x' a LHS in a RHS part of full update expression. It is
  /// important for non-associative operations.
  bool IsXLHSInRHSPart;
  BinaryOperatorKind Op;
  SourceLocation OpLoc;
  /// \brief true if the source expression is a postfix unary operation, false
  /// if it is a prefix unary operation.
  bool IsPostfixUpdate;

public:
  OpenMPAtomicUpdateChecker(Sema &SemaRef)
      : SemaRef(SemaRef), X(nullptr), E(nullptr), UpdateExpr(nullptr),
        IsXLHSInRHSPart(false), Op(BO_PtrMemD), IsPostfixUpdate(false) {}
  /// \brief Check specified statement that it is suitable for 'atomic update'
  /// constructs and extract 'x', 'expr' and Operation from the original
  /// expression. If DiagId and NoteId == 0, then only check is performed
  /// without error notification.
  /// \param DiagId Diagnostic which should be emitted if error is found.
  /// \param NoteId Diagnostic note for the main error message.
  /// \return true if statement is not an update expression, false otherwise.
  bool checkStatement(Stmt *S, unsigned DiagId = 0, unsigned NoteId = 0);
  /// \brief Return the 'x' lvalue part of the source atomic expression.
  Expr *getX() const { return X; }
  /// \brief Return the 'expr' rvalue part of the source atomic expression.
  Expr *getExpr() const { return E; }
  /// \brief Return the update expression used in calculation of the updated
  /// value. Always has form 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  Expr *getUpdateExpr() const { return UpdateExpr; }
  /// \brief Return true if 'x' is LHS in RHS part of full update expression,
  /// false otherwise.
  bool isXLHSInRHSPart() const { return IsXLHSInRHSPart; }

  /// \brief true if the source expression is a postfix unary operation, false
  /// if it is a prefix unary operation.
  bool isPostfixUpdate() const { return IsPostfixUpdate; }

private:
  bool checkBinaryOperation(BinaryOperator *AtomicBinOp, unsigned DiagId = 0,
                            unsigned NoteId = 0);
};
} // namespace

bool OpenMPAtomicUpdateChecker::checkBinaryOperation(
    BinaryOperator *AtomicBinOp, unsigned DiagId, unsigned NoteId) {
  ExprAnalysisErrorCode ErrorFound = NoError;
  SourceLocation ErrorLoc, NoteLoc;
  SourceRange ErrorRange, NoteRange;
  // Allowed constructs are:
  //  x = x binop expr;
  //  x = expr binop x;
  if (AtomicBinOp->getOpcode() == BO_Assign) {
    X = AtomicBinOp->getLHS();
    if (auto *AtomicInnerBinOp = dyn_cast<BinaryOperator>(
            AtomicBinOp->getRHS()->IgnoreParenImpCasts())) {
      if (AtomicInnerBinOp->isMultiplicativeOp() ||
          AtomicInnerBinOp->isAdditiveOp() || AtomicInnerBinOp->isShiftOp() ||
          AtomicInnerBinOp->isBitwiseOp()) {
        Op = AtomicInnerBinOp->getOpcode();
        OpLoc = AtomicInnerBinOp->getOperatorLoc();
        auto *LHS = AtomicInnerBinOp->getLHS();
        auto *RHS = AtomicInnerBinOp->getRHS();
        llvm::FoldingSetNodeID XId, LHSId, RHSId;
        X->IgnoreParenImpCasts()->Profile(XId, SemaRef.getASTContext(),
                                          /*Canonical=*/true);
        LHS->IgnoreParenImpCasts()->Profile(LHSId, SemaRef.getASTContext(),
                                            /*Canonical=*/true);
        RHS->IgnoreParenImpCasts()->Profile(RHSId, SemaRef.getASTContext(),
                                            /*Canonical=*/true);
        if (XId == LHSId) {
          E = RHS;
          IsXLHSInRHSPart = true;
        } else if (XId == RHSId) {
          E = LHS;
          IsXLHSInRHSPart = false;
        } else {
          ErrorLoc = AtomicInnerBinOp->getExprLoc();
          ErrorRange = AtomicInnerBinOp->getSourceRange();
          NoteLoc = X->getExprLoc();
          NoteRange = X->getSourceRange();
          ErrorFound = NotAnUpdateExpression;
        }
      } else {
        ErrorLoc = AtomicInnerBinOp->getExprLoc();
        ErrorRange = AtomicInnerBinOp->getSourceRange();
        NoteLoc = AtomicInnerBinOp->getOperatorLoc();
        NoteRange = SourceRange(NoteLoc, NoteLoc);
        ErrorFound = NotABinaryOperator;
      }
    } else {
      NoteLoc = ErrorLoc = AtomicBinOp->getRHS()->getExprLoc();
      NoteRange = ErrorRange = AtomicBinOp->getRHS()->getSourceRange();
      ErrorFound = NotABinaryExpression;
    }
  } else {
    ErrorLoc = AtomicBinOp->getExprLoc();
    ErrorRange = AtomicBinOp->getSourceRange();
    NoteLoc = AtomicBinOp->getOperatorLoc();
    NoteRange = SourceRange(NoteLoc, NoteLoc);
    ErrorFound = NotAnAssignmentOp;
  }
  if (ErrorFound != NoError && DiagId != 0 && NoteId != 0) {
    SemaRef.Diag(ErrorLoc, DiagId) << ErrorRange;
    SemaRef.Diag(NoteLoc, NoteId) << ErrorFound << NoteRange;
    return true;
  } else if (SemaRef.CurContext->isDependentContext())
    E = X = UpdateExpr = nullptr;
  return ErrorFound != NoError;
}

bool OpenMPAtomicUpdateChecker::checkStatement(Stmt *S, unsigned DiagId,
                                               unsigned NoteId) {
  ExprAnalysisErrorCode ErrorFound = NoError;
  SourceLocation ErrorLoc, NoteLoc;
  SourceRange ErrorRange, NoteRange;
  // Allowed constructs are:
  //  x++;
  //  x--;
  //  ++x;
  //  --x;
  //  x binop= expr;
  //  x = x binop expr;
  //  x = expr binop x;
  if (auto *AtomicBody = dyn_cast<Expr>(S)) {
    AtomicBody = AtomicBody->IgnoreParenImpCasts();
    if (AtomicBody->getType()->isScalarType() ||
        AtomicBody->isInstantiationDependent()) {
      if (auto *AtomicCompAssignOp = dyn_cast<CompoundAssignOperator>(
              AtomicBody->IgnoreParenImpCasts())) {
        // Check for Compound Assignment Operation
        Op = BinaryOperator::getOpForCompoundAssignment(
            AtomicCompAssignOp->getOpcode());
        OpLoc = AtomicCompAssignOp->getOperatorLoc();
        E = AtomicCompAssignOp->getRHS();
        X = AtomicCompAssignOp->getLHS();
        IsXLHSInRHSPart = true;
      } else if (auto *AtomicBinOp = dyn_cast<BinaryOperator>(
                     AtomicBody->IgnoreParenImpCasts())) {
        // Check for Binary Operation
        if(checkBinaryOperation(AtomicBinOp, DiagId, NoteId))
          return true;
      } else if (auto *AtomicUnaryOp =
                 dyn_cast<UnaryOperator>(AtomicBody->IgnoreParenImpCasts())) {
        // Check for Unary Operation
        if (AtomicUnaryOp->isIncrementDecrementOp()) {
          IsPostfixUpdate = AtomicUnaryOp->isPostfix();
          Op = AtomicUnaryOp->isIncrementOp() ? BO_Add : BO_Sub;
          OpLoc = AtomicUnaryOp->getOperatorLoc();
          X = AtomicUnaryOp->getSubExpr();
          E = SemaRef.ActOnIntegerConstant(OpLoc, /*uint64_t Val=*/1).get();
          IsXLHSInRHSPart = true;
        } else {
          ErrorFound = NotAnUnaryIncDecExpression;
          ErrorLoc = AtomicUnaryOp->getExprLoc();
          ErrorRange = AtomicUnaryOp->getSourceRange();
          NoteLoc = AtomicUnaryOp->getOperatorLoc();
          NoteRange = SourceRange(NoteLoc, NoteLoc);
        }
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorFound = NotABinaryOrUnaryExpression;
        NoteLoc = ErrorLoc = AtomicBody->getExprLoc();
        NoteRange = ErrorRange = AtomicBody->getSourceRange();
      }
    } else {
      ErrorFound = NotAScalarType;
      NoteLoc = ErrorLoc = AtomicBody->getLocStart();
      NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
    }
  } else {
    ErrorFound = NotAnExpression;
    NoteLoc = ErrorLoc = S->getLocStart();
    NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
  }
  if (ErrorFound != NoError && DiagId != 0 && NoteId != 0) {
    SemaRef.Diag(ErrorLoc, DiagId) << ErrorRange;
    SemaRef.Diag(NoteLoc, NoteId) << ErrorFound << NoteRange;
    return true;
  } else if (SemaRef.CurContext->isDependentContext())
    E = X = UpdateExpr = nullptr;
  if (ErrorFound == NoError && E && X) {
    // Build an update expression of form 'OpaqueValueExpr(x) binop
    // OpaqueValueExpr(expr)' or 'OpaqueValueExpr(expr) binop
    // OpaqueValueExpr(x)' and then cast it to the type of the 'x' expression.
    auto *OVEX = new (SemaRef.getASTContext())
        OpaqueValueExpr(X->getExprLoc(), X->getType(), VK_RValue);
    auto *OVEExpr = new (SemaRef.getASTContext())
        OpaqueValueExpr(E->getExprLoc(), E->getType(), VK_RValue);
    auto Update =
        SemaRef.CreateBuiltinBinOp(OpLoc, Op, IsXLHSInRHSPart ? OVEX : OVEExpr,
                                   IsXLHSInRHSPart ? OVEExpr : OVEX);
    if (Update.isInvalid())
      return true;
    Update = SemaRef.PerformImplicitConversion(Update.get(), X->getType(),
                                               Sema::AA_Casting);
    if (Update.isInvalid())
      return true;
    UpdateExpr = Update.get();
  }
  return ErrorFound != NoError;
}

StmtResult Sema::ActOnOpenMPAtomicDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  auto CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  OpenMPClauseKind AtomicKind = OMPC_unknown;
  SourceLocation AtomicKindLoc;
  for (auto *C : Clauses) {
    if (C->getClauseKind() == OMPC_read || C->getClauseKind() == OMPC_write ||
        C->getClauseKind() == OMPC_update ||
        C->getClauseKind() == OMPC_capture) {
      if (AtomicKind != OMPC_unknown) {
        Diag(C->getLocStart(), diag::err_omp_atomic_several_clauses)
            << SourceRange(C->getLocStart(), C->getLocEnd());
        Diag(AtomicKindLoc, diag::note_omp_atomic_previous_clause)
            << getOpenMPClauseName(AtomicKind);
      } else {
        AtomicKind = C->getClauseKind();
        AtomicKindLoc = C->getLocStart();
      }
    }
  }

  auto Body = CS->getCapturedStmt();
  if (auto *EWC = dyn_cast<ExprWithCleanups>(Body))
    Body = EWC->getSubExpr();

  Expr *X = nullptr;
  Expr *V = nullptr;
  Expr *E = nullptr;
  Expr *UE = nullptr;
  bool IsXLHSInRHSPart = false;
  bool IsPostfixUpdate = false;
  // OpenMP [2.12.6, atomic Construct]
  // In the next expressions:
  // * x and v (as applicable) are both l-value expressions with scalar type.
  // * During the execution of an atomic region, multiple syntactic
  // occurrences of x must designate the same storage location.
  // * Neither of v and expr (as applicable) may access the storage location
  // designated by x.
  // * Neither of x and expr (as applicable) may access the storage location
  // designated by v.
  // * expr is an expression with scalar type.
  // * binop is one of +, *, -, /, &, ^, |, <<, or >>.
  // * binop, binop=, ++, and -- are not overloaded operators.
  // * The expression x binop expr must be numerically equivalent to x binop
  // (expr). This requirement is satisfied if the operators in expr have
  // precedence greater than binop, or by using parentheses around expr or
  // subexpressions of expr.
  // * The expression expr binop x must be numerically equivalent to (expr)
  // binop x. This requirement is satisfied if the operators in expr have
  // precedence equal to or greater than binop, or by using parentheses around
  // expr or subexpressions of expr.
  // * For forms that allow multiple occurrences of x, the number of times
  // that x is evaluated is unspecified.
  if (AtomicKind == OMPC_read) {
    enum {
      NotAnExpression,
      NotAnAssignmentOp,
      NotAScalarType,
      NotAnLValue,
      NoError
    } ErrorFound = NoError;
    SourceLocation ErrorLoc, NoteLoc;
    SourceRange ErrorRange, NoteRange;
    // If clause is read:
    //  v = x;
    if (auto AtomicBody = dyn_cast<Expr>(Body)) {
      auto AtomicBinOp =
          dyn_cast<BinaryOperator>(AtomicBody->IgnoreParenImpCasts());
      if (AtomicBinOp && AtomicBinOp->getOpcode() == BO_Assign) {
        X = AtomicBinOp->getRHS()->IgnoreParenImpCasts();
        V = AtomicBinOp->getLHS()->IgnoreParenImpCasts();
        if ((X->isInstantiationDependent() || X->getType()->isScalarType()) &&
            (V->isInstantiationDependent() || V->getType()->isScalarType())) {
          if (!X->isLValue() || !V->isLValue()) {
            auto NotLValueExpr = X->isLValue() ? V : X;
            ErrorFound = NotAnLValue;
            ErrorLoc = AtomicBinOp->getExprLoc();
            ErrorRange = AtomicBinOp->getSourceRange();
            NoteLoc = NotLValueExpr->getExprLoc();
            NoteRange = NotLValueExpr->getSourceRange();
          }
        } else if (!X->isInstantiationDependent() ||
                   !V->isInstantiationDependent()) {
          auto NotScalarExpr =
              (X->isInstantiationDependent() || X->getType()->isScalarType())
                  ? V
                  : X;
          ErrorFound = NotAScalarType;
          ErrorLoc = AtomicBinOp->getExprLoc();
          ErrorRange = AtomicBinOp->getSourceRange();
          NoteLoc = NotScalarExpr->getExprLoc();
          NoteRange = NotScalarExpr->getSourceRange();
        }
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorFound = NotAnAssignmentOp;
        ErrorLoc = AtomicBody->getExprLoc();
        ErrorRange = AtomicBody->getSourceRange();
        NoteLoc = AtomicBinOp ? AtomicBinOp->getOperatorLoc()
                              : AtomicBody->getExprLoc();
        NoteRange = AtomicBinOp ? AtomicBinOp->getSourceRange()
                                : AtomicBody->getSourceRange();
      }
    } else {
      ErrorFound = NotAnExpression;
      NoteLoc = ErrorLoc = Body->getLocStart();
      NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
    }
    if (ErrorFound != NoError) {
      Diag(ErrorLoc, diag::err_omp_atomic_read_not_expression_statement)
          << ErrorRange;
      Diag(NoteLoc, diag::note_omp_atomic_read_write) << ErrorFound
                                                      << NoteRange;
      return StmtError();
    } else if (CurContext->isDependentContext())
      V = X = nullptr;
  } else if (AtomicKind == OMPC_write) {
    enum {
      NotAnExpression,
      NotAnAssignmentOp,
      NotAScalarType,
      NotAnLValue,
      NoError
    } ErrorFound = NoError;
    SourceLocation ErrorLoc, NoteLoc;
    SourceRange ErrorRange, NoteRange;
    // If clause is write:
    //  x = expr;
    if (auto AtomicBody = dyn_cast<Expr>(Body)) {
      auto AtomicBinOp =
          dyn_cast<BinaryOperator>(AtomicBody->IgnoreParenImpCasts());
      if (AtomicBinOp && AtomicBinOp->getOpcode() == BO_Assign) {
        X = AtomicBinOp->getLHS();
        E = AtomicBinOp->getRHS();
        if ((X->isInstantiationDependent() || X->getType()->isScalarType()) &&
            (E->isInstantiationDependent() || E->getType()->isScalarType())) {
          if (!X->isLValue()) {
            ErrorFound = NotAnLValue;
            ErrorLoc = AtomicBinOp->getExprLoc();
            ErrorRange = AtomicBinOp->getSourceRange();
            NoteLoc = X->getExprLoc();
            NoteRange = X->getSourceRange();
          }
        } else if (!X->isInstantiationDependent() ||
                   !E->isInstantiationDependent()) {
          auto NotScalarExpr =
              (X->isInstantiationDependent() || X->getType()->isScalarType())
                  ? E
                  : X;
          ErrorFound = NotAScalarType;
          ErrorLoc = AtomicBinOp->getExprLoc();
          ErrorRange = AtomicBinOp->getSourceRange();
          NoteLoc = NotScalarExpr->getExprLoc();
          NoteRange = NotScalarExpr->getSourceRange();
        }
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorFound = NotAnAssignmentOp;
        ErrorLoc = AtomicBody->getExprLoc();
        ErrorRange = AtomicBody->getSourceRange();
        NoteLoc = AtomicBinOp ? AtomicBinOp->getOperatorLoc()
                              : AtomicBody->getExprLoc();
        NoteRange = AtomicBinOp ? AtomicBinOp->getSourceRange()
                                : AtomicBody->getSourceRange();
      }
    } else {
      ErrorFound = NotAnExpression;
      NoteLoc = ErrorLoc = Body->getLocStart();
      NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
    }
    if (ErrorFound != NoError) {
      Diag(ErrorLoc, diag::err_omp_atomic_write_not_expression_statement)
          << ErrorRange;
      Diag(NoteLoc, diag::note_omp_atomic_read_write) << ErrorFound
                                                      << NoteRange;
      return StmtError();
    } else if (CurContext->isDependentContext())
      E = X = nullptr;
  } else if (AtomicKind == OMPC_update || AtomicKind == OMPC_unknown) {
    // If clause is update:
    //  x++;
    //  x--;
    //  ++x;
    //  --x;
    //  x binop= expr;
    //  x = x binop expr;
    //  x = expr binop x;
    OpenMPAtomicUpdateChecker Checker(*this);
    if (Checker.checkStatement(
            Body, (AtomicKind == OMPC_update)
                      ? diag::err_omp_atomic_update_not_expression_statement
                      : diag::err_omp_atomic_not_expression_statement,
            diag::note_omp_atomic_update))
      return StmtError();
    if (!CurContext->isDependentContext()) {
      E = Checker.getExpr();
      X = Checker.getX();
      UE = Checker.getUpdateExpr();
      IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
    }
  } else if (AtomicKind == OMPC_capture) {
    enum {
      NotAnAssignmentOp,
      NotACompoundStatement,
      NotTwoSubstatements,
      NotASpecificExpression,
      NoError
    } ErrorFound = NoError;
    SourceLocation ErrorLoc, NoteLoc;
    SourceRange ErrorRange, NoteRange;
    if (auto *AtomicBody = dyn_cast<Expr>(Body)) {
      // If clause is a capture:
      //  v = x++;
      //  v = x--;
      //  v = ++x;
      //  v = --x;
      //  v = x binop= expr;
      //  v = x = x binop expr;
      //  v = x = expr binop x;
      auto *AtomicBinOp =
          dyn_cast<BinaryOperator>(AtomicBody->IgnoreParenImpCasts());
      if (AtomicBinOp && AtomicBinOp->getOpcode() == BO_Assign) {
        V = AtomicBinOp->getLHS();
        Body = AtomicBinOp->getRHS()->IgnoreParenImpCasts();
        OpenMPAtomicUpdateChecker Checker(*this);
        if (Checker.checkStatement(
                Body, diag::err_omp_atomic_capture_not_expression_statement,
                diag::note_omp_atomic_update))
          return StmtError();
        E = Checker.getExpr();
        X = Checker.getX();
        UE = Checker.getUpdateExpr();
        IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
        IsPostfixUpdate = Checker.isPostfixUpdate();
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorLoc = AtomicBody->getExprLoc();
        ErrorRange = AtomicBody->getSourceRange();
        NoteLoc = AtomicBinOp ? AtomicBinOp->getOperatorLoc()
                              : AtomicBody->getExprLoc();
        NoteRange = AtomicBinOp ? AtomicBinOp->getSourceRange()
                                : AtomicBody->getSourceRange();
        ErrorFound = NotAnAssignmentOp;
      }
      if (ErrorFound != NoError) {
        Diag(ErrorLoc, diag::err_omp_atomic_capture_not_expression_statement)
            << ErrorRange;
        Diag(NoteLoc, diag::note_omp_atomic_capture) << ErrorFound << NoteRange;
        return StmtError();
      } else if (CurContext->isDependentContext()) {
        UE = V = E = X = nullptr;
      }
    } else {
      // If clause is a capture:
      //  { v = x; x = expr; }
      //  { v = x; x++; }
      //  { v = x; x--; }
      //  { v = x; ++x; }
      //  { v = x; --x; }
      //  { v = x; x binop= expr; }
      //  { v = x; x = x binop expr; }
      //  { v = x; x = expr binop x; }
      //  { x++; v = x; }
      //  { x--; v = x; }
      //  { ++x; v = x; }
      //  { --x; v = x; }
      //  { x binop= expr; v = x; }
      //  { x = x binop expr; v = x; }
      //  { x = expr binop x; v = x; }
      if (auto *CS = dyn_cast<CompoundStmt>(Body)) {
        // Check that this is { expr1; expr2; }
        if (CS->size() == 2) {
          auto *First = CS->body_front();
          auto *Second = CS->body_back();
          if (auto *EWC = dyn_cast<ExprWithCleanups>(First))
            First = EWC->getSubExpr()->IgnoreParenImpCasts();
          if (auto *EWC = dyn_cast<ExprWithCleanups>(Second))
            Second = EWC->getSubExpr()->IgnoreParenImpCasts();
          // Need to find what subexpression is 'v' and what is 'x'.
          OpenMPAtomicUpdateChecker Checker(*this);
          bool IsUpdateExprFound = !Checker.checkStatement(Second);
          BinaryOperator *BinOp = nullptr;
          if (IsUpdateExprFound) {
            BinOp = dyn_cast<BinaryOperator>(First);
            IsUpdateExprFound = BinOp && BinOp->getOpcode() == BO_Assign;
          }
          if (IsUpdateExprFound && !CurContext->isDependentContext()) {
            //  { v = x; x++; }
            //  { v = x; x--; }
            //  { v = x; ++x; }
            //  { v = x; --x; }
            //  { v = x; x binop= expr; }
            //  { v = x; x = x binop expr; }
            //  { v = x; x = expr binop x; }
            // Check that the first expression has form v = x.
            auto *PossibleX = BinOp->getRHS()->IgnoreParenImpCasts();
            llvm::FoldingSetNodeID XId, PossibleXId;
            Checker.getX()->Profile(XId, Context, /*Canonical=*/true);
            PossibleX->Profile(PossibleXId, Context, /*Canonical=*/true);
            IsUpdateExprFound = XId == PossibleXId;
            if (IsUpdateExprFound) {
              V = BinOp->getLHS();
              X = Checker.getX();
              E = Checker.getExpr();
              UE = Checker.getUpdateExpr();
              IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
              IsPostfixUpdate = true;
            }
          }
          if (!IsUpdateExprFound) {
            IsUpdateExprFound = !Checker.checkStatement(First);
            BinOp = nullptr;
            if (IsUpdateExprFound) {
              BinOp = dyn_cast<BinaryOperator>(Second);
              IsUpdateExprFound = BinOp && BinOp->getOpcode() == BO_Assign;
            }
            if (IsUpdateExprFound && !CurContext->isDependentContext()) {
              //  { x++; v = x; }
              //  { x--; v = x; }
              //  { ++x; v = x; }
              //  { --x; v = x; }
              //  { x binop= expr; v = x; }
              //  { x = x binop expr; v = x; }
              //  { x = expr binop x; v = x; }
              // Check that the second expression has form v = x.
              auto *PossibleX = BinOp->getRHS()->IgnoreParenImpCasts();
              llvm::FoldingSetNodeID XId, PossibleXId;
              Checker.getX()->Profile(XId, Context, /*Canonical=*/true);
              PossibleX->Profile(PossibleXId, Context, /*Canonical=*/true);
              IsUpdateExprFound = XId == PossibleXId;
              if (IsUpdateExprFound) {
                V = BinOp->getLHS();
                X = Checker.getX();
                E = Checker.getExpr();
                UE = Checker.getUpdateExpr();
                IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
                IsPostfixUpdate = false;
              }
            }
          }
          if (!IsUpdateExprFound) {
            //  { v = x; x = expr; }
            auto *FirstExpr = dyn_cast<Expr>(First);
            auto *SecondExpr = dyn_cast<Expr>(Second);
            if (!FirstExpr || !SecondExpr ||
                !(FirstExpr->isInstantiationDependent() ||
                  SecondExpr->isInstantiationDependent())) {
              auto *FirstBinOp = dyn_cast<BinaryOperator>(First);
              if (!FirstBinOp || FirstBinOp->getOpcode() != BO_Assign) {
                ErrorFound = NotAnAssignmentOp;
                NoteLoc = ErrorLoc = FirstBinOp ? FirstBinOp->getOperatorLoc()
                                                : First->getLocStart();
                NoteRange = ErrorRange = FirstBinOp
                                             ? FirstBinOp->getSourceRange()
                                             : SourceRange(ErrorLoc, ErrorLoc);
              } else {
                auto *SecondBinOp = dyn_cast<BinaryOperator>(Second);
                if (!SecondBinOp || SecondBinOp->getOpcode() != BO_Assign) {
                  ErrorFound = NotAnAssignmentOp;
                  NoteLoc = ErrorLoc = SecondBinOp
                                           ? SecondBinOp->getOperatorLoc()
                                           : Second->getLocStart();
                  NoteRange = ErrorRange =
                      SecondBinOp ? SecondBinOp->getSourceRange()
                                  : SourceRange(ErrorLoc, ErrorLoc);
                } else {
                  auto *PossibleXRHSInFirst =
                      FirstBinOp->getRHS()->IgnoreParenImpCasts();
                  auto *PossibleXLHSInSecond =
                      SecondBinOp->getLHS()->IgnoreParenImpCasts();
                  llvm::FoldingSetNodeID X1Id, X2Id;
                  PossibleXRHSInFirst->Profile(X1Id, Context,
                                               /*Canonical=*/true);
                  PossibleXLHSInSecond->Profile(X2Id, Context,
                                                /*Canonical=*/true);
                  IsUpdateExprFound = X1Id == X2Id;
                  if (IsUpdateExprFound) {
                    V = FirstBinOp->getLHS();
                    X = SecondBinOp->getLHS();
                    E = SecondBinOp->getRHS();
                    UE = nullptr;
                    IsXLHSInRHSPart = false;
                    IsPostfixUpdate = true;
                  } else {
                    ErrorFound = NotASpecificExpression;
                    ErrorLoc = FirstBinOp->getExprLoc();
                    ErrorRange = FirstBinOp->getSourceRange();
                    NoteLoc = SecondBinOp->getLHS()->getExprLoc();
                    NoteRange = SecondBinOp->getRHS()->getSourceRange();
                  }
                }
              }
            }
          }
        } else {
          NoteLoc = ErrorLoc = Body->getLocStart();
          NoteRange = ErrorRange =
              SourceRange(Body->getLocStart(), Body->getLocStart());
          ErrorFound = NotTwoSubstatements;
        }
      } else {
        NoteLoc = ErrorLoc = Body->getLocStart();
        NoteRange = ErrorRange =
            SourceRange(Body->getLocStart(), Body->getLocStart());
        ErrorFound = NotACompoundStatement;
      }
      if (ErrorFound != NoError) {
        Diag(ErrorLoc, diag::err_omp_atomic_capture_not_compound_statement)
            << ErrorRange;
        Diag(NoteLoc, diag::note_omp_atomic_capture) << ErrorFound << NoteRange;
        return StmtError();
      } else if (CurContext->isDependentContext()) {
        UE = V = E = X = nullptr;
      }
    }
  }

  getCurFunction()->setHasBranchProtectedScope();

  return OMPAtomicDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                    X, V, E, UE, IsXLHSInRHSPart,
                                    IsPostfixUpdate);
}

StmtResult Sema::ActOnOpenMPTargetDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  CapturedStmt *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  // OpenMP [2.16, Nesting of Regions]
  // If specified, a teams construct must be contained within a target
  // construct. That target construct must contain no statements or directives
  // outside of the teams construct.
  if (DSAStack->hasInnerTeamsRegion()) {
    auto S = AStmt->IgnoreContainers(/*IgnoreCaptured*/ true);
    bool OMPTeamsFound = true;
    if (auto *CS = dyn_cast<CompoundStmt>(S)) {
      auto I = CS->body_begin();
      while (I != CS->body_end()) {
        auto OED = dyn_cast<OMPExecutableDirective>(*I);
        if (!OED || !isOpenMPTeamsDirective(OED->getDirectiveKind())) {
          OMPTeamsFound = false;
          break;
        }
        ++I;
      }
      assert(I != CS->body_end() && "Not found statement");
      S = *I;
    }
    if (!OMPTeamsFound) {
      Diag(StartLoc, diag::err_omp_target_contains_not_only_teams);
      Diag(DSAStack->getInnerTeamsRegionLoc(),
           diag::note_omp_nested_teams_construct_here);
      Diag(S->getLocStart(), diag::note_omp_nested_statement_here)
          << isa<OMPExecutableDirective>(S);
      return StmtError();
    }
  }

  getCurFunction()->setHasBranchProtectedScope();

  return OMPTargetDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPTargetDataDirective(ArrayRef<OMPClause *> Clauses,
                                                Stmt *AStmt,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");

  getCurFunction()->setHasBranchProtectedScope();

  return OMPTargetDataDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                        AStmt);
}

StmtResult Sema::ActOnOpenMPTeamsDirective(ArrayRef<OMPClause *> Clauses,
                                           Stmt *AStmt, SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  CapturedStmt *CS = cast<CapturedStmt>(AStmt);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CS->getCapturedDecl()->setNothrow();

  getCurFunction()->setHasBranchProtectedScope();

  return OMPTeamsDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult
Sema::ActOnOpenMPCancellationPointDirective(SourceLocation StartLoc,
                                            SourceLocation EndLoc,
                                            OpenMPDirectiveKind CancelRegion) {
  if (CancelRegion != OMPD_parallel && CancelRegion != OMPD_for &&
      CancelRegion != OMPD_sections && CancelRegion != OMPD_taskgroup) {
    Diag(StartLoc, diag::err_omp_wrong_cancel_region)
        << getOpenMPDirectiveName(CancelRegion);
    return StmtError();
  }
  if (DSAStack->isParentNowaitRegion()) {
    Diag(StartLoc, diag::err_omp_parent_cancel_region_nowait) << 0;
    return StmtError();
  }
  if (DSAStack->isParentOrderedRegion()) {
    Diag(StartLoc, diag::err_omp_parent_cancel_region_ordered) << 0;
    return StmtError();
  }
  return OMPCancellationPointDirective::Create(Context, StartLoc, EndLoc,
                                               CancelRegion);
}

StmtResult Sema::ActOnOpenMPCancelDirective(ArrayRef<OMPClause *> Clauses,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc,
                                            OpenMPDirectiveKind CancelRegion) {
  if (CancelRegion != OMPD_parallel && CancelRegion != OMPD_for &&
      CancelRegion != OMPD_sections && CancelRegion != OMPD_taskgroup) {
    Diag(StartLoc, diag::err_omp_wrong_cancel_region)
        << getOpenMPDirectiveName(CancelRegion);
    return StmtError();
  }
  if (DSAStack->isParentNowaitRegion()) {
    Diag(StartLoc, diag::err_omp_parent_cancel_region_nowait) << 1;
    return StmtError();
  }
  if (DSAStack->isParentOrderedRegion()) {
    Diag(StartLoc, diag::err_omp_parent_cancel_region_ordered) << 1;
    return StmtError();
  }
  DSAStack->setParentCancelRegion(/*Cancel=*/true);
  return OMPCancelDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                    CancelRegion);
}

static bool checkGrainsizeNumTasksClauses(Sema &S,
                                          ArrayRef<OMPClause *> Clauses) {
  OMPClause *PrevClause = nullptr;
  bool ErrorFound = false;
  for (auto *C : Clauses) {
    if (C->getClauseKind() == OMPC_grainsize ||
        C->getClauseKind() == OMPC_num_tasks) {
      if (!PrevClause)
        PrevClause = C;
      else if (PrevClause->getClauseKind() != C->getClauseKind()) {
        S.Diag(C->getLocStart(),
               diag::err_omp_grainsize_num_tasks_mutually_exclusive)
            << getOpenMPClauseName(C->getClauseKind())
            << getOpenMPClauseName(PrevClause->getClauseKind());
        S.Diag(PrevClause->getLocStart(),
               diag::note_omp_previous_grainsize_num_tasks)
            << getOpenMPClauseName(PrevClause->getClauseKind());
        ErrorFound = true;
      }
    }
  }
  return ErrorFound;
}

StmtResult Sema::ActOnOpenMPTaskLoopDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      CheckOpenMPLoop(OMPD_taskloop, getCollapseNumberExpr(Clauses),
                      /*OrderedLoopCountExpr=*/nullptr, AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // The grainsize clause and num_tasks clause are mutually exclusive and may
  // not appear on the same taskloop directive.
  if (checkGrainsizeNumTasksClauses(*this, Clauses))
    return StmtError();

  getCurFunction()->setHasBranchProtectedScope();
  return OMPTaskLoopDirective::Create(Context, StartLoc, EndLoc,
                                      NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPTaskLoopSimdDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopDirective::HelperExprs B;
  // In presence of clause 'collapse' or 'ordered' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      CheckOpenMPLoop(OMPD_taskloop_simd, getCollapseNumberExpr(Clauses),
                      /*OrderedLoopCountExpr=*/nullptr, AStmt, *this, *DSAStack,
                      VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  // OpenMP, [2.9.2 taskloop Construct, Restrictions]
  // The grainsize clause and num_tasks clause are mutually exclusive and may
  // not appear on the same taskloop directive.
  if (checkGrainsizeNumTasksClauses(*this, Clauses))
    return StmtError();

  getCurFunction()->setHasBranchProtectedScope();
  return OMPTaskLoopSimdDirective::Create(Context, StartLoc, EndLoc,
                                          NestedLoopCount, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOpenMPDistributeDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc,
    llvm::DenseMap<VarDecl *, Expr *> &VarsWithImplicitDSA) {
  if (!AStmt)
    return StmtError();

  assert(isa<CapturedStmt>(AStmt) && "Captured statement expected");
  OMPLoopDirective::HelperExprs B;
  // In presence of clause 'collapse' with number of loops, it will
  // define the nested loops number.
  unsigned NestedLoopCount =
      CheckOpenMPLoop(OMPD_distribute, getCollapseNumberExpr(Clauses),
                      nullptr /*ordered not a clause on distribute*/, AStmt,
                      *this, *DSAStack, VarsWithImplicitDSA, B);
  if (NestedLoopCount == 0)
    return StmtError();

  assert((CurContext->isDependentContext() || B.builtAll()) &&
         "omp for loop exprs were not built");

  getCurFunction()->setHasBranchProtectedScope();
  return OMPDistributeDirective::Create(Context, StartLoc, EndLoc,
                                        NestedLoopCount, Clauses, AStmt, B);
}

OMPClause *Sema::ActOnOpenMPSingleExprClause(OpenMPClauseKind Kind, Expr *Expr,
                                             SourceLocation StartLoc,
                                             SourceLocation LParenLoc,
                                             SourceLocation EndLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_final:
    Res = ActOnOpenMPFinalClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_num_threads:
    Res = ActOnOpenMPNumThreadsClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_safelen:
    Res = ActOnOpenMPSafelenClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_simdlen:
    Res = ActOnOpenMPSimdlenClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_collapse:
    Res = ActOnOpenMPCollapseClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_ordered:
    Res = ActOnOpenMPOrderedClause(StartLoc, EndLoc, LParenLoc, Expr);
    break;
  case OMPC_device:
    Res = ActOnOpenMPDeviceClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_num_teams:
    Res = ActOnOpenMPNumTeamsClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_thread_limit:
    Res = ActOnOpenMPThreadLimitClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_priority:
    Res = ActOnOpenMPPriorityClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_grainsize:
    Res = ActOnOpenMPGrainsizeClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_num_tasks:
    Res = ActOnOpenMPNumTasksClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_hint:
    Res = ActOnOpenMPHintClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_if:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_schedule:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_threadprivate:
  case OMPC_flush:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_seq_cst:
  case OMPC_depend:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_map:
  case OMPC_nogroup:
  case OMPC_unknown:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPIfClause(OpenMPDirectiveKind NameModifier,
                                     Expr *Condition, SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation NameModifierLoc,
                                     SourceLocation ColonLoc,
                                     SourceLocation EndLoc) {
  Expr *ValExpr = Condition;
  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = ActOnBooleanCondition(DSAStack->getCurScope(),
                                           Condition->getExprLoc(), Condition);
    if (Val.isInvalid())
      return nullptr;

    ValExpr = Val.get();
  }

  return new (Context) OMPIfClause(NameModifier, ValExpr, StartLoc, LParenLoc,
                                   NameModifierLoc, ColonLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPFinalClause(Expr *Condition,
                                        SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc) {
  Expr *ValExpr = Condition;
  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = ActOnBooleanCondition(DSAStack->getCurScope(),
                                           Condition->getExprLoc(), Condition);
    if (Val.isInvalid())
      return nullptr;

    ValExpr = Val.get();
  }

  return new (Context) OMPFinalClause(ValExpr, StartLoc, LParenLoc, EndLoc);
}
ExprResult Sema::PerformOpenMPImplicitIntegerConversion(SourceLocation Loc,
                                                        Expr *Op) {
  if (!Op)
    return ExprError();

  class IntConvertDiagnoser : public ICEConvertDiagnoser {
  public:
    IntConvertDiagnoser()
        : ICEConvertDiagnoser(/*AllowScopedEnumerations*/ false, false, true) {}
    SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                         QualType T) override {
      return S.Diag(Loc, diag::err_omp_not_integral) << T;
    }
    SemaDiagnosticBuilder diagnoseIncomplete(Sema &S, SourceLocation Loc,
                                             QualType T) override {
      return S.Diag(Loc, diag::err_omp_incomplete_type) << T;
    }
    SemaDiagnosticBuilder diagnoseExplicitConv(Sema &S, SourceLocation Loc,
                                               QualType T,
                                               QualType ConvTy) override {
      return S.Diag(Loc, diag::err_omp_explicit_conversion) << T << ConvTy;
    }
    SemaDiagnosticBuilder noteExplicitConv(Sema &S, CXXConversionDecl *Conv,
                                           QualType ConvTy) override {
      return S.Diag(Conv->getLocation(), diag::note_omp_conversion_here)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    SemaDiagnosticBuilder diagnoseAmbiguous(Sema &S, SourceLocation Loc,
                                            QualType T) override {
      return S.Diag(Loc, diag::err_omp_ambiguous_conversion) << T;
    }
    SemaDiagnosticBuilder noteAmbiguous(Sema &S, CXXConversionDecl *Conv,
                                        QualType ConvTy) override {
      return S.Diag(Conv->getLocation(), diag::note_omp_conversion_here)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    SemaDiagnosticBuilder diagnoseConversion(Sema &, SourceLocation, QualType,
                                             QualType) override {
      llvm_unreachable("conversion functions are permitted");
    }
  } ConvertDiagnoser;
  return PerformContextualImplicitConversion(Loc, Op, ConvertDiagnoser);
}

static bool IsNonNegativeIntegerValue(Expr *&ValExpr, Sema &SemaRef,
                                      OpenMPClauseKind CKind,
                                      bool StrictlyPositive) {
  if (!ValExpr->isTypeDependent() && !ValExpr->isValueDependent() &&
      !ValExpr->isInstantiationDependent()) {
    SourceLocation Loc = ValExpr->getExprLoc();
    ExprResult Value =
        SemaRef.PerformOpenMPImplicitIntegerConversion(Loc, ValExpr);
    if (Value.isInvalid())
      return false;

    ValExpr = Value.get();
    // The expression must evaluate to a non-negative integer value.
    llvm::APSInt Result;
    if (ValExpr->isIntegerConstantExpr(Result, SemaRef.Context) &&
        Result.isSigned() &&
        !((!StrictlyPositive && Result.isNonNegative()) ||
          (StrictlyPositive && Result.isStrictlyPositive()))) {
      SemaRef.Diag(Loc, diag::err_omp_negative_expression_in_clause)
          << getOpenMPClauseName(CKind) << (StrictlyPositive ? 1 : 0)
          << ValExpr->getSourceRange();
      return false;
    }
  }
  return true;
}

OMPClause *Sema::ActOnOpenMPNumThreadsClause(Expr *NumThreads,
                                             SourceLocation StartLoc,
                                             SourceLocation LParenLoc,
                                             SourceLocation EndLoc) {
  Expr *ValExpr = NumThreads;

  // OpenMP [2.5, Restrictions]
  //  The num_threads expression must evaluate to a positive integer value.
  if (!IsNonNegativeIntegerValue(ValExpr, *this, OMPC_num_threads,
                                 /*StrictlyPositive=*/true))
    return nullptr;

  return new (Context)
      OMPNumThreadsClause(ValExpr, StartLoc, LParenLoc, EndLoc);
}

ExprResult Sema::VerifyPositiveIntegerConstantInClause(Expr *E,
                                                       OpenMPClauseKind CKind,
                                                       bool StrictlyPositive) {
  if (!E)
    return ExprError();
  if (E->isValueDependent() || E->isTypeDependent() ||
      E->isInstantiationDependent() || E->containsUnexpandedParameterPack())
    return E;
  llvm::APSInt Result;
  ExprResult ICE = VerifyIntegerConstantExpression(E, &Result);
  if (ICE.isInvalid())
    return ExprError();
  if ((StrictlyPositive && !Result.isStrictlyPositive()) ||
      (!StrictlyPositive && !Result.isNonNegative())) {
    Diag(E->getExprLoc(), diag::err_omp_negative_expression_in_clause)
        << getOpenMPClauseName(CKind) << (StrictlyPositive ? 1 : 0)
        << E->getSourceRange();
    return ExprError();
  }
  if (CKind == OMPC_aligned && !Result.isPowerOf2()) {
    Diag(E->getExprLoc(), diag::warn_omp_alignment_not_power_of_two)
        << E->getSourceRange();
    return ExprError();
  }
  if (CKind == OMPC_collapse && DSAStack->getAssociatedLoops() == 1)
    DSAStack->setAssociatedLoops(Result.getExtValue());
  else if (CKind == OMPC_ordered)
    DSAStack->setAssociatedLoops(Result.getExtValue());
  return ICE;
}

OMPClause *Sema::ActOnOpenMPSafelenClause(Expr *Len, SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  // OpenMP [2.8.1, simd construct, Description]
  // The parameter of the safelen clause must be a constant
  // positive integer expression.
  ExprResult Safelen = VerifyPositiveIntegerConstantInClause(Len, OMPC_safelen);
  if (Safelen.isInvalid())
    return nullptr;
  return new (Context)
      OMPSafelenClause(Safelen.get(), StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSimdlenClause(Expr *Len, SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  // OpenMP [2.8.1, simd construct, Description]
  // The parameter of the simdlen clause must be a constant
  // positive integer expression.
  ExprResult Simdlen = VerifyPositiveIntegerConstantInClause(Len, OMPC_simdlen);
  if (Simdlen.isInvalid())
    return nullptr;
  return new (Context)
      OMPSimdlenClause(Simdlen.get(), StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPCollapseClause(Expr *NumForLoops,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  // OpenMP [2.7.1, loop construct, Description]
  // OpenMP [2.8.1, simd construct, Description]
  // OpenMP [2.9.6, distribute construct, Description]
  // The parameter of the collapse clause must be a constant
  // positive integer expression.
  ExprResult NumForLoopsResult =
      VerifyPositiveIntegerConstantInClause(NumForLoops, OMPC_collapse);
  if (NumForLoopsResult.isInvalid())
    return nullptr;
  return new (Context)
      OMPCollapseClause(NumForLoopsResult.get(), StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPOrderedClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc,
                                          SourceLocation LParenLoc,
                                          Expr *NumForLoops) {
  // OpenMP [2.7.1, loop construct, Description]
  // OpenMP [2.8.1, simd construct, Description]
  // OpenMP [2.9.6, distribute construct, Description]
  // The parameter of the ordered clause must be a constant
  // positive integer expression if any.
  if (NumForLoops && LParenLoc.isValid()) {
    ExprResult NumForLoopsResult =
        VerifyPositiveIntegerConstantInClause(NumForLoops, OMPC_ordered);
    if (NumForLoopsResult.isInvalid())
      return nullptr;
    NumForLoops = NumForLoopsResult.get();
  } else
    NumForLoops = nullptr;
  DSAStack->setOrderedRegion(/*IsOrdered=*/true, NumForLoops);
  return new (Context)
      OMPOrderedClause(NumForLoops, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSimpleClause(
    OpenMPClauseKind Kind, unsigned Argument, SourceLocation ArgumentLoc,
    SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation EndLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_default:
    Res =
        ActOnOpenMPDefaultClause(static_cast<OpenMPDefaultClauseKind>(Argument),
                                 ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_proc_bind:
    Res = ActOnOpenMPProcBindClause(
        static_cast<OpenMPProcBindClauseKind>(Argument), ArgumentLoc, StartLoc,
        LParenLoc, EndLoc);
    break;
  case OMPC_if:
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_collapse:
  case OMPC_schedule:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_threadprivate:
  case OMPC_flush:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_seq_cst:
  case OMPC_depend:
  case OMPC_device:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_map:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_nogroup:
  case OMPC_num_tasks:
  case OMPC_hint:
  case OMPC_unknown:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

static std::string
getListOfPossibleValues(OpenMPClauseKind K, unsigned First, unsigned Last,
                        ArrayRef<unsigned> Exclude = llvm::None) {
  std::string Values;
  unsigned Bound = Last >= 2 ? Last - 2 : 0;
  unsigned Skipped = Exclude.size();
  auto S = Exclude.begin(), E = Exclude.end();
  for (unsigned i = First; i < Last; ++i) {
    if (std::find(S, E, i) != E) {
      --Skipped;
      continue;
    }
    Values += "'";
    Values += getOpenMPSimpleClauseTypeName(K, i);
    Values += "'";
    if (i == Bound - Skipped)
      Values += " or ";
    else if (i != Bound + 1 - Skipped)
      Values += ", ";
  }
  return Values;
}

OMPClause *Sema::ActOnOpenMPDefaultClause(OpenMPDefaultClauseKind Kind,
                                          SourceLocation KindKwLoc,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  if (Kind == OMPC_DEFAULT_unknown) {
    static_assert(OMPC_DEFAULT_unknown > 0,
                  "OMPC_DEFAULT_unknown not greater than 0");
    Diag(KindKwLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_default, /*First=*/0,
                                   /*Last=*/OMPC_DEFAULT_unknown)
        << getOpenMPClauseName(OMPC_default);
    return nullptr;
  }
  switch (Kind) {
  case OMPC_DEFAULT_none:
    DSAStack->setDefaultDSANone(KindKwLoc);
    break;
  case OMPC_DEFAULT_shared:
    DSAStack->setDefaultDSAShared(KindKwLoc);
    break;
  case OMPC_DEFAULT_unknown:
    llvm_unreachable("Clause kind is not allowed.");
    break;
  }
  return new (Context)
      OMPDefaultClause(Kind, KindKwLoc, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPProcBindClause(OpenMPProcBindClauseKind Kind,
                                           SourceLocation KindKwLoc,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  if (Kind == OMPC_PROC_BIND_unknown) {
    Diag(KindKwLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_proc_bind, /*First=*/0,
                                   /*Last=*/OMPC_PROC_BIND_unknown)
        << getOpenMPClauseName(OMPC_proc_bind);
    return nullptr;
  }
  return new (Context)
      OMPProcBindClause(Kind, KindKwLoc, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSingleExprWithArgClause(
    OpenMPClauseKind Kind, ArrayRef<unsigned> Argument, Expr *Expr,
    SourceLocation StartLoc, SourceLocation LParenLoc,
    ArrayRef<SourceLocation> ArgumentLoc, SourceLocation DelimLoc,
    SourceLocation EndLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_schedule:
    enum { Modifier1, Modifier2, ScheduleKind, NumberOfElements };
    assert(Argument.size() == NumberOfElements &&
           ArgumentLoc.size() == NumberOfElements);
    Res = ActOnOpenMPScheduleClause(
        static_cast<OpenMPScheduleClauseModifier>(Argument[Modifier1]),
        static_cast<OpenMPScheduleClauseModifier>(Argument[Modifier2]),
        static_cast<OpenMPScheduleClauseKind>(Argument[ScheduleKind]), Expr,
        StartLoc, LParenLoc, ArgumentLoc[Modifier1], ArgumentLoc[Modifier2],
        ArgumentLoc[ScheduleKind], DelimLoc, EndLoc);
    break;
  case OMPC_if:
    assert(Argument.size() == 1 && ArgumentLoc.size() == 1);
    Res = ActOnOpenMPIfClause(static_cast<OpenMPDirectiveKind>(Argument.back()),
                              Expr, StartLoc, LParenLoc, ArgumentLoc.back(),
                              DelimLoc, EndLoc);
    break;
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_collapse:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_threadprivate:
  case OMPC_flush:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_seq_cst:
  case OMPC_depend:
  case OMPC_device:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_map:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_nogroup:
  case OMPC_num_tasks:
  case OMPC_hint:
  case OMPC_unknown:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

static bool checkScheduleModifiers(Sema &S, OpenMPScheduleClauseModifier M1,
                                   OpenMPScheduleClauseModifier M2,
                                   SourceLocation M1Loc, SourceLocation M2Loc) {
  if (M1 == OMPC_SCHEDULE_MODIFIER_unknown && M1Loc.isValid()) {
    SmallVector<unsigned, 2> Excluded;
    if (M2 != OMPC_SCHEDULE_MODIFIER_unknown)
      Excluded.push_back(M2);
    if (M2 == OMPC_SCHEDULE_MODIFIER_nonmonotonic)
      Excluded.push_back(OMPC_SCHEDULE_MODIFIER_monotonic);
    if (M2 == OMPC_SCHEDULE_MODIFIER_monotonic)
      Excluded.push_back(OMPC_SCHEDULE_MODIFIER_nonmonotonic);
    S.Diag(M1Loc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_schedule,
                                   /*First=*/OMPC_SCHEDULE_MODIFIER_unknown + 1,
                                   /*Last=*/OMPC_SCHEDULE_MODIFIER_last,
                                   Excluded)
        << getOpenMPClauseName(OMPC_schedule);
    return true;
  }
  return false;
}

OMPClause *Sema::ActOnOpenMPScheduleClause(
    OpenMPScheduleClauseModifier M1, OpenMPScheduleClauseModifier M2,
    OpenMPScheduleClauseKind Kind, Expr *ChunkSize, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation M1Loc, SourceLocation M2Loc,
    SourceLocation KindLoc, SourceLocation CommaLoc, SourceLocation EndLoc) {
  if (checkScheduleModifiers(*this, M1, M2, M1Loc, M2Loc) ||
      checkScheduleModifiers(*this, M2, M1, M2Loc, M1Loc))
    return nullptr;
  // OpenMP, 2.7.1, Loop Construct, Restrictions
  // Either the monotonic modifier or the nonmonotonic modifier can be specified
  // but not both.
  if ((M1 == M2 && M1 != OMPC_SCHEDULE_MODIFIER_unknown) ||
      (M1 == OMPC_SCHEDULE_MODIFIER_monotonic &&
       M2 == OMPC_SCHEDULE_MODIFIER_nonmonotonic) ||
      (M1 == OMPC_SCHEDULE_MODIFIER_nonmonotonic &&
       M2 == OMPC_SCHEDULE_MODIFIER_monotonic)) {
    Diag(M2Loc, diag::err_omp_unexpected_schedule_modifier)
        << getOpenMPSimpleClauseTypeName(OMPC_schedule, M2)
        << getOpenMPSimpleClauseTypeName(OMPC_schedule, M1);
    return nullptr;
  }
  if (Kind == OMPC_SCHEDULE_unknown) {
    std::string Values;
    if (M1Loc.isInvalid() && M2Loc.isInvalid()) {
      unsigned Exclude[] = {OMPC_SCHEDULE_unknown};
      Values = getListOfPossibleValues(OMPC_schedule, /*First=*/0,
                                       /*Last=*/OMPC_SCHEDULE_MODIFIER_last,
                                       Exclude);
    } else {
      Values = getListOfPossibleValues(OMPC_schedule, /*First=*/0,
                                       /*Last=*/OMPC_SCHEDULE_unknown);
    }
    Diag(KindLoc, diag::err_omp_unexpected_clause_value)
        << Values << getOpenMPClauseName(OMPC_schedule);
    return nullptr;
  }
  // OpenMP, 2.7.1, Loop Construct, Restrictions
  // The nonmonotonic modifier can only be specified with schedule(dynamic) or
  // schedule(guided).
  if ((M1 == OMPC_SCHEDULE_MODIFIER_nonmonotonic ||
       M2 == OMPC_SCHEDULE_MODIFIER_nonmonotonic) &&
      Kind != OMPC_SCHEDULE_dynamic && Kind != OMPC_SCHEDULE_guided) {
    Diag(M1 == OMPC_SCHEDULE_MODIFIER_nonmonotonic ? M1Loc : M2Loc,
         diag::err_omp_schedule_nonmonotonic_static);
    return nullptr;
  }
  Expr *ValExpr = ChunkSize;
  Expr *HelperValExpr = nullptr;
  if (ChunkSize) {
    if (!ChunkSize->isValueDependent() && !ChunkSize->isTypeDependent() &&
        !ChunkSize->isInstantiationDependent() &&
        !ChunkSize->containsUnexpandedParameterPack()) {
      SourceLocation ChunkSizeLoc = ChunkSize->getLocStart();
      ExprResult Val =
          PerformOpenMPImplicitIntegerConversion(ChunkSizeLoc, ChunkSize);
      if (Val.isInvalid())
        return nullptr;

      ValExpr = Val.get();

      // OpenMP [2.7.1, Restrictions]
      //  chunk_size must be a loop invariant integer expression with a positive
      //  value.
      llvm::APSInt Result;
      if (ValExpr->isIntegerConstantExpr(Result, Context)) {
        if (Result.isSigned() && !Result.isStrictlyPositive()) {
          Diag(ChunkSizeLoc, diag::err_omp_negative_expression_in_clause)
              << "schedule" << 1 << ChunkSize->getSourceRange();
          return nullptr;
        }
      } else if (isParallelOrTaskRegion(DSAStack->getCurrentDirective())) {
        auto *ImpVar = buildVarDecl(*this, ChunkSize->getExprLoc(),
                                    ChunkSize->getType(), ".chunk.");
        auto *ImpVarRef = buildDeclRefExpr(*this, ImpVar, ChunkSize->getType(),
                                           ChunkSize->getExprLoc(),
                                           /*RefersToCapture=*/true);
        HelperValExpr = ImpVarRef;
      }
    }
  }

  return new (Context)
      OMPScheduleClause(StartLoc, LParenLoc, KindLoc, CommaLoc, EndLoc, Kind,
                        ValExpr, HelperValExpr, M1, M1Loc, M2, M2Loc);
}

OMPClause *Sema::ActOnOpenMPClause(OpenMPClauseKind Kind,
                                   SourceLocation StartLoc,
                                   SourceLocation EndLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_ordered:
    Res = ActOnOpenMPOrderedClause(StartLoc, EndLoc);
    break;
  case OMPC_nowait:
    Res = ActOnOpenMPNowaitClause(StartLoc, EndLoc);
    break;
  case OMPC_untied:
    Res = ActOnOpenMPUntiedClause(StartLoc, EndLoc);
    break;
  case OMPC_mergeable:
    Res = ActOnOpenMPMergeableClause(StartLoc, EndLoc);
    break;
  case OMPC_read:
    Res = ActOnOpenMPReadClause(StartLoc, EndLoc);
    break;
  case OMPC_write:
    Res = ActOnOpenMPWriteClause(StartLoc, EndLoc);
    break;
  case OMPC_update:
    Res = ActOnOpenMPUpdateClause(StartLoc, EndLoc);
    break;
  case OMPC_capture:
    Res = ActOnOpenMPCaptureClause(StartLoc, EndLoc);
    break;
  case OMPC_seq_cst:
    Res = ActOnOpenMPSeqCstClause(StartLoc, EndLoc);
    break;
  case OMPC_threads:
    Res = ActOnOpenMPThreadsClause(StartLoc, EndLoc);
    break;
  case OMPC_simd:
    Res = ActOnOpenMPSIMDClause(StartLoc, EndLoc);
    break;
  case OMPC_nogroup:
    Res = ActOnOpenMPNogroupClause(StartLoc, EndLoc);
    break;
  case OMPC_if:
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_collapse:
  case OMPC_schedule:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_threadprivate:
  case OMPC_flush:
  case OMPC_depend:
  case OMPC_device:
  case OMPC_map:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_num_tasks:
  case OMPC_hint:
  case OMPC_unknown:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPNowaitClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  DSAStack->setNowaitRegion();
  return new (Context) OMPNowaitClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPUntiedClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OMPUntiedClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPMergeableClause(SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  return new (Context) OMPMergeableClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPReadClause(SourceLocation StartLoc,
                                       SourceLocation EndLoc) {
  return new (Context) OMPReadClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPWriteClause(SourceLocation StartLoc,
                                        SourceLocation EndLoc) {
  return new (Context) OMPWriteClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPUpdateClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OMPUpdateClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPCaptureClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPCaptureClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSeqCstClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OMPSeqCstClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPThreadsClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPThreadsClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSIMDClause(SourceLocation StartLoc,
                                       SourceLocation EndLoc) {
  return new (Context) OMPSIMDClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPNogroupClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPNogroupClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPVarListClause(
    OpenMPClauseKind Kind, ArrayRef<Expr *> VarList, Expr *TailExpr,
    SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation ColonLoc,
    SourceLocation EndLoc, CXXScopeSpec &ReductionIdScopeSpec,
    const DeclarationNameInfo &ReductionId, OpenMPDependClauseKind DepKind,
    OpenMPLinearClauseKind LinKind, OpenMPMapClauseKind MapTypeModifier, 
    OpenMPMapClauseKind MapType, SourceLocation DepLinMapLoc) {
  OMPClause *Res = nullptr;
  switch (Kind) {
  case OMPC_private:
    Res = ActOnOpenMPPrivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_firstprivate:
    Res = ActOnOpenMPFirstprivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_lastprivate:
    Res = ActOnOpenMPLastprivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_shared:
    Res = ActOnOpenMPSharedClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_reduction:
    Res = ActOnOpenMPReductionClause(VarList, StartLoc, LParenLoc, ColonLoc,
                                     EndLoc, ReductionIdScopeSpec, ReductionId);
    break;
  case OMPC_linear:
    Res = ActOnOpenMPLinearClause(VarList, TailExpr, StartLoc, LParenLoc,
                                  LinKind, DepLinMapLoc, ColonLoc, EndLoc);
    break;
  case OMPC_aligned:
    Res = ActOnOpenMPAlignedClause(VarList, TailExpr, StartLoc, LParenLoc,
                                   ColonLoc, EndLoc);
    break;
  case OMPC_copyin:
    Res = ActOnOpenMPCopyinClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_copyprivate:
    Res = ActOnOpenMPCopyprivateClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_flush:
    Res = ActOnOpenMPFlushClause(VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_depend:
    Res = ActOnOpenMPDependClause(DepKind, DepLinMapLoc, ColonLoc, VarList, 
                                  StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_map:
    Res = ActOnOpenMPMapClause(MapTypeModifier, MapType, DepLinMapLoc, ColonLoc,
                               VarList, StartLoc, LParenLoc, EndLoc);
    break;
  case OMPC_if:
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_collapse:
  case OMPC_default:
  case OMPC_proc_bind:
  case OMPC_schedule:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_threadprivate:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_seq_cst:
  case OMPC_device:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_nogroup:
  case OMPC_num_tasks:
  case OMPC_hint:
  case OMPC_unknown:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPPrivateClause(ArrayRef<Expr *> VarList,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> PrivateCopies;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP private clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      PrivateCopies.push_back(nullptr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.9.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      PrivateCopies.push_back(nullptr);
      continue;
    }

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.3]
    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_private_incomplete_type)) {
      continue;
    }
    Type = Type.getNonReferenceType();

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD, false);
    if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_private) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(DVar.CKind)
                                          << getOpenMPClauseName(OMPC_private);
      ReportOriginalDSA(*this, DSAStack, VD, DVar);
      continue;
    }

    // Variably modified types are not supported for tasks.
    if (!Type->isAnyPointerType() && Type->isVariablyModifiedType() &&
        DSAStack->getCurrentDirective() == OMPD_task) {
      Diag(ELoc, diag::err_omp_variably_modified_type_not_supported)
          << getOpenMPClauseName(OMPC_private) << Type
          << getOpenMPDirectiveName(DSAStack->getCurrentDirective());
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a private
    //  clause requires an accessible, unambiguous default constructor for the
    //  class type.
    // Generate helper private variable and initialize it with the default
    // value. The address of the original variable is replaced by the address of
    // the new private variable in CodeGen. This new variable is not added to
    // IdResolver, so the code in the OpenMP region uses original variable for
    // proper diagnostics.
    Type = Type.getUnqualifiedType();
    auto VDPrivate = buildVarDecl(*this, DE->getExprLoc(), Type, VD->getName(),
                                  VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    ActOnUninitializedDecl(VDPrivate, /*TypeMayContainAuto=*/false);
    if (VDPrivate->isInvalidDecl())
      continue;
    auto VDPrivateRefExpr = buildDeclRefExpr(
        *this, VDPrivate, DE->getType().getUnqualifiedType(), DE->getExprLoc());

    DSAStack->addDSA(VD, DE, OMPC_private);
    Vars.push_back(DE);
    PrivateCopies.push_back(VDPrivateRefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OMPPrivateClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars,
                                  PrivateCopies);
}

namespace {
class DiagsUninitializedSeveretyRAII {
private:
  DiagnosticsEngine &Diags;
  SourceLocation SavedLoc;
  bool IsIgnored;

public:
  DiagsUninitializedSeveretyRAII(DiagnosticsEngine &Diags, SourceLocation Loc,
                                 bool IsIgnored)
      : Diags(Diags), SavedLoc(Loc), IsIgnored(IsIgnored) {
    if (!IsIgnored) {
      Diags.setSeverity(/*Diag*/ diag::warn_uninit_self_reference_in_init,
                        /*Map*/ diag::Severity::Ignored, Loc);
    }
  }
  ~DiagsUninitializedSeveretyRAII() {
    if (!IsIgnored)
      Diags.popMappings(SavedLoc);
  }
};
}

OMPClause *Sema::ActOnOpenMPFirstprivateClause(ArrayRef<Expr *> VarList,
                                               SourceLocation StartLoc,
                                               SourceLocation LParenLoc,
                                               SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> PrivateCopies;
  SmallVector<Expr *, 8> Inits;
  bool IsImplicitClause =
      StartLoc.isInvalid() && LParenLoc.isInvalid() && EndLoc.isInvalid();
  auto ImplicitClauseLoc = DSAStack->getConstructLoc();

  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP firstprivate clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      PrivateCopies.push_back(nullptr);
      Inits.push_back(nullptr);
      continue;
    }

    SourceLocation ELoc =
        IsImplicitClause ? ImplicitClauseLoc : RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.9.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      PrivateCopies.push_back(nullptr);
      Inits.push_back(nullptr);
      continue;
    }

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.3]
    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_firstprivate_incomplete_type)) {
      continue;
    }
    Type = Type.getNonReferenceType();

    // OpenMP [2.9.3.4, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a private
    //  clause requires an accessible, unambiguous copy constructor for the
    //  class type.
    auto ElemType = Context.getBaseElementType(Type).getNonReferenceType();

    // If an implicit firstprivate variable found it was checked already.
    if (!IsImplicitClause) {
      DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD, false);
      bool IsConstant = ElemType.isConstant(Context);
      // OpenMP [2.4.13, Data-sharing Attribute Clauses]
      //  A list item that specifies a given variable may not appear in more
      // than one clause on the same directive, except that a variable may be
      //  specified in both firstprivate and lastprivate clauses.
      if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_firstprivate &&
          DVar.CKind != OMPC_lastprivate && DVar.RefExpr) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(DVar.CKind)
            << getOpenMPClauseName(OMPC_firstprivate);
        ReportOriginalDSA(*this, DSAStack, VD, DVar);
        continue;
      }

      // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
      // in a Construct]
      //  Variables with the predetermined data-sharing attributes may not be
      //  listed in data-sharing attributes clauses, except for the cases
      //  listed below. For these exceptions only, listing a predetermined
      //  variable in a data-sharing attribute clause is allowed and overrides
      //  the variable's predetermined data-sharing attributes.
      // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
      // in a Construct, C/C++, p.2]
      //  Variables with const-qualified type having no mutable member may be
      //  listed in a firstprivate clause, even if they are static data members.
      if (!(IsConstant || VD->isStaticDataMember()) && !DVar.RefExpr &&
          DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_shared) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(DVar.CKind)
            << getOpenMPClauseName(OMPC_firstprivate);
        ReportOriginalDSA(*this, DSAStack, VD, DVar);
        continue;
      }

      OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
      // OpenMP [2.9.3.4, Restrictions, p.2]
      //  A list item that is private within a parallel region must not appear
      //  in a firstprivate clause on a worksharing construct if any of the
      //  worksharing regions arising from the worksharing construct ever bind
      //  to any of the parallel regions arising from the parallel construct.
      if (isOpenMPWorksharingDirective(CurrDir) &&
          !isOpenMPParallelDirective(CurrDir)) {
        DVar = DSAStack->getImplicitDSA(VD, true);
        if (DVar.CKind != OMPC_shared &&
            (isOpenMPParallelDirective(DVar.DKind) ||
             DVar.DKind == OMPD_unknown)) {
          Diag(ELoc, diag::err_omp_required_access)
              << getOpenMPClauseName(OMPC_firstprivate)
              << getOpenMPClauseName(OMPC_shared);
          ReportOriginalDSA(*this, DSAStack, VD, DVar);
          continue;
        }
      }
      // OpenMP [2.9.3.4, Restrictions, p.3]
      //  A list item that appears in a reduction clause of a parallel construct
      //  must not appear in a firstprivate clause on a worksharing or task
      //  construct if any of the worksharing or task regions arising from the
      //  worksharing or task construct ever bind to any of the parallel regions
      //  arising from the parallel construct.
      // OpenMP [2.9.3.4, Restrictions, p.4]
      //  A list item that appears in a reduction clause in worksharing
      //  construct must not appear in a firstprivate clause in a task construct
      //  encountered during execution of any of the worksharing regions arising
      //  from the worksharing construct.
      if (CurrDir == OMPD_task) {
        DVar =
            DSAStack->hasInnermostDSA(VD, MatchesAnyClause(OMPC_reduction),
                                      [](OpenMPDirectiveKind K) -> bool {
                                        return isOpenMPParallelDirective(K) ||
                                               isOpenMPWorksharingDirective(K);
                                      },
                                      false);
        if (DVar.CKind == OMPC_reduction &&
            (isOpenMPParallelDirective(DVar.DKind) ||
             isOpenMPWorksharingDirective(DVar.DKind))) {
          Diag(ELoc, diag::err_omp_parallel_reduction_in_task_firstprivate)
              << getOpenMPDirectiveName(DVar.DKind);
          ReportOriginalDSA(*this, DSAStack, VD, DVar);
          continue;
        }
      }

      // OpenMP 4.5 [2.15.3.4, Restrictions, p.3]
      // A list item that is private within a teams region must not appear in a
      // firstprivate clause on a distribute construct if any of the distribute
      // regions arising from the distribute construct ever bind to any of the
      // teams regions arising from the teams construct.
      // OpenMP 4.5 [2.15.3.4, Restrictions, p.3]
      // A list item that appears in a reduction clause of a teams construct
      // must not appear in a firstprivate clause on a distribute construct if
      // any of the distribute regions arising from the distribute construct
      // ever bind to any of the teams regions arising from the teams construct.
      // OpenMP 4.5 [2.10.8, Distribute Construct, p.3]
      // A list item may appear in a firstprivate or lastprivate clause but not
      // both.
      if (CurrDir == OMPD_distribute) {
        DVar = DSAStack->hasInnermostDSA(VD, MatchesAnyClause(OMPC_private),
                                         [](OpenMPDirectiveKind K) -> bool {
                                           return isOpenMPTeamsDirective(K);
                                         },
                                         false);
        if (DVar.CKind == OMPC_private && isOpenMPTeamsDirective(DVar.DKind)) {
          Diag(ELoc, diag::err_omp_firstprivate_distribute_private_teams);
          ReportOriginalDSA(*this, DSAStack, VD, DVar);
          continue;
        }
        DVar = DSAStack->hasInnermostDSA(VD, MatchesAnyClause(OMPC_reduction),
                                         [](OpenMPDirectiveKind K) -> bool {
                                           return isOpenMPTeamsDirective(K);
                                         },
                                         false);
        if (DVar.CKind == OMPC_reduction &&
            isOpenMPTeamsDirective(DVar.DKind)) {
          Diag(ELoc, diag::err_omp_firstprivate_distribute_in_teams_reduction);
          ReportOriginalDSA(*this, DSAStack, VD, DVar);
          continue;
        }
        DVar = DSAStack->getTopDSA(VD, false);
        if (DVar.CKind == OMPC_lastprivate) {
          Diag(ELoc, diag::err_omp_firstprivate_and_lastprivate_in_distribute);
          ReportOriginalDSA(*this, DSAStack, VD, DVar);
          continue;
        }
      }
    }

    // Variably modified types are not supported for tasks.
    if (!Type->isAnyPointerType() && Type->isVariablyModifiedType() &&
        DSAStack->getCurrentDirective() == OMPD_task) {
      Diag(ELoc, diag::err_omp_variably_modified_type_not_supported)
          << getOpenMPClauseName(OMPC_firstprivate) << Type
          << getOpenMPDirectiveName(DSAStack->getCurrentDirective());
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    Type = Type.getUnqualifiedType();
    auto VDPrivate = buildVarDecl(*this, ELoc, Type, VD->getName(),
                                  VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    // Generate helper private variable and initialize it with the value of the
    // original variable. The address of the original variable is replaced by
    // the address of the new private variable in the CodeGen. This new variable
    // is not added to IdResolver, so the code in the OpenMP region uses
    // original variable for proper diagnostics and variable capturing.
    Expr *VDInitRefExpr = nullptr;
    // For arrays generate initializer for single element and replace it by the
    // original array element in CodeGen.
    if (Type->isArrayType()) {
      auto VDInit =
          buildVarDecl(*this, DE->getExprLoc(), ElemType, VD->getName());
      VDInitRefExpr = buildDeclRefExpr(*this, VDInit, ElemType, ELoc);
      auto Init = DefaultLvalueConversion(VDInitRefExpr).get();
      ElemType = ElemType.getUnqualifiedType();
      auto *VDInitTemp = buildVarDecl(*this, DE->getLocStart(), ElemType,
                                      ".firstprivate.temp");
      InitializedEntity Entity =
          InitializedEntity::InitializeVariable(VDInitTemp);
      InitializationKind Kind = InitializationKind::CreateCopy(ELoc, ELoc);

      InitializationSequence InitSeq(*this, Entity, Kind, Init);
      ExprResult Result = InitSeq.Perform(*this, Entity, Kind, Init);
      if (Result.isInvalid())
        VDPrivate->setInvalidDecl();
      else
        VDPrivate->setInit(Result.getAs<Expr>());
      // Remove temp variable declaration.
      Context.Deallocate(VDInitTemp);
    } else {
      auto *VDInit =
          buildVarDecl(*this, DE->getLocStart(), Type, ".firstprivate.temp");
      VDInitRefExpr =
          buildDeclRefExpr(*this, VDInit, DE->getType(), DE->getExprLoc());
      AddInitializerToDecl(VDPrivate,
                           DefaultLvalueConversion(VDInitRefExpr).get(),
                           /*DirectInit=*/false, /*TypeMayContainAuto=*/false);
    }
    if (VDPrivate->isInvalidDecl()) {
      if (IsImplicitClause) {
        Diag(DE->getExprLoc(),
             diag::note_omp_task_predetermined_firstprivate_here);
      }
      continue;
    }
    CurContext->addDecl(VDPrivate);
    auto VDPrivateRefExpr = buildDeclRefExpr(
        *this, VDPrivate, DE->getType().getUnqualifiedType(), DE->getExprLoc());
    DSAStack->addDSA(VD, DE, OMPC_firstprivate);
    Vars.push_back(DE);
    PrivateCopies.push_back(VDPrivateRefExpr);
    Inits.push_back(VDInitRefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OMPFirstprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                       Vars, PrivateCopies, Inits);
}

OMPClause *Sema::ActOnOpenMPLastprivateClause(ArrayRef<Expr *> VarList,
                                              SourceLocation StartLoc,
                                              SourceLocation LParenLoc,
                                              SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> SrcExprs;
  SmallVector<Expr *, 8> DstExprs;
  SmallVector<Expr *, 8> AssignmentOps;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP lastprivate clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      SrcExprs.push_back(nullptr);
      DstExprs.push_back(nullptr);
      AssignmentOps.push_back(nullptr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.3.5, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or structure
    //  element) cannot appear in a lastprivate clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      SrcExprs.push_back(nullptr);
      DstExprs.push_back(nullptr);
      AssignmentOps.push_back(nullptr);
      continue;
    }

    // OpenMP [2.14.3.5, Restrictions, C/C++, p.2]
    //  A variable that appears in a lastprivate clause must not have an
    //  incomplete type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_lastprivate_incomplete_type)) {
      continue;
    }
    Type = Type.getNonReferenceType();

    // OpenMP [2.14.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD, false);
    if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_lastprivate &&
        DVar.CKind != OMPC_firstprivate &&
        (DVar.CKind != OMPC_private || DVar.RefExpr != nullptr)) {
      Diag(ELoc, diag::err_omp_wrong_dsa)
          << getOpenMPClauseName(DVar.CKind)
          << getOpenMPClauseName(OMPC_lastprivate);
      ReportOriginalDSA(*this, DSAStack, VD, DVar);
      continue;
    }

    OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
    // OpenMP [2.14.3.5, Restrictions, p.2]
    // A list item that is private within a parallel region, or that appears in
    // the reduction clause of a parallel construct, must not appear in a
    // lastprivate clause on a worksharing construct if any of the corresponding
    // worksharing regions ever binds to any of the corresponding parallel
    // regions.
    DSAStackTy::DSAVarData TopDVar = DVar;
    if (isOpenMPWorksharingDirective(CurrDir) &&
        !isOpenMPParallelDirective(CurrDir)) {
      DVar = DSAStack->getImplicitDSA(VD, true);
      if (DVar.CKind != OMPC_shared) {
        Diag(ELoc, diag::err_omp_required_access)
            << getOpenMPClauseName(OMPC_lastprivate)
            << getOpenMPClauseName(OMPC_shared);
        ReportOriginalDSA(*this, DSAStack, VD, DVar);
        continue;
      }
    }
    // OpenMP [2.14.3.5, Restrictions, C++, p.1,2]
    //  A variable of class type (or array thereof) that appears in a
    //  lastprivate clause requires an accessible, unambiguous default
    //  constructor for the class type, unless the list item is also specified
    //  in a firstprivate clause.
    //  A variable of class type (or array thereof) that appears in a
    //  lastprivate clause requires an accessible, unambiguous copy assignment
    //  operator for the class type.
    Type = Context.getBaseElementType(Type).getNonReferenceType();
    auto *SrcVD = buildVarDecl(*this, DE->getLocStart(),
                               Type.getUnqualifiedType(), ".lastprivate.src",
                               VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    auto *PseudoSrcExpr = buildDeclRefExpr(
        *this, SrcVD, Type.getUnqualifiedType(), DE->getExprLoc());
    auto *DstVD =
        buildVarDecl(*this, DE->getLocStart(), Type, ".lastprivate.dst",
                     VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    auto *PseudoDstExpr =
        buildDeclRefExpr(*this, DstVD, Type, DE->getExprLoc());
    // For arrays generate assignment operation for single element and replace
    // it by the original array element in CodeGen.
    auto AssignmentOp = BuildBinOp(/*S=*/nullptr, DE->getExprLoc(), BO_Assign,
                                   PseudoDstExpr, PseudoSrcExpr);
    if (AssignmentOp.isInvalid())
      continue;
    AssignmentOp = ActOnFinishFullExpr(AssignmentOp.get(), DE->getExprLoc(),
                                       /*DiscardedValue=*/true);
    if (AssignmentOp.isInvalid())
      continue;

    // OpenMP 4.5 [2.10.8, Distribute Construct, p.3]
    // A list item may appear in a firstprivate or lastprivate clause but not
    // both.
    if (CurrDir == OMPD_distribute) {
      DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD, false);
      if (DVar.CKind == OMPC_firstprivate) {
        Diag(ELoc, diag::err_omp_firstprivate_and_lastprivate_in_distribute);
        ReportOriginalDSA(*this, DSAStack, VD, DVar);
        continue;
      }
    }

    if (TopDVar.CKind != OMPC_firstprivate)
      DSAStack->addDSA(VD, DE, OMPC_lastprivate);
    Vars.push_back(DE);
    SrcExprs.push_back(PseudoSrcExpr);
    DstExprs.push_back(PseudoDstExpr);
    AssignmentOps.push_back(AssignmentOp.get());
  }

  if (Vars.empty())
    return nullptr;

  return OMPLastprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                      Vars, SrcExprs, DstExprs, AssignmentOps);
}

OMPClause *Sema::ActOnOpenMPSharedClause(ArrayRef<Expr *> VarList,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP shared clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.3.2, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or structure
    //  element) cannot appear in a shared unless it is a static data member
    //  of a C++ class.
    DeclRefExpr *DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      continue;
    }

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD, false);
    if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_shared &&
        DVar.RefExpr) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(DVar.CKind)
                                          << getOpenMPClauseName(OMPC_shared);
      ReportOriginalDSA(*this, DSAStack, VD, DVar);
      continue;
    }

    DSAStack->addDSA(VD, DE, OMPC_shared);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return nullptr;

  return OMPSharedClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars);
}

namespace {
class DSARefChecker : public StmtVisitor<DSARefChecker, bool> {
  DSAStackTy *Stack;

public:
  bool VisitDeclRefExpr(DeclRefExpr *E) {
    if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
      DSAStackTy::DSAVarData DVar = Stack->getTopDSA(VD, false);
      if (DVar.CKind == OMPC_shared && !DVar.RefExpr)
        return false;
      if (DVar.CKind != OMPC_unknown)
        return true;
      DSAStackTy::DSAVarData DVarPrivate =
          Stack->hasDSA(VD, isOpenMPPrivate, MatchesAlways(), false);
      if (DVarPrivate.CKind != OMPC_unknown)
        return true;
      return false;
    }
    return false;
  }
  bool VisitStmt(Stmt *S) {
    for (auto Child : S->children()) {
      if (Child && Visit(Child))
        return true;
    }
    return false;
  }
  explicit DSARefChecker(DSAStackTy *S) : Stack(S) {}
};
} // namespace

OMPClause *Sema::ActOnOpenMPReductionClause(
    ArrayRef<Expr *> VarList, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation EndLoc,
    CXXScopeSpec &ReductionIdScopeSpec,
    const DeclarationNameInfo &ReductionId) {
  // TODO: Allow scope specification search when 'declare reduction' is
  // supported.
  assert(ReductionIdScopeSpec.isEmpty() &&
         "No support for scoped reduction identifiers yet.");

  auto DN = ReductionId.getName();
  auto OOK = DN.getCXXOverloadedOperator();
  BinaryOperatorKind BOK = BO_Comma;

  // OpenMP [2.14.3.6, reduction clause]
  // C
  // reduction-identifier is either an identifier or one of the following
  // operators: +, -, *,  &, |, ^, && and ||
  // C++
  // reduction-identifier is either an id-expression or one of the following
  // operators: +, -, *, &, |, ^, && and ||
  // FIXME: Only 'min' and 'max' identifiers are supported for now.
  switch (OOK) {
  case OO_Plus:
  case OO_Minus:
    BOK = BO_Add;
    break;
  case OO_Star:
    BOK = BO_Mul;
    break;
  case OO_Amp:
    BOK = BO_And;
    break;
  case OO_Pipe:
    BOK = BO_Or;
    break;
  case OO_Caret:
    BOK = BO_Xor;
    break;
  case OO_AmpAmp:
    BOK = BO_LAnd;
    break;
  case OO_PipePipe:
    BOK = BO_LOr;
    break;
  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_Slash:
  case OO_Percent:
  case OO_Tilde:
  case OO_Exclaim:
  case OO_Equal:
  case OO_Less:
  case OO_Greater:
  case OO_LessEqual:
  case OO_GreaterEqual:
  case OO_PlusEqual:
  case OO_MinusEqual:
  case OO_StarEqual:
  case OO_SlashEqual:
  case OO_PercentEqual:
  case OO_CaretEqual:
  case OO_AmpEqual:
  case OO_PipeEqual:
  case OO_LessLess:
  case OO_GreaterGreater:
  case OO_LessLessEqual:
  case OO_GreaterGreaterEqual:
  case OO_EqualEqual:
  case OO_ExclaimEqual:
  case OO_PlusPlus:
  case OO_MinusMinus:
  case OO_Comma:
  case OO_ArrowStar:
  case OO_Arrow:
  case OO_Call:
  case OO_Subscript:
  case OO_Conditional:
  case OO_Coawait:
  case NUM_OVERLOADED_OPERATORS:
    llvm_unreachable("Unexpected reduction identifier");
  case OO_None:
    if (auto II = DN.getAsIdentifierInfo()) {
      if (II->isStr("max"))
        BOK = BO_GT;
      else if (II->isStr("min"))
        BOK = BO_LT;
    }
    break;
  }
  SourceRange ReductionIdRange;
  if (ReductionIdScopeSpec.isValid()) {
    ReductionIdRange.setBegin(ReductionIdScopeSpec.getBeginLoc());
  }
  ReductionIdRange.setEnd(ReductionId.getEndLoc());
  if (BOK == BO_Comma) {
    // Not allowed reduction identifier is found.
    Diag(ReductionId.getLocStart(), diag::err_omp_unknown_reduction_identifier)
        << ReductionIdRange;
    return nullptr;
  }

  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> Privates;
  SmallVector<Expr *, 8> LHSs;
  SmallVector<Expr *, 8> RHSs;
  SmallVector<Expr *, 8> ReductionOps;
  for (auto RefExpr : VarList) {
    assert(RefExpr && "nullptr expr in OpenMP reduction clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      Privates.push_back(nullptr);
      LHSs.push_back(nullptr);
      RHSs.push_back(nullptr);
      ReductionOps.push_back(nullptr);
      continue;
    }

    if (RefExpr->isTypeDependent() || RefExpr->isValueDependent() ||
        RefExpr->isInstantiationDependent() ||
        RefExpr->containsUnexpandedParameterPack()) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      Privates.push_back(nullptr);
      LHSs.push_back(nullptr);
      RHSs.push_back(nullptr);
      ReductionOps.push_back(nullptr);
      continue;
    }

    auto ELoc = RefExpr->getExprLoc();
    auto ERange = RefExpr->getSourceRange();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable or array section, subject to the restrictions
    //  specified in Section 2.4 on page 42 and in each of the sections
    // describing clauses and directives for which a list appears.
    // OpenMP  [2.14.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    auto *DE = dyn_cast<DeclRefExpr>(RefExpr);
    auto *ASE = dyn_cast<ArraySubscriptExpr>(RefExpr);
    auto *OASE = dyn_cast<OMPArraySectionExpr>(RefExpr);
    if (!ASE && !OASE && (!DE || !isa<VarDecl>(DE->getDecl()))) {
      Diag(ELoc, diag::err_omp_expected_var_name_or_array_item) << ERange;
      continue;
    }
    QualType Type;
    VarDecl *VD = nullptr;
    if (DE) {
      auto D = DE->getDecl();
      VD = cast<VarDecl>(D);
      Type = VD->getType();
    } else if (ASE) {
      Type = ASE->getType();
      auto *Base = ASE->getBase()->IgnoreParenImpCasts();
      while (auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
        Base = TempASE->getBase()->IgnoreParenImpCasts();
      DE = dyn_cast<DeclRefExpr>(Base);
      if (DE)
        VD = dyn_cast<VarDecl>(DE->getDecl());
      if (!VD) {
        Diag(Base->getExprLoc(), diag::err_omp_expected_base_var_name)
            << 0 << Base->getSourceRange();
        continue;
      }
    } else if (OASE) {
      auto BaseType = OMPArraySectionExpr::getBaseOriginalType(OASE->getBase());
      if (auto *ATy = BaseType->getAsArrayTypeUnsafe())
        Type = ATy->getElementType();
      else
        Type = BaseType->getPointeeType();
      auto *Base = OASE->getBase()->IgnoreParenImpCasts();
      while (auto *TempOASE = dyn_cast<OMPArraySectionExpr>(Base))
        Base = TempOASE->getBase()->IgnoreParenImpCasts();
      while (auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
        Base = TempASE->getBase()->IgnoreParenImpCasts();
      DE = dyn_cast<DeclRefExpr>(Base);
      if (DE)
        VD = dyn_cast<VarDecl>(DE->getDecl());
      if (!VD) {
        Diag(Base->getExprLoc(), diag::err_omp_expected_base_var_name)
            << 1 << Base->getSourceRange();
        continue;
      }
    }

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.3]
    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_reduction_incomplete_type))
      continue;
    // OpenMP [2.14.3.6, reduction clause, Restrictions]
    // Arrays may not appear in a reduction clause.
    if (Type.getNonReferenceType()->isArrayType()) {
      Diag(ELoc, diag::err_omp_reduction_type_array) << Type << ERange;
      if (!ASE && !OASE) {
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
      }
      continue;
    }
    // OpenMP [2.14.3.6, reduction clause, Restrictions]
    // A list item that appears in a reduction clause must not be
    // const-qualified.
    if (Type.getNonReferenceType().isConstant(Context)) {
      Diag(ELoc, diag::err_omp_const_reduction_list_item)
          << getOpenMPClauseName(OMPC_reduction) << Type << ERange;
      if (!ASE && !OASE) {
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
      }
      continue;
    }
    // OpenMP [2.9.3.6, Restrictions, C/C++, p.4]
    //  If a list-item is a reference type then it must bind to the same object
    //  for all threads of the team.
    if (!ASE && !OASE) {
      VarDecl *VDDef = VD->getDefinition();
      if (Type->isReferenceType() && VDDef) {
        DSARefChecker Check(DSAStack);
        if (Check.Visit(VDDef->getInit())) {
          Diag(ELoc, diag::err_omp_reduction_ref_type_arg) << ERange;
          Diag(VDDef->getLocation(), diag::note_defined_here) << VDDef;
          continue;
        }
      }
    }
    // OpenMP [2.14.3.6, reduction clause, Restrictions]
    // The type of a list item that appears in a reduction clause must be valid
    // for the reduction-identifier. For a max or min reduction in C, the type
    // of the list item must be an allowed arithmetic data type: char, int,
    // float, double, or _Bool, possibly modified with long, short, signed, or
    // unsigned. For a max or min reduction in C++, the type of the list item
    // must be an allowed arithmetic data type: char, wchar_t, int, float,
    // double, or bool, possibly modified with long, short, signed, or unsigned.
    if ((BOK == BO_GT || BOK == BO_LT) &&
        !(Type->isScalarType() ||
          (getLangOpts().CPlusPlus && Type->isArithmeticType()))) {
      Diag(ELoc, diag::err_omp_clause_not_arithmetic_type_arg)
          << getLangOpts().CPlusPlus;
      if (!ASE && !OASE) {
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
      }
      continue;
    }
    if ((BOK == BO_OrAssign || BOK == BO_AndAssign || BOK == BO_XorAssign) &&
        !getLangOpts().CPlusPlus && Type->isFloatingType()) {
      Diag(ELoc, diag::err_omp_clause_floating_type_arg);
      if (!ASE && !OASE) {
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
      }
      continue;
    }
    // OpenMP [2.14.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    // OpenMP [2.14.3.6, Restrictions, p.3]
    //  Any number of reduction clauses can be specified on the directive,
    //  but a list item can appear only once in the reduction clauses for that
    //  directive.
    DSAStackTy::DSAVarData DVar;
    DVar = DSAStack->getTopDSA(VD, false);
    if (DVar.CKind == OMPC_reduction) {
      Diag(ELoc, diag::err_omp_once_referenced)
          << getOpenMPClauseName(OMPC_reduction);
      if (DVar.RefExpr) {
        Diag(DVar.RefExpr->getExprLoc(), diag::note_omp_referenced);
      }
    } else if (DVar.CKind != OMPC_unknown) {
      Diag(ELoc, diag::err_omp_wrong_dsa)
          << getOpenMPClauseName(DVar.CKind)
          << getOpenMPClauseName(OMPC_reduction);
      ReportOriginalDSA(*this, DSAStack, VD, DVar);
      continue;
    }

    // OpenMP [2.14.3.6, Restrictions, p.1]
    //  A list item that appears in a reduction clause of a worksharing
    //  construct must be shared in the parallel regions to which any of the
    //  worksharing regions arising from the worksharing construct bind.
    OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
    if (isOpenMPWorksharingDirective(CurrDir) &&
        !isOpenMPParallelDirective(CurrDir)) {
      DVar = DSAStack->getImplicitDSA(VD, true);
      if (DVar.CKind != OMPC_shared) {
        Diag(ELoc, diag::err_omp_required_access)
            << getOpenMPClauseName(OMPC_reduction)
            << getOpenMPClauseName(OMPC_shared);
        ReportOriginalDSA(*this, DSAStack, VD, DVar);
        continue;
      }
    }

    Type = Type.getNonLValueExprType(Context).getUnqualifiedType();
    auto *LHSVD = buildVarDecl(*this, ELoc, Type, ".reduction.lhs",
                               VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    auto *RHSVD = buildVarDecl(*this, ELoc, Type, VD->getName(),
                               VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    auto PrivateTy = Type;
    if (OASE) {
      // For array sections only:
      // Create pseudo array type for private copy. The size for this array will
      // be generated during codegen.
      // For array subscripts or single variables Private Ty is the same as Type
      // (type of the variable or single array element).
      PrivateTy = Context.getVariableArrayType(
          Type, new (Context) OpaqueValueExpr(SourceLocation(),
                                              Context.getSizeType(), VK_RValue),
          ArrayType::Normal, /*IndexTypeQuals=*/0, SourceRange());
    }
    // Private copy.
    auto *PrivateVD = buildVarDecl(*this, ELoc, PrivateTy, VD->getName(),
                                   VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    // Add initializer for private variable.
    Expr *Init = nullptr;
    switch (BOK) {
    case BO_Add:
    case BO_Xor:
    case BO_Or:
    case BO_LOr:
      // '+', '-', '^', '|', '||' reduction ops - initializer is '0'.
      if (Type->isScalarType() || Type->isAnyComplexType()) {
        Init = ActOnIntegerConstant(ELoc, /*Val=*/0).get();
      }
      break;
    case BO_Mul:
    case BO_LAnd:
      if (Type->isScalarType() || Type->isAnyComplexType()) {
        // '*' and '&&' reduction ops - initializer is '1'.
        Init = ActOnIntegerConstant(ELoc, /*Val=*/1).get();
      }
      break;
    case BO_And: {
      // '&' reduction op - initializer is '~0'.
      QualType OrigType = Type;
      if (auto *ComplexTy = OrigType->getAs<ComplexType>()) {
        Type = ComplexTy->getElementType();
      }
      if (Type->isRealFloatingType()) {
        llvm::APFloat InitValue =
            llvm::APFloat::getAllOnesValue(Context.getTypeSize(Type),
                                           /*isIEEE=*/true);
        Init = FloatingLiteral::Create(Context, InitValue, /*isexact=*/true,
                                       Type, ELoc);
      } else if (Type->isScalarType()) {
        auto Size = Context.getTypeSize(Type);
        QualType IntTy = Context.getIntTypeForBitwidth(Size, /*Signed=*/0);
        llvm::APInt InitValue = llvm::APInt::getAllOnesValue(Size);
        Init = IntegerLiteral::Create(Context, InitValue, IntTy, ELoc);
      }
      if (Init && OrigType->isAnyComplexType()) {
        // Init = 0xFFFF + 0xFFFFi;
        auto *Im = new (Context) ImaginaryLiteral(Init, OrigType);
        Init = CreateBuiltinBinOp(ELoc, BO_Add, Init, Im).get();
      }
      Type = OrigType;
      break;
    }
    case BO_LT:
    case BO_GT: {
      // 'min' reduction op - initializer is 'Largest representable number in
      // the reduction list item type'.
      // 'max' reduction op - initializer is 'Least representable number in
      // the reduction list item type'.
      if (Type->isIntegerType() || Type->isPointerType()) {
        bool IsSigned = Type->hasSignedIntegerRepresentation();
        auto Size = Context.getTypeSize(Type);
        QualType IntTy =
            Context.getIntTypeForBitwidth(Size, /*Signed=*/IsSigned);
        llvm::APInt InitValue =
            (BOK != BO_LT)
                ? IsSigned ? llvm::APInt::getSignedMinValue(Size)
                           : llvm::APInt::getMinValue(Size)
                : IsSigned ? llvm::APInt::getSignedMaxValue(Size)
                           : llvm::APInt::getMaxValue(Size);
        Init = IntegerLiteral::Create(Context, InitValue, IntTy, ELoc);
        if (Type->isPointerType()) {
          // Cast to pointer type.
          auto CastExpr = BuildCStyleCastExpr(
              SourceLocation(), Context.getTrivialTypeSourceInfo(Type, ELoc),
              SourceLocation(), Init);
          if (CastExpr.isInvalid())
            continue;
          Init = CastExpr.get();
        }
      } else if (Type->isRealFloatingType()) {
        llvm::APFloat InitValue = llvm::APFloat::getLargest(
            Context.getFloatTypeSemantics(Type), BOK != BO_LT);
        Init = FloatingLiteral::Create(Context, InitValue, /*isexact=*/true,
                                       Type, ELoc);
      }
      break;
    }
    case BO_PtrMemD:
    case BO_PtrMemI:
    case BO_MulAssign:
    case BO_Div:
    case BO_Rem:
    case BO_Sub:
    case BO_Shl:
    case BO_Shr:
    case BO_LE:
    case BO_GE:
    case BO_EQ:
    case BO_NE:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
    case BO_Assign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_ShlAssign:
    case BO_ShrAssign:
    case BO_Comma:
      llvm_unreachable("Unexpected reduction operation");
    }
    if (Init) {
      AddInitializerToDecl(RHSVD, Init, /*DirectInit=*/false,
                           /*TypeMayContainAuto=*/false);
    } else
      ActOnUninitializedDecl(RHSVD, /*TypeMayContainAuto=*/false);
    if (!RHSVD->hasInit()) {
      Diag(ELoc, diag::err_omp_reduction_id_not_compatible) << Type
                                                            << ReductionIdRange;
      if (VD) {
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
      }
      continue;
    }
    // Store initializer for single element in private copy. Will be used during
    // codegen.
    PrivateVD->setInit(RHSVD->getInit());
    PrivateVD->setInitStyle(RHSVD->getInitStyle());
    auto *LHSDRE = buildDeclRefExpr(*this, LHSVD, Type, ELoc);
    auto *RHSDRE = buildDeclRefExpr(*this, RHSVD, Type, ELoc);
    auto *PrivateDRE = buildDeclRefExpr(*this, PrivateVD, PrivateTy, ELoc);
    ExprResult ReductionOp =
        BuildBinOp(DSAStack->getCurScope(), ReductionId.getLocStart(), BOK,
                   LHSDRE, RHSDRE);
    if (ReductionOp.isUsable()) {
      if (BOK != BO_LT && BOK != BO_GT) {
        ReductionOp =
            BuildBinOp(DSAStack->getCurScope(), ReductionId.getLocStart(),
                       BO_Assign, LHSDRE, ReductionOp.get());
      } else {
        auto *ConditionalOp = new (Context) ConditionalOperator(
            ReductionOp.get(), SourceLocation(), LHSDRE, SourceLocation(),
            RHSDRE, Type, VK_LValue, OK_Ordinary);
        ReductionOp =
            BuildBinOp(DSAStack->getCurScope(), ReductionId.getLocStart(),
                       BO_Assign, LHSDRE, ConditionalOp);
      }
      ReductionOp = ActOnFinishFullExpr(ReductionOp.get());
    }
    if (ReductionOp.isInvalid())
      continue;

    DSAStack->addDSA(VD, DE, OMPC_reduction);
    Vars.push_back(RefExpr);
    Privates.push_back(PrivateDRE);
    LHSs.push_back(LHSDRE);
    RHSs.push_back(RHSDRE);
    ReductionOps.push_back(ReductionOp.get());
  }

  if (Vars.empty())
    return nullptr;

  return OMPReductionClause::Create(
      Context, StartLoc, LParenLoc, ColonLoc, EndLoc, Vars,
      ReductionIdScopeSpec.getWithLocInContext(Context), ReductionId, Privates,
      LHSs, RHSs, ReductionOps);
}

OMPClause *Sema::ActOnOpenMPLinearClause(
    ArrayRef<Expr *> VarList, Expr *Step, SourceLocation StartLoc,
    SourceLocation LParenLoc, OpenMPLinearClauseKind LinKind,
    SourceLocation LinLoc, SourceLocation ColonLoc, SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> Privates;
  SmallVector<Expr *, 8> Inits;
  if ((!LangOpts.CPlusPlus && LinKind != OMPC_LINEAR_val) ||
      LinKind == OMPC_LINEAR_unknown) {
    Diag(LinLoc, diag::err_omp_wrong_linear_modifier) << LangOpts.CPlusPlus;
    LinKind = OMPC_LINEAR_val;
  }
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP linear clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      Privates.push_back(nullptr);
      Inits.push_back(nullptr);
      continue;
    }

    // OpenMP [2.14.3.7, linear clause]
    // A list item that appears in a linear clause is subject to the private
    // clause semantics described in Section 2.14.3.3 on page 159 except as
    // noted. In addition, the value of the new list item on each iteration
    // of the associated loop(s) corresponds to the value of the original
    // list item before entering the construct plus the logical number of
    // the iteration times linear-step.

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
      continue;
    }

    VarDecl *VD = cast<VarDecl>(DE->getDecl());

    // OpenMP [2.14.3.7, linear clause]
    //  A list-item cannot appear in more than one linear clause.
    //  A list-item that appears in a linear clause cannot appear in any
    //  other data-sharing attribute clause.
    DSAStackTy::DSAVarData DVar = DSAStack->getTopDSA(VD, false);
    if (DVar.RefExpr) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(DVar.CKind)
                                          << getOpenMPClauseName(OMPC_linear);
      ReportOriginalDSA(*this, DSAStack, VD, DVar);
      continue;
    }

    QualType QType = VD->getType();
    if (QType->isDependentType() || QType->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      Privates.push_back(nullptr);
      Inits.push_back(nullptr);
      continue;
    }

    // A variable must not have an incomplete type or a reference type.
    if (RequireCompleteType(ELoc, QType,
                            diag::err_omp_linear_incomplete_type)) {
      continue;
    }
    if ((LinKind == OMPC_LINEAR_uval || LinKind == OMPC_LINEAR_ref) &&
        !QType->isReferenceType()) {
      Diag(ELoc, diag::err_omp_wrong_linear_modifier_non_reference)
          << QType << getOpenMPSimpleClauseTypeName(OMPC_linear, LinKind);
      continue;
    }
    QType = QType.getNonReferenceType();

    // A list item must not be const-qualified.
    if (QType.isConstant(Context)) {
      Diag(ELoc, diag::err_omp_const_variable)
          << getOpenMPClauseName(OMPC_linear);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // A list item must be of integral or pointer type.
    QType = QType.getUnqualifiedType().getCanonicalType();
    const Type *Ty = QType.getTypePtrOrNull();
    if (!Ty || (!Ty->isDependentType() && !Ty->isIntegralType(Context) &&
                !Ty->isPointerType())) {
      Diag(ELoc, diag::err_omp_linear_expected_int_or_ptr) << QType;
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // Build private copy of original var.
    auto *Private = buildVarDecl(*this, ELoc, QType, VD->getName(),
                                 VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    auto *PrivateRef = buildDeclRefExpr(
        *this, Private, DE->getType().getUnqualifiedType(), DE->getExprLoc());
    // Build var to save initial value.
    VarDecl *Init = buildVarDecl(*this, ELoc, QType, ".linear.start");
    Expr *InitExpr;
    if (LinKind == OMPC_LINEAR_uval)
      InitExpr = VD->getInit();
    else
      InitExpr = DE;
    AddInitializerToDecl(Init, DefaultLvalueConversion(InitExpr).get(),
                         /*DirectInit*/ false, /*TypeMayContainAuto*/ false);
    auto InitRef = buildDeclRefExpr(
        *this, Init, DE->getType().getUnqualifiedType(), DE->getExprLoc());
    DSAStack->addDSA(VD, DE, OMPC_linear);
    Vars.push_back(DE);
    Privates.push_back(PrivateRef);
    Inits.push_back(InitRef);
  }

  if (Vars.empty())
    return nullptr;

  Expr *StepExpr = Step;
  Expr *CalcStepExpr = nullptr;
  if (Step && !Step->isValueDependent() && !Step->isTypeDependent() &&
      !Step->isInstantiationDependent() &&
      !Step->containsUnexpandedParameterPack()) {
    SourceLocation StepLoc = Step->getLocStart();
    ExprResult Val = PerformOpenMPImplicitIntegerConversion(StepLoc, Step);
    if (Val.isInvalid())
      return nullptr;
    StepExpr = Val.get();

    // Build var to save the step value.
    VarDecl *SaveVar =
        buildVarDecl(*this, StepLoc, StepExpr->getType(), ".linear.step");
    ExprResult SaveRef =
        buildDeclRefExpr(*this, SaveVar, StepExpr->getType(), StepLoc);
    ExprResult CalcStep =
        BuildBinOp(CurScope, StepLoc, BO_Assign, SaveRef.get(), StepExpr);
    CalcStep = ActOnFinishFullExpr(CalcStep.get());

    // Warn about zero linear step (it would be probably better specified as
    // making corresponding variables 'const').
    llvm::APSInt Result;
    bool IsConstant = StepExpr->isIntegerConstantExpr(Result, Context);
    if (IsConstant && !Result.isNegative() && !Result.isStrictlyPositive())
      Diag(StepLoc, diag::warn_omp_linear_step_zero) << Vars[0]
                                                     << (Vars.size() > 1);
    if (!IsConstant && CalcStep.isUsable()) {
      // Calculate the step beforehand instead of doing this on each iteration.
      // (This is not used if the number of iterations may be kfold-ed).
      CalcStepExpr = CalcStep.get();
    }
  }

  return OMPLinearClause::Create(Context, StartLoc, LParenLoc, LinKind, LinLoc,
                                 ColonLoc, EndLoc, Vars, Privates, Inits,
                                 StepExpr, CalcStepExpr);
}

static bool FinishOpenMPLinearClause(OMPLinearClause &Clause, DeclRefExpr *IV,
                                     Expr *NumIterations, Sema &SemaRef,
                                     Scope *S) {
  // Walk the vars and build update/final expressions for the CodeGen.
  SmallVector<Expr *, 8> Updates;
  SmallVector<Expr *, 8> Finals;
  Expr *Step = Clause.getStep();
  Expr *CalcStep = Clause.getCalcStep();
  // OpenMP [2.14.3.7, linear clause]
  // If linear-step is not specified it is assumed to be 1.
  if (Step == nullptr)
    Step = SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get();
  else if (CalcStep)
    Step = cast<BinaryOperator>(CalcStep)->getLHS();
  bool HasErrors = false;
  auto CurInit = Clause.inits().begin();
  auto CurPrivate = Clause.privates().begin();
  auto LinKind = Clause.getModifier();
  for (auto &RefExpr : Clause.varlists()) {
    Expr *InitExpr = *CurInit;

    // Build privatized reference to the current linear var.
    auto DE = cast<DeclRefExpr>(RefExpr);
    Expr *CapturedRef;
    if (LinKind == OMPC_LINEAR_uval)
      CapturedRef = cast<VarDecl>(DE->getDecl())->getInit();
    else
      CapturedRef =
          buildDeclRefExpr(SemaRef, cast<VarDecl>(DE->getDecl()),
                           DE->getType().getUnqualifiedType(), DE->getExprLoc(),
                           /*RefersToCapture=*/true);

    // Build update: Var = InitExpr + IV * Step
    ExprResult Update =
        BuildCounterUpdate(SemaRef, S, RefExpr->getExprLoc(), *CurPrivate,
                           InitExpr, IV, Step, /* Subtract */ false);
    Update = SemaRef.ActOnFinishFullExpr(Update.get(), DE->getLocStart(),
                                         /*DiscardedValue=*/true);

    // Build final: Var = InitExpr + NumIterations * Step
    ExprResult Final =
        BuildCounterUpdate(SemaRef, S, RefExpr->getExprLoc(), CapturedRef,
                           InitExpr, NumIterations, Step, /* Subtract */ false);
    Final = SemaRef.ActOnFinishFullExpr(Final.get(), DE->getLocStart(),
                                        /*DiscardedValue=*/true);
    if (!Update.isUsable() || !Final.isUsable()) {
      Updates.push_back(nullptr);
      Finals.push_back(nullptr);
      HasErrors = true;
    } else {
      Updates.push_back(Update.get());
      Finals.push_back(Final.get());
    }
    ++CurInit, ++CurPrivate;
  }
  Clause.setUpdates(Updates);
  Clause.setFinals(Finals);
  return HasErrors;
}

OMPClause *Sema::ActOnOpenMPAlignedClause(
    ArrayRef<Expr *> VarList, Expr *Alignment, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation ColonLoc, SourceLocation EndLoc) {

  SmallVector<Expr *, 8> Vars;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP aligned clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    DeclRefExpr *DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
      continue;
    }

    VarDecl *VD = cast<VarDecl>(DE->getDecl());

    // OpenMP  [2.8.1, simd construct, Restrictions]
    // The type of list items appearing in the aligned clause must be
    // array, pointer, reference to array, or reference to pointer.
    QualType QType = VD->getType();
    QType = QType.getNonReferenceType().getUnqualifiedType().getCanonicalType();
    const Type *Ty = QType.getTypePtrOrNull();
    if (!Ty || (!Ty->isDependentType() && !Ty->isArrayType() &&
                !Ty->isPointerType())) {
      Diag(ELoc, diag::err_omp_aligned_expected_array_or_ptr)
          << QType << getLangOpts().CPlusPlus << RefExpr->getSourceRange();
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP  [2.8.1, simd construct, Restrictions]
    // A list-item cannot appear in more than one aligned clause.
    if (DeclRefExpr *PrevRef = DSAStack->addUniqueAligned(VD, DE)) {
      Diag(ELoc, diag::err_omp_aligned_twice) << RefExpr->getSourceRange();
      Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
          << getOpenMPClauseName(OMPC_aligned);
      continue;
    }

    Vars.push_back(DE);
  }

  // OpenMP [2.8.1, simd construct, Description]
  // The parameter of the aligned clause, alignment, must be a constant
  // positive integer expression.
  // If no optional parameter is specified, implementation-defined default
  // alignments for SIMD instructions on the target platforms are assumed.
  if (Alignment != nullptr) {
    ExprResult AlignResult =
        VerifyPositiveIntegerConstantInClause(Alignment, OMPC_aligned);
    if (AlignResult.isInvalid())
      return nullptr;
    Alignment = AlignResult.get();
  }
  if (Vars.empty())
    return nullptr;

  return OMPAlignedClause::Create(Context, StartLoc, LParenLoc, ColonLoc,
                                  EndLoc, Vars, Alignment);
}

OMPClause *Sema::ActOnOpenMPCopyinClause(ArrayRef<Expr *> VarList,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> SrcExprs;
  SmallVector<Expr *, 8> DstExprs;
  SmallVector<Expr *, 8> AssignmentOps;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP copyin clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      SrcExprs.push_back(nullptr);
      DstExprs.push_back(nullptr);
      AssignmentOps.push_back(nullptr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.4.1, Restrictions, p.1]
    //  A list item that appears in a copyin clause must be threadprivate.
    DeclRefExpr *DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
      continue;
    }

    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      SrcExprs.push_back(nullptr);
      DstExprs.push_back(nullptr);
      AssignmentOps.push_back(nullptr);
      continue;
    }

    // OpenMP [2.14.4.1, Restrictions, C/C++, p.1]
    //  A list item that appears in a copyin clause must be threadprivate.
    if (!DSAStack->isThreadPrivate(VD)) {
      Diag(ELoc, diag::err_omp_required_access)
          << getOpenMPClauseName(OMPC_copyin)
          << getOpenMPDirectiveName(OMPD_threadprivate);
      continue;
    }

    // OpenMP [2.14.4.1, Restrictions, C/C++, p.2]
    //  A variable of class type (or array thereof) that appears in a
    //  copyin clause requires an accessible, unambiguous copy assignment
    //  operator for the class type.
    auto ElemType = Context.getBaseElementType(Type).getNonReferenceType();
    auto *SrcVD =
        buildVarDecl(*this, DE->getLocStart(), ElemType.getUnqualifiedType(),
                     ".copyin.src", VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    auto *PseudoSrcExpr = buildDeclRefExpr(
        *this, SrcVD, ElemType.getUnqualifiedType(), DE->getExprLoc());
    auto *DstVD =
        buildVarDecl(*this, DE->getLocStart(), ElemType, ".copyin.dst",
                     VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    auto *PseudoDstExpr =
        buildDeclRefExpr(*this, DstVD, ElemType, DE->getExprLoc());
    // For arrays generate assignment operation for single element and replace
    // it by the original array element in CodeGen.
    auto AssignmentOp = BuildBinOp(/*S=*/nullptr, DE->getExprLoc(), BO_Assign,
                                   PseudoDstExpr, PseudoSrcExpr);
    if (AssignmentOp.isInvalid())
      continue;
    AssignmentOp = ActOnFinishFullExpr(AssignmentOp.get(), DE->getExprLoc(),
                                       /*DiscardedValue=*/true);
    if (AssignmentOp.isInvalid())
      continue;

    DSAStack->addDSA(VD, DE, OMPC_copyin);
    Vars.push_back(DE);
    SrcExprs.push_back(PseudoSrcExpr);
    DstExprs.push_back(PseudoDstExpr);
    AssignmentOps.push_back(AssignmentOp.get());
  }

  if (Vars.empty())
    return nullptr;

  return OMPCopyinClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars,
                                 SrcExprs, DstExprs, AssignmentOps);
}

OMPClause *Sema::ActOnOpenMPCopyprivateClause(ArrayRef<Expr *> VarList,
                                              SourceLocation StartLoc,
                                              SourceLocation LParenLoc,
                                              SourceLocation EndLoc) {
  SmallVector<Expr *, 8> Vars;
  SmallVector<Expr *, 8> SrcExprs;
  SmallVector<Expr *, 8> DstExprs;
  SmallVector<Expr *, 8> AssignmentOps;
  for (auto &RefExpr : VarList) {
    assert(RefExpr && "NULL expr in OpenMP copyprivate clause.");
    if (isa<DependentScopeDeclRefExpr>(RefExpr)) {
      // It will be analyzed later.
      Vars.push_back(RefExpr);
      SrcExprs.push_back(nullptr);
      DstExprs.push_back(nullptr);
      AssignmentOps.push_back(nullptr);
      continue;
    }

    SourceLocation ELoc = RefExpr->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.4.1, Restrictions, p.1]
    //  A list item that appears in a copyin clause must be threadprivate.
    DeclRefExpr *DE = dyn_cast<DeclRefExpr>(RefExpr);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << RefExpr->getSourceRange();
      continue;
    }

    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(DE);
      SrcExprs.push_back(nullptr);
      DstExprs.push_back(nullptr);
      AssignmentOps.push_back(nullptr);
      continue;
    }

    // OpenMP [2.14.4.2, Restrictions, p.2]
    //  A list item that appears in a copyprivate clause may not appear in a
    //  private or firstprivate clause on the single construct.
    if (!DSAStack->isThreadPrivate(VD)) {
      auto DVar = DSAStack->getTopDSA(VD, false);
      if (DVar.CKind != OMPC_unknown && DVar.CKind != OMPC_copyprivate &&
          DVar.RefExpr) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(DVar.CKind)
            << getOpenMPClauseName(OMPC_copyprivate);
        ReportOriginalDSA(*this, DSAStack, VD, DVar);
        continue;
      }

      // OpenMP [2.11.4.2, Restrictions, p.1]
      //  All list items that appear in a copyprivate clause must be either
      //  threadprivate or private in the enclosing context.
      if (DVar.CKind == OMPC_unknown) {
        DVar = DSAStack->getImplicitDSA(VD, false);
        if (DVar.CKind == OMPC_shared) {
          Diag(ELoc, diag::err_omp_required_access)
              << getOpenMPClauseName(OMPC_copyprivate)
              << "threadprivate or private in the enclosing context";
          ReportOriginalDSA(*this, DSAStack, VD, DVar);
          continue;
        }
      }
    }

    // Variably modified types are not supported.
    if (!Type->isAnyPointerType() && Type->isVariablyModifiedType()) {
      Diag(ELoc, diag::err_omp_variably_modified_type_not_supported)
          << getOpenMPClauseName(OMPC_copyprivate) << Type
          << getOpenMPDirectiveName(DSAStack->getCurrentDirective());
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.14.4.1, Restrictions, C/C++, p.2]
    //  A variable of class type (or array thereof) that appears in a
    //  copyin clause requires an accessible, unambiguous copy assignment
    //  operator for the class type.
    Type = Context.getBaseElementType(Type.getNonReferenceType())
               .getUnqualifiedType();
    auto *SrcVD =
        buildVarDecl(*this, DE->getLocStart(), Type, ".copyprivate.src",
                     VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    auto *PseudoSrcExpr =
        buildDeclRefExpr(*this, SrcVD, Type, DE->getExprLoc());
    auto *DstVD =
        buildVarDecl(*this, DE->getLocStart(), Type, ".copyprivate.dst",
                     VD->hasAttrs() ? &VD->getAttrs() : nullptr);
    auto *PseudoDstExpr =
        buildDeclRefExpr(*this, DstVD, Type, DE->getExprLoc());
    auto AssignmentOp = BuildBinOp(/*S=*/nullptr, DE->getExprLoc(), BO_Assign,
                                   PseudoDstExpr, PseudoSrcExpr);
    if (AssignmentOp.isInvalid())
      continue;
    AssignmentOp = ActOnFinishFullExpr(AssignmentOp.get(), DE->getExprLoc(),
                                       /*DiscardedValue=*/true);
    if (AssignmentOp.isInvalid())
      continue;

    // No need to mark vars as copyprivate, they are already threadprivate or
    // implicitly private.
    Vars.push_back(DE);
    SrcExprs.push_back(PseudoSrcExpr);
    DstExprs.push_back(PseudoDstExpr);
    AssignmentOps.push_back(AssignmentOp.get());
  }

  if (Vars.empty())
    return nullptr;

  return OMPCopyprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                      Vars, SrcExprs, DstExprs, AssignmentOps);
}

OMPClause *Sema::ActOnOpenMPFlushClause(ArrayRef<Expr *> VarList,
                                        SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc) {
  if (VarList.empty())
    return nullptr;

  return OMPFlushClause::Create(Context, StartLoc, LParenLoc, EndLoc, VarList);
}

OMPClause *
Sema::ActOnOpenMPDependClause(OpenMPDependClauseKind DepKind,
                              SourceLocation DepLoc, SourceLocation ColonLoc,
                              ArrayRef<Expr *> VarList, SourceLocation StartLoc,
                              SourceLocation LParenLoc, SourceLocation EndLoc) {
  if (DSAStack->getCurrentDirective() == OMPD_ordered &&
      DepKind != OMPC_DEPEND_source && DepKind != OMPC_DEPEND_sink) {
    Diag(DepLoc, diag::err_omp_unexpected_clause_value)
        << "'source' or 'sink'" << getOpenMPClauseName(OMPC_depend);
    return nullptr;
  }
  if (DSAStack->getCurrentDirective() != OMPD_ordered &&
      (DepKind == OMPC_DEPEND_unknown || DepKind == OMPC_DEPEND_source ||
       DepKind == OMPC_DEPEND_sink)) {
    unsigned Except[] = {OMPC_DEPEND_source, OMPC_DEPEND_sink};
    Diag(DepLoc, diag::err_omp_unexpected_clause_value)
        << getListOfPossibleValues(OMPC_depend, /*First=*/0,
                                   /*Last=*/OMPC_DEPEND_unknown, Except)
        << getOpenMPClauseName(OMPC_depend);
    return nullptr;
  }
  SmallVector<Expr *, 8> Vars;
  llvm::APSInt DepCounter(/*BitWidth=*/32);
  llvm::APSInt TotalDepCount(/*BitWidth=*/32);
  if (DepKind == OMPC_DEPEND_sink) {
    if (auto *OrderedCountExpr = DSAStack->getParentOrderedRegionParam()) {
      TotalDepCount = OrderedCountExpr->EvaluateKnownConstInt(Context);
      TotalDepCount.setIsUnsigned(/*Val=*/true);
    }
  }
  if ((DepKind != OMPC_DEPEND_sink && DepKind != OMPC_DEPEND_source) ||
      DSAStack->getParentOrderedRegionParam()) {
    for (auto &RefExpr : VarList) {
      assert(RefExpr && "NULL expr in OpenMP shared clause.");
      if (isa<DependentScopeDeclRefExpr>(RefExpr) ||
          (DepKind == OMPC_DEPEND_sink && CurContext->isDependentContext())) {
        // It will be analyzed later.
        Vars.push_back(RefExpr);
        continue;
      }

      SourceLocation ELoc = RefExpr->getExprLoc();
      auto *SimpleExpr = RefExpr->IgnoreParenCasts();
      if (DepKind == OMPC_DEPEND_sink) {
        if (DepCounter >= TotalDepCount) {
          Diag(ELoc, diag::err_omp_depend_sink_unexpected_expr);
          continue;
        }
        ++DepCounter;
        // OpenMP  [2.13.9, Summary]
        // depend(dependence-type : vec), where dependence-type is:
        // 'sink' and where vec is the iteration vector, which has the form:
        //  x1 [+- d1], x2 [+- d2 ], . . . , xn [+- dn]
        // where n is the value specified by the ordered clause in the loop
        // directive, xi denotes the loop iteration variable of the i-th nested
        // loop associated with the loop directive, and di is a constant
        // non-negative integer.
        SimpleExpr = SimpleExpr->IgnoreImplicit();
        auto *DE = dyn_cast<DeclRefExpr>(SimpleExpr);
        if (!DE) {
          OverloadedOperatorKind OOK = OO_None;
          SourceLocation OOLoc;
          Expr *LHS, *RHS;
          if (auto *BO = dyn_cast<BinaryOperator>(SimpleExpr)) {
            OOK = BinaryOperator::getOverloadedOperator(BO->getOpcode());
            OOLoc = BO->getOperatorLoc();
            LHS = BO->getLHS()->IgnoreParenImpCasts();
            RHS = BO->getRHS()->IgnoreParenImpCasts();
          } else if (auto *OCE = dyn_cast<CXXOperatorCallExpr>(SimpleExpr)) {
            OOK = OCE->getOperator();
            OOLoc = OCE->getOperatorLoc();
            LHS = OCE->getArg(/*Arg=*/0)->IgnoreParenImpCasts();
            RHS = OCE->getArg(/*Arg=*/1)->IgnoreParenImpCasts();
          } else if (auto *MCE = dyn_cast<CXXMemberCallExpr>(SimpleExpr)) {
            OOK = MCE->getMethodDecl()
                      ->getNameInfo()
                      .getName()
                      .getCXXOverloadedOperator();
            OOLoc = MCE->getCallee()->getExprLoc();
            LHS = MCE->getImplicitObjectArgument()->IgnoreParenImpCasts();
            RHS = MCE->getArg(/*Arg=*/0)->IgnoreParenImpCasts();
          } else {
            Diag(ELoc, diag::err_omp_depend_sink_wrong_expr);
            continue;
          }
          DE = dyn_cast<DeclRefExpr>(LHS);
          if (!DE) {
            Diag(LHS->getExprLoc(),
                 diag::err_omp_depend_sink_expected_loop_iteration)
                << DSAStack->getParentLoopControlVariable(
                    DepCounter.getZExtValue());
            continue;
          }
          if (OOK != OO_Plus && OOK != OO_Minus) {
            Diag(OOLoc, diag::err_omp_depend_sink_expected_plus_minus);
            continue;
          }
          ExprResult Res = VerifyPositiveIntegerConstantInClause(
              RHS, OMPC_depend, /*StrictlyPositive=*/false);
          if (Res.isInvalid())
            continue;
        }
        auto *VD = dyn_cast<VarDecl>(DE->getDecl());
        if (!CurContext->isDependentContext() &&
            DSAStack->getParentOrderedRegionParam() &&
            (!VD || DepCounter != DSAStack->isParentLoopControlVariable(VD))) {
          Diag(DE->getExprLoc(),
               diag::err_omp_depend_sink_expected_loop_iteration)
              << DSAStack->getParentLoopControlVariable(
                  DepCounter.getZExtValue());
          continue;
        }
      } else {
        // OpenMP  [2.11.1.1, Restrictions, p.3]
        //  A variable that is part of another variable (such as a field of a
        //  structure) but is not an array element or an array section cannot
        //  appear  in a depend clause.
        auto *DE = dyn_cast<DeclRefExpr>(SimpleExpr);
        auto *ASE = dyn_cast<ArraySubscriptExpr>(SimpleExpr);
        auto *OASE = dyn_cast<OMPArraySectionExpr>(SimpleExpr);
        if (!RefExpr->IgnoreParenImpCasts()->isLValue() ||
            (!ASE && !DE && !OASE) || (DE && !isa<VarDecl>(DE->getDecl())) ||
            (ASE && !ASE->getBase()->getType()->isAnyPointerType() &&
             !ASE->getBase()->getType()->isArrayType())) {
          Diag(ELoc, diag::err_omp_expected_var_name_or_array_item)
              << RefExpr->getSourceRange();
          continue;
        }
      }

      Vars.push_back(RefExpr->IgnoreParenImpCasts());
    }

    if (!CurContext->isDependentContext() && DepKind == OMPC_DEPEND_sink &&
        TotalDepCount > VarList.size() &&
        DSAStack->getParentOrderedRegionParam()) {
      Diag(EndLoc, diag::err_omp_depend_sink_expected_loop_iteration)
          << DSAStack->getParentLoopControlVariable(VarList.size() + 1);
    }
    if (DepKind != OMPC_DEPEND_source && DepKind != OMPC_DEPEND_sink &&
        Vars.empty())
      return nullptr;
  }

  return OMPDependClause::Create(Context, StartLoc, LParenLoc, EndLoc, DepKind,
                                 DepLoc, ColonLoc, Vars);
}

OMPClause *Sema::ActOnOpenMPDeviceClause(Expr *Device, SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  Expr *ValExpr = Device;

  // OpenMP [2.9.1, Restrictions]
  // The device expression must evaluate to a non-negative integer value.
  if (!IsNonNegativeIntegerValue(ValExpr, *this, OMPC_device,
                                 /*StrictlyPositive=*/false))
    return nullptr;

  return new (Context) OMPDeviceClause(ValExpr, StartLoc, LParenLoc, EndLoc);
}

static bool IsCXXRecordForMappable(Sema &SemaRef, SourceLocation Loc,
                                   DSAStackTy *Stack, CXXRecordDecl *RD) {
  if (!RD || RD->isInvalidDecl())
    return true;

  if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD))
    if (auto *CTD = CTSD->getSpecializedTemplate())
      RD = CTD->getTemplatedDecl();
  auto QTy = SemaRef.Context.getRecordType(RD);
  if (RD->isDynamicClass()) {
    SemaRef.Diag(Loc, diag::err_omp_not_mappable_type) << QTy;
    SemaRef.Diag(RD->getLocation(), diag::note_omp_polymorphic_in_target);
    return false;
  }
  auto *DC = RD;
  bool IsCorrect = true;
  for (auto *I : DC->decls()) {
    if (I) {
      if (auto *MD = dyn_cast<CXXMethodDecl>(I)) {
        if (MD->isStatic()) {
          SemaRef.Diag(Loc, diag::err_omp_not_mappable_type) << QTy;
          SemaRef.Diag(MD->getLocation(),
                       diag::note_omp_static_member_in_target);
          IsCorrect = false;
        }
      } else if (auto *VD = dyn_cast<VarDecl>(I)) {
        if (VD->isStaticDataMember()) {
          SemaRef.Diag(Loc, diag::err_omp_not_mappable_type) << QTy;
          SemaRef.Diag(VD->getLocation(),
                       diag::note_omp_static_member_in_target);
          IsCorrect = false;
        }
      }
    }
  }

  for (auto &I : RD->bases()) {
    if (!IsCXXRecordForMappable(SemaRef, I.getLocStart(), Stack,
                                I.getType()->getAsCXXRecordDecl()))
      IsCorrect = false;
  }
  return IsCorrect;
}

static bool CheckTypeMappable(SourceLocation SL, SourceRange SR, Sema &SemaRef,
                              DSAStackTy *Stack, QualType QTy) {
  NamedDecl *ND;
  if (QTy->isIncompleteType(&ND)) {
    SemaRef.Diag(SL, diag::err_incomplete_type) << QTy << SR;
    return false;
  } else if (CXXRecordDecl *RD = dyn_cast_or_null<CXXRecordDecl>(ND)) {
    if (!RD->isInvalidDecl() &&
        !IsCXXRecordForMappable(SemaRef, SL, Stack, RD))
      return false;
  }
  return true;
}

OMPClause *Sema::ActOnOpenMPMapClause(
    OpenMPMapClauseKind MapTypeModifier, OpenMPMapClauseKind MapType,
    SourceLocation MapLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
    SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation EndLoc) {
  SmallVector<Expr *, 4> Vars;

  for (auto &RE : VarList) {
    assert(RE && "Null expr in omp map");
    if (isa<DependentScopeDeclRefExpr>(RE)) {
      // It will be analyzed later.
      Vars.push_back(RE);
      continue;
    }
    SourceLocation ELoc = RE->getExprLoc();

    // OpenMP [2.14.5, Restrictions]
    //  A variable that is part of another variable (such as field of a
    //  structure) but is not an array element or an array section cannot appear
    //  in a map clause.
    auto *VE = RE->IgnoreParenLValueCasts();

    if (VE->isValueDependent() || VE->isTypeDependent() ||
        VE->isInstantiationDependent() ||
        VE->containsUnexpandedParameterPack()) {
      // It will be analyzed later.
      Vars.push_back(RE);
      continue;
    }

    auto *SimpleExpr = RE->IgnoreParenCasts();
    auto *DE = dyn_cast<DeclRefExpr>(SimpleExpr);
    auto *ASE = dyn_cast<ArraySubscriptExpr>(SimpleExpr);
    auto *OASE = dyn_cast<OMPArraySectionExpr>(SimpleExpr);

    if (!RE->IgnoreParenImpCasts()->isLValue() ||
        (!OASE && !ASE && !DE) ||
        (DE && !isa<VarDecl>(DE->getDecl())) ||
        (ASE && !ASE->getBase()->getType()->isAnyPointerType() &&
         !ASE->getBase()->getType()->isArrayType())) {
      Diag(ELoc, diag::err_omp_expected_var_name_or_array_item)
        << RE->getSourceRange();
      continue;
    }

    Decl *D = nullptr;
    if (DE) {
      D = DE->getDecl();
    } else if (ASE) {
      auto *B = ASE->getBase()->IgnoreParenCasts();
      D = dyn_cast<DeclRefExpr>(B)->getDecl();
    } else if (OASE) {
      auto *B = OASE->getBase();
      D = dyn_cast<DeclRefExpr>(B)->getDecl();
    }
    assert(D && "Null decl on map clause.");
    auto *VD = cast<VarDecl>(D);

    // OpenMP [2.14.5, Restrictions, p.8]
    // threadprivate variables cannot appear in a map clause.
    if (DSAStack->isThreadPrivate(VD)) {
      auto DVar = DSAStack->getTopDSA(VD, false);
      Diag(ELoc, diag::err_omp_threadprivate_in_map);
      ReportOriginalDSA(*this, DSAStack, VD, DVar);
      continue;
    }

    // OpenMP [2.14.5, Restrictions, p.2]
    //  At most one list item can be an array item derived from a given variable
    //  in map clauses of the same construct.
    // OpenMP [2.14.5, Restrictions, p.3]
    //  List items of map clauses in the same construct must not share original
    //  storage.
    // OpenMP [2.14.5, Restrictions, C/C++, p.2]
    //  A variable for which the type is pointer, reference to array, or
    //  reference to pointer and an array section derived from that variable
    //  must not appear as list items of map clauses of the same construct.
    DSAStackTy::MapInfo MI = DSAStack->IsMappedInCurrentRegion(VD);
    if (MI.RefExpr) {
      Diag(ELoc, diag::err_omp_map_shared_storage) << ELoc;
      Diag(MI.RefExpr->getExprLoc(), diag::note_used_here)
          << MI.RefExpr->getSourceRange();
      continue;
    }

    // OpenMP [2.14.5, Restrictions, C/C++, p.3,4]
    //  A variable for which the type is pointer, reference to array, or
    //  reference to pointer must not appear as a list item if the enclosing
    //  device data environment already contains an array section derived from
    //  that variable.
    //  An array section derived from a variable for which the type is pointer,
    //  reference to array, or reference to pointer must not appear as a list
    //  item if the enclosing device data environment already contains that
    //  variable.
    QualType Type = VD->getType();
    MI = DSAStack->getMapInfoForVar(VD);
    if (MI.RefExpr && (isa<DeclRefExpr>(MI.RefExpr->IgnoreParenLValueCasts()) !=
                       isa<DeclRefExpr>(VE)) &&
        (Type->isPointerType() || Type->isReferenceType())) {
      Diag(ELoc, diag::err_omp_map_shared_storage) << ELoc;
      Diag(MI.RefExpr->getExprLoc(), diag::note_used_here)
          << MI.RefExpr->getSourceRange();
      continue;
    }

    // OpenMP [2.14.5, Restrictions, C/C++, p.7]
    //  A list item must have a mappable type.
    if (!CheckTypeMappable(VE->getExprLoc(), VE->getSourceRange(), *this,
                           DSAStack, Type))
      continue;

    Vars.push_back(RE);
    MI.RefExpr = RE;
    DSAStack->addMapInfoForVar(VD, MI);
  }
  if (Vars.empty())
    return nullptr;

  return OMPMapClause::Create(Context, StartLoc, LParenLoc, EndLoc, Vars,
                              MapTypeModifier, MapType, MapLoc);
}

OMPClause *Sema::ActOnOpenMPNumTeamsClause(Expr *NumTeams, 
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  Expr *ValExpr = NumTeams;

  // OpenMP [teams Constrcut, Restrictions]
  // The num_teams expression must evaluate to a positive integer value.
  if (!IsNonNegativeIntegerValue(ValExpr, *this, OMPC_num_teams,
                                 /*StrictlyPositive=*/true))
    return nullptr;

  return new (Context) OMPNumTeamsClause(ValExpr, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPThreadLimitClause(Expr *ThreadLimit,
                                              SourceLocation StartLoc,
                                              SourceLocation LParenLoc,
                                              SourceLocation EndLoc) {
  Expr *ValExpr = ThreadLimit;

  // OpenMP [teams Constrcut, Restrictions]
  // The thread_limit expression must evaluate to a positive integer value.
  if (!IsNonNegativeIntegerValue(ValExpr, *this, OMPC_thread_limit,
                                 /*StrictlyPositive=*/true))
    return nullptr;

  return new (Context) OMPThreadLimitClause(ValExpr, StartLoc, LParenLoc,
                                            EndLoc);
}

OMPClause *Sema::ActOnOpenMPPriorityClause(Expr *Priority,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  Expr *ValExpr = Priority;

  // OpenMP [2.9.1, task Constrcut]
  // The priority-value is a non-negative numerical scalar expression.
  if (!IsNonNegativeIntegerValue(ValExpr, *this, OMPC_priority,
                                 /*StrictlyPositive=*/false))
    return nullptr;

  return new (Context) OMPPriorityClause(ValExpr, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPGrainsizeClause(Expr *Grainsize,
                                            SourceLocation StartLoc,
                                            SourceLocation LParenLoc,
                                            SourceLocation EndLoc) {
  Expr *ValExpr = Grainsize;

  // OpenMP [2.9.2, taskloop Constrcut]
  // The parameter of the grainsize clause must be a positive integer
  // expression.
  if (!IsNonNegativeIntegerValue(ValExpr, *this, OMPC_grainsize,
                                 /*StrictlyPositive=*/true))
    return nullptr;

  return new (Context) OMPGrainsizeClause(ValExpr, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPNumTasksClause(Expr *NumTasks,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  Expr *ValExpr = NumTasks;

  // OpenMP [2.9.2, taskloop Constrcut]
  // The parameter of the num_tasks clause must be a positive integer
  // expression.
  if (!IsNonNegativeIntegerValue(ValExpr, *this, OMPC_num_tasks,
                                 /*StrictlyPositive=*/true))
    return nullptr;

  return new (Context) OMPNumTasksClause(ValExpr, StartLoc, LParenLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPHintClause(Expr *Hint, SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc) {
  // OpenMP [2.13.2, critical construct, Description]
  // ... where hint-expression is an integer constant expression that evaluates
  // to a valid lock hint.
  ExprResult HintExpr = VerifyPositiveIntegerConstantInClause(Hint, OMPC_hint);
  if (HintExpr.isInvalid())
    return nullptr;
  return new (Context)
      OMPHintClause(HintExpr.get(), StartLoc, LParenLoc, EndLoc);
}

