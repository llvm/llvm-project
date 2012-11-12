//== ProgramState.h - Path-sensitive "State" for tracking values -*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the state of the program along the analysisa path.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_VALUESTATE_H
#define LLVM_CLANG_GR_VALUESTATE_H

#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicTypeInfo.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/Environment.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/Store.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/TaintTag.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"

namespace llvm {
class APSInt;
class BumpPtrAllocator;
}

namespace clang {
class ASTContext;

namespace ento {

class CallEvent;
class CallEventManager;

typedef ConstraintManager* (*ConstraintManagerCreator)(ProgramStateManager&,
                                                       SubEngine*);
typedef StoreManager* (*StoreManagerCreator)(ProgramStateManager&);

//===----------------------------------------------------------------------===//
// ProgramStateTrait - Traits used by the Generic Data Map of a ProgramState.
//===----------------------------------------------------------------------===//

template <typename T> struct ProgramStatePartialTrait;

template <typename T> struct ProgramStateTrait {
  typedef typename T::data_type data_type;
  static inline void *MakeVoidPtr(data_type D) { return (void*) D; }
  static inline data_type MakeData(void *const* P) {
    return P ? (data_type) *P : (data_type) 0;
  }
};

/// \class ProgramState
/// ProgramState - This class encapsulates:
///
///    1. A mapping from expressions to values (Environment)
///    2. A mapping from locations to values (Store)
///    3. Constraints on symbolic values (GenericDataMap)
///
///  Together these represent the "abstract state" of a program.
///
///  ProgramState is intended to be used as a functional object; that is,
///  once it is created and made "persistent" in a FoldingSet, its
///  values will never change.
class ProgramState : public llvm::FoldingSetNode {
public:
  typedef llvm::ImmutableSet<llvm::APSInt*>                IntSetTy;
  typedef llvm::ImmutableMap<void*, void*>                 GenericDataMap;

private:
  void operator=(const ProgramState& R) LLVM_DELETED_FUNCTION;

  friend class ProgramStateManager;
  friend class ExplodedGraph;
  friend class ExplodedNode;

  ProgramStateManager *stateMgr;
  Environment Env;           // Maps a Stmt to its current SVal.
  Store store;               // Maps a location to its current value.
  GenericDataMap   GDM;      // Custom data stored by a client of this class.
  unsigned refCount;

  /// makeWithStore - Return a ProgramState with the same values as the current
  ///  state with the exception of using the specified Store.
  ProgramStateRef makeWithStore(const StoreRef &store) const;

  void setStore(const StoreRef &storeRef);

public:
  /// This ctor is used when creating the first ProgramState object.
  ProgramState(ProgramStateManager *mgr, const Environment& env,
          StoreRef st, GenericDataMap gdm);
    
  /// Copy ctor - We must explicitly define this or else the "Next" ptr
  ///  in FoldingSetNode will also get copied.
  ProgramState(const ProgramState &RHS);
  
  ~ProgramState();

  /// Return the ProgramStateManager associated with this state.
  ProgramStateManager &getStateManager() const {
    return *stateMgr;
  }
  
  /// Return the ConstraintManager.
  ConstraintManager &getConstraintManager() const;

  /// getEnvironment - Return the environment associated with this state.
  ///  The environment is the mapping from expressions to values.
  const Environment& getEnvironment() const { return Env; }

  /// Return the store associated with this state.  The store
  ///  is a mapping from locations to values.
  Store getStore() const { return store; }

  
  /// getGDM - Return the generic data map associated with this state.
  GenericDataMap getGDM() const { return GDM; }

  void setGDM(GenericDataMap gdm) { GDM = gdm; }

  /// Profile - Profile the contents of a ProgramState object for use in a
  ///  FoldingSet.  Two ProgramState objects are considered equal if they
  ///  have the same Environment, Store, and GenericDataMap.
  static void Profile(llvm::FoldingSetNodeID& ID, const ProgramState *V) {
    V->Env.Profile(ID);
    ID.AddPointer(V->store);
    V->GDM.Profile(ID);
  }

  /// Profile - Used to profile the contents of this object for inclusion
  ///  in a FoldingSet.
  void Profile(llvm::FoldingSetNodeID& ID) const {
    Profile(ID, this);
  }

  BasicValueFactory &getBasicVals() const;
  SymbolManager &getSymbolManager() const;

  //==---------------------------------------------------------------------==//
  // Constraints on values.
  //==---------------------------------------------------------------------==//
  //
  // Each ProgramState records constraints on symbolic values.  These constraints
  // are managed using the ConstraintManager associated with a ProgramStateManager.
  // As constraints gradually accrue on symbolic values, added constraints
  // may conflict and indicate that a state is infeasible (as no real values
  // could satisfy all the constraints).  This is the principal mechanism
  // for modeling path-sensitivity in ExprEngine/ProgramState.
  //
  // Various "assume" methods form the interface for adding constraints to
  // symbolic values.  A call to 'assume' indicates an assumption being placed
  // on one or symbolic values.  'assume' methods take the following inputs:
  //
  //  (1) A ProgramState object representing the current state.
  //
  //  (2) The assumed constraint (which is specific to a given "assume" method).
  //
  //  (3) A binary value "Assumption" that indicates whether the constraint is
  //      assumed to be true or false.
  //
  // The output of "assume*" is a new ProgramState object with the added constraints.
  // If no new state is feasible, NULL is returned.
  //

  ProgramStateRef assume(DefinedOrUnknownSVal cond, bool assumption) const;

  /// This method assumes both "true" and "false" for 'cond', and
  ///  returns both corresponding states.  It's shorthand for doing
  ///  'assume' twice.
  std::pair<ProgramStateRef , ProgramStateRef >
  assume(DefinedOrUnknownSVal cond) const;

  ProgramStateRef assumeInBound(DefinedOrUnknownSVal idx,
                               DefinedOrUnknownSVal upperBound,
                               bool assumption,
                               QualType IndexType = QualType()) const;

  /// Utility method for getting regions.
  const VarRegion* getRegion(const VarDecl *D, const LocationContext *LC) const;

  //==---------------------------------------------------------------------==//
  // Binding and retrieving values to/from the environment and symbolic store.
  //==---------------------------------------------------------------------==//

  /// \brief Create a new state with the specified CompoundLiteral binding.
  /// \param CL the compound literal expression (the binding key)
  /// \param LC the LocationContext of the binding
  /// \param V the value to bind.
  ProgramStateRef bindCompoundLiteral(const CompoundLiteralExpr *CL,
                                      const LocationContext *LC,
                                      SVal V) const;

  /// Create a new state by binding the value 'V' to the statement 'S' in the
  /// state's environment.
  ProgramStateRef BindExpr(const Stmt *S, const LocationContext *LCtx,
                               SVal V, bool Invalidate = true) const;

  /// Create a new state by binding the value 'V' and location 'locaton' to the
  /// statement 'S' in the state's environment.
  ProgramStateRef bindExprAndLocation(const Stmt *S,
                                          const LocationContext *LCtx,
                                          SVal location, SVal V) const;

  ProgramStateRef bindLoc(Loc location,
                          SVal V,
                          bool notifyChanges = true) const;

  ProgramStateRef bindLoc(SVal location, SVal V) const;

  ProgramStateRef bindDefault(SVal loc, SVal V) const;

  ProgramStateRef killBinding(Loc LV) const;

  /// invalidateRegions - Returns the state with bindings for the given regions
  ///  cleared from the store. The regions are provided as a continuous array
  ///  from Begin to End. Optionally invalidates global regions as well.
  ProgramStateRef invalidateRegions(ArrayRef<const MemRegion *> Regions,
                               const Expr *E, unsigned BlockCount,
                               const LocationContext *LCtx,
                               StoreManager::InvalidatedSymbols *IS = 0,
                               const CallEvent *Call = 0) const;

  /// enterStackFrame - Returns the state for entry to the given stack frame,
  ///  preserving the current state.
  ProgramStateRef enterStackFrame(const CallEvent &Call,
                                  const StackFrameContext *CalleeCtx) const;

  /// Get the lvalue for a variable reference.
  Loc getLValue(const VarDecl *D, const LocationContext *LC) const;

  Loc getLValue(const CompoundLiteralExpr *literal, 
                const LocationContext *LC) const;

  /// Get the lvalue for an ivar reference.
  SVal getLValue(const ObjCIvarDecl *decl, SVal base) const;

  /// Get the lvalue for a field reference.
  SVal getLValue(const FieldDecl *decl, SVal Base) const;

  /// Get the lvalue for an indirect field reference.
  SVal getLValue(const IndirectFieldDecl *decl, SVal Base) const;

  /// Get the lvalue for an array index.
  SVal getLValue(QualType ElementType, SVal Idx, SVal Base) const;

  /// Returns the SVal bound to the statement 'S' in the state's environment.
  SVal getSVal(const Stmt *S, const LocationContext *LCtx) const;
  
  SVal getSValAsScalarOrLoc(const Stmt *Ex, const LocationContext *LCtx) const;

  /// \brief Return the value bound to the specified location.
  /// Returns UnknownVal() if none found.
  SVal getSVal(Loc LV, QualType T = QualType()) const;

  /// Returns the "raw" SVal bound to LV before any value simplfication.
  SVal getRawSVal(Loc LV, QualType T= QualType()) const;

  /// \brief Return the value bound to the specified location.
  /// Returns UnknownVal() if none found.
  SVal getSVal(const MemRegion* R) const;

  SVal getSValAsScalarOrLoc(const MemRegion *R) const;
  
  /// \brief Visits the symbols reachable from the given SVal using the provided
  /// SymbolVisitor.
  ///
  /// This is a convenience API. Consider using ScanReachableSymbols class
  /// directly when making multiple scans on the same state with the same
  /// visitor to avoid repeated initialization cost.
  /// \sa ScanReachableSymbols
  bool scanReachableSymbols(SVal val, SymbolVisitor& visitor) const;
  
  /// \brief Visits the symbols reachable from the SVals in the given range
  /// using the provided SymbolVisitor.
  bool scanReachableSymbols(const SVal *I, const SVal *E,
                            SymbolVisitor &visitor) const;
  
  /// \brief Visits the symbols reachable from the regions in the given
  /// MemRegions range using the provided SymbolVisitor.
  bool scanReachableSymbols(const MemRegion * const *I, 
                            const MemRegion * const *E,
                            SymbolVisitor &visitor) const;

  template <typename CB> CB scanReachableSymbols(SVal val) const;
  template <typename CB> CB scanReachableSymbols(const SVal *beg,
                                                 const SVal *end) const;
  
  template <typename CB> CB
  scanReachableSymbols(const MemRegion * const *beg,
                       const MemRegion * const *end) const;

  /// Create a new state in which the statement is marked as tainted.
  ProgramStateRef addTaint(const Stmt *S, const LocationContext *LCtx,
                               TaintTagType Kind = TaintTagGeneric) const;

  /// Create a new state in which the symbol is marked as tainted.
  ProgramStateRef addTaint(SymbolRef S,
                               TaintTagType Kind = TaintTagGeneric) const;

  /// Create a new state in which the region symbol is marked as tainted.
  ProgramStateRef addTaint(const MemRegion *R,
                               TaintTagType Kind = TaintTagGeneric) const;

  /// Check if the statement is tainted in the current state.
  bool isTainted(const Stmt *S, const LocationContext *LCtx,
                 TaintTagType Kind = TaintTagGeneric) const;
  bool isTainted(SVal V, TaintTagType Kind = TaintTagGeneric) const;
  bool isTainted(SymbolRef Sym, TaintTagType Kind = TaintTagGeneric) const;
  bool isTainted(const MemRegion *Reg, TaintTagType Kind=TaintTagGeneric) const;

  /// \brief Get dynamic type information for a region.
  DynamicTypeInfo getDynamicTypeInfo(const MemRegion *Reg) const;

  /// \brief Set dynamic type information of the region; return the new state.
  ProgramStateRef setDynamicTypeInfo(const MemRegion *Reg,
                                     DynamicTypeInfo NewTy) const;

  /// \brief Set dynamic type information of the region; return the new state.
  ProgramStateRef setDynamicTypeInfo(const MemRegion *Reg,
                                     QualType NewTy,
                                     bool CanBeSubClassed = true) const {
    return setDynamicTypeInfo(Reg, DynamicTypeInfo(NewTy, CanBeSubClassed));
  }

  //==---------------------------------------------------------------------==//
  // Accessing the Generic Data Map (GDM).
  //==---------------------------------------------------------------------==//

  void *const* FindGDM(void *K) const;

  template<typename T>
  ProgramStateRef add(typename ProgramStateTrait<T>::key_type K) const;

  template <typename T>
  typename ProgramStateTrait<T>::data_type
  get() const {
    return ProgramStateTrait<T>::MakeData(FindGDM(ProgramStateTrait<T>::GDMIndex()));
  }

  template<typename T>
  typename ProgramStateTrait<T>::lookup_type
  get(typename ProgramStateTrait<T>::key_type key) const {
    void *const* d = FindGDM(ProgramStateTrait<T>::GDMIndex());
    return ProgramStateTrait<T>::Lookup(ProgramStateTrait<T>::MakeData(d), key);
  }

  template <typename T>
  typename ProgramStateTrait<T>::context_type get_context() const;


  template<typename T>
  ProgramStateRef remove(typename ProgramStateTrait<T>::key_type K) const;

  template<typename T>
  ProgramStateRef remove(typename ProgramStateTrait<T>::key_type K,
                        typename ProgramStateTrait<T>::context_type C) const;
  template <typename T>
  ProgramStateRef remove() const;

  template<typename T>
  ProgramStateRef set(typename ProgramStateTrait<T>::data_type D) const;

  template<typename T>
  ProgramStateRef set(typename ProgramStateTrait<T>::key_type K,
                     typename ProgramStateTrait<T>::value_type E) const;

  template<typename T>
  ProgramStateRef set(typename ProgramStateTrait<T>::key_type K,
                     typename ProgramStateTrait<T>::value_type E,
                     typename ProgramStateTrait<T>::context_type C) const;

  template<typename T>
  bool contains(typename ProgramStateTrait<T>::key_type key) const {
    void *const* d = FindGDM(ProgramStateTrait<T>::GDMIndex());
    return ProgramStateTrait<T>::Contains(ProgramStateTrait<T>::MakeData(d), key);
  }

  // Pretty-printing.
  void print(raw_ostream &Out, const char *nl = "\n",
             const char *sep = "") const;
  void printDOT(raw_ostream &Out) const;
  void printTaint(raw_ostream &Out, const char *nl = "\n",
                  const char *sep = "") const;

  void dump() const;
  void dumpTaint() const;

private:
  friend void ProgramStateRetain(const ProgramState *state);
  friend void ProgramStateRelease(const ProgramState *state);
  
  ProgramStateRef 
  invalidateRegionsImpl(ArrayRef<const MemRegion *> Regions,
                        const Expr *E, unsigned BlockCount,
                        const LocationContext *LCtx,
                        StoreManager::InvalidatedSymbols &IS,
                        const CallEvent *Call) const;
};

//===----------------------------------------------------------------------===//
// ProgramStateManager - Factory object for ProgramStates.
//===----------------------------------------------------------------------===//

class ProgramStateManager {
  friend class ProgramState;
  friend void ProgramStateRelease(const ProgramState *state);
private:
  /// Eng - The SubEngine that owns this state manager.
  SubEngine *Eng; /* Can be null. */

  EnvironmentManager                   EnvMgr;
  OwningPtr<StoreManager>              StoreMgr;
  OwningPtr<ConstraintManager>         ConstraintMgr;

  ProgramState::GenericDataMap::Factory     GDMFactory;

  typedef llvm::DenseMap<void*,std::pair<void*,void (*)(void*)> > GDMContextsTy;
  GDMContextsTy GDMContexts;

  /// StateSet - FoldingSet containing all the states created for analyzing
  ///  a particular function.  This is used to unique states.
  llvm::FoldingSet<ProgramState> StateSet;

  /// Object that manages the data for all created SVals.
  OwningPtr<SValBuilder> svalBuilder;

  /// Manages memory for created CallEvents.
  OwningPtr<CallEventManager> CallEventMgr;

  /// A BumpPtrAllocator to allocate states.
  llvm::BumpPtrAllocator &Alloc;
  
  /// A vector of ProgramStates that we can reuse.
  std::vector<ProgramState *> freeStates;

public:
  ProgramStateManager(ASTContext &Ctx,
                 StoreManagerCreator CreateStoreManager,
                 ConstraintManagerCreator CreateConstraintManager,
                 llvm::BumpPtrAllocator& alloc,
                 SubEngine *subeng);

  ~ProgramStateManager();

  ProgramStateRef getInitialState(const LocationContext *InitLoc);

  ASTContext &getContext() { return svalBuilder->getContext(); }
  const ASTContext &getContext() const { return svalBuilder->getContext(); }

  BasicValueFactory &getBasicVals() {
    return svalBuilder->getBasicValueFactory();
  }

  SValBuilder &getSValBuilder() {
    return *svalBuilder;
  }

  SymbolManager &getSymbolManager() {
    return svalBuilder->getSymbolManager();
  }
  const SymbolManager &getSymbolManager() const {
    return svalBuilder->getSymbolManager();
  }

  llvm::BumpPtrAllocator& getAllocator() { return Alloc; }

  MemRegionManager& getRegionManager() {
    return svalBuilder->getRegionManager();
  }
  const MemRegionManager& getRegionManager() const {
    return svalBuilder->getRegionManager();
  }

  CallEventManager &getCallEventManager() { return *CallEventMgr; }

  StoreManager& getStoreManager() { return *StoreMgr; }
  ConstraintManager& getConstraintManager() { return *ConstraintMgr; }
  SubEngine* getOwningEngine() { return Eng; }

  ProgramStateRef removeDeadBindings(ProgramStateRef St,
                                    const StackFrameContext *LCtx,
                                    SymbolReaper& SymReaper);

public:

  SVal ArrayToPointer(Loc Array) {
    return StoreMgr->ArrayToPointer(Array);
  }

  // Methods that manipulate the GDM.
  ProgramStateRef addGDM(ProgramStateRef St, void *Key, void *Data);
  ProgramStateRef removeGDM(ProgramStateRef state, void *Key);

  // Methods that query & manipulate the Store.

  void iterBindings(ProgramStateRef state, StoreManager::BindingsHandler& F) {
    StoreMgr->iterBindings(state->getStore(), F);
  }

  ProgramStateRef getPersistentState(ProgramState &Impl);
  ProgramStateRef getPersistentStateWithGDM(ProgramStateRef FromState,
                                           ProgramStateRef GDMState);

  bool haveEqualEnvironments(ProgramStateRef S1, ProgramStateRef S2) {
    return S1->Env == S2->Env;
  }

  bool haveEqualStores(ProgramStateRef S1, ProgramStateRef S2) {
    return S1->store == S2->store;
  }

  //==---------------------------------------------------------------------==//
  // Generic Data Map methods.
  //==---------------------------------------------------------------------==//
  //
  // ProgramStateManager and ProgramState support a "generic data map" that allows
  // different clients of ProgramState objects to embed arbitrary data within a
  // ProgramState object.  The generic data map is essentially an immutable map
  // from a "tag" (that acts as the "key" for a client) and opaque values.
  // Tags/keys and values are simply void* values.  The typical way that clients
  // generate unique tags are by taking the address of a static variable.
  // Clients are responsible for ensuring that data values referred to by a
  // the data pointer are immutable (and thus are essentially purely functional
  // data).
  //
  // The templated methods below use the ProgramStateTrait<T> class
  // to resolve keys into the GDM and to return data values to clients.
  //

  // Trait based GDM dispatch.
  template <typename T>
  ProgramStateRef set(ProgramStateRef st, typename ProgramStateTrait<T>::data_type D) {
    return addGDM(st, ProgramStateTrait<T>::GDMIndex(),
                  ProgramStateTrait<T>::MakeVoidPtr(D));
  }

  template<typename T>
  ProgramStateRef set(ProgramStateRef st,
                     typename ProgramStateTrait<T>::key_type K,
                     typename ProgramStateTrait<T>::value_type V,
                     typename ProgramStateTrait<T>::context_type C) {

    return addGDM(st, ProgramStateTrait<T>::GDMIndex(),
     ProgramStateTrait<T>::MakeVoidPtr(ProgramStateTrait<T>::Set(st->get<T>(), K, V, C)));
  }

  template <typename T>
  ProgramStateRef add(ProgramStateRef st,
                     typename ProgramStateTrait<T>::key_type K,
                     typename ProgramStateTrait<T>::context_type C) {
    return addGDM(st, ProgramStateTrait<T>::GDMIndex(),
        ProgramStateTrait<T>::MakeVoidPtr(ProgramStateTrait<T>::Add(st->get<T>(), K, C)));
  }

  template <typename T>
  ProgramStateRef remove(ProgramStateRef st,
                        typename ProgramStateTrait<T>::key_type K,
                        typename ProgramStateTrait<T>::context_type C) {

    return addGDM(st, ProgramStateTrait<T>::GDMIndex(),
     ProgramStateTrait<T>::MakeVoidPtr(ProgramStateTrait<T>::Remove(st->get<T>(), K, C)));
  }

  template <typename T>
  ProgramStateRef remove(ProgramStateRef st) {
    return removeGDM(st, ProgramStateTrait<T>::GDMIndex());
  }

  void *FindGDMContext(void *index,
                       void *(*CreateContext)(llvm::BumpPtrAllocator&),
                       void  (*DeleteContext)(void*));

  template <typename T>
  typename ProgramStateTrait<T>::context_type get_context() {
    void *p = FindGDMContext(ProgramStateTrait<T>::GDMIndex(),
                             ProgramStateTrait<T>::CreateContext,
                             ProgramStateTrait<T>::DeleteContext);

    return ProgramStateTrait<T>::MakeContext(p);
  }

  void EndPath(ProgramStateRef St) {
    ConstraintMgr->EndPath(St);
  }
};


//===----------------------------------------------------------------------===//
// Out-of-line method definitions for ProgramState.
//===----------------------------------------------------------------------===//

inline ConstraintManager &ProgramState::getConstraintManager() const {
  return stateMgr->getConstraintManager();
}
  
inline const VarRegion* ProgramState::getRegion(const VarDecl *D,
                                                const LocationContext *LC) const 
{
  return getStateManager().getRegionManager().getVarRegion(D, LC);
}

inline ProgramStateRef ProgramState::assume(DefinedOrUnknownSVal Cond,
                                      bool Assumption) const {
  if (Cond.isUnknown())
    return this;
  
  return getStateManager().ConstraintMgr->assume(this, cast<DefinedSVal>(Cond),
                                                 Assumption);
}
  
inline std::pair<ProgramStateRef , ProgramStateRef >
ProgramState::assume(DefinedOrUnknownSVal Cond) const {
  if (Cond.isUnknown())
    return std::make_pair(this, this);
  
  return getStateManager().ConstraintMgr->assumeDual(this,
                                                     cast<DefinedSVal>(Cond));
}

inline ProgramStateRef ProgramState::bindLoc(SVal LV, SVal V) const {
  return !isa<Loc>(LV) ? this : bindLoc(cast<Loc>(LV), V);
}

inline Loc ProgramState::getLValue(const VarDecl *VD,
                               const LocationContext *LC) const {
  return getStateManager().StoreMgr->getLValueVar(VD, LC);
}

inline Loc ProgramState::getLValue(const CompoundLiteralExpr *literal,
                               const LocationContext *LC) const {
  return getStateManager().StoreMgr->getLValueCompoundLiteral(literal, LC);
}

inline SVal ProgramState::getLValue(const ObjCIvarDecl *D, SVal Base) const {
  return getStateManager().StoreMgr->getLValueIvar(D, Base);
}

inline SVal ProgramState::getLValue(const FieldDecl *D, SVal Base) const {
  return getStateManager().StoreMgr->getLValueField(D, Base);
}

inline SVal ProgramState::getLValue(const IndirectFieldDecl *D,
                                    SVal Base) const {
  StoreManager &SM = *getStateManager().StoreMgr;
  for (IndirectFieldDecl::chain_iterator I = D->chain_begin(),
                                         E = D->chain_end();
       I != E; ++I) {
    Base = SM.getLValueField(cast<FieldDecl>(*I), Base);
  }

  return Base;
}

inline SVal ProgramState::getLValue(QualType ElementType, SVal Idx, SVal Base) const{
  if (NonLoc *N = dyn_cast<NonLoc>(&Idx))
    return getStateManager().StoreMgr->getLValueElement(ElementType, *N, Base);
  return UnknownVal();
}

inline SVal ProgramState::getSVal(const Stmt *Ex,
                                  const LocationContext *LCtx) const{
  return Env.getSVal(EnvironmentEntry(Ex, LCtx),
                     *getStateManager().svalBuilder);
}

inline SVal
ProgramState::getSValAsScalarOrLoc(const Stmt *S,
                                   const LocationContext *LCtx) const {
  if (const Expr *Ex = dyn_cast<Expr>(S)) {
    QualType T = Ex->getType();
    if (Ex->isGLValue() || Loc::isLocType(T) || T->isIntegerType())
      return getSVal(S, LCtx);
  }

  return UnknownVal();
}

inline SVal ProgramState::getRawSVal(Loc LV, QualType T) const {
  return getStateManager().StoreMgr->getBinding(getStore(), LV, T);
}

inline SVal ProgramState::getSVal(const MemRegion* R) const {
  return getStateManager().StoreMgr->getBinding(getStore(),
                                                loc::MemRegionVal(R));
}

inline BasicValueFactory &ProgramState::getBasicVals() const {
  return getStateManager().getBasicVals();
}

inline SymbolManager &ProgramState::getSymbolManager() const {
  return getStateManager().getSymbolManager();
}

template<typename T>
ProgramStateRef ProgramState::add(typename ProgramStateTrait<T>::key_type K) const {
  return getStateManager().add<T>(this, K, get_context<T>());
}

template <typename T>
typename ProgramStateTrait<T>::context_type ProgramState::get_context() const {
  return getStateManager().get_context<T>();
}

template<typename T>
ProgramStateRef ProgramState::remove(typename ProgramStateTrait<T>::key_type K) const {
  return getStateManager().remove<T>(this, K, get_context<T>());
}

template<typename T>
ProgramStateRef ProgramState::remove(typename ProgramStateTrait<T>::key_type K,
                               typename ProgramStateTrait<T>::context_type C) const {
  return getStateManager().remove<T>(this, K, C);
}

template <typename T>
ProgramStateRef ProgramState::remove() const {
  return getStateManager().remove<T>(this);
}

template<typename T>
ProgramStateRef ProgramState::set(typename ProgramStateTrait<T>::data_type D) const {
  return getStateManager().set<T>(this, D);
}

template<typename T>
ProgramStateRef ProgramState::set(typename ProgramStateTrait<T>::key_type K,
                            typename ProgramStateTrait<T>::value_type E) const {
  return getStateManager().set<T>(this, K, E, get_context<T>());
}

template<typename T>
ProgramStateRef ProgramState::set(typename ProgramStateTrait<T>::key_type K,
                            typename ProgramStateTrait<T>::value_type E,
                            typename ProgramStateTrait<T>::context_type C) const {
  return getStateManager().set<T>(this, K, E, C);
}

template <typename CB>
CB ProgramState::scanReachableSymbols(SVal val) const {
  CB cb(this);
  scanReachableSymbols(val, cb);
  return cb;
}
  
template <typename CB>
CB ProgramState::scanReachableSymbols(const SVal *beg, const SVal *end) const {
  CB cb(this);
  scanReachableSymbols(beg, end, cb);
  return cb;
}

template <typename CB>
CB ProgramState::scanReachableSymbols(const MemRegion * const *beg,
                                 const MemRegion * const *end) const {
  CB cb(this);
  scanReachableSymbols(beg, end, cb);
  return cb;
}

/// \class ScanReachableSymbols
/// A Utility class that allows to visit the reachable symbols using a custom
/// SymbolVisitor.
class ScanReachableSymbols {
  typedef llvm::DenseMap<const void*, unsigned> VisitedItems;

  VisitedItems visited;
  ProgramStateRef state;
  SymbolVisitor &visitor;
public:

  ScanReachableSymbols(ProgramStateRef st, SymbolVisitor& v)
    : state(st), visitor(v) {}

  bool scan(nonloc::CompoundVal val);
  bool scan(SVal val);
  bool scan(const MemRegion *R);
  bool scan(const SymExpr *sym);
};

} // end ento namespace

} // end clang namespace

#endif
