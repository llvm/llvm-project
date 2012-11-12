//===- CallEvent.h - Wrapper for all function and method calls ----*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file defines CallEvent and its subclasses, which represent path-
/// sensitive instances of different kinds of function and method calls
/// (C, C++, and Objective-C).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_PATHSENSITIVE_CALL
#define LLVM_CLANG_STATICANALYZER_PATHSENSITIVE_CALL

#include "clang/Basic/SourceManager.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/PointerIntPair.h"

namespace clang {
class ProgramPoint;
class ProgramPointTag;

namespace ento {

enum CallEventKind {
  CE_Function,
  CE_Block,
  CE_BEG_SIMPLE_CALLS = CE_Function,
  CE_END_SIMPLE_CALLS = CE_Block,
  CE_CXXMember,
  CE_CXXMemberOperator,
  CE_CXXDestructor,
  CE_BEG_CXX_INSTANCE_CALLS = CE_CXXMember,
  CE_END_CXX_INSTANCE_CALLS = CE_CXXDestructor,
  CE_CXXConstructor,
  CE_CXXAllocator,
  CE_BEG_FUNCTION_CALLS = CE_Function,
  CE_END_FUNCTION_CALLS = CE_CXXAllocator,
  CE_ObjCMessage
};

class CallEvent;
class CallEventManager;

template<typename T = CallEvent>
class CallEventRef : public IntrusiveRefCntPtr<const T> {
public:
  CallEventRef(const T *Call) : IntrusiveRefCntPtr<const T>(Call) {}
  CallEventRef(const CallEventRef &Orig) : IntrusiveRefCntPtr<const T>(Orig) {}

  CallEventRef<T> cloneWithState(ProgramStateRef State) const {
    return this->getPtr()->template cloneWithState<T>(State);
  }

  // Allow implicit conversions to a superclass type, since CallEventRef
  // behaves like a pointer-to-const.
  template <typename SuperT>
  operator CallEventRef<SuperT> () const {
    return this->getPtr();
  }
};

/// \class RuntimeDefinition 
/// \brief Defines the runtime definition of the called function.
/// 
/// Encapsulates the information we have about which Decl will be used 
/// when the call is executed on the given path. When dealing with dynamic
/// dispatch, the information is based on DynamicTypeInfo and might not be 
/// precise.
class RuntimeDefinition {
  /// The Declaration of the function which could be called at runtime.
  /// NULL if not available.
  const Decl *D;

  /// The region representing an object (ObjC/C++) on which the method is
  /// called. With dynamic dispatch, the method definition depends on the
  /// runtime type of this object. NULL when the DynamicTypeInfo is
  /// precise.
  const MemRegion *R;

public:
  RuntimeDefinition(): D(0), R(0) {}
  RuntimeDefinition(const Decl *InD): D(InD), R(0) {}
  RuntimeDefinition(const Decl *InD, const MemRegion *InR): D(InD), R(InR) {}
  const Decl *getDecl() { return D; }
    
  /// \brief Check if the definition we have is precise. 
  /// If not, it is possible that the call dispatches to another definition at 
  /// execution time.
  bool mayHaveOtherDefinitions() { return R != 0; }
  
  /// When other definitions are possible, returns the region whose runtime type 
  /// determines the method definition.
  const MemRegion *getDispatchRegion() { return R; }
};

/// \brief Represents an abstract call to a function or method along a
/// particular path.
///
/// CallEvents are created through the factory methods of CallEventManager.
///
/// CallEvents should always be cheap to create and destroy. In order for
/// CallEventManager to be able to re-use CallEvent-sized memory blocks,
/// subclasses of CallEvent may not add any data members to the base class.
/// Use the "Data" and "Location" fields instead.
class CallEvent {
public:
  typedef CallEventKind Kind;

private:
  ProgramStateRef State;
  const LocationContext *LCtx;
  llvm::PointerUnion<const Expr *, const Decl *> Origin;

  void operator=(const CallEvent &) LLVM_DELETED_FUNCTION;

protected:
  // This is user data for subclasses.
  const void *Data;

  // This is user data for subclasses.
  // This should come right before RefCount, so that the two fields can be
  // packed together on LP64 platforms.
  SourceLocation Location;

private:
  mutable unsigned RefCount;

  template <typename T> friend struct llvm::IntrusiveRefCntPtrInfo;
  void Retain() const { ++RefCount; }
  void Release() const;

protected:
  friend class CallEventManager;

  CallEvent(const Expr *E, ProgramStateRef state, const LocationContext *lctx)
    : State(state), LCtx(lctx), Origin(E), RefCount(0) {}

  CallEvent(const Decl *D, ProgramStateRef state, const LocationContext *lctx)
    : State(state), LCtx(lctx), Origin(D), RefCount(0) {}

  // DO NOT MAKE PUBLIC
  CallEvent(const CallEvent &Original)
    : State(Original.State), LCtx(Original.LCtx), Origin(Original.Origin),
      Data(Original.Data), Location(Original.Location), RefCount(0) {}

  /// Copies this CallEvent, with vtable intact, into a new block of memory.
  virtual void cloneTo(void *Dest) const = 0;

  /// \brief Get the value of arbitrary expressions at this point in the path.
  SVal getSVal(const Stmt *S) const {
    return getState()->getSVal(S, getLocationContext());
  }


  typedef SmallVectorImpl<const MemRegion *> RegionList;

  /// \brief Used to specify non-argument regions that will be invalidated as a
  /// result of this call.
  virtual void getExtraInvalidatedRegions(RegionList &Regions) const {}

public:
  virtual ~CallEvent() {}

  /// \brief Returns the kind of call this is.
  virtual Kind getKind() const = 0;

  /// \brief Returns the declaration of the function or method that will be
  /// called. May be null.
  virtual const Decl *getDecl() const {
    return Origin.dyn_cast<const Decl *>();
  }

  /// \brief The state in which the call is being evaluated.
  ProgramStateRef getState() const {
    return State;
  }

  /// \brief The context in which the call is being evaluated.
  const LocationContext *getLocationContext() const {
    return LCtx;
  }

  /// \brief Returns the definition of the function or method that will be
  /// called.
  virtual RuntimeDefinition getRuntimeDefinition() const = 0;

  /// \brief Returns the expression whose value will be the result of this call.
  /// May be null.
  const Expr *getOriginExpr() const {
    return Origin.dyn_cast<const Expr *>();
  }

  /// \brief Returns the number of arguments (explicit and implicit).
  ///
  /// Note that this may be greater than the number of parameters in the
  /// callee's declaration, and that it may include arguments not written in
  /// the source.
  virtual unsigned getNumArgs() const = 0;

  /// \brief Returns true if the callee is known to be from a system header.
  bool isInSystemHeader() const {
    const Decl *D = getDecl();
    if (!D)
      return false;

    SourceLocation Loc = D->getLocation();
    if (Loc.isValid()) {
      const SourceManager &SM =
        getState()->getStateManager().getContext().getSourceManager();
      return SM.isInSystemHeader(D->getLocation());
    }

    // Special case for implicitly-declared global operator new/delete.
    // These should be considered system functions.
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
      return FD->isOverloadedOperator() && FD->isImplicit() && FD->isGlobal();

    return false;
  }

  /// \brief Returns a source range for the entire call, suitable for
  /// outputting in diagnostics.
  virtual SourceRange getSourceRange() const {
    return getOriginExpr()->getSourceRange();
  }

  /// \brief Returns the value of a given argument at the time of the call.
  virtual SVal getArgSVal(unsigned Index) const;

  /// \brief Returns the expression associated with a given argument.
  /// May be null if this expression does not appear in the source.
  virtual const Expr *getArgExpr(unsigned Index) const { return 0; }

  /// \brief Returns the source range for errors associated with this argument.
  ///
  /// May be invalid if the argument is not written in the source.
  virtual SourceRange getArgSourceRange(unsigned Index) const;

  /// \brief Returns the result type, adjusted for references.
  QualType getResultType() const;

  /// \brief Returns the return value of the call.
  ///
  /// This should only be called if the CallEvent was created using a state in
  /// which the return value has already been bound to the origin expression.
  SVal getReturnValue() const;

  /// \brief Returns true if any of the arguments appear to represent callbacks.
  bool hasNonZeroCallbackArg() const;

  /// \brief Returns true if any of the arguments are known to escape to long-
  /// term storage, even if this method will not modify them.
  // NOTE: The exact semantics of this are still being defined!
  // We don't really want a list of hardcoded exceptions in the long run,
  // but we don't want duplicated lists of known APIs in the short term either.
  virtual bool argumentsMayEscape() const {
    return hasNonZeroCallbackArg();
  }

  /// \brief Returns true if the callee is an externally-visible function in the
  /// top-level namespace, such as \c malloc.
  ///
  /// You can use this call to determine that a particular function really is
  /// a library function and not, say, a C++ member function with the same name.
  ///
  /// If a name is provided, the function must additionally match the given
  /// name.
  ///
  /// Note that this deliberately excludes C++ library functions in the \c std
  /// namespace, but will include C library functions accessed through the
  /// \c std namespace. This also does not check if the function is declared
  /// as 'extern "C"', or if it uses C++ name mangling.
  // FIXME: Add a helper for checking namespaces.
  // FIXME: Move this down to AnyFunctionCall once checkers have more
  // precise callbacks.
  bool isGlobalCFunction(StringRef SpecificName = StringRef()) const;

  /// \brief Returns the name of the callee, if its name is a simple identifier.
  ///
  /// Note that this will fail for Objective-C methods, blocks, and C++
  /// overloaded operators. The former is named by a Selector rather than a
  /// simple identifier, and the latter two do not have names.
  // FIXME: Move this down to AnyFunctionCall once checkers have more
  // precise callbacks.
  const IdentifierInfo *getCalleeIdentifier() const {
    const NamedDecl *ND = dyn_cast_or_null<NamedDecl>(getDecl());
    if (!ND)
      return 0;
    return ND->getIdentifier();
  }

  /// \brief Returns an appropriate ProgramPoint for this call.
  ProgramPoint getProgramPoint(bool IsPreVisit = false,
                               const ProgramPointTag *Tag = 0) const;

  /// \brief Returns a new state with all argument regions invalidated.
  ///
  /// This accepts an alternate state in case some processing has already
  /// occurred.
  ProgramStateRef invalidateRegions(unsigned BlockCount,
                                    ProgramStateRef Orig = 0) const;

  typedef std::pair<Loc, SVal> FrameBindingTy;
  typedef SmallVectorImpl<FrameBindingTy> BindingsTy;

  /// Populates the given SmallVector with the bindings in the callee's stack
  /// frame at the start of this call.
  virtual void getInitialStackFrameContents(const StackFrameContext *CalleeCtx,
                                            BindingsTy &Bindings) const = 0;

  /// Returns a copy of this CallEvent, but using the given state.
  template <typename T>
  CallEventRef<T> cloneWithState(ProgramStateRef NewState) const;

  /// Returns a copy of this CallEvent, but using the given state.
  CallEventRef<> cloneWithState(ProgramStateRef NewState) const {
    return cloneWithState<CallEvent>(NewState);
  }

  /// \brief Returns true if this is a statement is a function or method call
  /// of some kind.
  static bool isCallStmt(const Stmt *S);

  /// \brief Returns the result type of a function, method declaration.
  static QualType getDeclaredResultType(const Decl *D);

  // Iterator access to formal parameters and their types.
private:
  typedef std::const_mem_fun_t<QualType, ParmVarDecl> get_type_fun;
  
public:
  typedef const ParmVarDecl * const *param_iterator;

  /// Returns an iterator over the call's formal parameters.
  ///
  /// If UseDefinitionParams is set, this will return the parameter decls
  /// used in the callee's definition (suitable for inlining). Most of the
  /// time it is better to use the decl found by name lookup, which likely
  /// carries more annotations.
  ///
  /// Remember that the number of formal parameters may not match the number
  /// of arguments for all calls. However, the first parameter will always
  /// correspond with the argument value returned by \c getArgSVal(0).
  ///
  /// If the call has no accessible declaration (or definition, if
  /// \p UseDefinitionParams is set), \c param_begin() will be equal to
  /// \c param_end().
  virtual param_iterator param_begin() const =0;
  /// \sa param_begin()
  virtual param_iterator param_end() const = 0;

  typedef llvm::mapped_iterator<param_iterator, get_type_fun>
    param_type_iterator;

  /// Returns an iterator over the types of the call's formal parameters.
  ///
  /// This uses the callee decl found by default name lookup rather than the
  /// definition because it represents a public interface, and probably has
  /// more annotations.
  param_type_iterator param_type_begin() const {
    return llvm::map_iterator(param_begin(),
                              get_type_fun(&ParmVarDecl::getType));
  }
  /// \sa param_type_begin()
  param_type_iterator param_type_end() const {
    return llvm::map_iterator(param_end(), get_type_fun(&ParmVarDecl::getType));
  }

  // For debugging purposes only
  void dump(raw_ostream &Out) const;
  LLVM_ATTRIBUTE_USED void dump() const;
};


/// \brief Represents a call to any sort of function that might have a
/// FunctionDecl.
class AnyFunctionCall : public CallEvent {
protected:
  AnyFunctionCall(const Expr *E, ProgramStateRef St,
                  const LocationContext *LCtx)
    : CallEvent(E, St, LCtx) {}
  AnyFunctionCall(const Decl *D, ProgramStateRef St,
                  const LocationContext *LCtx)
    : CallEvent(D, St, LCtx) {}
  AnyFunctionCall(const AnyFunctionCall &Other) : CallEvent(Other) {}

public:
  // This function is overridden by subclasses, but they must return
  // a FunctionDecl.
  virtual const FunctionDecl *getDecl() const {
    return cast<FunctionDecl>(CallEvent::getDecl());
  }

  virtual RuntimeDefinition getRuntimeDefinition() const {
    const FunctionDecl *FD = getDecl();
    // Note that the AnalysisDeclContext will have the FunctionDecl with
    // the definition (if one exists).
    if (FD) {
      AnalysisDeclContext *AD =
        getLocationContext()->getAnalysisDeclContext()->
        getManager()->getContext(FD);
      if (AD->getBody())
        return RuntimeDefinition(AD->getDecl());
    }

    return RuntimeDefinition();
  }

  virtual bool argumentsMayEscape() const;

  virtual void getInitialStackFrameContents(const StackFrameContext *CalleeCtx,
                                            BindingsTy &Bindings) const;

  virtual param_iterator param_begin() const;
  virtual param_iterator param_end() const;

  static bool classof(const CallEvent *CA) {
    return CA->getKind() >= CE_BEG_FUNCTION_CALLS &&
           CA->getKind() <= CE_END_FUNCTION_CALLS;
  }
};

/// \brief Represents a call to a non-C++ function, written as a CallExpr.
class SimpleCall : public AnyFunctionCall {
protected:
  SimpleCall(const CallExpr *CE, ProgramStateRef St,
             const LocationContext *LCtx)
    : AnyFunctionCall(CE, St, LCtx) {}
  SimpleCall(const SimpleCall &Other) : AnyFunctionCall(Other) {}

public:
  virtual const CallExpr *getOriginExpr() const {
    return cast<CallExpr>(AnyFunctionCall::getOriginExpr());
  }

  virtual const FunctionDecl *getDecl() const;

  virtual unsigned getNumArgs() const { return getOriginExpr()->getNumArgs(); }

  virtual const Expr *getArgExpr(unsigned Index) const {
    return getOriginExpr()->getArg(Index);
  }

  static bool classof(const CallEvent *CA) {
    return CA->getKind() >= CE_BEG_SIMPLE_CALLS &&
           CA->getKind() <= CE_END_SIMPLE_CALLS;
  }
};

/// \brief Represents a C function or static C++ member function call.
///
/// Example: \c fun()
class FunctionCall : public SimpleCall {
  friend class CallEventManager;

protected:
  FunctionCall(const CallExpr *CE, ProgramStateRef St,
               const LocationContext *LCtx)
    : SimpleCall(CE, St, LCtx) {}

  FunctionCall(const FunctionCall &Other) : SimpleCall(Other) {}
  virtual void cloneTo(void *Dest) const { new (Dest) FunctionCall(*this); }

public:
  virtual Kind getKind() const { return CE_Function; }

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_Function;
  }
};

/// \brief Represents a call to a block.
///
/// Example: <tt>^{ /* ... */ }()</tt>
class BlockCall : public SimpleCall {
  friend class CallEventManager;

protected:
  BlockCall(const CallExpr *CE, ProgramStateRef St,
            const LocationContext *LCtx)
    : SimpleCall(CE, St, LCtx) {}

  BlockCall(const BlockCall &Other) : SimpleCall(Other) {}
  virtual void cloneTo(void *Dest) const { new (Dest) BlockCall(*this); }

  virtual void getExtraInvalidatedRegions(RegionList &Regions) const;

public:
  /// \brief Returns the region associated with this instance of the block.
  ///
  /// This may be NULL if the block's origin is unknown.
  const BlockDataRegion *getBlockRegion() const;

  /// \brief Gets the declaration of the block.
  ///
  /// This is not an override of getDecl() because AnyFunctionCall has already
  /// assumed that it's a FunctionDecl.
  const BlockDecl *getBlockDecl() const {
    const BlockDataRegion *BR = getBlockRegion();
    if (!BR)
      return 0;
    return BR->getDecl();
  }

  virtual RuntimeDefinition getRuntimeDefinition() const {
    return RuntimeDefinition(getBlockDecl());
  }

  virtual void getInitialStackFrameContents(const StackFrameContext *CalleeCtx,
                                            BindingsTy &Bindings) const;

  virtual param_iterator param_begin() const;
  virtual param_iterator param_end() const;

  virtual Kind getKind() const { return CE_Block; }

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_Block;
  }
};

/// \brief Represents a non-static C++ member function call, no matter how
/// it is written.
class CXXInstanceCall : public AnyFunctionCall {
protected:
  virtual void getExtraInvalidatedRegions(RegionList &Regions) const;

  CXXInstanceCall(const CallExpr *CE, ProgramStateRef St,
                  const LocationContext *LCtx)
    : AnyFunctionCall(CE, St, LCtx) {}
  CXXInstanceCall(const FunctionDecl *D, ProgramStateRef St,
                  const LocationContext *LCtx)
    : AnyFunctionCall(D, St, LCtx) {}


  CXXInstanceCall(const CXXInstanceCall &Other) : AnyFunctionCall(Other) {}

public:
  /// \brief Returns the expression representing the implicit 'this' object.
  virtual const Expr *getCXXThisExpr() const { return 0; }

  /// \brief Returns the value of the implicit 'this' object.
  virtual SVal getCXXThisVal() const;

  virtual const FunctionDecl *getDecl() const;

  virtual RuntimeDefinition getRuntimeDefinition() const;

  virtual void getInitialStackFrameContents(const StackFrameContext *CalleeCtx,
                                            BindingsTy &Bindings) const;

  static bool classof(const CallEvent *CA) {
    return CA->getKind() >= CE_BEG_CXX_INSTANCE_CALLS &&
           CA->getKind() <= CE_END_CXX_INSTANCE_CALLS;
  }
};

/// \brief Represents a non-static C++ member function call.
///
/// Example: \c obj.fun()
class CXXMemberCall : public CXXInstanceCall {
  friend class CallEventManager;

protected:
  CXXMemberCall(const CXXMemberCallExpr *CE, ProgramStateRef St,
                const LocationContext *LCtx)
    : CXXInstanceCall(CE, St, LCtx) {}

  CXXMemberCall(const CXXMemberCall &Other) : CXXInstanceCall(Other) {}
  virtual void cloneTo(void *Dest) const { new (Dest) CXXMemberCall(*this); }

public:
  virtual const CXXMemberCallExpr *getOriginExpr() const {
    return cast<CXXMemberCallExpr>(CXXInstanceCall::getOriginExpr());
  }

  virtual unsigned getNumArgs() const {
    if (const CallExpr *CE = getOriginExpr())
      return CE->getNumArgs();
    return 0;
  }

  virtual const Expr *getArgExpr(unsigned Index) const {
    return getOriginExpr()->getArg(Index);
  }

  virtual const Expr *getCXXThisExpr() const;
  
  virtual RuntimeDefinition getRuntimeDefinition() const;

  virtual Kind getKind() const { return CE_CXXMember; }

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_CXXMember;
  }
};

/// \brief Represents a C++ overloaded operator call where the operator is
/// implemented as a non-static member function.
///
/// Example: <tt>iter + 1</tt>
class CXXMemberOperatorCall : public CXXInstanceCall {
  friend class CallEventManager;

protected:
  CXXMemberOperatorCall(const CXXOperatorCallExpr *CE, ProgramStateRef St,
                        const LocationContext *LCtx)
    : CXXInstanceCall(CE, St, LCtx) {}

  CXXMemberOperatorCall(const CXXMemberOperatorCall &Other)
    : CXXInstanceCall(Other) {}
  virtual void cloneTo(void *Dest) const {
    new (Dest) CXXMemberOperatorCall(*this);
  }

public:
  virtual const CXXOperatorCallExpr *getOriginExpr() const {
    return cast<CXXOperatorCallExpr>(CXXInstanceCall::getOriginExpr());
  }

  virtual unsigned getNumArgs() const {
    return getOriginExpr()->getNumArgs() - 1;
  }
  virtual const Expr *getArgExpr(unsigned Index) const {
    return getOriginExpr()->getArg(Index + 1);
  }

  virtual const Expr *getCXXThisExpr() const;

  virtual Kind getKind() const { return CE_CXXMemberOperator; }

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_CXXMemberOperator;
  }
};

/// \brief Represents an implicit call to a C++ destructor.
///
/// This can occur at the end of a scope (for automatic objects), at the end
/// of a full-expression (for temporaries), or as part of a delete.
class CXXDestructorCall : public CXXInstanceCall {
  friend class CallEventManager;

protected:
  typedef llvm::PointerIntPair<const MemRegion *, 1, bool> DtorDataTy;

  /// Creates an implicit destructor.
  ///
  /// \param DD The destructor that will be called.
  /// \param Trigger The statement whose completion causes this destructor call.
  /// \param Target The object region to be destructed.
  /// \param St The path-sensitive state at this point in the program.
  /// \param LCtx The location context at this point in the program.
  CXXDestructorCall(const CXXDestructorDecl *DD, const Stmt *Trigger,
                    const MemRegion *Target, bool IsBaseDestructor,
                    ProgramStateRef St, const LocationContext *LCtx)
    : CXXInstanceCall(DD, St, LCtx) {
    Data = DtorDataTy(Target, IsBaseDestructor).getOpaqueValue();
    Location = Trigger->getLocEnd();
  }

  CXXDestructorCall(const CXXDestructorCall &Other) : CXXInstanceCall(Other) {}
  virtual void cloneTo(void *Dest) const { new (Dest) CXXDestructorCall(*this); }

public:
  virtual SourceRange getSourceRange() const { return Location; }
  virtual unsigned getNumArgs() const { return 0; }

  virtual RuntimeDefinition getRuntimeDefinition() const;

  /// \brief Returns the value of the implicit 'this' object.
  virtual SVal getCXXThisVal() const;

  /// Returns true if this is a call to a base class destructor.
  bool isBaseDestructor() const {
    return DtorDataTy::getFromOpaqueValue(Data).getInt();
  }

  virtual Kind getKind() const { return CE_CXXDestructor; }

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_CXXDestructor;
  }
};

/// \brief Represents a call to a C++ constructor.
///
/// Example: \c T(1)
class CXXConstructorCall : public AnyFunctionCall {
  friend class CallEventManager;

protected:
  /// Creates a constructor call.
  ///
  /// \param CE The constructor expression as written in the source.
  /// \param Target The region where the object should be constructed. If NULL,
  ///               a new symbolic region will be used.
  /// \param St The path-sensitive state at this point in the program.
  /// \param LCtx The location context at this point in the program.
  CXXConstructorCall(const CXXConstructExpr *CE, const MemRegion *Target,
                     ProgramStateRef St, const LocationContext *LCtx)
    : AnyFunctionCall(CE, St, LCtx) {
    Data = Target;
  }

  CXXConstructorCall(const CXXConstructorCall &Other) : AnyFunctionCall(Other){}
  virtual void cloneTo(void *Dest) const { new (Dest) CXXConstructorCall(*this); }

  virtual void getExtraInvalidatedRegions(RegionList &Regions) const;

public:
  virtual const CXXConstructExpr *getOriginExpr() const {
    return cast<CXXConstructExpr>(AnyFunctionCall::getOriginExpr());
  }

  virtual const CXXConstructorDecl *getDecl() const {
    return getOriginExpr()->getConstructor();
  }

  virtual unsigned getNumArgs() const { return getOriginExpr()->getNumArgs(); }

  virtual const Expr *getArgExpr(unsigned Index) const {
    return getOriginExpr()->getArg(Index);
  }

  /// \brief Returns the value of the implicit 'this' object.
  SVal getCXXThisVal() const;

  virtual void getInitialStackFrameContents(const StackFrameContext *CalleeCtx,
                                            BindingsTy &Bindings) const;

  virtual Kind getKind() const { return CE_CXXConstructor; }

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_CXXConstructor;
  }
};

/// \brief Represents the memory allocation call in a C++ new-expression.
///
/// This is a call to "operator new".
class CXXAllocatorCall : public AnyFunctionCall {
  friend class CallEventManager;

protected:
  CXXAllocatorCall(const CXXNewExpr *E, ProgramStateRef St,
                   const LocationContext *LCtx)
    : AnyFunctionCall(E, St, LCtx) {}

  CXXAllocatorCall(const CXXAllocatorCall &Other) : AnyFunctionCall(Other) {}
  virtual void cloneTo(void *Dest) const { new (Dest) CXXAllocatorCall(*this); }

public:
  virtual const CXXNewExpr *getOriginExpr() const {
    return cast<CXXNewExpr>(AnyFunctionCall::getOriginExpr());
  }

  virtual const FunctionDecl *getDecl() const {
    return getOriginExpr()->getOperatorNew();
  }

  virtual unsigned getNumArgs() const {
    return getOriginExpr()->getNumPlacementArgs() + 1;
  }

  virtual const Expr *getArgExpr(unsigned Index) const {
    // The first argument of an allocator call is the size of the allocation.
    if (Index == 0)
      return 0;
    return getOriginExpr()->getPlacementArg(Index - 1);
  }

  virtual Kind getKind() const { return CE_CXXAllocator; }

  static bool classof(const CallEvent *CE) {
    return CE->getKind() == CE_CXXAllocator;
  }
};

/// \brief Represents the ways an Objective-C message send can occur.
//
// Note to maintainers: OCM_Message should always be last, since it does not
// need to fit in the Data field's low bits.
enum ObjCMessageKind {
  OCM_PropertyAccess,
  OCM_Subscript,
  OCM_Message
};

/// \brief Represents any expression that calls an Objective-C method.
///
/// This includes all of the kinds listed in ObjCMessageKind.
class ObjCMethodCall : public CallEvent {
  friend class CallEventManager;

  const PseudoObjectExpr *getContainingPseudoObjectExpr() const;

protected:
  ObjCMethodCall(const ObjCMessageExpr *Msg, ProgramStateRef St,
                 const LocationContext *LCtx)
    : CallEvent(Msg, St, LCtx) {
    Data = 0;
  }

  ObjCMethodCall(const ObjCMethodCall &Other) : CallEvent(Other) {}
  virtual void cloneTo(void *Dest) const { new (Dest) ObjCMethodCall(*this); }

  virtual void getExtraInvalidatedRegions(RegionList &Regions) const;

  /// Check if the selector may have multiple definitions (may have overrides).
  virtual bool canBeOverridenInSubclass(ObjCInterfaceDecl *IDecl,
                                        Selector Sel) const;

public:
  virtual const ObjCMessageExpr *getOriginExpr() const {
    return cast<ObjCMessageExpr>(CallEvent::getOriginExpr());
  }
  virtual const ObjCMethodDecl *getDecl() const {
    return getOriginExpr()->getMethodDecl();
  }
  virtual unsigned getNumArgs() const {
    return getOriginExpr()->getNumArgs();
  }
  virtual const Expr *getArgExpr(unsigned Index) const {
    return getOriginExpr()->getArg(Index);
  }

  bool isInstanceMessage() const {
    return getOriginExpr()->isInstanceMessage();
  }
  ObjCMethodFamily getMethodFamily() const {
    return getOriginExpr()->getMethodFamily();
  }
  Selector getSelector() const {
    return getOriginExpr()->getSelector();
  }

  virtual SourceRange getSourceRange() const;

  /// \brief Returns the value of the receiver at the time of this call.
  SVal getReceiverSVal() const;

  /// \brief Return the value of 'self' if available.
  SVal getSelfSVal() const;

  /// \brief Get the interface for the receiver.
  ///
  /// This works whether this is an instance message or a class message.
  /// However, it currently just uses the static type of the receiver.
  const ObjCInterfaceDecl *getReceiverInterface() const {
    return getOriginExpr()->getReceiverInterface();
  }

  /// \brief Checks if the receiver refers to 'self' or 'super'.
  bool isReceiverSelfOrSuper() const;

  /// Returns how the message was written in the source (property access,
  /// subscript, or explicit message send).
  ObjCMessageKind getMessageKind() const;

  /// Returns true if this property access or subscript is a setter (has the
  /// form of an assignment).
  bool isSetter() const {
    switch (getMessageKind()) {
    case OCM_Message:
      llvm_unreachable("This is not a pseudo-object access!");
    case OCM_PropertyAccess:
      return getNumArgs() > 0;
    case OCM_Subscript:
      return getNumArgs() > 1;
    }
    llvm_unreachable("Unknown message kind");
  }

  virtual RuntimeDefinition getRuntimeDefinition() const;

  virtual void getInitialStackFrameContents(const StackFrameContext *CalleeCtx,
                                            BindingsTy &Bindings) const;

  virtual param_iterator param_begin() const;
  virtual param_iterator param_end() const;

  virtual Kind getKind() const { return CE_ObjCMessage; }

  static bool classof(const CallEvent *CA) {
    return CA->getKind() == CE_ObjCMessage;
  }
};


/// \brief Manages the lifetime of CallEvent objects.
///
/// CallEventManager provides a way to create arbitrary CallEvents "on the
/// stack" as if they were value objects by keeping a cache of CallEvent-sized
/// memory blocks. The CallEvents created by CallEventManager are only valid
/// for the lifetime of the OwnedCallEvent that holds them; right now these
/// objects cannot be copied and ownership cannot be transferred.
class CallEventManager {
  friend class CallEvent;

  llvm::BumpPtrAllocator &Alloc;
  SmallVector<void *, 8> Cache;

  void reclaim(const void *Memory) {
    Cache.push_back(const_cast<void *>(Memory));
  }

  /// Returns memory that can be initialized as a CallEvent.
  void *allocate() {
    if (Cache.empty())
      return Alloc.Allocate<FunctionCall>();
    else
      return Cache.pop_back_val();
  }

  template <typename T, typename Arg>
  T *create(Arg A, ProgramStateRef St, const LocationContext *LCtx) {
    return new (allocate()) T(A, St, LCtx);
  }

  template <typename T, typename Arg1, typename Arg2>
  T *create(Arg1 A1, Arg2 A2, ProgramStateRef St, const LocationContext *LCtx) {
    return new (allocate()) T(A1, A2, St, LCtx);
  }

  template <typename T, typename Arg1, typename Arg2, typename Arg3>
  T *create(Arg1 A1, Arg2 A2, Arg3 A3, ProgramStateRef St,
            const LocationContext *LCtx) {
    return new (allocate()) T(A1, A2, A3, St, LCtx);
  }

  template <typename T, typename Arg1, typename Arg2, typename Arg3,
            typename Arg4>
  T *create(Arg1 A1, Arg2 A2, Arg3 A3, Arg4 A4, ProgramStateRef St,
            const LocationContext *LCtx) {
    return new (allocate()) T(A1, A2, A3, A4, St, LCtx);
  }

public:
  CallEventManager(llvm::BumpPtrAllocator &alloc) : Alloc(alloc) {}


  CallEventRef<>
  getCaller(const StackFrameContext *CalleeCtx, ProgramStateRef State);


  CallEventRef<>
  getSimpleCall(const CallExpr *E, ProgramStateRef State,
                const LocationContext *LCtx);

  CallEventRef<ObjCMethodCall>
  getObjCMethodCall(const ObjCMessageExpr *E, ProgramStateRef State,
                    const LocationContext *LCtx) {
    return create<ObjCMethodCall>(E, State, LCtx);
  }

  CallEventRef<CXXConstructorCall>
  getCXXConstructorCall(const CXXConstructExpr *E, const MemRegion *Target,
                        ProgramStateRef State, const LocationContext *LCtx) {
    return create<CXXConstructorCall>(E, Target, State, LCtx);
  }

  CallEventRef<CXXDestructorCall>
  getCXXDestructorCall(const CXXDestructorDecl *DD, const Stmt *Trigger,
                       const MemRegion *Target, bool IsBase,
                       ProgramStateRef State, const LocationContext *LCtx) {
    return create<CXXDestructorCall>(DD, Trigger, Target, IsBase, State, LCtx);
  }

  CallEventRef<CXXAllocatorCall>
  getCXXAllocatorCall(const CXXNewExpr *E, ProgramStateRef State,
                      const LocationContext *LCtx) {
    return create<CXXAllocatorCall>(E, State, LCtx);
  }
};


template <typename T>
CallEventRef<T> CallEvent::cloneWithState(ProgramStateRef NewState) const {
  assert(isa<T>(*this) && "Cloning to unrelated type");
  assert(sizeof(T) == sizeof(CallEvent) && "Subclasses may not add fields");

  if (NewState == State)
    return cast<T>(this);

  CallEventManager &Mgr = State->getStateManager().getCallEventManager();
  T *Copy = static_cast<T *>(Mgr.allocate());
  cloneTo(Copy);
  assert(Copy->getKind() == this->getKind() && "Bad copy");

  Copy->State = NewState;
  return Copy;
}

inline void CallEvent::Release() const {
  assert(RefCount > 0 && "Reference count is already zero.");
  --RefCount;

  if (RefCount > 0)
    return;

  CallEventManager &Mgr = State->getStateManager().getCallEventManager();
  Mgr.reclaim(this);

  this->~CallEvent();
}

} // end namespace ento
} // end namespace clang

namespace llvm {
  // Support isa<>, cast<>, and dyn_cast<> for CallEventRef.
  template<class T> struct simplify_type< clang::ento::CallEventRef<T> > {
    typedef const T *SimpleType;

    static SimpleType
    getSimplifiedValue(const clang::ento::CallEventRef<T>& Val) {
      return Val.getPtr();
    }
  };
}

#endif
