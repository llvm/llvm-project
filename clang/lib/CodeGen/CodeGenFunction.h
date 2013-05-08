//===-- CodeGenFunction.h - Per-Function state for LLVM CodeGen -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-function state used for llvm translation.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CODEGENFUNCTION_H
#define CLANG_CODEGEN_CODEGENFUNCTION_H

#include "CGBuilder.h"
#include "CGDebugInfo.h"
#include "CGValue.h"
#include "CodeGenModule.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Type.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ValueHandle.h"

namespace llvm {
  class BasicBlock;
  class LLVMContext;
  class MDNode;
  class Module;
  class SwitchInst;
  class Twine;
  class Value;
  class CallSite;
}

namespace clang {
  class ASTContext;
  class BlockDecl;
  class CXXDestructorDecl;
  class CXXForRangeStmt;
  class CXXTryStmt;
  class Decl;
  class LabelDecl;
  class EnumConstantDecl;
  class FunctionDecl;
  class FunctionProtoType;
  class LabelStmt;
  class ObjCContainerDecl;
  class ObjCInterfaceDecl;
  class ObjCIvarDecl;
  class ObjCMethodDecl;
  class ObjCImplementationDecl;
  class ObjCPropertyImplDecl;
  class TargetInfo;
  class TargetCodeGenInfo;
  class VarDecl;
  class ObjCForCollectionStmt;
  class ObjCAtTryStmt;
  class ObjCAtThrowStmt;
  class ObjCAtSynchronizedStmt;
  class ObjCAutoreleasePoolStmt;

namespace CodeGen {
  class CodeGenTypes;
  class CGFunctionInfo;
  class CGRecordLayout;
  class CGBlockInfo;
  class CGCXXABI;
  class BlockFlags;
  class BlockFieldFlags;

/// The kind of evaluation to perform on values of a particular
/// type.  Basically, is the code in CGExprScalar, CGExprComplex, or
/// CGExprAgg?
///
/// TODO: should vectors maybe be split out into their own thing?
enum TypeEvaluationKind {
  TEK_Scalar,
  TEK_Complex,
  TEK_Aggregate
};

/// A branch fixup.  These are required when emitting a goto to a
/// label which hasn't been emitted yet.  The goto is optimistically
/// emitted as a branch to the basic block for the label, and (if it
/// occurs in a scope with non-trivial cleanups) a fixup is added to
/// the innermost cleanup.  When a (normal) cleanup is popped, any
/// unresolved fixups in that scope are threaded through the cleanup.
struct BranchFixup {
  /// The block containing the terminator which needs to be modified
  /// into a switch if this fixup is resolved into the current scope.
  /// If null, LatestBranch points directly to the destination.
  llvm::BasicBlock *OptimisticBranchBlock;

  /// The ultimate destination of the branch.
  ///
  /// This can be set to null to indicate that this fixup was
  /// successfully resolved.
  llvm::BasicBlock *Destination;

  /// The destination index value.
  unsigned DestinationIndex;

  /// The initial branch of the fixup.
  llvm::BranchInst *InitialBranch;
};

template <class T> struct InvariantValue {
  typedef T type;
  typedef T saved_type;
  static bool needsSaving(type value) { return false; }
  static saved_type save(CodeGenFunction &CGF, type value) { return value; }
  static type restore(CodeGenFunction &CGF, saved_type value) { return value; }
};

/// A metaprogramming class for ensuring that a value will dominate an
/// arbitrary position in a function.
template <class T> struct DominatingValue : InvariantValue<T> {};

template <class T, bool mightBeInstruction =
            llvm::is_base_of<llvm::Value, T>::value &&
            !llvm::is_base_of<llvm::Constant, T>::value &&
            !llvm::is_base_of<llvm::BasicBlock, T>::value>
struct DominatingPointer;
template <class T> struct DominatingPointer<T,false> : InvariantValue<T*> {};
// template <class T> struct DominatingPointer<T,true> at end of file

template <class T> struct DominatingValue<T*> : DominatingPointer<T> {};

enum CleanupKind {
  EHCleanup = 0x1,
  NormalCleanup = 0x2,
  NormalAndEHCleanup = EHCleanup | NormalCleanup,

  InactiveCleanup = 0x4,
  InactiveEHCleanup = EHCleanup | InactiveCleanup,
  InactiveNormalCleanup = NormalCleanup | InactiveCleanup,
  InactiveNormalAndEHCleanup = NormalAndEHCleanup | InactiveCleanup
};

/// A stack of scopes which respond to exceptions, including cleanups
/// and catch blocks.
class EHScopeStack {
public:
  /// A saved depth on the scope stack.  This is necessary because
  /// pushing scopes onto the stack invalidates iterators.
  class stable_iterator {
    friend class EHScopeStack;

    /// Offset from StartOfData to EndOfBuffer.
    ptrdiff_t Size;

    stable_iterator(ptrdiff_t Size) : Size(Size) {}

  public:
    static stable_iterator invalid() { return stable_iterator(-1); }
    stable_iterator() : Size(-1) {}

    bool isValid() const { return Size >= 0; }

    /// Returns true if this scope encloses I.
    /// Returns false if I is invalid.
    /// This scope must be valid.
    bool encloses(stable_iterator I) const { return Size <= I.Size; }

    /// Returns true if this scope strictly encloses I: that is,
    /// if it encloses I and is not I.
    /// Returns false is I is invalid.
    /// This scope must be valid.
    bool strictlyEncloses(stable_iterator I) const { return Size < I.Size; }

    friend bool operator==(stable_iterator A, stable_iterator B) {
      return A.Size == B.Size;
    }
    friend bool operator!=(stable_iterator A, stable_iterator B) {
      return A.Size != B.Size;
    }
  };

  /// Information for lazily generating a cleanup.  Subclasses must be
  /// POD-like: cleanups will not be destructed, and they will be
  /// allocated on the cleanup stack and freely copied and moved
  /// around.
  ///
  /// Cleanup implementations should generally be declared in an
  /// anonymous namespace.
  class Cleanup {
    // Anchor the construction vtable.
    virtual void anchor();
  public:
    /// Generation flags.
    class Flags {
      enum {
        F_IsForEH             = 0x1,
        F_IsNormalCleanupKind = 0x2,
        F_IsEHCleanupKind     = 0x4
      };
      unsigned flags;

    public:
      Flags() : flags(0) {}

      /// isForEH - true if the current emission is for an EH cleanup.
      bool isForEHCleanup() const { return flags & F_IsForEH; }
      bool isForNormalCleanup() const { return !isForEHCleanup(); }
      void setIsForEHCleanup() { flags |= F_IsForEH; }

      bool isNormalCleanupKind() const { return flags & F_IsNormalCleanupKind; }
      void setIsNormalCleanupKind() { flags |= F_IsNormalCleanupKind; }

      /// isEHCleanupKind - true if the cleanup was pushed as an EH
      /// cleanup.
      bool isEHCleanupKind() const { return flags & F_IsEHCleanupKind; }
      void setIsEHCleanupKind() { flags |= F_IsEHCleanupKind; }
    };

    // Provide a virtual destructor to suppress a very common warning
    // that unfortunately cannot be suppressed without this.  Cleanups
    // should not rely on this destructor ever being called.
    virtual ~Cleanup() {}

    /// Emit the cleanup.  For normal cleanups, this is run in the
    /// same EH context as when the cleanup was pushed, i.e. the
    /// immediately-enclosing context of the cleanup scope.  For
    /// EH cleanups, this is run in a terminate context.
    ///
    // \param flags cleanup kind.
    virtual void Emit(CodeGenFunction &CGF, Flags flags) = 0;
  };

  /// ConditionalCleanupN stores the saved form of its N parameters,
  /// then restores them and performs the cleanup.
  template <class T, class A0>
  class ConditionalCleanup1 : public Cleanup {
    typedef typename DominatingValue<A0>::saved_type A0_saved;
    A0_saved a0_saved;

    void Emit(CodeGenFunction &CGF, Flags flags) {
      A0 a0 = DominatingValue<A0>::restore(CGF, a0_saved);
      T(a0).Emit(CGF, flags);
    }

  public:
    ConditionalCleanup1(A0_saved a0)
      : a0_saved(a0) {}
  };

  template <class T, class A0, class A1>
  class ConditionalCleanup2 : public Cleanup {
    typedef typename DominatingValue<A0>::saved_type A0_saved;
    typedef typename DominatingValue<A1>::saved_type A1_saved;
    A0_saved a0_saved;
    A1_saved a1_saved;

    void Emit(CodeGenFunction &CGF, Flags flags) {
      A0 a0 = DominatingValue<A0>::restore(CGF, a0_saved);
      A1 a1 = DominatingValue<A1>::restore(CGF, a1_saved);
      T(a0, a1).Emit(CGF, flags);
    }

  public:
    ConditionalCleanup2(A0_saved a0, A1_saved a1)
      : a0_saved(a0), a1_saved(a1) {}
  };

  template <class T, class A0, class A1, class A2>
  class ConditionalCleanup3 : public Cleanup {
    typedef typename DominatingValue<A0>::saved_type A0_saved;
    typedef typename DominatingValue<A1>::saved_type A1_saved;
    typedef typename DominatingValue<A2>::saved_type A2_saved;
    A0_saved a0_saved;
    A1_saved a1_saved;
    A2_saved a2_saved;
    
    void Emit(CodeGenFunction &CGF, Flags flags) {
      A0 a0 = DominatingValue<A0>::restore(CGF, a0_saved);
      A1 a1 = DominatingValue<A1>::restore(CGF, a1_saved);
      A2 a2 = DominatingValue<A2>::restore(CGF, a2_saved);
      T(a0, a1, a2).Emit(CGF, flags);
    }
    
  public:
    ConditionalCleanup3(A0_saved a0, A1_saved a1, A2_saved a2)
      : a0_saved(a0), a1_saved(a1), a2_saved(a2) {}
  };

  template <class T, class A0, class A1, class A2, class A3>
  class ConditionalCleanup4 : public Cleanup {
    typedef typename DominatingValue<A0>::saved_type A0_saved;
    typedef typename DominatingValue<A1>::saved_type A1_saved;
    typedef typename DominatingValue<A2>::saved_type A2_saved;
    typedef typename DominatingValue<A3>::saved_type A3_saved;
    A0_saved a0_saved;
    A1_saved a1_saved;
    A2_saved a2_saved;
    A3_saved a3_saved;
    
    void Emit(CodeGenFunction &CGF, Flags flags) {
      A0 a0 = DominatingValue<A0>::restore(CGF, a0_saved);
      A1 a1 = DominatingValue<A1>::restore(CGF, a1_saved);
      A2 a2 = DominatingValue<A2>::restore(CGF, a2_saved);
      A3 a3 = DominatingValue<A3>::restore(CGF, a3_saved);
      T(a0, a1, a2, a3).Emit(CGF, flags);
    }
    
  public:
    ConditionalCleanup4(A0_saved a0, A1_saved a1, A2_saved a2, A3_saved a3)
      : a0_saved(a0), a1_saved(a1), a2_saved(a2), a3_saved(a3) {}
  };

private:
  // The implementation for this class is in CGException.h and
  // CGException.cpp; the definition is here because it's used as a
  // member of CodeGenFunction.

  /// The start of the scope-stack buffer, i.e. the allocated pointer
  /// for the buffer.  All of these pointers are either simultaneously
  /// null or simultaneously valid.
  char *StartOfBuffer;

  /// The end of the buffer.
  char *EndOfBuffer;

  /// The first valid entry in the buffer.
  char *StartOfData;

  /// The innermost normal cleanup on the stack.
  stable_iterator InnermostNormalCleanup;

  /// The innermost EH scope on the stack.
  stable_iterator InnermostEHScope;

  /// The current set of branch fixups.  A branch fixup is a jump to
  /// an as-yet unemitted label, i.e. a label for which we don't yet
  /// know the EH stack depth.  Whenever we pop a cleanup, we have
  /// to thread all the current branch fixups through it.
  ///
  /// Fixups are recorded as the Use of the respective branch or
  /// switch statement.  The use points to the final destination.
  /// When popping out of a cleanup, these uses are threaded through
  /// the cleanup and adjusted to point to the new cleanup.
  ///
  /// Note that branches are allowed to jump into protected scopes
  /// in certain situations;  e.g. the following code is legal:
  ///     struct A { ~A(); }; // trivial ctor, non-trivial dtor
  ///     goto foo;
  ///     A a;
  ///    foo:
  ///     bar();
  SmallVector<BranchFixup, 8> BranchFixups;

  char *allocate(size_t Size);

  void *pushCleanup(CleanupKind K, size_t DataSize);

public:
  EHScopeStack() : StartOfBuffer(0), EndOfBuffer(0), StartOfData(0),
                   InnermostNormalCleanup(stable_end()),
                   InnermostEHScope(stable_end()) {}
  ~EHScopeStack() { delete[] StartOfBuffer; }

  // Variadic templates would make this not terrible.

  /// Push a lazily-created cleanup on the stack.
  template <class T>
  void pushCleanup(CleanupKind Kind) {
    void *Buffer = pushCleanup(Kind, sizeof(T));
    Cleanup *Obj = new(Buffer) T();
    (void) Obj;
  }

  /// Push a lazily-created cleanup on the stack.
  template <class T, class A0>
  void pushCleanup(CleanupKind Kind, A0 a0) {
    void *Buffer = pushCleanup(Kind, sizeof(T));
    Cleanup *Obj = new(Buffer) T(a0);
    (void) Obj;
  }

  /// Push a lazily-created cleanup on the stack.
  template <class T, class A0, class A1>
  void pushCleanup(CleanupKind Kind, A0 a0, A1 a1) {
    void *Buffer = pushCleanup(Kind, sizeof(T));
    Cleanup *Obj = new(Buffer) T(a0, a1);
    (void) Obj;
  }

  /// Push a lazily-created cleanup on the stack.
  template <class T, class A0, class A1, class A2>
  void pushCleanup(CleanupKind Kind, A0 a0, A1 a1, A2 a2) {
    void *Buffer = pushCleanup(Kind, sizeof(T));
    Cleanup *Obj = new(Buffer) T(a0, a1, a2);
    (void) Obj;
  }

  /// Push a lazily-created cleanup on the stack.
  template <class T, class A0, class A1, class A2, class A3>
  void pushCleanup(CleanupKind Kind, A0 a0, A1 a1, A2 a2, A3 a3) {
    void *Buffer = pushCleanup(Kind, sizeof(T));
    Cleanup *Obj = new(Buffer) T(a0, a1, a2, a3);
    (void) Obj;
  }

  /// Push a lazily-created cleanup on the stack.
  template <class T, class A0, class A1, class A2, class A3, class A4>
  void pushCleanup(CleanupKind Kind, A0 a0, A1 a1, A2 a2, A3 a3, A4 a4) {
    void *Buffer = pushCleanup(Kind, sizeof(T));
    Cleanup *Obj = new(Buffer) T(a0, a1, a2, a3, a4);
    (void) Obj;
  }

  // Feel free to add more variants of the following:

  /// Push a cleanup with non-constant storage requirements on the
  /// stack.  The cleanup type must provide an additional static method:
  ///   static size_t getExtraSize(size_t);
  /// The argument to this method will be the value N, which will also
  /// be passed as the first argument to the constructor.
  ///
  /// The data stored in the extra storage must obey the same
  /// restrictions as normal cleanup member data.
  ///
  /// The pointer returned from this method is valid until the cleanup
  /// stack is modified.
  template <class T, class A0, class A1, class A2>
  T *pushCleanupWithExtra(CleanupKind Kind, size_t N, A0 a0, A1 a1, A2 a2) {
    void *Buffer = pushCleanup(Kind, sizeof(T) + T::getExtraSize(N));
    return new (Buffer) T(N, a0, a1, a2);
  }

  /// Pops a cleanup scope off the stack.  This is private to CGCleanup.cpp.
  void popCleanup();

  /// Push a set of catch handlers on the stack.  The catch is
  /// uninitialized and will need to have the given number of handlers
  /// set on it.
  class EHCatchScope *pushCatch(unsigned NumHandlers);

  /// Pops a catch scope off the stack.  This is private to CGException.cpp.
  void popCatch();

  /// Push an exceptions filter on the stack.
  class EHFilterScope *pushFilter(unsigned NumFilters);

  /// Pops an exceptions filter off the stack.
  void popFilter();

  /// Push a terminate handler on the stack.
  void pushTerminate();

  /// Pops a terminate handler off the stack.
  void popTerminate();

  /// Determines whether the exception-scopes stack is empty.
  bool empty() const { return StartOfData == EndOfBuffer; }

  bool requiresLandingPad() const {
    return InnermostEHScope != stable_end();
  }

  /// Determines whether there are any normal cleanups on the stack.
  bool hasNormalCleanups() const {
    return InnermostNormalCleanup != stable_end();
  }

  /// Returns the innermost normal cleanup on the stack, or
  /// stable_end() if there are no normal cleanups.
  stable_iterator getInnermostNormalCleanup() const {
    return InnermostNormalCleanup;
  }
  stable_iterator getInnermostActiveNormalCleanup() const;

  stable_iterator getInnermostEHScope() const {
    return InnermostEHScope;
  }

  stable_iterator getInnermostActiveEHScope() const;

  /// An unstable reference to a scope-stack depth.  Invalidated by
  /// pushes but not pops.
  class iterator;

  /// Returns an iterator pointing to the innermost EH scope.
  iterator begin() const;

  /// Returns an iterator pointing to the outermost EH scope.
  iterator end() const;

  /// Create a stable reference to the top of the EH stack.  The
  /// returned reference is valid until that scope is popped off the
  /// stack.
  stable_iterator stable_begin() const {
    return stable_iterator(EndOfBuffer - StartOfData);
  }

  /// Create a stable reference to the bottom of the EH stack.
  static stable_iterator stable_end() {
    return stable_iterator(0);
  }

  /// Translates an iterator into a stable_iterator.
  stable_iterator stabilize(iterator it) const;

  /// Turn a stable reference to a scope depth into a unstable pointer
  /// to the EH stack.
  iterator find(stable_iterator save) const;

  /// Removes the cleanup pointed to by the given stable_iterator.
  void removeCleanup(stable_iterator save);

  /// Add a branch fixup to the current cleanup scope.
  BranchFixup &addBranchFixup() {
    assert(hasNormalCleanups() && "adding fixup in scope without cleanups");
    BranchFixups.push_back(BranchFixup());
    return BranchFixups.back();
  }

  unsigned getNumBranchFixups() const { return BranchFixups.size(); }
  BranchFixup &getBranchFixup(unsigned I) {
    assert(I < getNumBranchFixups());
    return BranchFixups[I];
  }

  /// Pops lazily-removed fixups from the end of the list.  This
  /// should only be called by procedures which have just popped a
  /// cleanup or resolved one or more fixups.
  void popNullFixups();

  /// Clears the branch-fixups list.  This should only be called by
  /// ResolveAllBranchFixups.
  void clearFixups() { BranchFixups.clear(); }
};

/// CodeGenFunction - This class organizes the per-function state that is used
/// while generating LLVM code.
class CodeGenFunction : public CodeGenTypeCache {
  CodeGenFunction(const CodeGenFunction &) LLVM_DELETED_FUNCTION;
  void operator=(const CodeGenFunction &) LLVM_DELETED_FUNCTION;

  friend class CGCXXABI;
public:
  /// A jump destination is an abstract label, branching to which may
  /// require a jump out through normal cleanups.
  struct JumpDest {
    JumpDest() : Block(0), ScopeDepth(), Index(0) {}
    JumpDest(llvm::BasicBlock *Block,
             EHScopeStack::stable_iterator Depth,
             unsigned Index)
      : Block(Block), ScopeDepth(Depth), Index(Index) {}

    bool isValid() const { return Block != 0; }
    llvm::BasicBlock *getBlock() const { return Block; }
    EHScopeStack::stable_iterator getScopeDepth() const { return ScopeDepth; }
    unsigned getDestIndex() const { return Index; }

    // This should be used cautiously.
    void setScopeDepth(EHScopeStack::stable_iterator depth) {
      ScopeDepth = depth;
    }

  private:
    llvm::BasicBlock *Block;
    EHScopeStack::stable_iterator ScopeDepth;
    unsigned Index;
  };

  CodeGenModule &CGM;  // Per-module state.
  const TargetInfo &Target;

  typedef std::pair<llvm::Value *, llvm::Value *> ComplexPairTy;
  CGBuilderTy Builder;

  /// CurFuncDecl - Holds the Decl for the current outermost
  /// non-closure context.
  const Decl *CurFuncDecl;
  /// CurCodeDecl - This is the inner-most code context, which includes blocks.
  const Decl *CurCodeDecl;
  const CGFunctionInfo *CurFnInfo;
  QualType FnRetTy;
  llvm::Function *CurFn;

  /// CurGD - The GlobalDecl for the current function being compiled.
  GlobalDecl CurGD;

  /// PrologueCleanupDepth - The cleanup depth enclosing all the
  /// cleanups associated with the parameters.
  EHScopeStack::stable_iterator PrologueCleanupDepth;

  /// ReturnBlock - Unified return block.
  JumpDest ReturnBlock;

  /// ReturnValue - The temporary alloca to hold the return value. This is null
  /// iff the function has no return value.
  llvm::Value *ReturnValue;

  /// AllocaInsertPoint - This is an instruction in the entry block before which
  /// we prefer to insert allocas.
  llvm::AssertingVH<llvm::Instruction> AllocaInsertPt;

  /// BoundsChecking - Emit run-time bounds checks. Higher values mean
  /// potentially higher performance penalties.
  unsigned char BoundsChecking;

  /// \brief Whether any type-checking sanitizers are enabled. If \c false,
  /// calls to EmitTypeCheck can be skipped.
  bool SanitizePerformTypeCheck;

  /// \brief Sanitizer options to use for this function.
  const SanitizerOptions *SanOpts;

  /// In ARC, whether we should autorelease the return value.
  bool AutoreleaseResult;

  const CodeGen::CGBlockInfo *BlockInfo;
  llvm::Value *BlockPointer;

  llvm::DenseMap<const VarDecl *, FieldDecl *> LambdaCaptureFields;
  FieldDecl *LambdaThisCaptureField;

  /// \brief A mapping from NRVO variables to the flags used to indicate
  /// when the NRVO has been applied to this variable.
  llvm::DenseMap<const VarDecl *, llvm::Value *> NRVOFlags;

  EHScopeStack EHStack;

  /// i32s containing the indexes of the cleanup destinations.
  llvm::AllocaInst *NormalCleanupDest;

  unsigned NextCleanupDestIndex;

  /// FirstBlockInfo - The head of a singly-linked-list of block layouts.
  CGBlockInfo *FirstBlockInfo;

  /// EHResumeBlock - Unified block containing a call to llvm.eh.resume.
  llvm::BasicBlock *EHResumeBlock;

  /// The exception slot.  All landing pads write the current exception pointer
  /// into this alloca.
  llvm::Value *ExceptionSlot;

  /// The selector slot.  Under the MandatoryCleanup model, all landing pads
  /// write the current selector value into this alloca.
  llvm::AllocaInst *EHSelectorSlot;

  /// Emits a landing pad for the current EH stack.
  llvm::BasicBlock *EmitLandingPad();

  llvm::BasicBlock *getInvokeDestImpl();

  template <class T>
  typename DominatingValue<T>::saved_type saveValueInCond(T value) {
    return DominatingValue<T>::save(*this, value);
  }

public:
  /// ObjCEHValueStack - Stack of Objective-C exception values, used for
  /// rethrows.
  SmallVector<llvm::Value*, 8> ObjCEHValueStack;

  /// A class controlling the emission of a finally block.
  class FinallyInfo {
    /// Where the catchall's edge through the cleanup should go.
    JumpDest RethrowDest;

    /// A function to call to enter the catch.
    llvm::Constant *BeginCatchFn;

    /// An i1 variable indicating whether or not the @finally is
    /// running for an exception.
    llvm::AllocaInst *ForEHVar;

    /// An i8* variable into which the exception pointer to rethrow
    /// has been saved.
    llvm::AllocaInst *SavedExnVar;

  public:
    void enter(CodeGenFunction &CGF, const Stmt *Finally,
               llvm::Constant *beginCatchFn, llvm::Constant *endCatchFn,
               llvm::Constant *rethrowFn);
    void exit(CodeGenFunction &CGF);
  };

  /// pushFullExprCleanup - Push a cleanup to be run at the end of the
  /// current full-expression.  Safe against the possibility that
  /// we're currently inside a conditionally-evaluated expression.
  template <class T, class A0>
  void pushFullExprCleanup(CleanupKind kind, A0 a0) {
    // If we're not in a conditional branch, or if none of the
    // arguments requires saving, then use the unconditional cleanup.
    if (!isInConditionalBranch())
      return EHStack.pushCleanup<T>(kind, a0);

    typename DominatingValue<A0>::saved_type a0_saved = saveValueInCond(a0);

    typedef EHScopeStack::ConditionalCleanup1<T, A0> CleanupType;
    EHStack.pushCleanup<CleanupType>(kind, a0_saved);
    initFullExprCleanup();
  }

  /// pushFullExprCleanup - Push a cleanup to be run at the end of the
  /// current full-expression.  Safe against the possibility that
  /// we're currently inside a conditionally-evaluated expression.
  template <class T, class A0, class A1>
  void pushFullExprCleanup(CleanupKind kind, A0 a0, A1 a1) {
    // If we're not in a conditional branch, or if none of the
    // arguments requires saving, then use the unconditional cleanup.
    if (!isInConditionalBranch())
      return EHStack.pushCleanup<T>(kind, a0, a1);

    typename DominatingValue<A0>::saved_type a0_saved = saveValueInCond(a0);
    typename DominatingValue<A1>::saved_type a1_saved = saveValueInCond(a1);

    typedef EHScopeStack::ConditionalCleanup2<T, A0, A1> CleanupType;
    EHStack.pushCleanup<CleanupType>(kind, a0_saved, a1_saved);
    initFullExprCleanup();
  }

  /// pushFullExprCleanup - Push a cleanup to be run at the end of the
  /// current full-expression.  Safe against the possibility that
  /// we're currently inside a conditionally-evaluated expression.
  template <class T, class A0, class A1, class A2>
  void pushFullExprCleanup(CleanupKind kind, A0 a0, A1 a1, A2 a2) {
    // If we're not in a conditional branch, or if none of the
    // arguments requires saving, then use the unconditional cleanup.
    if (!isInConditionalBranch()) {
      return EHStack.pushCleanup<T>(kind, a0, a1, a2);
    }
    
    typename DominatingValue<A0>::saved_type a0_saved = saveValueInCond(a0);
    typename DominatingValue<A1>::saved_type a1_saved = saveValueInCond(a1);
    typename DominatingValue<A2>::saved_type a2_saved = saveValueInCond(a2);
    
    typedef EHScopeStack::ConditionalCleanup3<T, A0, A1, A2> CleanupType;
    EHStack.pushCleanup<CleanupType>(kind, a0_saved, a1_saved, a2_saved);
    initFullExprCleanup();
  }

  /// pushFullExprCleanup - Push a cleanup to be run at the end of the
  /// current full-expression.  Safe against the possibility that
  /// we're currently inside a conditionally-evaluated expression.
  template <class T, class A0, class A1, class A2, class A3>
  void pushFullExprCleanup(CleanupKind kind, A0 a0, A1 a1, A2 a2, A3 a3) {
    // If we're not in a conditional branch, or if none of the
    // arguments requires saving, then use the unconditional cleanup.
    if (!isInConditionalBranch()) {
      return EHStack.pushCleanup<T>(kind, a0, a1, a2, a3);
    }
    
    typename DominatingValue<A0>::saved_type a0_saved = saveValueInCond(a0);
    typename DominatingValue<A1>::saved_type a1_saved = saveValueInCond(a1);
    typename DominatingValue<A2>::saved_type a2_saved = saveValueInCond(a2);
    typename DominatingValue<A3>::saved_type a3_saved = saveValueInCond(a3);
    
    typedef EHScopeStack::ConditionalCleanup4<T, A0, A1, A2, A3> CleanupType;
    EHStack.pushCleanup<CleanupType>(kind, a0_saved, a1_saved,
                                     a2_saved, a3_saved);
    initFullExprCleanup();
  }

  /// Set up the last cleaup that was pushed as a conditional
  /// full-expression cleanup.
  void initFullExprCleanup();

  /// PushDestructorCleanup - Push a cleanup to call the
  /// complete-object destructor of an object of the given type at the
  /// given address.  Does nothing if T is not a C++ class type with a
  /// non-trivial destructor.
  void PushDestructorCleanup(QualType T, llvm::Value *Addr);

  /// PushDestructorCleanup - Push a cleanup to call the
  /// complete-object variant of the given destructor on the object at
  /// the given address.
  void PushDestructorCleanup(const CXXDestructorDecl *Dtor,
                             llvm::Value *Addr);

  /// PopCleanupBlock - Will pop the cleanup entry on the stack and
  /// process all branch fixups.
  /// \param EHLoc - Optional debug location for EH code.
  void PopCleanupBlock(bool FallThroughIsBranchThrough = false,
                       SourceLocation EHLoc=SourceLocation());

  /// DeactivateCleanupBlock - Deactivates the given cleanup block.
  /// The block cannot be reactivated.  Pops it if it's the top of the
  /// stack.
  ///
  /// \param DominatingIP - An instruction which is known to
  ///   dominate the current IP (if set) and which lies along
  ///   all paths of execution between the current IP and the
  ///   the point at which the cleanup comes into scope.
  void DeactivateCleanupBlock(EHScopeStack::stable_iterator Cleanup,
                              llvm::Instruction *DominatingIP);

  /// ActivateCleanupBlock - Activates an initially-inactive cleanup.
  /// Cannot be used to resurrect a deactivated cleanup.
  ///
  /// \param DominatingIP - An instruction which is known to
  ///   dominate the current IP (if set) and which lies along
  ///   all paths of execution between the current IP and the
  ///   the point at which the cleanup comes into scope.
  void ActivateCleanupBlock(EHScopeStack::stable_iterator Cleanup,
                            llvm::Instruction *DominatingIP);

  /// \brief Enters a new scope for capturing cleanups, all of which
  /// will be executed once the scope is exited.
  class RunCleanupsScope {
    EHScopeStack::stable_iterator CleanupStackDepth;
    bool OldDidCallStackSave;
  protected:
    bool PerformCleanup;
  private:

    RunCleanupsScope(const RunCleanupsScope &) LLVM_DELETED_FUNCTION;
    void operator=(const RunCleanupsScope &) LLVM_DELETED_FUNCTION;

  protected:
    CodeGenFunction& CGF;

  public:
    /// \brief Enter a new cleanup scope.
    explicit RunCleanupsScope(CodeGenFunction &CGF)
      : PerformCleanup(true), CGF(CGF)
    {
      CleanupStackDepth = CGF.EHStack.stable_begin();
      OldDidCallStackSave = CGF.DidCallStackSave;
      CGF.DidCallStackSave = false;
    }

    /// \brief Exit this cleanup scope, emitting any accumulated
    /// cleanups.
    ~RunCleanupsScope() {
      if (PerformCleanup) {
        CGF.DidCallStackSave = OldDidCallStackSave;
        CGF.PopCleanupBlocks(CleanupStackDepth);
      }
    }

    /// \brief Determine whether this scope requires any cleanups.
    bool requiresCleanups() const {
      return CGF.EHStack.stable_begin() != CleanupStackDepth;
    }

    /// \brief Force the emission of cleanups now, instead of waiting
    /// until this object is destroyed.
    void ForceCleanup() {
      assert(PerformCleanup && "Already forced cleanup");
      CGF.DidCallStackSave = OldDidCallStackSave;
      CGF.PopCleanupBlocks(CleanupStackDepth);
      PerformCleanup = false;
    }
  };

  class LexicalScope: protected RunCleanupsScope {
    SourceRange Range;
    SmallVector<const LabelDecl*, 4> Labels;
    LexicalScope *ParentScope;

    LexicalScope(const LexicalScope &) LLVM_DELETED_FUNCTION;
    void operator=(const LexicalScope &) LLVM_DELETED_FUNCTION;

  public:
    /// \brief Enter a new cleanup scope.
    explicit LexicalScope(CodeGenFunction &CGF, SourceRange Range)
      : RunCleanupsScope(CGF), Range(Range), ParentScope(CGF.CurLexicalScope) {
      CGF.CurLexicalScope = this;
      if (CGDebugInfo *DI = CGF.getDebugInfo())
        DI->EmitLexicalBlockStart(CGF.Builder, Range.getBegin());
    }

    void addLabel(const LabelDecl *label) {
      assert(PerformCleanup && "adding label to dead scope?");
      Labels.push_back(label);
    }

    /// \brief Exit this cleanup scope, emitting any accumulated
    /// cleanups.
    ~LexicalScope() {
      if (CGDebugInfo *DI = CGF.getDebugInfo())
        DI->EmitLexicalBlockEnd(CGF.Builder, Range.getEnd());

      // If we should perform a cleanup, force them now.  Note that
      // this ends the cleanup scope before rescoping any labels.
      if (PerformCleanup) ForceCleanup();
    }

    /// \brief Force the emission of cleanups now, instead of waiting
    /// until this object is destroyed.
    void ForceCleanup() {
      CGF.CurLexicalScope = ParentScope;
      RunCleanupsScope::ForceCleanup();

      if (!Labels.empty())
        rescopeLabels();
    }

    void rescopeLabels();
  };


  /// PopCleanupBlocks - Takes the old cleanup stack size and emits
  /// the cleanup blocks that have been added.
  /// \param EHLoc - Optional debug location for EH code.
  void PopCleanupBlocks(EHScopeStack::stable_iterator OldCleanupStackSize,
                        SourceLocation EHLoc=SourceLocation());

  void ResolveBranchFixups(llvm::BasicBlock *Target);

  /// The given basic block lies in the current EH scope, but may be a
  /// target of a potentially scope-crossing jump; get a stable handle
  /// to which we can perform this jump later.
  JumpDest getJumpDestInCurrentScope(llvm::BasicBlock *Target) {
    return JumpDest(Target,
                    EHStack.getInnermostNormalCleanup(),
                    NextCleanupDestIndex++);
  }

  /// The given basic block lies in the current EH scope, but may be a
  /// target of a potentially scope-crossing jump; get a stable handle
  /// to which we can perform this jump later.
  JumpDest getJumpDestInCurrentScope(StringRef Name = StringRef()) {
    return getJumpDestInCurrentScope(createBasicBlock(Name));
  }

  /// EmitBranchThroughCleanup - Emit a branch from the current insert
  /// block through the normal cleanup handling code (if any) and then
  /// on to \arg Dest.
  void EmitBranchThroughCleanup(JumpDest Dest);
  
  /// isObviouslyBranchWithoutCleanups - Return true if a branch to the
  /// specified destination obviously has no cleanups to run.  'false' is always
  /// a conservatively correct answer for this method.
  bool isObviouslyBranchWithoutCleanups(JumpDest Dest) const;

  /// popCatchScope - Pops the catch scope at the top of the EHScope
  /// stack, emitting any required code (other than the catch handlers
  /// themselves).
  void popCatchScope();

  llvm::BasicBlock *getEHResumeBlock(bool isCleanup);
  llvm::BasicBlock *getEHDispatchBlock(EHScopeStack::stable_iterator scope);

  /// An object to manage conditionally-evaluated expressions.
  class ConditionalEvaluation {
    llvm::BasicBlock *StartBB;

  public:
    ConditionalEvaluation(CodeGenFunction &CGF)
      : StartBB(CGF.Builder.GetInsertBlock()) {}

    void begin(CodeGenFunction &CGF) {
      assert(CGF.OutermostConditional != this);
      if (!CGF.OutermostConditional)
        CGF.OutermostConditional = this;
    }

    void end(CodeGenFunction &CGF) {
      assert(CGF.OutermostConditional != 0);
      if (CGF.OutermostConditional == this)
        CGF.OutermostConditional = 0;
    }

    /// Returns a block which will be executed prior to each
    /// evaluation of the conditional code.
    llvm::BasicBlock *getStartingBlock() const {
      return StartBB;
    }
  };

  /// isInConditionalBranch - Return true if we're currently emitting
  /// one branch or the other of a conditional expression.
  bool isInConditionalBranch() const { return OutermostConditional != 0; }

  void setBeforeOutermostConditional(llvm::Value *value, llvm::Value *addr) {
    assert(isInConditionalBranch());
    llvm::BasicBlock *block = OutermostConditional->getStartingBlock();
    new llvm::StoreInst(value, addr, &block->back());    
  }

  /// An RAII object to record that we're evaluating a statement
  /// expression.
  class StmtExprEvaluation {
    CodeGenFunction &CGF;

    /// We have to save the outermost conditional: cleanups in a
    /// statement expression aren't conditional just because the
    /// StmtExpr is.
    ConditionalEvaluation *SavedOutermostConditional;

  public:
    StmtExprEvaluation(CodeGenFunction &CGF)
      : CGF(CGF), SavedOutermostConditional(CGF.OutermostConditional) {
      CGF.OutermostConditional = 0;
    }

    ~StmtExprEvaluation() {
      CGF.OutermostConditional = SavedOutermostConditional;
      CGF.EnsureInsertPoint();
    }
  };

  /// An object which temporarily prevents a value from being
  /// destroyed by aggressive peephole optimizations that assume that
  /// all uses of a value have been realized in the IR.
  class PeepholeProtection {
    llvm::Instruction *Inst;
    friend class CodeGenFunction;

  public:
    PeepholeProtection() : Inst(0) {}
  };

  /// A non-RAII class containing all the information about a bound
  /// opaque value.  OpaqueValueMapping, below, is a RAII wrapper for
  /// this which makes individual mappings very simple; using this
  /// class directly is useful when you have a variable number of
  /// opaque values or don't want the RAII functionality for some
  /// reason.
  class OpaqueValueMappingData {
    const OpaqueValueExpr *OpaqueValue;
    bool BoundLValue;
    CodeGenFunction::PeepholeProtection Protection;

    OpaqueValueMappingData(const OpaqueValueExpr *ov,
                           bool boundLValue)
      : OpaqueValue(ov), BoundLValue(boundLValue) {}
  public:
    OpaqueValueMappingData() : OpaqueValue(0) {}

    static bool shouldBindAsLValue(const Expr *expr) {
      // gl-values should be bound as l-values for obvious reasons.
      // Records should be bound as l-values because IR generation
      // always keeps them in memory.  Expressions of function type
      // act exactly like l-values but are formally required to be
      // r-values in C.
      return expr->isGLValue() ||
             expr->getType()->isRecordType() ||
             expr->getType()->isFunctionType();
    }

    static OpaqueValueMappingData bind(CodeGenFunction &CGF,
                                       const OpaqueValueExpr *ov,
                                       const Expr *e) {
      if (shouldBindAsLValue(ov))
        return bind(CGF, ov, CGF.EmitLValue(e));
      return bind(CGF, ov, CGF.EmitAnyExpr(e));
    }

    static OpaqueValueMappingData bind(CodeGenFunction &CGF,
                                       const OpaqueValueExpr *ov,
                                       const LValue &lv) {
      assert(shouldBindAsLValue(ov));
      CGF.OpaqueLValues.insert(std::make_pair(ov, lv));
      return OpaqueValueMappingData(ov, true);
    }

    static OpaqueValueMappingData bind(CodeGenFunction &CGF,
                                       const OpaqueValueExpr *ov,
                                       const RValue &rv) {
      assert(!shouldBindAsLValue(ov));
      CGF.OpaqueRValues.insert(std::make_pair(ov, rv));

      OpaqueValueMappingData data(ov, false);

      // Work around an extremely aggressive peephole optimization in
      // EmitScalarConversion which assumes that all other uses of a
      // value are extant.
      data.Protection = CGF.protectFromPeepholes(rv);

      return data;
    }

    bool isValid() const { return OpaqueValue != 0; }
    void clear() { OpaqueValue = 0; }

    void unbind(CodeGenFunction &CGF) {
      assert(OpaqueValue && "no data to unbind!");

      if (BoundLValue) {
        CGF.OpaqueLValues.erase(OpaqueValue);
      } else {
        CGF.OpaqueRValues.erase(OpaqueValue);
        CGF.unprotectFromPeepholes(Protection);
      }
    }
  };

  /// An RAII object to set (and then clear) a mapping for an OpaqueValueExpr.
  class OpaqueValueMapping {
    CodeGenFunction &CGF;
    OpaqueValueMappingData Data;

  public:
    static bool shouldBindAsLValue(const Expr *expr) {
      return OpaqueValueMappingData::shouldBindAsLValue(expr);
    }

    /// Build the opaque value mapping for the given conditional
    /// operator if it's the GNU ?: extension.  This is a common
    /// enough pattern that the convenience operator is really
    /// helpful.
    ///
    OpaqueValueMapping(CodeGenFunction &CGF,
                       const AbstractConditionalOperator *op) : CGF(CGF) {
      if (isa<ConditionalOperator>(op))
        // Leave Data empty.
        return;

      const BinaryConditionalOperator *e = cast<BinaryConditionalOperator>(op);
      Data = OpaqueValueMappingData::bind(CGF, e->getOpaqueValue(),
                                          e->getCommon());
    }

    OpaqueValueMapping(CodeGenFunction &CGF,
                       const OpaqueValueExpr *opaqueValue,
                       LValue lvalue)
      : CGF(CGF), Data(OpaqueValueMappingData::bind(CGF, opaqueValue, lvalue)) {
    }

    OpaqueValueMapping(CodeGenFunction &CGF,
                       const OpaqueValueExpr *opaqueValue,
                       RValue rvalue)
      : CGF(CGF), Data(OpaqueValueMappingData::bind(CGF, opaqueValue, rvalue)) {
    }

    void pop() {
      Data.unbind(CGF);
      Data.clear();
    }

    ~OpaqueValueMapping() {
      if (Data.isValid()) Data.unbind(CGF);
    }
  };
  
  /// getByrefValueFieldNumber - Given a declaration, returns the LLVM field
  /// number that holds the value.
  unsigned getByRefValueLLVMField(const ValueDecl *VD) const;

  /// BuildBlockByrefAddress - Computes address location of the
  /// variable which is declared as __block.
  llvm::Value *BuildBlockByrefAddress(llvm::Value *BaseAddr,
                                      const VarDecl *V);
private:
  CGDebugInfo *DebugInfo;
  bool DisableDebugInfo;

  /// If the current function returns 'this', use the field to keep track of
  /// the callee that returns 'this'.
  llvm::Value *CalleeWithThisReturn;

  /// DidCallStackSave - Whether llvm.stacksave has been called. Used to avoid
  /// calling llvm.stacksave for multiple VLAs in the same scope.
  bool DidCallStackSave;

  /// IndirectBranch - The first time an indirect goto is seen we create a block
  /// with an indirect branch.  Every time we see the address of a label taken,
  /// we add the label to the indirect goto.  Every subsequent indirect goto is
  /// codegen'd as a jump to the IndirectBranch's basic block.
  llvm::IndirectBrInst *IndirectBranch;

  /// LocalDeclMap - This keeps track of the LLVM allocas or globals for local C
  /// decls.
  typedef llvm::DenseMap<const Decl*, llvm::Value*> DeclMapTy;
  DeclMapTy LocalDeclMap;

  /// LabelMap - This keeps track of the LLVM basic block for each C label.
  llvm::DenseMap<const LabelDecl*, JumpDest> LabelMap;

  // BreakContinueStack - This keeps track of where break and continue
  // statements should jump to.
  struct BreakContinue {
    BreakContinue(JumpDest Break, JumpDest Continue)
      : BreakBlock(Break), ContinueBlock(Continue) {}

    JumpDest BreakBlock;
    JumpDest ContinueBlock;
  };
  SmallVector<BreakContinue, 8> BreakContinueStack;

  /// SwitchInsn - This is nearest current switch instruction. It is null if
  /// current context is not in a switch.
  llvm::SwitchInst *SwitchInsn;

  /// CaseRangeBlock - This block holds if condition check for last case
  /// statement range in current switch instruction.
  llvm::BasicBlock *CaseRangeBlock;

  /// OpaqueLValues - Keeps track of the current set of opaque value
  /// expressions.
  llvm::DenseMap<const OpaqueValueExpr *, LValue> OpaqueLValues;
  llvm::DenseMap<const OpaqueValueExpr *, RValue> OpaqueRValues;

  // VLASizeMap - This keeps track of the associated size for each VLA type.
  // We track this by the size expression rather than the type itself because
  // in certain situations, like a const qualifier applied to an VLA typedef,
  // multiple VLA types can share the same size expression.
  // FIXME: Maybe this could be a stack of maps that is pushed/popped as we
  // enter/leave scopes.
  llvm::DenseMap<const Expr*, llvm::Value*> VLASizeMap;

  /// A block containing a single 'unreachable' instruction.  Created
  /// lazily by getUnreachableBlock().
  llvm::BasicBlock *UnreachableBlock;

  /// Counts of the number return expressions in the function.
  unsigned NumReturnExprs;

  /// Count the number of simple (constant) return expressions in the function.
  unsigned NumSimpleReturnExprs;

  /// The last regular (non-return) debug location (breakpoint) in the function.
  SourceLocation LastStopPoint;

public:
  /// A scope within which we are constructing the fields of an object which
  /// might use a CXXDefaultInitExpr. This stashes away a 'this' value to use
  /// if we need to evaluate a CXXDefaultInitExpr within the evaluation.
  class FieldConstructionScope {
  public:
    FieldConstructionScope(CodeGenFunction &CGF, llvm::Value *This)
        : CGF(CGF), OldCXXDefaultInitExprThis(CGF.CXXDefaultInitExprThis) {
      CGF.CXXDefaultInitExprThis = This;
    }
    ~FieldConstructionScope() {
      CGF.CXXDefaultInitExprThis = OldCXXDefaultInitExprThis;
    }

  private:
    CodeGenFunction &CGF;
    llvm::Value *OldCXXDefaultInitExprThis;
  };

  /// The scope of a CXXDefaultInitExpr. Within this scope, the value of 'this'
  /// is overridden to be the object under construction.
  class CXXDefaultInitExprScope {
  public:
    CXXDefaultInitExprScope(CodeGenFunction &CGF)
        : CGF(CGF), OldCXXThisValue(CGF.CXXThisValue) {
      CGF.CXXThisValue = CGF.CXXDefaultInitExprThis;
    }
    ~CXXDefaultInitExprScope() {
      CGF.CXXThisValue = OldCXXThisValue;
    }

  public:
    CodeGenFunction &CGF;
    llvm::Value *OldCXXThisValue;
  };

private:
  /// CXXThisDecl - When generating code for a C++ member function,
  /// this will hold the implicit 'this' declaration.
  ImplicitParamDecl *CXXABIThisDecl;
  llvm::Value *CXXABIThisValue;
  llvm::Value *CXXThisValue;

  /// The value of 'this' to use when evaluating CXXDefaultInitExprs within
  /// this expression.
  llvm::Value *CXXDefaultInitExprThis;

  /// CXXStructorImplicitParamDecl - When generating code for a constructor or
  /// destructor, this will hold the implicit argument (e.g. VTT).
  ImplicitParamDecl *CXXStructorImplicitParamDecl;
  llvm::Value *CXXStructorImplicitParamValue;

  /// OutermostConditional - Points to the outermost active
  /// conditional control.  This is used so that we know if a
  /// temporary should be destroyed conditionally.
  ConditionalEvaluation *OutermostConditional;

  /// The current lexical scope.
  LexicalScope *CurLexicalScope;

  /// ByrefValueInfoMap - For each __block variable, contains a pair of the LLVM
  /// type as well as the field number that contains the actual data.
  llvm::DenseMap<const ValueDecl *, std::pair<llvm::Type *,
                                              unsigned> > ByRefValueInfo;

  llvm::BasicBlock *TerminateLandingPad;
  llvm::BasicBlock *TerminateHandler;
  llvm::BasicBlock *TrapBB;

  /// Add a kernel metadata node to the named metadata node 'opencl.kernels'.
  /// In the kernel metadata node, reference the kernel function and metadata 
  /// nodes for its optional attribute qualifiers (OpenCL 1.1 6.7.2):
  /// - A node for the vec_type_hint(<type>) qualifier contains string
  ///   "vec_type_hint", an undefined value of the <type> data type,
  ///   and a Boolean that is true if the <type> is integer and signed.
  /// - A node for the work_group_size_hint(X,Y,Z) qualifier contains string 
  ///   "work_group_size_hint", and three 32-bit integers X, Y and Z.
  /// - A node for the reqd_work_group_size(X,Y,Z) qualifier contains string 
  ///   "reqd_work_group_size", and three 32-bit integers X, Y and Z.
  void EmitOpenCLKernelMetadata(const FunctionDecl *FD, 
                                llvm::Function *Fn);

public:
  CodeGenFunction(CodeGenModule &cgm, bool suppressNewContext=false);
  ~CodeGenFunction();

  CodeGenTypes &getTypes() const { return CGM.getTypes(); }
  ASTContext &getContext() const { return CGM.getContext(); }
  /// Returns true if DebugInfo is actually initialized.
  bool maybeInitializeDebugInfo() {
    if (CGM.getModuleDebugInfo()) {
      DebugInfo = CGM.getModuleDebugInfo();
      return true;
    }
    return false;
  }
  CGDebugInfo *getDebugInfo() { 
    if (DisableDebugInfo) 
      return NULL;
    return DebugInfo; 
  }
  void disableDebugInfo() { DisableDebugInfo = true; }
  void enableDebugInfo() { DisableDebugInfo = false; }

  bool shouldUseFusedARCCalls() {
    return CGM.getCodeGenOpts().OptimizationLevel == 0;
  }

  const LangOptions &getLangOpts() const { return CGM.getLangOpts(); }

  /// Returns a pointer to the function's exception object and selector slot,
  /// which is assigned in every landing pad.
  llvm::Value *getExceptionSlot();
  llvm::Value *getEHSelectorSlot();

  /// Returns the contents of the function's exception object and selector
  /// slots.
  llvm::Value *getExceptionFromSlot();
  llvm::Value *getSelectorFromSlot();

  llvm::Value *getNormalCleanupDestSlot();

  llvm::BasicBlock *getUnreachableBlock() {
    if (!UnreachableBlock) {
      UnreachableBlock = createBasicBlock("unreachable");
      new llvm::UnreachableInst(getLLVMContext(), UnreachableBlock);
    }
    return UnreachableBlock;
  }

  llvm::BasicBlock *getInvokeDest() {
    if (!EHStack.requiresLandingPad()) return 0;
    return getInvokeDestImpl();
  }

  const TargetInfo &getTarget() const { return Target; }
  llvm::LLVMContext &getLLVMContext() { return CGM.getLLVMContext(); }

  //===--------------------------------------------------------------------===//
  //                                  Cleanups
  //===--------------------------------------------------------------------===//

  typedef void Destroyer(CodeGenFunction &CGF, llvm::Value *addr, QualType ty);

  void pushIrregularPartialArrayCleanup(llvm::Value *arrayBegin,
                                        llvm::Value *arrayEndPointer,
                                        QualType elementType,
                                        Destroyer *destroyer);
  void pushRegularPartialArrayCleanup(llvm::Value *arrayBegin,
                                      llvm::Value *arrayEnd,
                                      QualType elementType,
                                      Destroyer *destroyer);

  void pushDestroy(QualType::DestructionKind dtorKind,
                   llvm::Value *addr, QualType type);
  void pushEHDestroy(QualType::DestructionKind dtorKind,
                     llvm::Value *addr, QualType type);
  void pushDestroy(CleanupKind kind, llvm::Value *addr, QualType type,
                   Destroyer *destroyer, bool useEHCleanupForArray);
  void emitDestroy(llvm::Value *addr, QualType type, Destroyer *destroyer,
                   bool useEHCleanupForArray);
  llvm::Function *generateDestroyHelper(llvm::Constant *addr,
                                        QualType type,
                                        Destroyer *destroyer,
                                        bool useEHCleanupForArray);
  void emitArrayDestroy(llvm::Value *begin, llvm::Value *end,
                        QualType type, Destroyer *destroyer,
                        bool checkZeroLength, bool useEHCleanup);

  Destroyer *getDestroyer(QualType::DestructionKind destructionKind);

  /// Determines whether an EH cleanup is required to destroy a type
  /// with the given destruction kind.
  bool needsEHCleanup(QualType::DestructionKind kind) {
    switch (kind) {
    case QualType::DK_none:
      return false;
    case QualType::DK_cxx_destructor:
    case QualType::DK_objc_weak_lifetime:
      return getLangOpts().Exceptions;
    case QualType::DK_objc_strong_lifetime:
      return getLangOpts().Exceptions &&
             CGM.getCodeGenOpts().ObjCAutoRefCountExceptions;
    }
    llvm_unreachable("bad destruction kind");
  }

  CleanupKind getCleanupKind(QualType::DestructionKind kind) {
    return (needsEHCleanup(kind) ? NormalAndEHCleanup : NormalCleanup);
  }

  //===--------------------------------------------------------------------===//
  //                                  Objective-C
  //===--------------------------------------------------------------------===//

  void GenerateObjCMethod(const ObjCMethodDecl *OMD);

  void StartObjCMethod(const ObjCMethodDecl *MD,
                       const ObjCContainerDecl *CD,
                       SourceLocation StartLoc);

  /// GenerateObjCGetter - Synthesize an Objective-C property getter function.
  void GenerateObjCGetter(ObjCImplementationDecl *IMP,
                          const ObjCPropertyImplDecl *PID);
  void generateObjCGetterBody(const ObjCImplementationDecl *classImpl,
                              const ObjCPropertyImplDecl *propImpl,
                              const ObjCMethodDecl *GetterMothodDecl,
                              llvm::Constant *AtomicHelperFn);

  void GenerateObjCCtorDtorMethod(ObjCImplementationDecl *IMP,
                                  ObjCMethodDecl *MD, bool ctor);

  /// GenerateObjCSetter - Synthesize an Objective-C property setter function
  /// for the given property.
  void GenerateObjCSetter(ObjCImplementationDecl *IMP,
                          const ObjCPropertyImplDecl *PID);
  void generateObjCSetterBody(const ObjCImplementationDecl *classImpl,
                              const ObjCPropertyImplDecl *propImpl,
                              llvm::Constant *AtomicHelperFn);
  bool IndirectObjCSetterArg(const CGFunctionInfo &FI);
  bool IvarTypeWithAggrGCObjects(QualType Ty);

  //===--------------------------------------------------------------------===//
  //                                  Block Bits
  //===--------------------------------------------------------------------===//

  llvm::Value *EmitBlockLiteral(const BlockExpr *);
  llvm::Value *EmitBlockLiteral(const CGBlockInfo &Info);
  static void destroyBlockInfos(CGBlockInfo *info);
  llvm::Constant *BuildDescriptorBlockDecl(const BlockExpr *,
                                           const CGBlockInfo &Info,
                                           llvm::StructType *,
                                           llvm::Constant *BlockVarLayout);

  llvm::Function *GenerateBlockFunction(GlobalDecl GD,
                                        const CGBlockInfo &Info,
                                        const DeclMapTy &ldm,
                                        bool IsLambdaConversionToBlock);

  llvm::Constant *GenerateCopyHelperFunction(const CGBlockInfo &blockInfo);
  llvm::Constant *GenerateDestroyHelperFunction(const CGBlockInfo &blockInfo);
  llvm::Constant *GenerateObjCAtomicSetterCopyHelperFunction(
                                             const ObjCPropertyImplDecl *PID);
  llvm::Constant *GenerateObjCAtomicGetterCopyHelperFunction(
                                             const ObjCPropertyImplDecl *PID);
  llvm::Value *EmitBlockCopyAndAutorelease(llvm::Value *Block, QualType Ty);

  void BuildBlockRelease(llvm::Value *DeclPtr, BlockFieldFlags flags);

  class AutoVarEmission;

  void emitByrefStructureInit(const AutoVarEmission &emission);
  void enterByrefCleanup(const AutoVarEmission &emission);

  llvm::Value *LoadBlockStruct() {
    assert(BlockPointer && "no block pointer set!");
    return BlockPointer;
  }

  void AllocateBlockCXXThisPointer(const CXXThisExpr *E);
  void AllocateBlockDecl(const DeclRefExpr *E);
  llvm::Value *GetAddrOfBlockDecl(const VarDecl *var, bool ByRef);
  llvm::Type *BuildByRefType(const VarDecl *var);

  void GenerateCode(GlobalDecl GD, llvm::Function *Fn,
                    const CGFunctionInfo &FnInfo);
  void StartFunction(GlobalDecl GD,
                     QualType RetTy,
                     llvm::Function *Fn,
                     const CGFunctionInfo &FnInfo,
                     const FunctionArgList &Args,
                     SourceLocation StartLoc);

  void EmitConstructorBody(FunctionArgList &Args);
  void EmitDestructorBody(FunctionArgList &Args);
  void emitImplicitAssignmentOperatorBody(FunctionArgList &Args);
  void EmitFunctionBody(FunctionArgList &Args);

  void EmitForwardingCallToLambda(const CXXRecordDecl *Lambda,
                                  CallArgList &CallArgs);
  void EmitLambdaToBlockPointerBody(FunctionArgList &Args);
  void EmitLambdaBlockInvokeBody();
  void EmitLambdaDelegatingInvokeBody(const CXXMethodDecl *MD);
  void EmitLambdaStaticInvokeFunction(const CXXMethodDecl *MD);

  /// EmitReturnBlock - Emit the unified return block, trying to avoid its
  /// emission when possible.
  void EmitReturnBlock();

  /// FinishFunction - Complete IR generation of the current function. It is
  /// legal to call this function even if there is no current insertion point.
  void FinishFunction(SourceLocation EndLoc=SourceLocation());

  /// GenerateThunk - Generate a thunk for the given method.
  void GenerateThunk(llvm::Function *Fn, const CGFunctionInfo &FnInfo,
                     GlobalDecl GD, const ThunkInfo &Thunk);

  void GenerateVarArgsThunk(llvm::Function *Fn, const CGFunctionInfo &FnInfo,
                            GlobalDecl GD, const ThunkInfo &Thunk);

  void EmitCtorPrologue(const CXXConstructorDecl *CD, CXXCtorType Type,
                        FunctionArgList &Args);

  void EmitInitializerForField(FieldDecl *Field, LValue LHS, Expr *Init,
                               ArrayRef<VarDecl *> ArrayIndexes);

  /// InitializeVTablePointer - Initialize the vtable pointer of the given
  /// subobject.
  ///
  void InitializeVTablePointer(BaseSubobject Base,
                               const CXXRecordDecl *NearestVBase,
                               CharUnits OffsetFromNearestVBase,
                               llvm::Constant *VTable,
                               const CXXRecordDecl *VTableClass);

  typedef llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBasesSetTy;
  void InitializeVTablePointers(BaseSubobject Base,
                                const CXXRecordDecl *NearestVBase,
                                CharUnits OffsetFromNearestVBase,
                                bool BaseIsNonVirtualPrimaryBase,
                                llvm::Constant *VTable,
                                const CXXRecordDecl *VTableClass,
                                VisitedVirtualBasesSetTy& VBases);

  void InitializeVTablePointers(const CXXRecordDecl *ClassDecl);

  /// GetVTablePtr - Return the Value of the vtable pointer member pointed
  /// to by This.
  llvm::Value *GetVTablePtr(llvm::Value *This, llvm::Type *Ty);

  /// EnterDtorCleanups - Enter the cleanups necessary to complete the
  /// given phase of destruction for a destructor.  The end result
  /// should call destructors on members and base classes in reverse
  /// order of their construction.
  void EnterDtorCleanups(const CXXDestructorDecl *Dtor, CXXDtorType Type);

  /// ShouldInstrumentFunction - Return true if the current function should be
  /// instrumented with __cyg_profile_func_* calls
  bool ShouldInstrumentFunction();

  /// EmitFunctionInstrumentation - Emit LLVM code to call the specified
  /// instrumentation function with the current function and the call site, if
  /// function instrumentation is enabled.
  void EmitFunctionInstrumentation(const char *Fn);

  /// EmitMCountInstrumentation - Emit call to .mcount.
  void EmitMCountInstrumentation();

  /// EmitFunctionProlog - Emit the target specific LLVM code to load the
  /// arguments for the given function. This is also responsible for naming the
  /// LLVM function arguments.
  void EmitFunctionProlog(const CGFunctionInfo &FI,
                          llvm::Function *Fn,
                          const FunctionArgList &Args);

  /// EmitFunctionEpilog - Emit the target specific LLVM code to return the
  /// given temporary.
  void EmitFunctionEpilog(const CGFunctionInfo &FI, bool EmitRetDbgLoc);

  /// EmitStartEHSpec - Emit the start of the exception spec.
  void EmitStartEHSpec(const Decl *D);

  /// EmitEndEHSpec - Emit the end of the exception spec.
  void EmitEndEHSpec(const Decl *D);

  /// getTerminateLandingPad - Return a landing pad that just calls terminate.
  llvm::BasicBlock *getTerminateLandingPad();

  /// getTerminateHandler - Return a handler (not a landing pad, just
  /// a catch handler) that just calls terminate.  This is used when
  /// a terminate scope encloses a try.
  llvm::BasicBlock *getTerminateHandler();

  llvm::Type *ConvertTypeForMem(QualType T);
  llvm::Type *ConvertType(QualType T);
  llvm::Type *ConvertType(const TypeDecl *T) {
    return ConvertType(getContext().getTypeDeclType(T));
  }

  /// LoadObjCSelf - Load the value of self. This function is only valid while
  /// generating code for an Objective-C method.
  llvm::Value *LoadObjCSelf();

  /// TypeOfSelfObject - Return type of object that this self represents.
  QualType TypeOfSelfObject();

  /// hasAggregateLLVMType - Return true if the specified AST type will map into
  /// an aggregate LLVM type or is void.
  static TypeEvaluationKind getEvaluationKind(QualType T);

  static bool hasScalarEvaluationKind(QualType T) {
    return getEvaluationKind(T) == TEK_Scalar;
  }

  static bool hasAggregateEvaluationKind(QualType T) {
    return getEvaluationKind(T) == TEK_Aggregate;
  }

  /// createBasicBlock - Create an LLVM basic block.
  llvm::BasicBlock *createBasicBlock(const Twine &name = "",
                                     llvm::Function *parent = 0,
                                     llvm::BasicBlock *before = 0) {
#ifdef NDEBUG
    return llvm::BasicBlock::Create(getLLVMContext(), "", parent, before);
#else
    return llvm::BasicBlock::Create(getLLVMContext(), name, parent, before);
#endif
  }

  /// getBasicBlockForLabel - Return the LLVM basicblock that the specified
  /// label maps to.
  JumpDest getJumpDestForLabel(const LabelDecl *S);

  /// SimplifyForwardingBlocks - If the given basic block is only a branch to
  /// another basic block, simplify it. This assumes that no other code could
  /// potentially reference the basic block.
  void SimplifyForwardingBlocks(llvm::BasicBlock *BB);

  /// EmitBlock - Emit the given block \arg BB and set it as the insert point,
  /// adding a fall-through branch from the current insert block if
  /// necessary. It is legal to call this function even if there is no current
  /// insertion point.
  ///
  /// IsFinished - If true, indicates that the caller has finished emitting
  /// branches to the given block and does not expect to emit code into it. This
  /// means the block can be ignored if it is unreachable.
  void EmitBlock(llvm::BasicBlock *BB, bool IsFinished=false);

  /// EmitBlockAfterUses - Emit the given block somewhere hopefully
  /// near its uses, and leave the insertion point in it.
  void EmitBlockAfterUses(llvm::BasicBlock *BB);

  /// EmitBranch - Emit a branch to the specified basic block from the current
  /// insert block, taking care to avoid creation of branches from dummy
  /// blocks. It is legal to call this function even if there is no current
  /// insertion point.
  ///
  /// This function clears the current insertion point. The caller should follow
  /// calls to this function with calls to Emit*Block prior to generation new
  /// code.
  void EmitBranch(llvm::BasicBlock *Block);

  /// HaveInsertPoint - True if an insertion point is defined. If not, this
  /// indicates that the current code being emitted is unreachable.
  bool HaveInsertPoint() const {
    return Builder.GetInsertBlock() != 0;
  }

  /// EnsureInsertPoint - Ensure that an insertion point is defined so that
  /// emitted IR has a place to go. Note that by definition, if this function
  /// creates a block then that block is unreachable; callers may do better to
  /// detect when no insertion point is defined and simply skip IR generation.
  void EnsureInsertPoint() {
    if (!HaveInsertPoint())
      EmitBlock(createBasicBlock());
  }

  /// ErrorUnsupported - Print out an error that codegen doesn't support the
  /// specified stmt yet.
  void ErrorUnsupported(const Stmt *S, const char *Type,
                        bool OmitOnError=false);

  //===--------------------------------------------------------------------===//
  //                                  Helpers
  //===--------------------------------------------------------------------===//

  LValue MakeAddrLValue(llvm::Value *V, QualType T,
                        CharUnits Alignment = CharUnits()) {
    return LValue::MakeAddr(V, T, Alignment, getContext(),
                            CGM.getTBAAInfo(T));
  }

  LValue MakeNaturalAlignAddrLValue(llvm::Value *V, QualType T) {
    CharUnits Alignment;
    if (!T->isIncompleteType())
      Alignment = getContext().getTypeAlignInChars(T);
    return LValue::MakeAddr(V, T, Alignment, getContext(),
                            CGM.getTBAAInfo(T));
  }

  /// CreateTempAlloca - This creates a alloca and inserts it into the entry
  /// block. The caller is responsible for setting an appropriate alignment on
  /// the alloca.
  llvm::AllocaInst *CreateTempAlloca(llvm::Type *Ty,
                                     const Twine &Name = "tmp");

  /// InitTempAlloca - Provide an initial value for the given alloca.
  void InitTempAlloca(llvm::AllocaInst *Alloca, llvm::Value *Value);

  /// CreateIRTemp - Create a temporary IR object of the given type, with
  /// appropriate alignment. This routine should only be used when an temporary
  /// value needs to be stored into an alloca (for example, to avoid explicit
  /// PHI construction), but the type is the IR type, not the type appropriate
  /// for storing in memory.
  llvm::AllocaInst *CreateIRTemp(QualType T, const Twine &Name = "tmp");

  /// CreateMemTemp - Create a temporary memory object of the given type, with
  /// appropriate alignment.
  llvm::AllocaInst *CreateMemTemp(QualType T, const Twine &Name = "tmp");

  /// CreateAggTemp - Create a temporary memory object for the given
  /// aggregate type.
  AggValueSlot CreateAggTemp(QualType T, const Twine &Name = "tmp") {
    CharUnits Alignment = getContext().getTypeAlignInChars(T);
    return AggValueSlot::forAddr(CreateMemTemp(T, Name), Alignment,
                                 T.getQualifiers(),
                                 AggValueSlot::IsNotDestructed,
                                 AggValueSlot::DoesNotNeedGCBarriers,
                                 AggValueSlot::IsNotAliased);
  }

  /// Emit a cast to void* in the appropriate address space.
  llvm::Value *EmitCastToVoidPtr(llvm::Value *value);

  /// EvaluateExprAsBool - Perform the usual unary conversions on the specified
  /// expression and compare the result against zero, returning an Int1Ty value.
  llvm::Value *EvaluateExprAsBool(const Expr *E);

  /// EmitIgnoredExpr - Emit an expression in a context which ignores the result.
  void EmitIgnoredExpr(const Expr *E);

  /// EmitAnyExpr - Emit code to compute the specified expression which can have
  /// any type.  The result is returned as an RValue struct.  If this is an
  /// aggregate expression, the aggloc/agglocvolatile arguments indicate where
  /// the result should be returned.
  ///
  /// \param ignoreResult True if the resulting value isn't used.
  RValue EmitAnyExpr(const Expr *E,
                     AggValueSlot aggSlot = AggValueSlot::ignored(),
                     bool ignoreResult = false);

  // EmitVAListRef - Emit a "reference" to a va_list; this is either the address
  // or the value of the expression, depending on how va_list is defined.
  llvm::Value *EmitVAListRef(const Expr *E);

  /// EmitAnyExprToTemp - Similary to EmitAnyExpr(), however, the result will
  /// always be accessible even if no aggregate location is provided.
  RValue EmitAnyExprToTemp(const Expr *E);

  /// EmitAnyExprToMem - Emits the code necessary to evaluate an
  /// arbitrary expression into the given memory location.
  void EmitAnyExprToMem(const Expr *E, llvm::Value *Location,
                        Qualifiers Quals, bool IsInitializer);

  /// EmitExprAsInit - Emits the code necessary to initialize a
  /// location in memory with the given initializer.
  void EmitExprAsInit(const Expr *init, const ValueDecl *D,
                      LValue lvalue, bool capturedByInit);

  /// hasVolatileMember - returns true if aggregate type has a volatile
  /// member.
  bool hasVolatileMember(QualType T) {
    if (const RecordType *RT = T->getAs<RecordType>()) {
      const RecordDecl *RD = cast<RecordDecl>(RT->getDecl());
      return RD->hasVolatileMember();
    }
    return false;
  }
  /// EmitAggregateCopy - Emit an aggregate assignment.
  ///
  /// The difference to EmitAggregateCopy is that tail padding is not copied.
  /// This is required for correctness when assigning non-POD structures in C++.
  void EmitAggregateAssign(llvm::Value *DestPtr, llvm::Value *SrcPtr,
                           QualType EltTy) {
    bool IsVolatile = hasVolatileMember(EltTy);
    EmitAggregateCopy(DestPtr, SrcPtr, EltTy, IsVolatile, CharUnits::Zero(),
                      true);
  }

  /// EmitAggregateCopy - Emit an aggregate copy.
  ///
  /// \param isVolatile - True iff either the source or the destination is
  /// volatile.
  /// \param isAssignment - If false, allow padding to be copied.  This often
  /// yields more efficient.
  void EmitAggregateCopy(llvm::Value *DestPtr, llvm::Value *SrcPtr,
                         QualType EltTy, bool isVolatile=false,
                         CharUnits Alignment = CharUnits::Zero(),
                         bool isAssignment = false);

  /// StartBlock - Start new block named N. If insert block is a dummy block
  /// then reuse it.
  void StartBlock(const char *N);

  /// GetAddrOfLocalVar - Return the address of a local variable.
  llvm::Value *GetAddrOfLocalVar(const VarDecl *VD) {
    llvm::Value *Res = LocalDeclMap[VD];
    assert(Res && "Invalid argument to GetAddrOfLocalVar(), no decl!");
    return Res;
  }

  /// getOpaqueLValueMapping - Given an opaque value expression (which
  /// must be mapped to an l-value), return its mapping.
  const LValue &getOpaqueLValueMapping(const OpaqueValueExpr *e) {
    assert(OpaqueValueMapping::shouldBindAsLValue(e));

    llvm::DenseMap<const OpaqueValueExpr*,LValue>::iterator
      it = OpaqueLValues.find(e);
    assert(it != OpaqueLValues.end() && "no mapping for opaque value!");
    return it->second;
  }

  /// getOpaqueRValueMapping - Given an opaque value expression (which
  /// must be mapped to an r-value), return its mapping.
  const RValue &getOpaqueRValueMapping(const OpaqueValueExpr *e) {
    assert(!OpaqueValueMapping::shouldBindAsLValue(e));

    llvm::DenseMap<const OpaqueValueExpr*,RValue>::iterator
      it = OpaqueRValues.find(e);
    assert(it != OpaqueRValues.end() && "no mapping for opaque value!");
    return it->second;
  }

  /// getAccessedFieldNo - Given an encoded value and a result number, return
  /// the input field number being accessed.
  static unsigned getAccessedFieldNo(unsigned Idx, const llvm::Constant *Elts);

  llvm::BlockAddress *GetAddrOfLabel(const LabelDecl *L);
  llvm::BasicBlock *GetIndirectGotoBlock();

  /// EmitNullInitialization - Generate code to set a value of the given type to
  /// null, If the type contains data member pointers, they will be initialized
  /// to -1 in accordance with the Itanium C++ ABI.
  void EmitNullInitialization(llvm::Value *DestPtr, QualType Ty);

  // EmitVAArg - Generate code to get an argument from the passed in pointer
  // and update it accordingly. The return value is a pointer to the argument.
  // FIXME: We should be able to get rid of this method and use the va_arg
  // instruction in LLVM instead once it works well enough.
  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty);

  /// emitArrayLength - Compute the length of an array, even if it's a
  /// VLA, and drill down to the base element type.
  llvm::Value *emitArrayLength(const ArrayType *arrayType,
                               QualType &baseType,
                               llvm::Value *&addr);

  /// EmitVLASize - Capture all the sizes for the VLA expressions in
  /// the given variably-modified type and store them in the VLASizeMap.
  ///
  /// This function can be called with a null (unreachable) insert point.
  void EmitVariablyModifiedType(QualType Ty);

  /// getVLASize - Returns an LLVM value that corresponds to the size,
  /// in non-variably-sized elements, of a variable length array type,
  /// plus that largest non-variably-sized element type.  Assumes that
  /// the type has already been emitted with EmitVariablyModifiedType.
  std::pair<llvm::Value*,QualType> getVLASize(const VariableArrayType *vla);
  std::pair<llvm::Value*,QualType> getVLASize(QualType vla);

  /// LoadCXXThis - Load the value of 'this'. This function is only valid while
  /// generating code for an C++ member function.
  llvm::Value *LoadCXXThis() {
    assert(CXXThisValue && "no 'this' value for this function");
    return CXXThisValue;
  }

  /// LoadCXXVTT - Load the VTT parameter to base constructors/destructors have
  /// virtual bases.
  // FIXME: Every place that calls LoadCXXVTT is something
  // that needs to be abstracted properly.
  llvm::Value *LoadCXXVTT() {
    assert(CXXStructorImplicitParamValue && "no VTT value for this function");
    return CXXStructorImplicitParamValue;
  }

  /// LoadCXXStructorImplicitParam - Load the implicit parameter
  /// for a constructor/destructor.
  llvm::Value *LoadCXXStructorImplicitParam() {
    assert(CXXStructorImplicitParamValue &&
           "no implicit argument value for this function");
    return CXXStructorImplicitParamValue;
  }

  /// GetAddressOfBaseOfCompleteClass - Convert the given pointer to a
  /// complete class to the given direct base.
  llvm::Value *
  GetAddressOfDirectBaseInCompleteClass(llvm::Value *Value,
                                        const CXXRecordDecl *Derived,
                                        const CXXRecordDecl *Base,
                                        bool BaseIsVirtual);

  /// GetAddressOfBaseClass - This function will add the necessary delta to the
  /// load of 'this' and returns address of the base class.
  llvm::Value *GetAddressOfBaseClass(llvm::Value *Value,
                                     const CXXRecordDecl *Derived,
                                     CastExpr::path_const_iterator PathBegin,
                                     CastExpr::path_const_iterator PathEnd,
                                     bool NullCheckValue);

  llvm::Value *GetAddressOfDerivedClass(llvm::Value *Value,
                                        const CXXRecordDecl *Derived,
                                        CastExpr::path_const_iterator PathBegin,
                                        CastExpr::path_const_iterator PathEnd,
                                        bool NullCheckValue);

  llvm::Value *GetVirtualBaseClassOffset(llvm::Value *This,
                                         const CXXRecordDecl *ClassDecl,
                                         const CXXRecordDecl *BaseClassDecl);

  /// GetVTTParameter - Return the VTT parameter that should be passed to a
  /// base constructor/destructor with virtual bases.
  /// FIXME: VTTs are Itanium ABI-specific, so the definition should move
  /// to ItaniumCXXABI.cpp together with all the references to VTT.
  llvm::Value *GetVTTParameter(GlobalDecl GD, bool ForVirtualBase,
                               bool Delegating);

  void EmitDelegateCXXConstructorCall(const CXXConstructorDecl *Ctor,
                                      CXXCtorType CtorType,
                                      const FunctionArgList &Args);
  // It's important not to confuse this and the previous function. Delegating
  // constructors are the C++0x feature. The constructor delegate optimization
  // is used to reduce duplication in the base and complete consturctors where
  // they are substantially the same.
  void EmitDelegatingCXXConstructorCall(const CXXConstructorDecl *Ctor,
                                        const FunctionArgList &Args);
  void EmitCXXConstructorCall(const CXXConstructorDecl *D, CXXCtorType Type,
                              bool ForVirtualBase, bool Delegating,
                              llvm::Value *This,
                              CallExpr::const_arg_iterator ArgBeg,
                              CallExpr::const_arg_iterator ArgEnd);
  
  void EmitSynthesizedCXXCopyCtorCall(const CXXConstructorDecl *D,
                              llvm::Value *This, llvm::Value *Src,
                              CallExpr::const_arg_iterator ArgBeg,
                              CallExpr::const_arg_iterator ArgEnd);

  void EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                  const ConstantArrayType *ArrayTy,
                                  llvm::Value *ArrayPtr,
                                  CallExpr::const_arg_iterator ArgBeg,
                                  CallExpr::const_arg_iterator ArgEnd,
                                  bool ZeroInitialization = false);

  void EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                  llvm::Value *NumElements,
                                  llvm::Value *ArrayPtr,
                                  CallExpr::const_arg_iterator ArgBeg,
                                  CallExpr::const_arg_iterator ArgEnd,
                                  bool ZeroInitialization = false);

  static Destroyer destroyCXXObject;

  void EmitCXXDestructorCall(const CXXDestructorDecl *D, CXXDtorType Type,
                             bool ForVirtualBase, bool Delegating,
                             llvm::Value *This);

  void EmitNewArrayInitializer(const CXXNewExpr *E, QualType elementType,
                               llvm::Value *NewPtr, llvm::Value *NumElements);

  void EmitCXXTemporary(const CXXTemporary *Temporary, QualType TempType,
                        llvm::Value *Ptr);

  llvm::Value *EmitCXXNewExpr(const CXXNewExpr *E);
  void EmitCXXDeleteExpr(const CXXDeleteExpr *E);

  void EmitDeleteCall(const FunctionDecl *DeleteFD, llvm::Value *Ptr,
                      QualType DeleteTy);

  llvm::Value* EmitCXXTypeidExpr(const CXXTypeidExpr *E);
  llvm::Value *EmitDynamicCast(llvm::Value *V, const CXXDynamicCastExpr *DCE);
  llvm::Value* EmitCXXUuidofExpr(const CXXUuidofExpr *E);

  void MaybeEmitStdInitializerListCleanup(llvm::Value *loc, const Expr *init);
  void EmitStdInitializerListCleanup(llvm::Value *loc,
                                     const InitListExpr *init);

  /// \brief Situations in which we might emit a check for the suitability of a
  ///        pointer or glvalue.
  enum TypeCheckKind {
    /// Checking the operand of a load. Must be suitably sized and aligned.
    TCK_Load,
    /// Checking the destination of a store. Must be suitably sized and aligned.
    TCK_Store,
    /// Checking the bound value in a reference binding. Must be suitably sized
    /// and aligned, but is not required to refer to an object (until the
    /// reference is used), per core issue 453.
    TCK_ReferenceBinding,
    /// Checking the object expression in a non-static data member access. Must
    /// be an object within its lifetime.
    TCK_MemberAccess,
    /// Checking the 'this' pointer for a call to a non-static member function.
    /// Must be an object within its lifetime.
    TCK_MemberCall,
    /// Checking the 'this' pointer for a constructor call.
    TCK_ConstructorCall,
    /// Checking the operand of a static_cast to a derived pointer type. Must be
    /// null or an object within its lifetime.
    TCK_DowncastPointer,
    /// Checking the operand of a static_cast to a derived reference type. Must
    /// be an object within its lifetime.
    TCK_DowncastReference
  };

  /// \brief Emit a check that \p V is the address of storage of the
  /// appropriate size and alignment for an object of type \p Type.
  void EmitTypeCheck(TypeCheckKind TCK, SourceLocation Loc, llvm::Value *V,
                     QualType Type, CharUnits Alignment = CharUnits::Zero());

  /// \brief Emit a check that \p Base points into an array object, which
  /// we can access at index \p Index. \p Accessed should be \c false if we
  /// this expression is used as an lvalue, for instance in "&Arr[Idx]".
  void EmitBoundsCheck(const Expr *E, const Expr *Base, llvm::Value *Index,
                       QualType IndexType, bool Accessed);

  llvm::Value *EmitScalarPrePostIncDec(const UnaryOperator *E, LValue LV,
                                       bool isInc, bool isPre);
  ComplexPairTy EmitComplexPrePostIncDec(const UnaryOperator *E, LValue LV,
                                         bool isInc, bool isPre);
  //===--------------------------------------------------------------------===//
  //                            Declaration Emission
  //===--------------------------------------------------------------------===//

  /// EmitDecl - Emit a declaration.
  ///
  /// This function can be called with a null (unreachable) insert point.
  void EmitDecl(const Decl &D);

  /// EmitVarDecl - Emit a local variable declaration.
  ///
  /// This function can be called with a null (unreachable) insert point.
  void EmitVarDecl(const VarDecl &D);

  void EmitScalarInit(const Expr *init, const ValueDecl *D,
                      LValue lvalue, bool capturedByInit);
  void EmitScalarInit(llvm::Value *init, LValue lvalue);

  typedef void SpecialInitFn(CodeGenFunction &Init, const VarDecl &D,
                             llvm::Value *Address);

  /// EmitAutoVarDecl - Emit an auto variable declaration.
  ///
  /// This function can be called with a null (unreachable) insert point.
  void EmitAutoVarDecl(const VarDecl &D);

  class AutoVarEmission {
    friend class CodeGenFunction;

    const VarDecl *Variable;

    /// The alignment of the variable.
    CharUnits Alignment;

    /// The address of the alloca.  Null if the variable was emitted
    /// as a global constant.
    llvm::Value *Address;

    llvm::Value *NRVOFlag;

    /// True if the variable is a __block variable.
    bool IsByRef;

    /// True if the variable is of aggregate type and has a constant
    /// initializer.
    bool IsConstantAggregate;

    /// Non-null if we should use lifetime annotations.
    llvm::Value *SizeForLifetimeMarkers;

    struct Invalid {};
    AutoVarEmission(Invalid) : Variable(0) {}

    AutoVarEmission(const VarDecl &variable)
      : Variable(&variable), Address(0), NRVOFlag(0),
        IsByRef(false), IsConstantAggregate(false),
        SizeForLifetimeMarkers(0) {}

    bool wasEmittedAsGlobal() const { return Address == 0; }

  public:
    static AutoVarEmission invalid() { return AutoVarEmission(Invalid()); }

    bool useLifetimeMarkers() const { return SizeForLifetimeMarkers != 0; }
    llvm::Value *getSizeForLifetimeMarkers() const {
      assert(useLifetimeMarkers());
      return SizeForLifetimeMarkers;
    }

    /// Returns the raw, allocated address, which is not necessarily
    /// the address of the object itself.
    llvm::Value *getAllocatedAddress() const {
      return Address;
    }

    /// Returns the address of the object within this declaration.
    /// Note that this does not chase the forwarding pointer for
    /// __block decls.
    llvm::Value *getObjectAddress(CodeGenFunction &CGF) const {
      if (!IsByRef) return Address;

      return CGF.Builder.CreateStructGEP(Address,
                                         CGF.getByRefValueLLVMField(Variable),
                                         Variable->getNameAsString());
    }
  };
  AutoVarEmission EmitAutoVarAlloca(const VarDecl &var);
  void EmitAutoVarInit(const AutoVarEmission &emission);
  void EmitAutoVarCleanups(const AutoVarEmission &emission);  
  void emitAutoVarTypeCleanup(const AutoVarEmission &emission,
                              QualType::DestructionKind dtorKind);

  void EmitStaticVarDecl(const VarDecl &D,
                         llvm::GlobalValue::LinkageTypes Linkage);

  /// EmitParmDecl - Emit a ParmVarDecl or an ImplicitParamDecl.
  void EmitParmDecl(const VarDecl &D, llvm::Value *Arg, unsigned ArgNo);

  /// protectFromPeepholes - Protect a value that we're intending to
  /// store to the side, but which will probably be used later, from
  /// aggressive peepholing optimizations that might delete it.
  ///
  /// Pass the result to unprotectFromPeepholes to declare that
  /// protection is no longer required.
  ///
  /// There's no particular reason why this shouldn't apply to
  /// l-values, it's just that no existing peepholes work on pointers.
  PeepholeProtection protectFromPeepholes(RValue rvalue);
  void unprotectFromPeepholes(PeepholeProtection protection);

  //===--------------------------------------------------------------------===//
  //                             Statement Emission
  //===--------------------------------------------------------------------===//

  /// EmitStopPoint - Emit a debug stoppoint if we are emitting debug info.
  void EmitStopPoint(const Stmt *S);

  /// EmitStmt - Emit the code for the statement \arg S. It is legal to call
  /// this function even if there is no current insertion point.
  ///
  /// This function may clear the current insertion point; callers should use
  /// EnsureInsertPoint if they wish to subsequently generate code without first
  /// calling EmitBlock, EmitBranch, or EmitStmt.
  void EmitStmt(const Stmt *S);

  /// EmitSimpleStmt - Try to emit a "simple" statement which does not
  /// necessarily require an insertion point or debug information; typically
  /// because the statement amounts to a jump or a container of other
  /// statements.
  ///
  /// \return True if the statement was handled.
  bool EmitSimpleStmt(const Stmt *S);

  RValue EmitCompoundStmt(const CompoundStmt &S, bool GetLast = false,
                          AggValueSlot AVS = AggValueSlot::ignored());
  RValue EmitCompoundStmtWithoutScope(const CompoundStmt &S,
                                      bool GetLast = false, AggValueSlot AVS =
                                          AggValueSlot::ignored());

  /// EmitLabel - Emit the block for the given label. It is legal to call this
  /// function even if there is no current insertion point.
  void EmitLabel(const LabelDecl *D); // helper for EmitLabelStmt.

  void EmitLabelStmt(const LabelStmt &S);
  void EmitAttributedStmt(const AttributedStmt &S);
  void EmitGotoStmt(const GotoStmt &S);
  void EmitIndirectGotoStmt(const IndirectGotoStmt &S);
  void EmitIfStmt(const IfStmt &S);
  void EmitWhileStmt(const WhileStmt &S);
  void EmitDoStmt(const DoStmt &S);
  void EmitForStmt(const ForStmt &S);
  void EmitReturnStmt(const ReturnStmt &S);
  void EmitDeclStmt(const DeclStmt &S);
  void EmitBreakStmt(const BreakStmt &S);
  void EmitContinueStmt(const ContinueStmt &S);
  void EmitSwitchStmt(const SwitchStmt &S);
  void EmitDefaultStmt(const DefaultStmt &S);
  void EmitCaseStmt(const CaseStmt &S);
  void EmitCaseStmtRange(const CaseStmt &S);
  void EmitAsmStmt(const AsmStmt &S);
  void EmitCapturedStmt(const CapturedStmt &S);

  void EmitObjCForCollectionStmt(const ObjCForCollectionStmt &S);
  void EmitObjCAtTryStmt(const ObjCAtTryStmt &S);
  void EmitObjCAtThrowStmt(const ObjCAtThrowStmt &S);
  void EmitObjCAtSynchronizedStmt(const ObjCAtSynchronizedStmt &S);
  void EmitObjCAutoreleasePoolStmt(const ObjCAutoreleasePoolStmt &S);

  llvm::Constant *getUnwindResumeFn();
  llvm::Constant *getUnwindResumeOrRethrowFn();
  void EnterCXXTryStmt(const CXXTryStmt &S, bool IsFnTryBlock = false);
  void ExitCXXTryStmt(const CXXTryStmt &S, bool IsFnTryBlock = false);

  void EmitCXXTryStmt(const CXXTryStmt &S);
  void EmitCXXForRangeStmt(const CXXForRangeStmt &S);

  //===--------------------------------------------------------------------===//
  //                         LValue Expression Emission
  //===--------------------------------------------------------------------===//

  /// GetUndefRValue - Get an appropriate 'undef' rvalue for the given type.
  RValue GetUndefRValue(QualType Ty);

  /// EmitUnsupportedRValue - Emit a dummy r-value using the type of E
  /// and issue an ErrorUnsupported style diagnostic (using the
  /// provided Name).
  RValue EmitUnsupportedRValue(const Expr *E,
                               const char *Name);

  /// EmitUnsupportedLValue - Emit a dummy l-value using the type of E and issue
  /// an ErrorUnsupported style diagnostic (using the provided Name).
  LValue EmitUnsupportedLValue(const Expr *E,
                               const char *Name);

  /// EmitLValue - Emit code to compute a designator that specifies the location
  /// of the expression.
  ///
  /// This can return one of two things: a simple address or a bitfield
  /// reference.  In either case, the LLVM Value* in the LValue structure is
  /// guaranteed to be an LLVM pointer type.
  ///
  /// If this returns a bitfield reference, nothing about the pointee type of
  /// the LLVM value is known: For example, it may not be a pointer to an
  /// integer.
  ///
  /// If this returns a normal address, and if the lvalue's C type is fixed
  /// size, this method guarantees that the returned pointer type will point to
  /// an LLVM type of the same size of the lvalue's type.  If the lvalue has a
  /// variable length type, this is not possible.
  ///
  LValue EmitLValue(const Expr *E);

  /// \brief Same as EmitLValue but additionally we generate checking code to
  /// guard against undefined behavior.  This is only suitable when we know
  /// that the address will be used to access the object.
  LValue EmitCheckedLValue(const Expr *E, TypeCheckKind TCK);

  RValue convertTempToRValue(llvm::Value *addr, QualType type);

  void EmitAtomicInit(Expr *E, LValue lvalue);

  RValue EmitAtomicLoad(LValue lvalue,
                        AggValueSlot slot = AggValueSlot::ignored());

  void EmitAtomicStore(RValue rvalue, LValue lvalue, bool isInit);

  /// EmitToMemory - Change a scalar value from its value
  /// representation to its in-memory representation.
  llvm::Value *EmitToMemory(llvm::Value *Value, QualType Ty);

  /// EmitFromMemory - Change a scalar value from its memory
  /// representation to its value representation.
  llvm::Value *EmitFromMemory(llvm::Value *Value, QualType Ty);

  /// EmitLoadOfScalar - Load a scalar value from an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.
  llvm::Value *EmitLoadOfScalar(llvm::Value *Addr, bool Volatile,
                                unsigned Alignment, QualType Ty,
                                llvm::MDNode *TBAAInfo = 0,
                                QualType TBAABaseTy = QualType(),
                                uint64_t TBAAOffset = 0);

  /// EmitLoadOfScalar - Load a scalar value from an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.  The l-value must be a simple
  /// l-value.
  llvm::Value *EmitLoadOfScalar(LValue lvalue);

  /// EmitStoreOfScalar - Store a scalar value to an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.
  void EmitStoreOfScalar(llvm::Value *Value, llvm::Value *Addr,
                         bool Volatile, unsigned Alignment, QualType Ty,
                         llvm::MDNode *TBAAInfo = 0, bool isInit = false,
                         QualType TBAABaseTy = QualType(),
                         uint64_t TBAAOffset = 0);

  /// EmitStoreOfScalar - Store a scalar value to an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.  The l-value must be a simple
  /// l-value.  The isInit flag indicates whether this is an initialization.
  /// If so, atomic qualifiers are ignored and the store is always non-atomic.
  void EmitStoreOfScalar(llvm::Value *value, LValue lvalue, bool isInit=false);

  /// EmitLoadOfLValue - Given an expression that represents a value lvalue,
  /// this method emits the address of the lvalue, then loads the result as an
  /// rvalue, returning the rvalue.
  RValue EmitLoadOfLValue(LValue V);
  RValue EmitLoadOfExtVectorElementLValue(LValue V);
  RValue EmitLoadOfBitfieldLValue(LValue LV);

  /// EmitStoreThroughLValue - Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void EmitStoreThroughLValue(RValue Src, LValue Dst, bool isInit=false);
  void EmitStoreThroughExtVectorComponentLValue(RValue Src, LValue Dst);

  /// EmitStoreThroughLValue - Store Src into Dst with same constraints as
  /// EmitStoreThroughLValue.
  ///
  /// \param Result [out] - If non-null, this will be set to a Value* for the
  /// bit-field contents after the store, appropriate for use as the result of
  /// an assignment to the bit-field.
  void EmitStoreThroughBitfieldLValue(RValue Src, LValue Dst,
                                      llvm::Value **Result=0);

  /// Emit an l-value for an assignment (simple or compound) of complex type.
  LValue EmitComplexAssignmentLValue(const BinaryOperator *E);
  LValue EmitComplexCompoundAssignmentLValue(const CompoundAssignOperator *E);

  // Note: only available for agg return types
  LValue EmitBinaryOperatorLValue(const BinaryOperator *E);
  LValue EmitCompoundAssignmentLValue(const CompoundAssignOperator *E);
  // Note: only available for agg return types
  LValue EmitCallExprLValue(const CallExpr *E);
  // Note: only available for agg return types
  LValue EmitVAArgExprLValue(const VAArgExpr *E);
  LValue EmitDeclRefLValue(const DeclRefExpr *E);
  LValue EmitStringLiteralLValue(const StringLiteral *E);
  LValue EmitObjCEncodeExprLValue(const ObjCEncodeExpr *E);
  LValue EmitPredefinedLValue(const PredefinedExpr *E);
  LValue EmitUnaryOpLValue(const UnaryOperator *E);
  LValue EmitArraySubscriptExpr(const ArraySubscriptExpr *E,
                                bool Accessed = false);
  LValue EmitExtVectorElementExpr(const ExtVectorElementExpr *E);
  LValue EmitMemberExpr(const MemberExpr *E);
  LValue EmitObjCIsaExpr(const ObjCIsaExpr *E);
  LValue EmitCompoundLiteralLValue(const CompoundLiteralExpr *E);
  LValue EmitInitListLValue(const InitListExpr *E);
  LValue EmitConditionalOperatorLValue(const AbstractConditionalOperator *E);
  LValue EmitCastLValue(const CastExpr *E);
  LValue EmitNullInitializationLValue(const CXXScalarValueInitExpr *E);
  LValue EmitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *E);
  LValue EmitOpaqueValueLValue(const OpaqueValueExpr *e);

  RValue EmitRValueForField(LValue LV, const FieldDecl *FD);

  class ConstantEmission {
    llvm::PointerIntPair<llvm::Constant*, 1, bool> ValueAndIsReference;
    ConstantEmission(llvm::Constant *C, bool isReference)
      : ValueAndIsReference(C, isReference) {}
  public:
    ConstantEmission() {}
    static ConstantEmission forReference(llvm::Constant *C) {
      return ConstantEmission(C, true);
    }
    static ConstantEmission forValue(llvm::Constant *C) {
      return ConstantEmission(C, false);
    }

    operator bool() const { return ValueAndIsReference.getOpaqueValue() != 0; }

    bool isReference() const { return ValueAndIsReference.getInt(); }
    LValue getReferenceLValue(CodeGenFunction &CGF, Expr *refExpr) const {
      assert(isReference());
      return CGF.MakeNaturalAlignAddrLValue(ValueAndIsReference.getPointer(),
                                            refExpr->getType());
    }

    llvm::Constant *getValue() const {
      assert(!isReference());
      return ValueAndIsReference.getPointer();
    }
  };

  ConstantEmission tryEmitAsConstant(DeclRefExpr *refExpr);

  RValue EmitPseudoObjectRValue(const PseudoObjectExpr *e,
                                AggValueSlot slot = AggValueSlot::ignored());
  LValue EmitPseudoObjectLValue(const PseudoObjectExpr *e);

  llvm::Value *EmitIvarOffset(const ObjCInterfaceDecl *Interface,
                              const ObjCIvarDecl *Ivar);
  LValue EmitLValueForField(LValue Base, const FieldDecl* Field);
  LValue EmitLValueForLambdaField(const FieldDecl *Field);

  /// EmitLValueForFieldInitialization - Like EmitLValueForField, except that
  /// if the Field is a reference, this will return the address of the reference
  /// and not the address of the value stored in the reference.
  LValue EmitLValueForFieldInitialization(LValue Base,
                                          const FieldDecl* Field);

  LValue EmitLValueForIvar(QualType ObjectTy,
                           llvm::Value* Base, const ObjCIvarDecl *Ivar,
                           unsigned CVRQualifiers);

  LValue EmitCXXConstructLValue(const CXXConstructExpr *E);
  LValue EmitCXXBindTemporaryLValue(const CXXBindTemporaryExpr *E);
  LValue EmitLambdaLValue(const LambdaExpr *E);
  LValue EmitCXXTypeidLValue(const CXXTypeidExpr *E);
  LValue EmitCXXUuidofLValue(const CXXUuidofExpr *E);

  LValue EmitObjCMessageExprLValue(const ObjCMessageExpr *E);
  LValue EmitObjCIvarRefLValue(const ObjCIvarRefExpr *E);
  LValue EmitStmtExprLValue(const StmtExpr *E);
  LValue EmitPointerToDataMemberBinaryExpr(const BinaryOperator *E);
  LValue EmitObjCSelectorLValue(const ObjCSelectorExpr *E);
  void   EmitDeclRefExprDbgValue(const DeclRefExpr *E, llvm::Constant *Init);

  //===--------------------------------------------------------------------===//
  //                         Scalar Expression Emission
  //===--------------------------------------------------------------------===//

  /// EmitCall - Generate a call of the given function, expecting the given
  /// result type, and using the given argument list which specifies both the
  /// LLVM arguments and the types they were derived from.
  ///
  /// \param TargetDecl - If given, the decl of the function in a direct call;
  /// used to set attributes on the call (noreturn, etc.).
  RValue EmitCall(const CGFunctionInfo &FnInfo,
                  llvm::Value *Callee,
                  ReturnValueSlot ReturnValue,
                  const CallArgList &Args,
                  const Decl *TargetDecl = 0,
                  llvm::Instruction **callOrInvoke = 0);

  RValue EmitCall(QualType FnType, llvm::Value *Callee,
                  ReturnValueSlot ReturnValue,
                  CallExpr::const_arg_iterator ArgBeg,
                  CallExpr::const_arg_iterator ArgEnd,
                  const Decl *TargetDecl = 0);
  RValue EmitCallExpr(const CallExpr *E,
                      ReturnValueSlot ReturnValue = ReturnValueSlot());

  llvm::CallInst *EmitRuntimeCall(llvm::Value *callee,
                                  const Twine &name = "");
  llvm::CallInst *EmitRuntimeCall(llvm::Value *callee,
                                  ArrayRef<llvm::Value*> args,
                                  const Twine &name = "");
  llvm::CallInst *EmitNounwindRuntimeCall(llvm::Value *callee,
                                          const Twine &name = "");
  llvm::CallInst *EmitNounwindRuntimeCall(llvm::Value *callee,
                                          ArrayRef<llvm::Value*> args,
                                          const Twine &name = "");

  llvm::CallSite EmitCallOrInvoke(llvm::Value *Callee,
                                  ArrayRef<llvm::Value *> Args,
                                  const Twine &Name = "");
  llvm::CallSite EmitCallOrInvoke(llvm::Value *Callee,
                                  const Twine &Name = "");
  llvm::CallSite EmitRuntimeCallOrInvoke(llvm::Value *callee,
                                         ArrayRef<llvm::Value*> args,
                                         const Twine &name = "");
  llvm::CallSite EmitRuntimeCallOrInvoke(llvm::Value *callee,
                                         const Twine &name = "");
  void EmitNoreturnRuntimeCallOrInvoke(llvm::Value *callee,
                                       ArrayRef<llvm::Value*> args);

  llvm::Value *BuildVirtualCall(const CXXMethodDecl *MD, llvm::Value *This,
                                llvm::Type *Ty);
  llvm::Value *BuildVirtualCall(const CXXDestructorDecl *DD, CXXDtorType Type,
                                llvm::Value *This, llvm::Type *Ty);
  llvm::Value *BuildAppleKextVirtualCall(const CXXMethodDecl *MD, 
                                         NestedNameSpecifier *Qual,
                                         llvm::Type *Ty);
  
  llvm::Value *BuildAppleKextVirtualDestructorCall(const CXXDestructorDecl *DD,
                                                   CXXDtorType Type, 
                                                   const CXXRecordDecl *RD);

  RValue EmitCXXMemberCall(const CXXMethodDecl *MD,
                           SourceLocation CallLoc,
                           llvm::Value *Callee,
                           ReturnValueSlot ReturnValue,
                           llvm::Value *This,
                           llvm::Value *ImplicitParam,
                           QualType ImplicitParamTy,
                           CallExpr::const_arg_iterator ArgBeg,
                           CallExpr::const_arg_iterator ArgEnd);
  RValue EmitCXXMemberCallExpr(const CXXMemberCallExpr *E,
                               ReturnValueSlot ReturnValue);
  RValue EmitCXXMemberPointerCallExpr(const CXXMemberCallExpr *E,
                                      ReturnValueSlot ReturnValue);

  llvm::Value *EmitCXXOperatorMemberCallee(const CXXOperatorCallExpr *E,
                                           const CXXMethodDecl *MD,
                                           llvm::Value *This);
  RValue EmitCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *E,
                                       const CXXMethodDecl *MD,
                                       ReturnValueSlot ReturnValue);

  RValue EmitCUDAKernelCallExpr(const CUDAKernelCallExpr *E,
                                ReturnValueSlot ReturnValue);


  RValue EmitBuiltinExpr(const FunctionDecl *FD,
                         unsigned BuiltinID, const CallExpr *E);

  RValue EmitBlockCallExpr(const CallExpr *E, ReturnValueSlot ReturnValue);

  /// EmitTargetBuiltinExpr - Emit the given builtin call. Returns 0 if the call
  /// is unhandled by the current target.
  llvm::Value *EmitTargetBuiltinExpr(unsigned BuiltinID, const CallExpr *E);

  llvm::Value *EmitAArch64BuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  llvm::Value *EmitARMBuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  llvm::Value *EmitNeonCall(llvm::Function *F,
                            SmallVectorImpl<llvm::Value*> &O,
                            const char *name,
                            unsigned shift = 0, bool rightshift = false);
  llvm::Value *EmitNeonSplat(llvm::Value *V, llvm::Constant *Idx);
  llvm::Value *EmitNeonShiftVector(llvm::Value *V, llvm::Type *Ty,
                                   bool negateForRightShift);

  llvm::Value *BuildVector(ArrayRef<llvm::Value*> Ops);
  llvm::Value *EmitX86BuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  llvm::Value *EmitPPCBuiltinExpr(unsigned BuiltinID, const CallExpr *E);

  llvm::Value *EmitObjCProtocolExpr(const ObjCProtocolExpr *E);
  llvm::Value *EmitObjCStringLiteral(const ObjCStringLiteral *E);
  llvm::Value *EmitObjCBoxedExpr(const ObjCBoxedExpr *E);
  llvm::Value *EmitObjCArrayLiteral(const ObjCArrayLiteral *E);
  llvm::Value *EmitObjCDictionaryLiteral(const ObjCDictionaryLiteral *E);
  llvm::Value *EmitObjCCollectionLiteral(const Expr *E,
                                const ObjCMethodDecl *MethodWithObjects);
  llvm::Value *EmitObjCSelectorExpr(const ObjCSelectorExpr *E);
  RValue EmitObjCMessageExpr(const ObjCMessageExpr *E,
                             ReturnValueSlot Return = ReturnValueSlot());

  /// Retrieves the default cleanup kind for an ARC cleanup.
  /// Except under -fobjc-arc-eh, ARC cleanups are normal-only.
  CleanupKind getARCCleanupKind() {
    return CGM.getCodeGenOpts().ObjCAutoRefCountExceptions
             ? NormalAndEHCleanup : NormalCleanup;
  }

  // ARC primitives.
  void EmitARCInitWeak(llvm::Value *value, llvm::Value *addr);
  void EmitARCDestroyWeak(llvm::Value *addr);
  llvm::Value *EmitARCLoadWeak(llvm::Value *addr);
  llvm::Value *EmitARCLoadWeakRetained(llvm::Value *addr);
  llvm::Value *EmitARCStoreWeak(llvm::Value *value, llvm::Value *addr,
                                bool ignored);
  void EmitARCCopyWeak(llvm::Value *dst, llvm::Value *src);
  void EmitARCMoveWeak(llvm::Value *dst, llvm::Value *src);
  llvm::Value *EmitARCRetainAutorelease(QualType type, llvm::Value *value);
  llvm::Value *EmitARCRetainAutoreleaseNonBlock(llvm::Value *value);
  llvm::Value *EmitARCStoreStrong(LValue lvalue, llvm::Value *value,
                                  bool resultIgnored);
  llvm::Value *EmitARCStoreStrongCall(llvm::Value *addr, llvm::Value *value,
                                      bool resultIgnored);
  llvm::Value *EmitARCRetain(QualType type, llvm::Value *value);
  llvm::Value *EmitARCRetainNonBlock(llvm::Value *value);
  llvm::Value *EmitARCRetainBlock(llvm::Value *value, bool mandatory);
  void EmitARCDestroyStrong(llvm::Value *addr, ARCPreciseLifetime_t precise);
  void EmitARCRelease(llvm::Value *value, ARCPreciseLifetime_t precise);
  llvm::Value *EmitARCAutorelease(llvm::Value *value);
  llvm::Value *EmitARCAutoreleaseReturnValue(llvm::Value *value);
  llvm::Value *EmitARCRetainAutoreleaseReturnValue(llvm::Value *value);
  llvm::Value *EmitARCRetainAutoreleasedReturnValue(llvm::Value *value);

  std::pair<LValue,llvm::Value*>
  EmitARCStoreAutoreleasing(const BinaryOperator *e);
  std::pair<LValue,llvm::Value*>
  EmitARCStoreStrong(const BinaryOperator *e, bool ignored);

  llvm::Value *EmitObjCThrowOperand(const Expr *expr);

  llvm::Value *EmitObjCProduceObject(QualType T, llvm::Value *Ptr);
  llvm::Value *EmitObjCConsumeObject(QualType T, llvm::Value *Ptr);
  llvm::Value *EmitObjCExtendObjectLifetime(QualType T, llvm::Value *Ptr);

  llvm::Value *EmitARCExtendBlockObject(const Expr *expr);
  llvm::Value *EmitARCRetainScalarExpr(const Expr *expr);
  llvm::Value *EmitARCRetainAutoreleaseScalarExpr(const Expr *expr);

  void EmitARCIntrinsicUse(llvm::ArrayRef<llvm::Value*> values);

  static Destroyer destroyARCStrongImprecise;
  static Destroyer destroyARCStrongPrecise;
  static Destroyer destroyARCWeak;

  void EmitObjCAutoreleasePoolPop(llvm::Value *Ptr); 
  llvm::Value *EmitObjCAutoreleasePoolPush();
  llvm::Value *EmitObjCMRRAutoreleasePoolPush();
  void EmitObjCAutoreleasePoolCleanup(llvm::Value *Ptr);
  void EmitObjCMRRAutoreleasePoolPop(llvm::Value *Ptr); 

  /// EmitReferenceBindingToExpr - Emits a reference binding to the passed in
  /// expression. Will emit a temporary variable if E is not an LValue.
  RValue EmitReferenceBindingToExpr(const Expr* E,
                                    const NamedDecl *InitializedDecl);

  //===--------------------------------------------------------------------===//
  //                           Expression Emission
  //===--------------------------------------------------------------------===//

  // Expressions are broken into three classes: scalar, complex, aggregate.

  /// EmitScalarExpr - Emit the computation of the specified expression of LLVM
  /// scalar type, returning the result.
  llvm::Value *EmitScalarExpr(const Expr *E , bool IgnoreResultAssign = false);

  /// EmitScalarConversion - Emit a conversion from the specified type to the
  /// specified destination type, both of which are LLVM scalar types.
  llvm::Value *EmitScalarConversion(llvm::Value *Src, QualType SrcTy,
                                    QualType DstTy);

  /// EmitComplexToScalarConversion - Emit a conversion from the specified
  /// complex type to the specified destination type, where the destination type
  /// is an LLVM scalar type.
  llvm::Value *EmitComplexToScalarConversion(ComplexPairTy Src, QualType SrcTy,
                                             QualType DstTy);


  /// EmitAggExpr - Emit the computation of the specified expression
  /// of aggregate type.  The result is computed into the given slot,
  /// which may be null to indicate that the value is not needed.
  void EmitAggExpr(const Expr *E, AggValueSlot AS);

  /// EmitAggExprToLValue - Emit the computation of the specified expression of
  /// aggregate type into a temporary LValue.
  LValue EmitAggExprToLValue(const Expr *E);

  /// EmitGCMemmoveCollectable - Emit special API for structs with object
  /// pointers.
  void EmitGCMemmoveCollectable(llvm::Value *DestPtr, llvm::Value *SrcPtr,
                                QualType Ty);

  /// EmitExtendGCLifetime - Given a pointer to an Objective-C object,
  /// make sure it survives garbage collection until this point.
  void EmitExtendGCLifetime(llvm::Value *object);

  /// EmitComplexExpr - Emit the computation of the specified expression of
  /// complex type, returning the result.
  ComplexPairTy EmitComplexExpr(const Expr *E,
                                bool IgnoreReal = false,
                                bool IgnoreImag = false);

  /// EmitComplexExprIntoLValue - Emit the given expression of complex
  /// type and place its result into the specified l-value.
  void EmitComplexExprIntoLValue(const Expr *E, LValue dest, bool isInit);

  /// EmitStoreOfComplex - Store a complex number into the specified l-value.
  void EmitStoreOfComplex(ComplexPairTy V, LValue dest, bool isInit);

  /// EmitLoadOfComplex - Load a complex number from the specified l-value.
  ComplexPairTy EmitLoadOfComplex(LValue src);

  /// CreateStaticVarDecl - Create a zero-initialized LLVM global for
  /// a static local variable.
  llvm::GlobalVariable *CreateStaticVarDecl(const VarDecl &D,
                                            const char *Separator,
                                       llvm::GlobalValue::LinkageTypes Linkage);

  /// AddInitializerToStaticVarDecl - Add the initializer for 'D' to the
  /// global variable that has already been created for it.  If the initializer
  /// has a different type than GV does, this may free GV and return a different
  /// one.  Otherwise it just returns GV.
  llvm::GlobalVariable *
  AddInitializerToStaticVarDecl(const VarDecl &D,
                                llvm::GlobalVariable *GV);


  /// EmitCXXGlobalVarDeclInit - Create the initializer for a C++
  /// variable with global storage.
  void EmitCXXGlobalVarDeclInit(const VarDecl &D, llvm::Constant *DeclPtr,
                                bool PerformInit);

  /// Call atexit() with a function that passes the given argument to
  /// the given function.
  void registerGlobalDtorWithAtExit(llvm::Constant *fn, llvm::Constant *addr);

  /// Emit code in this function to perform a guarded variable
  /// initialization.  Guarded initializations are used when it's not
  /// possible to prove that an initialization will be done exactly
  /// once, e.g. with a static local variable or a static data member
  /// of a class template.
  void EmitCXXGuardedInit(const VarDecl &D, llvm::GlobalVariable *DeclPtr,
                          bool PerformInit);

  /// GenerateCXXGlobalInitFunc - Generates code for initializing global
  /// variables.
  void GenerateCXXGlobalInitFunc(llvm::Function *Fn,
                                 ArrayRef<llvm::Constant *> Decls,
                                 llvm::GlobalVariable *Guard = 0);

  /// GenerateCXXGlobalDtorsFunc - Generates code for destroying global
  /// variables.
  void GenerateCXXGlobalDtorsFunc(llvm::Function *Fn,
                                  const std::vector<std::pair<llvm::WeakVH,
                                  llvm::Constant*> > &DtorsAndObjects);

  void GenerateCXXGlobalVarDeclInitFunc(llvm::Function *Fn,
                                        const VarDecl *D,
                                        llvm::GlobalVariable *Addr,
                                        bool PerformInit);

  void EmitCXXConstructExpr(const CXXConstructExpr *E, AggValueSlot Dest);
  
  void EmitSynthesizedCXXCopyCtor(llvm::Value *Dest, llvm::Value *Src,
                                  const Expr *Exp);

  void enterFullExpression(const ExprWithCleanups *E) {
    if (E->getNumObjects() == 0) return;
    enterNonTrivialFullExpression(E);
  }
  void enterNonTrivialFullExpression(const ExprWithCleanups *E);

  void EmitCXXThrowExpr(const CXXThrowExpr *E, bool KeepInsertionPoint = true);

  void EmitLambdaExpr(const LambdaExpr *E, AggValueSlot Dest);

  RValue EmitAtomicExpr(AtomicExpr *E, llvm::Value *Dest = 0);

  //===--------------------------------------------------------------------===//
  //                         Annotations Emission
  //===--------------------------------------------------------------------===//

  /// Emit an annotation call (intrinsic or builtin).
  llvm::Value *EmitAnnotationCall(llvm::Value *AnnotationFn,
                                  llvm::Value *AnnotatedVal,
                                  StringRef AnnotationStr,
                                  SourceLocation Location);

  /// Emit local annotations for the local variable V, declared by D.
  void EmitVarAnnotations(const VarDecl *D, llvm::Value *V);

  /// Emit field annotations for the given field & value. Returns the
  /// annotation result.
  llvm::Value *EmitFieldAnnotations(const FieldDecl *D, llvm::Value *V);

  //===--------------------------------------------------------------------===//
  //                             Internal Helpers
  //===--------------------------------------------------------------------===//

  /// ContainsLabel - Return true if the statement contains a label in it.  If
  /// this statement is not executed normally, it not containing a label means
  /// that we can just remove the code.
  static bool ContainsLabel(const Stmt *S, bool IgnoreCaseStmts = false);

  /// containsBreak - Return true if the statement contains a break out of it.
  /// If the statement (recursively) contains a switch or loop with a break
  /// inside of it, this is fine.
  static bool containsBreak(const Stmt *S);
  
  /// ConstantFoldsToSimpleInteger - If the specified expression does not fold
  /// to a constant, or if it does but contains a label, return false.  If it
  /// constant folds return true and set the boolean result in Result.
  bool ConstantFoldsToSimpleInteger(const Expr *Cond, bool &Result);

  /// ConstantFoldsToSimpleInteger - If the specified expression does not fold
  /// to a constant, or if it does but contains a label, return false.  If it
  /// constant folds return true and set the folded value.
  bool ConstantFoldsToSimpleInteger(const Expr *Cond, llvm::APSInt &Result);
  
  /// EmitBranchOnBoolExpr - Emit a branch on a boolean condition (e.g. for an
  /// if statement) to the specified blocks.  Based on the condition, this might
  /// try to simplify the codegen of the conditional based on the branch.
  void EmitBranchOnBoolExpr(const Expr *Cond, llvm::BasicBlock *TrueBlock,
                            llvm::BasicBlock *FalseBlock);

  /// \brief Emit a description of a type in a format suitable for passing to
  /// a runtime sanitizer handler.
  llvm::Constant *EmitCheckTypeDescriptor(QualType T);

  /// \brief Convert a value into a format suitable for passing to a runtime
  /// sanitizer handler.
  llvm::Value *EmitCheckValue(llvm::Value *V);

  /// \brief Emit a description of a source location in a format suitable for
  /// passing to a runtime sanitizer handler.
  llvm::Constant *EmitCheckSourceLocation(SourceLocation Loc);

  /// \brief Specify under what conditions this check can be recovered
  enum CheckRecoverableKind {
    /// Always terminate program execution if this check fails
    CRK_Unrecoverable,
    /// Check supports recovering, allows user to specify which
    CRK_Recoverable,
    /// Runtime conditionally aborts, always need to support recovery.
    CRK_AlwaysRecoverable
  };

  /// \brief Create a basic block that will call a handler function in a
  /// sanitizer runtime with the provided arguments, and create a conditional
  /// branch to it.
  void EmitCheck(llvm::Value *Checked, StringRef CheckName,
                 ArrayRef<llvm::Constant *> StaticArgs,
                 ArrayRef<llvm::Value *> DynamicArgs,
                 CheckRecoverableKind Recoverable);

  /// \brief Create a basic block that will call the trap intrinsic, and emit a
  /// conditional branch to it, for the -ftrapv checks.
  void EmitTrapCheck(llvm::Value *Checked);

  /// EmitCallArg - Emit a single call argument.
  void EmitCallArg(CallArgList &args, const Expr *E, QualType ArgType);

  /// EmitDelegateCallArg - We are performing a delegate call; that
  /// is, the current function is delegating to another one.  Produce
  /// a r-value suitable for passing the given parameter.
  void EmitDelegateCallArg(CallArgList &args, const VarDecl *param);

  /// SetFPAccuracy - Set the minimum required accuracy of the given floating
  /// point operation, expressed as the maximum relative error in ulp.
  void SetFPAccuracy(llvm::Value *Val, float Accuracy);

private:
  llvm::MDNode *getRangeForLoadFromType(QualType Ty);
  void EmitReturnOfRValue(RValue RV, QualType Ty);

  /// ExpandTypeFromArgs - Reconstruct a structure of type \arg Ty
  /// from function arguments into \arg Dst. See ABIArgInfo::Expand.
  ///
  /// \param AI - The first function argument of the expansion.
  /// \return The argument following the last expanded function
  /// argument.
  llvm::Function::arg_iterator
  ExpandTypeFromArgs(QualType Ty, LValue Dst,
                     llvm::Function::arg_iterator AI);

  /// ExpandTypeToArgs - Expand an RValue \arg Src, with the LLVM type for \arg
  /// Ty, into individual arguments on the provided vector \arg Args. See
  /// ABIArgInfo::Expand.
  void ExpandTypeToArgs(QualType Ty, RValue Src,
                        SmallVector<llvm::Value*, 16> &Args,
                        llvm::FunctionType *IRFuncTy);

  llvm::Value* EmitAsmInput(const TargetInfo::ConstraintInfo &Info,
                            const Expr *InputExpr, std::string &ConstraintStr);

  llvm::Value* EmitAsmInputLValue(const TargetInfo::ConstraintInfo &Info,
                                  LValue InputValue, QualType InputType,
                                  std::string &ConstraintStr);

  /// EmitCallArgs - Emit call arguments for a function.
  /// The CallArgTypeInfo parameter is used for iterating over the known
  /// argument types of the function being called.
  template<typename T>
  void EmitCallArgs(CallArgList& Args, const T* CallArgTypeInfo,
                    CallExpr::const_arg_iterator ArgBeg,
                    CallExpr::const_arg_iterator ArgEnd) {
      CallExpr::const_arg_iterator Arg = ArgBeg;

    // First, use the argument types that the type info knows about
    if (CallArgTypeInfo) {
      for (typename T::arg_type_iterator I = CallArgTypeInfo->arg_type_begin(),
           E = CallArgTypeInfo->arg_type_end(); I != E; ++I, ++Arg) {
        assert(Arg != ArgEnd && "Running over edge of argument list!");
        QualType ArgType = *I;
#ifndef NDEBUG
        QualType ActualArgType = Arg->getType();
        if (ArgType->isPointerType() && ActualArgType->isPointerType()) {
          QualType ActualBaseType =
            ActualArgType->getAs<PointerType>()->getPointeeType();
          QualType ArgBaseType =
            ArgType->getAs<PointerType>()->getPointeeType();
          if (ArgBaseType->isVariableArrayType()) {
            if (const VariableArrayType *VAT =
                getContext().getAsVariableArrayType(ActualBaseType)) {
              if (!VAT->getSizeExpr())
                ActualArgType = ArgType;
            }
          }
        }
        assert(getContext().getCanonicalType(ArgType.getNonReferenceType()).
               getTypePtr() ==
               getContext().getCanonicalType(ActualArgType).getTypePtr() &&
               "type mismatch in call argument!");
#endif
        EmitCallArg(Args, *Arg, ArgType);
      }

      // Either we've emitted all the call args, or we have a call to a
      // variadic function.
      assert((Arg == ArgEnd || CallArgTypeInfo->isVariadic()) &&
             "Extra arguments in non-variadic function!");

    }

    // If we still have any arguments, emit them using the type of the argument.
    for (; Arg != ArgEnd; ++Arg)
      EmitCallArg(Args, *Arg, Arg->getType());
  }

  const TargetCodeGenInfo &getTargetHooks() const {
    return CGM.getTargetCodeGenInfo();
  }

  void EmitDeclMetadata();

  CodeGenModule::ByrefHelpers *
  buildByrefHelpers(llvm::StructType &byrefType,
                    const AutoVarEmission &emission);

  void AddObjCARCExceptionMetadata(llvm::Instruction *Inst);

  /// GetPointeeAlignment - Given an expression with a pointer type, emit the
  /// value and compute our best estimate of the alignment of the pointee.
  std::pair<llvm::Value*, unsigned> EmitPointerWithAlignment(const Expr *Addr);
};

/// Helper class with most of the code for saving a value for a
/// conditional expression cleanup.
struct DominatingLLVMValue {
  typedef llvm::PointerIntPair<llvm::Value*, 1, bool> saved_type;

  /// Answer whether the given value needs extra work to be saved.
  static bool needsSaving(llvm::Value *value) {
    // If it's not an instruction, we don't need to save.
    if (!isa<llvm::Instruction>(value)) return false;

    // If it's an instruction in the entry block, we don't need to save.
    llvm::BasicBlock *block = cast<llvm::Instruction>(value)->getParent();
    return (block != &block->getParent()->getEntryBlock());
  }

  /// Try to save the given value.
  static saved_type save(CodeGenFunction &CGF, llvm::Value *value) {
    if (!needsSaving(value)) return saved_type(value, false);

    // Otherwise we need an alloca.
    llvm::Value *alloca =
      CGF.CreateTempAlloca(value->getType(), "cond-cleanup.save");
    CGF.Builder.CreateStore(value, alloca);

    return saved_type(alloca, true);
  }

  static llvm::Value *restore(CodeGenFunction &CGF, saved_type value) {
    if (!value.getInt()) return value.getPointer();
    return CGF.Builder.CreateLoad(value.getPointer());
  }
};

/// A partial specialization of DominatingValue for llvm::Values that
/// might be llvm::Instructions.
template <class T> struct DominatingPointer<T,true> : DominatingLLVMValue {
  typedef T *type;
  static type restore(CodeGenFunction &CGF, saved_type value) {
    return static_cast<T*>(DominatingLLVMValue::restore(CGF, value));
  }
};

/// A specialization of DominatingValue for RValue.
template <> struct DominatingValue<RValue> {
  typedef RValue type;
  class saved_type {
    enum Kind { ScalarLiteral, ScalarAddress, AggregateLiteral,
                AggregateAddress, ComplexAddress };

    llvm::Value *Value;
    Kind K;
    saved_type(llvm::Value *v, Kind k) : Value(v), K(k) {}

  public:
    static bool needsSaving(RValue value);
    static saved_type save(CodeGenFunction &CGF, RValue value);
    RValue restore(CodeGenFunction &CGF);

    // implementations in CGExprCXX.cpp
  };

  static bool needsSaving(type value) {
    return saved_type::needsSaving(value);
  }
  static saved_type save(CodeGenFunction &CGF, type value) {
    return saved_type::save(CGF, value);
  }
  static type restore(CodeGenFunction &CGF, saved_type value) {
    return value.restore(CGF);
  }
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
