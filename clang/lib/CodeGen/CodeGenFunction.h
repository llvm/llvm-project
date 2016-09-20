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

#ifndef LLVM_CLANG_LIB_CODEGEN_CODEGENFUNCTION_H
#define LLVM_CLANG_LIB_CODEGEN_CODEGENFUNCTION_H

#include "CGBuilder.h"
#include "CGDebugInfo.h"
#include "CGLoopInfo.h"
#include "CGValue.h"
#include "CodeGenModule.h"
#include "CodeGenPGO.h"
#include "EHScopeStack.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprOpenMP.h"
#include "clang/AST/Type.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/CapturedStmt.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/SanitizerStats.h"

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
class BlockByrefHelpers;
class BlockByrefInfo;
class BlockFlags;
class BlockFieldFlags;
class RegionCodeGenTy;
class TargetCodeGenInfo;
struct OMPTaskDataTy;

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

/// CodeGenFunction - This class organizes the per-function state that is used
/// while generating LLVM code.
class CodeGenFunction : public CodeGenTypeCache {
  CodeGenFunction(const CodeGenFunction &) = delete;
  void operator=(const CodeGenFunction &) = delete;

  friend class CGCXXABI;
public:
  /// A jump destination is an abstract label, branching to which may
  /// require a jump out through normal cleanups.
  struct JumpDest {
    JumpDest() : Block(nullptr), ScopeDepth(), Index(0) {}
    JumpDest(llvm::BasicBlock *Block,
             EHScopeStack::stable_iterator Depth,
             unsigned Index)
      : Block(Block), ScopeDepth(Depth), Index(Index) {}

    bool isValid() const { return Block != nullptr; }
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
  LoopInfoStack LoopStack;
  CGBuilderTy Builder;

  /// \brief CGBuilder insert helper. This function is called after an
  /// instruction is created using Builder.
  void InsertHelper(llvm::Instruction *I, const llvm::Twine &Name,
                    llvm::BasicBlock *BB,
                    llvm::BasicBlock::iterator InsertPt) const;

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

  /// ReturnValue - The temporary alloca to hold the return
  /// value. This is invalid iff the function has no return value.
  Address ReturnValue;

  /// AllocaInsertPoint - This is an instruction in the entry block before which
  /// we prefer to insert allocas.
  llvm::AssertingVH<llvm::Instruction> AllocaInsertPt;

  /// \brief API for captured statement code generation.
  class CGCapturedStmtInfo {
  public:
    explicit CGCapturedStmtInfo(CapturedRegionKind K = CR_Default)
        : Kind(K), ThisValue(nullptr), CXXThisFieldDecl(nullptr) {}
    explicit CGCapturedStmtInfo(const CapturedStmt &S,
                                CapturedRegionKind K = CR_Default)
      : Kind(K), ThisValue(nullptr), CXXThisFieldDecl(nullptr) {

      RecordDecl::field_iterator Field =
        S.getCapturedRecordDecl()->field_begin();
      for (CapturedStmt::const_capture_iterator I = S.capture_begin(),
                                                E = S.capture_end();
           I != E; ++I, ++Field) {
        if (I->capturesThis())
          CXXThisFieldDecl = *Field;
        else if (I->capturesVariable())
          CaptureFields[I->getCapturedVar()] = *Field;
        else if (I->capturesVariableByCopy())
          CaptureFields[I->getCapturedVar()] = *Field;
      }
    }

    virtual ~CGCapturedStmtInfo();

    CapturedRegionKind getKind() const { return Kind; }

    virtual void setContextValue(llvm::Value *V) { ThisValue = V; }
    // \brief Retrieve the value of the context parameter.
    virtual llvm::Value *getContextValue() const { return ThisValue; }

    /// \brief Lookup the captured field decl for a variable.
    virtual const FieldDecl *lookup(const VarDecl *VD) const {
      return CaptureFields.lookup(VD);
    }

    bool isCXXThisExprCaptured() const { return getThisFieldDecl() != nullptr; }
    virtual FieldDecl *getThisFieldDecl() const { return CXXThisFieldDecl; }

    static bool classof(const CGCapturedStmtInfo *) {
      return true;
    }

    /// \brief Emit the captured statement body.
    virtual void EmitBody(CodeGenFunction &CGF, const Stmt *S) {
      CGF.incrementProfileCounter(S);
      CGF.EmitStmt(S);
    }

    /// \brief Get the name of the capture helper.
    virtual StringRef getHelperName() const { return "__captured_stmt"; }

  private:
    /// \brief The kind of captured statement being generated.
    CapturedRegionKind Kind;

    /// \brief Keep the map between VarDecl and FieldDecl.
    llvm::SmallDenseMap<const VarDecl *, FieldDecl *> CaptureFields;

    /// \brief The base address of the captured record, passed in as the first
    /// argument of the parallel region function.
    llvm::Value *ThisValue;

    /// \brief Captured 'this' type.
    FieldDecl *CXXThisFieldDecl;
  };
  CGCapturedStmtInfo *CapturedStmtInfo;

  /// \brief RAII for correct setting/restoring of CapturedStmtInfo.
  class CGCapturedStmtRAII {
  private:
    CodeGenFunction &CGF;
    CGCapturedStmtInfo *PrevCapturedStmtInfo;
  public:
    CGCapturedStmtRAII(CodeGenFunction &CGF,
                       CGCapturedStmtInfo *NewCapturedStmtInfo)
        : CGF(CGF), PrevCapturedStmtInfo(CGF.CapturedStmtInfo) {
      CGF.CapturedStmtInfo = NewCapturedStmtInfo;
    }
    ~CGCapturedStmtRAII() { CGF.CapturedStmtInfo = PrevCapturedStmtInfo; }
  };

  /// \brief Sanitizers enabled for this function.
  SanitizerSet SanOpts;

  /// \brief True if CodeGen currently emits code implementing sanitizer checks.
  bool IsSanitizerScope;

  /// \brief RAII object to set/unset CodeGenFunction::IsSanitizerScope.
  class SanitizerScope {
    CodeGenFunction *CGF;
  public:
    SanitizerScope(CodeGenFunction *CGF);
    ~SanitizerScope();
  };

  /// In C++, whether we are code generating a thunk.  This controls whether we
  /// should emit cleanups.
  bool CurFuncIsThunk;

  /// In ARC, whether we should autorelease the return value.
  bool AutoreleaseResult;

  /// Whether we processed a Microsoft-style asm block during CodeGen. These can
  /// potentially set the return value.
  bool SawAsmBlock;

  const FunctionDecl *CurSEHParent = nullptr;

  /// True if the current function is an outlined SEH helper. This can be a
  /// finally block or filter expression.
  bool IsOutlinedSEHHelper;

  const CodeGen::CGBlockInfo *BlockInfo;
  llvm::Value *BlockPointer;

  llvm::DenseMap<const VarDecl *, FieldDecl *> LambdaCaptureFields;
  FieldDecl *LambdaThisCaptureField;

  /// \brief A mapping from NRVO variables to the flags used to indicate
  /// when the NRVO has been applied to this variable.
  llvm::DenseMap<const VarDecl *, llvm::Value *> NRVOFlags;

  EHScopeStack EHStack;
  llvm::SmallVector<char, 256> LifetimeExtendedCleanupStack;
  llvm::SmallVector<const JumpDest *, 2> SEHTryEpilogueStack;

  llvm::Instruction *CurrentFuncletPad = nullptr;

  class CallLifetimeEnd final : public EHScopeStack::Cleanup {
    llvm::Value *Addr;
    llvm::Value *Size;

  public:
    CallLifetimeEnd(Address addr, llvm::Value *size)
        : Addr(addr.getPointer()), Size(size) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      CGF.EmitLifetimeEnd(Size, Addr);
    }
  };

  /// Header for data within LifetimeExtendedCleanupStack.
  struct LifetimeExtendedCleanupHeader {
    /// The size of the following cleanup object.
    unsigned Size;
    /// The kind of cleanup to push: a value from the CleanupKind enumeration.
    CleanupKind Kind;

    size_t getSize() const { return Size; }
    CleanupKind getKind() const { return Kind; }
  };

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

  /// A stack of exception code slots. Entering an __except block pushes a slot
  /// on the stack and leaving pops one. The __exception_code() intrinsic loads
  /// a value from the top of the stack.
  SmallVector<Address, 1> SEHCodeSlotStack;

  /// Value returned by __exception_info intrinsic.
  llvm::Value *SEHInfo = nullptr;

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

  /// Returns true inside SEH __try blocks.
  bool isSEHTryScope() const { return !SEHTryEpilogueStack.empty(); }

  /// Returns true while emitting a cleanuppad.
  bool isCleanupPadScope() const {
    return CurrentFuncletPad && isa<llvm::CleanupPadInst>(CurrentFuncletPad);
  }

  /// pushFullExprCleanup - Push a cleanup to be run at the end of the
  /// current full-expression.  Safe against the possibility that
  /// we're currently inside a conditionally-evaluated expression.
  template <class T, class... As>
  void pushFullExprCleanup(CleanupKind kind, As... A) {
    // If we're not in a conditional branch, or if none of the
    // arguments requires saving, then use the unconditional cleanup.
    if (!isInConditionalBranch())
      return EHStack.pushCleanup<T>(kind, A...);

    // Stash values in a tuple so we can guarantee the order of saves.
    typedef std::tuple<typename DominatingValue<As>::saved_type...> SavedTuple;
    SavedTuple Saved{saveValueInCond(A)...};

    typedef EHScopeStack::ConditionalCleanup<T, As...> CleanupType;
    EHStack.pushCleanupTuple<CleanupType>(kind, Saved);
    initFullExprCleanup();
  }

  /// \brief Queue a cleanup to be pushed after finishing the current
  /// full-expression.
  template <class T, class... As>
  void pushCleanupAfterFullExpr(CleanupKind Kind, As... A) {
    assert(!isInConditionalBranch() && "can't defer conditional cleanup");

    LifetimeExtendedCleanupHeader Header = { sizeof(T), Kind };

    size_t OldSize = LifetimeExtendedCleanupStack.size();
    LifetimeExtendedCleanupStack.resize(
        LifetimeExtendedCleanupStack.size() + sizeof(Header) + Header.Size);

    static_assert(sizeof(Header) % llvm::AlignOf<T>::Alignment == 0,
                  "Cleanup will be allocated on misaligned address");
    char *Buffer = &LifetimeExtendedCleanupStack[OldSize];
    new (Buffer) LifetimeExtendedCleanupHeader(Header);
    new (Buffer + sizeof(Header)) T(A...);
  }

  /// Set up the last cleaup that was pushed as a conditional
  /// full-expression cleanup.
  void initFullExprCleanup();

  /// PushDestructorCleanup - Push a cleanup to call the
  /// complete-object destructor of an object of the given type at the
  /// given address.  Does nothing if T is not a C++ class type with a
  /// non-trivial destructor.
  void PushDestructorCleanup(QualType T, Address Addr);

  /// PushDestructorCleanup - Push a cleanup to call the
  /// complete-object variant of the given destructor on the object at
  /// the given address.
  void PushDestructorCleanup(const CXXDestructorDecl *Dtor, Address Addr);

  /// PopCleanupBlock - Will pop the cleanup entry on the stack and
  /// process all branch fixups.
  void PopCleanupBlock(bool FallThroughIsBranchThrough = false);

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
    size_t LifetimeExtendedCleanupStackSize;
    bool OldDidCallStackSave;
  protected:
    bool PerformCleanup;
  private:

    RunCleanupsScope(const RunCleanupsScope &) = delete;
    void operator=(const RunCleanupsScope &) = delete;

  protected:
    CodeGenFunction& CGF;

  public:
    /// \brief Enter a new cleanup scope.
    explicit RunCleanupsScope(CodeGenFunction &CGF)
      : PerformCleanup(true), CGF(CGF)
    {
      CleanupStackDepth = CGF.EHStack.stable_begin();
      LifetimeExtendedCleanupStackSize =
          CGF.LifetimeExtendedCleanupStack.size();
      OldDidCallStackSave = CGF.DidCallStackSave;
      CGF.DidCallStackSave = false;
    }

    /// \brief Exit this cleanup scope, emitting any accumulated
    /// cleanups.
    ~RunCleanupsScope() {
      if (PerformCleanup) {
        CGF.DidCallStackSave = OldDidCallStackSave;
        CGF.PopCleanupBlocks(CleanupStackDepth,
                             LifetimeExtendedCleanupStackSize);
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
      CGF.PopCleanupBlocks(CleanupStackDepth,
                           LifetimeExtendedCleanupStackSize);
      PerformCleanup = false;
    }
  };

  class LexicalScope : public RunCleanupsScope {
    SourceRange Range;
    SmallVector<const LabelDecl*, 4> Labels;
    LexicalScope *ParentScope;

    LexicalScope(const LexicalScope &) = delete;
    void operator=(const LexicalScope &) = delete;

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
      if (PerformCleanup) {
        ApplyDebugLocation DL(CGF, Range.getEnd());
        ForceCleanup();
      }
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

  typedef llvm::DenseMap<const Decl *, Address> DeclMapTy;

  /// \brief The scope used to remap some variables as private in the OpenMP
  /// loop body (or other captured region emitted without outlining), and to
  /// restore old vars back on exit.
  class OMPPrivateScope : public RunCleanupsScope {
    DeclMapTy SavedLocals;
    DeclMapTy SavedPrivates;

  private:
    OMPPrivateScope(const OMPPrivateScope &) = delete;
    void operator=(const OMPPrivateScope &) = delete;

  public:
    /// \brief Enter a new OpenMP private scope.
    explicit OMPPrivateScope(CodeGenFunction &CGF) : RunCleanupsScope(CGF) {}

    /// \brief Registers \a LocalVD variable as a private and apply \a
    /// PrivateGen function for it to generate corresponding private variable.
    /// \a PrivateGen returns an address of the generated private variable.
    /// \return true if the variable is registered as private, false if it has
    /// been privatized already.
    bool
    addPrivate(const VarDecl *LocalVD,
               llvm::function_ref<Address()> PrivateGen) {
      assert(PerformCleanup && "adding private to dead scope");

      // Only save it once.
      if (SavedLocals.count(LocalVD)) return false;

      // Copy the existing local entry to SavedLocals.
      auto it = CGF.LocalDeclMap.find(LocalVD);
      if (it != CGF.LocalDeclMap.end()) {
        SavedLocals.insert({LocalVD, it->second});
      } else {
        SavedLocals.insert({LocalVD, Address::invalid()});
      }

      // Generate the private entry.
      Address Addr = PrivateGen();
      QualType VarTy = LocalVD->getType();
      if (VarTy->isReferenceType()) {
        Address Temp = CGF.CreateMemTemp(VarTy);
        CGF.Builder.CreateStore(Addr.getPointer(), Temp);
        Addr = Temp;
      }
      SavedPrivates.insert({LocalVD, Addr});

      return true;
    }

    /// \brief Privatizes local variables previously registered as private.
    /// Registration is separate from the actual privatization to allow
    /// initializers use values of the original variables, not the private one.
    /// This is important, for example, if the private variable is a class
    /// variable initialized by a constructor that references other private
    /// variables. But at initialization original variables must be used, not
    /// private copies.
    /// \return true if at least one variable was privatized, false otherwise.
    bool Privatize() {
      copyInto(SavedPrivates, CGF.LocalDeclMap);
      SavedPrivates.clear();
      return !SavedLocals.empty();
    }

    void ForceCleanup() {
      RunCleanupsScope::ForceCleanup();
      copyInto(SavedLocals, CGF.LocalDeclMap);
      SavedLocals.clear();
    }

    /// \brief Exit scope - all the mapped variables are restored.
    ~OMPPrivateScope() {
      if (PerformCleanup)
        ForceCleanup();
    }

    /// Checks if the global variable is captured in current function. 
    bool isGlobalVarCaptured(const VarDecl *VD) const {
      return !VD->isLocalVarDeclOrParm() && CGF.LocalDeclMap.count(VD) > 0;
    }

  private:
    /// Copy all the entries in the source map over the corresponding
    /// entries in the destination, which must exist.
    static void copyInto(const DeclMapTy &src, DeclMapTy &dest) {
      for (auto &pair : src) {
        if (!pair.second.isValid()) {
          dest.erase(pair.first);
          continue;
        }

        auto it = dest.find(pair.first);
        if (it != dest.end()) {
          it->second = pair.second;
        } else {
          dest.insert(pair);
        }
      }
    }
  };

  /// \brief Takes the old cleanup stack size and emits the cleanup blocks
  /// that have been added.
  void PopCleanupBlocks(EHScopeStack::stable_iterator OldCleanupStackSize);

  /// \brief Takes the old cleanup stack size and emits the cleanup blocks
  /// that have been added, then adds all lifetime-extended cleanups from
  /// the given position to the stack.
  void PopCleanupBlocks(EHScopeStack::stable_iterator OldCleanupStackSize,
                        size_t OldLifetimeExtendedStackSize);

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
  llvm::BasicBlock *getMSVCDispatchBlock(EHScopeStack::stable_iterator scope);

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
      assert(CGF.OutermostConditional != nullptr);
      if (CGF.OutermostConditional == this)
        CGF.OutermostConditional = nullptr;
    }

    /// Returns a block which will be executed prior to each
    /// evaluation of the conditional code.
    llvm::BasicBlock *getStartingBlock() const {
      return StartBB;
    }
  };

  /// isInConditionalBranch - Return true if we're currently emitting
  /// one branch or the other of a conditional expression.
  bool isInConditionalBranch() const { return OutermostConditional != nullptr; }

  void setBeforeOutermostConditional(llvm::Value *value, Address addr) {
    assert(isInConditionalBranch());
    llvm::BasicBlock *block = OutermostConditional->getStartingBlock();
    auto store = new llvm::StoreInst(value, addr.getPointer(), &block->back());
    store->setAlignment(addr.getAlignment().getQuantity());
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
      CGF.OutermostConditional = nullptr;
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
    PeepholeProtection() : Inst(nullptr) {}
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
    OpaqueValueMappingData() : OpaqueValue(nullptr) {}

    static bool shouldBindAsLValue(const Expr *expr) {
      // gl-values should be bound as l-values for obvious reasons.
      // Records should be bound as l-values because IR generation
      // always keeps them in memory.  Expressions of function type
      // act exactly like l-values but are formally required to be
      // r-values in C.
      return expr->isGLValue() ||
             expr->getType()->isFunctionType() ||
             hasAggregateEvaluationKind(expr->getType());
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

    bool isValid() const { return OpaqueValue != nullptr; }
    void clear() { OpaqueValue = nullptr; }

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
  
private:
  CGDebugInfo *DebugInfo;
  bool DisableDebugInfo;

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
  DeclMapTy LocalDeclMap;

  /// SizeArguments - If a ParmVarDecl had the pass_object_size attribute, this
  /// will contain a mapping from said ParmVarDecl to its implicit "object_size"
  /// parameter.
  llvm::SmallDenseMap<const ParmVarDecl *, const ImplicitParamDecl *, 2>
      SizeArguments;

  /// Track escaped local variables with auto storage. Used during SEH
  /// outlining to produce a call to llvm.localescape.
  llvm::DenseMap<llvm::AllocaInst *, int> EscapedLocals;

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

  CodeGenPGO PGO;

  /// Calculate branch weights appropriate for PGO data
  llvm::MDNode *createProfileWeights(uint64_t TrueCount, uint64_t FalseCount);
  llvm::MDNode *createProfileWeights(ArrayRef<uint64_t> Weights);
  llvm::MDNode *createProfileWeightsForLoop(const Stmt *Cond,
                                            uint64_t LoopCount);

public:
  /// Increment the profiler's counter for the given statement.
  void incrementProfileCounter(const Stmt *S) {
    if (CGM.getCodeGenOpts().hasProfileClangInstr())
      PGO.emitCounterIncrement(Builder, S);
    PGO.setCurrentStmt(S);
  }

  /// Get the profiler's count for the given statement.
  uint64_t getProfileCount(const Stmt *S) {
    Optional<uint64_t> Count = PGO.getStmtCount(S);
    if (!Count.hasValue())
      return 0;
    return *Count;
  }

  /// Set the profiler's current count.
  void setCurrentProfileCount(uint64_t Count) {
    PGO.setCurrentRegionCount(Count);
  }

  /// Get the profiler's current count. This is generally the count for the most
  /// recently incremented counter.
  uint64_t getCurrentProfileCount() {
    return PGO.getCurrentRegionCount();
  }

private:

  /// SwitchInsn - This is nearest current switch instruction. It is null if
  /// current context is not in a switch.
  llvm::SwitchInst *SwitchInsn;
  /// The branch weights of SwitchInsn when doing instrumentation based PGO.
  SmallVector<uint64_t, 16> *SwitchWeights;

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
    FieldConstructionScope(CodeGenFunction &CGF, Address This)
        : CGF(CGF), OldCXXDefaultInitExprThis(CGF.CXXDefaultInitExprThis) {
      CGF.CXXDefaultInitExprThis = This;
    }
    ~FieldConstructionScope() {
      CGF.CXXDefaultInitExprThis = OldCXXDefaultInitExprThis;
    }

  private:
    CodeGenFunction &CGF;
    Address OldCXXDefaultInitExprThis;
  };

  /// The scope of a CXXDefaultInitExpr. Within this scope, the value of 'this'
  /// is overridden to be the object under construction.
  class CXXDefaultInitExprScope {
  public:
    CXXDefaultInitExprScope(CodeGenFunction &CGF)
      : CGF(CGF), OldCXXThisValue(CGF.CXXThisValue),
        OldCXXThisAlignment(CGF.CXXThisAlignment) {
      CGF.CXXThisValue = CGF.CXXDefaultInitExprThis.getPointer();
      CGF.CXXThisAlignment = CGF.CXXDefaultInitExprThis.getAlignment();
    }
    ~CXXDefaultInitExprScope() {
      CGF.CXXThisValue = OldCXXThisValue;
      CGF.CXXThisAlignment = OldCXXThisAlignment;
    }

  public:
    CodeGenFunction &CGF;
    llvm::Value *OldCXXThisValue;
    CharUnits OldCXXThisAlignment;
  };

  class InlinedInheritingConstructorScope {
  public:
    InlinedInheritingConstructorScope(CodeGenFunction &CGF, GlobalDecl GD)
        : CGF(CGF), OldCurGD(CGF.CurGD), OldCurFuncDecl(CGF.CurFuncDecl),
          OldCurCodeDecl(CGF.CurCodeDecl),
          OldCXXABIThisDecl(CGF.CXXABIThisDecl),
          OldCXXABIThisValue(CGF.CXXABIThisValue),
          OldCXXThisValue(CGF.CXXThisValue),
          OldCXXABIThisAlignment(CGF.CXXABIThisAlignment),
          OldCXXThisAlignment(CGF.CXXThisAlignment),
          OldReturnValue(CGF.ReturnValue), OldFnRetTy(CGF.FnRetTy),
          OldCXXInheritedCtorInitExprArgs(
              std::move(CGF.CXXInheritedCtorInitExprArgs)) {
      CGF.CurGD = GD;
      CGF.CurFuncDecl = CGF.CurCodeDecl =
          cast<CXXConstructorDecl>(GD.getDecl());
      CGF.CXXABIThisDecl = nullptr;
      CGF.CXXABIThisValue = nullptr;
      CGF.CXXThisValue = nullptr;
      CGF.CXXABIThisAlignment = CharUnits();
      CGF.CXXThisAlignment = CharUnits();
      CGF.ReturnValue = Address::invalid();
      CGF.FnRetTy = QualType();
      CGF.CXXInheritedCtorInitExprArgs.clear();
    }
    ~InlinedInheritingConstructorScope() {
      CGF.CurGD = OldCurGD;
      CGF.CurFuncDecl = OldCurFuncDecl;
      CGF.CurCodeDecl = OldCurCodeDecl;
      CGF.CXXABIThisDecl = OldCXXABIThisDecl;
      CGF.CXXABIThisValue = OldCXXABIThisValue;
      CGF.CXXThisValue = OldCXXThisValue;
      CGF.CXXABIThisAlignment = OldCXXABIThisAlignment;
      CGF.CXXThisAlignment = OldCXXThisAlignment;
      CGF.ReturnValue = OldReturnValue;
      CGF.FnRetTy = OldFnRetTy;
      CGF.CXXInheritedCtorInitExprArgs =
          std::move(OldCXXInheritedCtorInitExprArgs);
    }

  private:
    CodeGenFunction &CGF;
    GlobalDecl OldCurGD;
    const Decl *OldCurFuncDecl;
    const Decl *OldCurCodeDecl;
    ImplicitParamDecl *OldCXXABIThisDecl;
    llvm::Value *OldCXXABIThisValue;
    llvm::Value *OldCXXThisValue;
    CharUnits OldCXXABIThisAlignment;
    CharUnits OldCXXThisAlignment;
    Address OldReturnValue;
    QualType OldFnRetTy;
    CallArgList OldCXXInheritedCtorInitExprArgs;
  };

private:
  /// CXXThisDecl - When generating code for a C++ member function,
  /// this will hold the implicit 'this' declaration.
  ImplicitParamDecl *CXXABIThisDecl;
  llvm::Value *CXXABIThisValue;
  llvm::Value *CXXThisValue;
  CharUnits CXXABIThisAlignment;
  CharUnits CXXThisAlignment;

  /// The value of 'this' to use when evaluating CXXDefaultInitExprs within
  /// this expression.
  Address CXXDefaultInitExprThis = Address::invalid();

  /// The values of function arguments to use when evaluating
  /// CXXInheritedCtorInitExprs within this context.
  CallArgList CXXInheritedCtorInitExprArgs;

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

  /// The current source location that should be used for exception
  /// handling code.
  SourceLocation CurEHLocation;

  /// BlockByrefInfos - For each __block variable, contains
  /// information about the layout of the variable.
  llvm::DenseMap<const ValueDecl *, BlockByrefInfo> BlockByrefInfos;

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
  CGDebugInfo *getDebugInfo() { 
    if (DisableDebugInfo) 
      return nullptr;
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
  Address getExceptionSlot();
  Address getEHSelectorSlot();

  /// Returns the contents of the function's exception object and selector
  /// slots.
  llvm::Value *getExceptionFromSlot();
  llvm::Value *getSelectorFromSlot();

  Address getNormalCleanupDestSlot();

  llvm::BasicBlock *getUnreachableBlock() {
    if (!UnreachableBlock) {
      UnreachableBlock = createBasicBlock("unreachable");
      new llvm::UnreachableInst(getLLVMContext(), UnreachableBlock);
    }
    return UnreachableBlock;
  }

  llvm::BasicBlock *getInvokeDest() {
    if (!EHStack.requiresLandingPad()) return nullptr;
    return getInvokeDestImpl();
  }

  bool currentFunctionUsesSEHTry() const { return CurSEHParent != nullptr; }

  const TargetInfo &getTarget() const { return Target; }
  llvm::LLVMContext &getLLVMContext() { return CGM.getLLVMContext(); }

  //===--------------------------------------------------------------------===//
  //                                  Cleanups
  //===--------------------------------------------------------------------===//

  typedef void Destroyer(CodeGenFunction &CGF, Address addr, QualType ty);

  void pushIrregularPartialArrayCleanup(llvm::Value *arrayBegin,
                                        Address arrayEndPointer,
                                        QualType elementType,
                                        CharUnits elementAlignment,
                                        Destroyer *destroyer);
  void pushRegularPartialArrayCleanup(llvm::Value *arrayBegin,
                                      llvm::Value *arrayEnd,
                                      QualType elementType,
                                      CharUnits elementAlignment,
                                      Destroyer *destroyer);

  void pushDestroy(QualType::DestructionKind dtorKind,
                   Address addr, QualType type);
  void pushEHDestroy(QualType::DestructionKind dtorKind,
                     Address addr, QualType type);
  void pushDestroy(CleanupKind kind, Address addr, QualType type,
                   Destroyer *destroyer, bool useEHCleanupForArray);
  void pushLifetimeExtendedDestroy(CleanupKind kind, Address addr,
                                   QualType type, Destroyer *destroyer,
                                   bool useEHCleanupForArray);
  void pushCallObjectDeleteCleanup(const FunctionDecl *OperatorDelete,
                                   llvm::Value *CompletePtr,
                                   QualType ElementType);
  void pushStackRestore(CleanupKind kind, Address SPMem);
  void emitDestroy(Address addr, QualType type, Destroyer *destroyer,
                   bool useEHCleanupForArray);
  llvm::Function *generateDestroyHelper(Address addr, QualType type,
                                        Destroyer *destroyer,
                                        bool useEHCleanupForArray,
                                        const VarDecl *VD);
  void emitArrayDestroy(llvm::Value *begin, llvm::Value *end,
                        QualType elementType, CharUnits elementAlign,
                        Destroyer *destroyer,
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

  void StartObjCMethod(const ObjCMethodDecl *MD, const ObjCContainerDecl *CD);

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

  //===--------------------------------------------------------------------===//
  //                                  Block Bits
  //===--------------------------------------------------------------------===//

  llvm::Value *EmitBlockLiteral(const BlockExpr *);
  llvm::Value *EmitBlockLiteral(const CGBlockInfo &Info);
  static void destroyBlockInfos(CGBlockInfo *info);

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

  void setBlockContextParameter(const ImplicitParamDecl *D, unsigned argNum,
                                llvm::Value *ptr);

  Address LoadBlockStruct();
  Address GetAddrOfBlockDecl(const VarDecl *var, bool ByRef);

  /// BuildBlockByrefAddress - Computes the location of the
  /// data in a variable which is declared as __block.
  Address emitBlockByrefAddress(Address baseAddr, const VarDecl *V,
                                bool followForward = true);
  Address emitBlockByrefAddress(Address baseAddr,
                                const BlockByrefInfo &info,
                                bool followForward,
                                const llvm::Twine &name);

  const BlockByrefInfo &getBlockByrefInfo(const VarDecl *var);

  QualType BuildFunctionArgList(GlobalDecl GD, FunctionArgList &Args);

  void GenerateCode(GlobalDecl GD, llvm::Function *Fn,
                    const CGFunctionInfo &FnInfo);
  /// \brief Emit code for the start of a function.
  /// \param Loc       The location to be associated with the function.
  /// \param StartLoc  The location of the function body.
  void StartFunction(GlobalDecl GD,
                     QualType RetTy,
                     llvm::Function *Fn,
                     const CGFunctionInfo &FnInfo,
                     const FunctionArgList &Args,
                     SourceLocation Loc = SourceLocation(),
                     SourceLocation StartLoc = SourceLocation());

  void EmitConstructorBody(FunctionArgList &Args);
  void EmitDestructorBody(FunctionArgList &Args);
  void emitImplicitAssignmentOperatorBody(FunctionArgList &Args);
  void EmitFunctionBody(FunctionArgList &Args, const Stmt *Body);
  void EmitBlockWithFallThrough(llvm::BasicBlock *BB, const Stmt *S);

  void EmitForwardingCallToLambda(const CXXMethodDecl *LambdaCallOperator,
                                  CallArgList &CallArgs);
  void EmitLambdaToBlockPointerBody(FunctionArgList &Args);
  void EmitLambdaBlockInvokeBody();
  void EmitLambdaDelegatingInvokeBody(const CXXMethodDecl *MD);
  void EmitLambdaStaticInvokeFunction(const CXXMethodDecl *MD);
  void EmitAsanPrologueOrEpilogue(bool Prologue);

  /// \brief Emit the unified return block, trying to avoid its emission when
  /// possible.
  /// \return The debug location of the user written return statement if the
  /// return block is is avoided.
  llvm::DebugLoc EmitReturnBlock();

  /// FinishFunction - Complete IR generation of the current function. It is
  /// legal to call this function even if there is no current insertion point.
  void FinishFunction(SourceLocation EndLoc=SourceLocation());

  void StartThunk(llvm::Function *Fn, GlobalDecl GD,
                  const CGFunctionInfo &FnInfo);

  void EmitCallAndReturnForThunk(llvm::Value *Callee, const ThunkInfo *Thunk);

  void FinishThunk();

  /// Emit a musttail call for a thunk with a potentially adjusted this pointer.
  void EmitMustTailThunk(const CXXMethodDecl *MD, llvm::Value *AdjustedThisPtr,
                         llvm::Value *Callee);

  /// Generate a thunk for the given method.
  void generateThunk(llvm::Function *Fn, const CGFunctionInfo &FnInfo,
                     GlobalDecl GD, const ThunkInfo &Thunk);

  llvm::Function *GenerateVarArgsThunk(llvm::Function *Fn,
                                       const CGFunctionInfo &FnInfo,
                                       GlobalDecl GD, const ThunkInfo &Thunk);

  void EmitCtorPrologue(const CXXConstructorDecl *CD, CXXCtorType Type,
                        FunctionArgList &Args);

  void EmitInitializerForField(FieldDecl *Field, LValue LHS, Expr *Init,
                               ArrayRef<VarDecl *> ArrayIndexes);

  /// Struct with all informations about dynamic [sub]class needed to set vptr.
  struct VPtr {
    BaseSubobject Base;
    const CXXRecordDecl *NearestVBase;
    CharUnits OffsetFromNearestVBase;
    const CXXRecordDecl *VTableClass;
  };

  /// Initialize the vtable pointer of the given subobject.
  void InitializeVTablePointer(const VPtr &vptr);

  typedef llvm::SmallVector<VPtr, 4> VPtrsVector;

  typedef llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBasesSetTy;
  VPtrsVector getVTablePointers(const CXXRecordDecl *VTableClass);

  void getVTablePointers(BaseSubobject Base, const CXXRecordDecl *NearestVBase,
                         CharUnits OffsetFromNearestVBase,
                         bool BaseIsNonVirtualPrimaryBase,
                         const CXXRecordDecl *VTableClass,
                         VisitedVirtualBasesSetTy &VBases, VPtrsVector &vptrs);

  void InitializeVTablePointers(const CXXRecordDecl *ClassDecl);

  /// GetVTablePtr - Return the Value of the vtable pointer member pointed
  /// to by This.
  llvm::Value *GetVTablePtr(Address This, llvm::Type *VTableTy,
                            const CXXRecordDecl *VTableClass);

  enum CFITypeCheckKind {
    CFITCK_VCall,
    CFITCK_NVCall,
    CFITCK_DerivedCast,
    CFITCK_UnrelatedCast,
    CFITCK_ICall,
  };

  /// \brief Derived is the presumed address of an object of type T after a
  /// cast. If T is a polymorphic class type, emit a check that the virtual
  /// table for Derived belongs to a class derived from T.
  void EmitVTablePtrCheckForCast(QualType T, llvm::Value *Derived,
                                 bool MayBeNull, CFITypeCheckKind TCK,
                                 SourceLocation Loc);

  /// EmitVTablePtrCheckForCall - Virtual method MD is being called via VTable.
  /// If vptr CFI is enabled, emit a check that VTable is valid.
  void EmitVTablePtrCheckForCall(const CXXRecordDecl *RD, llvm::Value *VTable,
                                 CFITypeCheckKind TCK, SourceLocation Loc);

  /// EmitVTablePtrCheck - Emit a check that VTable is a valid virtual table for
  /// RD using llvm.type.test.
  void EmitVTablePtrCheck(const CXXRecordDecl *RD, llvm::Value *VTable,
                          CFITypeCheckKind TCK, SourceLocation Loc);

  /// If whole-program virtual table optimization is enabled, emit an assumption
  /// that VTable is a member of RD's type identifier. Or, if vptr CFI is
  /// enabled, emit a check that VTable is a member of RD's type identifier.
  void EmitTypeMetadataCodeForVCall(const CXXRecordDecl *RD,
                                    llvm::Value *VTable, SourceLocation Loc);

  /// Returns whether we should perform a type checked load when loading a
  /// virtual function for virtual calls to members of RD. This is generally
  /// true when both vcall CFI and whole-program-vtables are enabled.
  bool ShouldEmitVTableTypeCheckedLoad(const CXXRecordDecl *RD);

  /// Emit a type checked load from the given vtable.
  llvm::Value *EmitVTableTypeCheckedLoad(const CXXRecordDecl *RD, llvm::Value *VTable,
                                         uint64_t VTableByteOffset);

  /// CanDevirtualizeMemberFunctionCalls - Checks whether virtual calls on given
  /// expr can be devirtualized.
  bool CanDevirtualizeMemberFunctionCall(const Expr *Base,
                                         const CXXMethodDecl *MD);

  /// EnterDtorCleanups - Enter the cleanups necessary to complete the
  /// given phase of destruction for a destructor.  The end result
  /// should call destructors on members and base classes in reverse
  /// order of their construction.
  void EnterDtorCleanups(const CXXDestructorDecl *Dtor, CXXDtorType Type);

  /// ShouldInstrumentFunction - Return true if the current function should be
  /// instrumented with __cyg_profile_func_* calls
  bool ShouldInstrumentFunction();

  /// ShouldXRayInstrument - Return true if the current function should be
  /// instrumented with XRay nop sleds.
  bool ShouldXRayInstrumentFunction() const;

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
  void EmitFunctionEpilog(const CGFunctionInfo &FI, bool EmitRetDbgLoc,
                          SourceLocation EndLoc);

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
                                     llvm::Function *parent = nullptr,
                                     llvm::BasicBlock *before = nullptr) {
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
    return Builder.GetInsertBlock() != nullptr;
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
  void ErrorUnsupported(const Stmt *S, const char *Type);

  //===--------------------------------------------------------------------===//
  //                                  Helpers
  //===--------------------------------------------------------------------===//

  LValue MakeAddrLValue(Address Addr, QualType T,
                        AlignmentSource AlignSource = AlignmentSource::Type) {
    return LValue::MakeAddr(Addr, T, getContext(), AlignSource,
                            CGM.getTBAAInfo(T));
  }

  LValue MakeAddrLValue(llvm::Value *V, QualType T, CharUnits Alignment,
                        AlignmentSource AlignSource = AlignmentSource::Type) {
    return LValue::MakeAddr(Address(V, Alignment), T, getContext(),
                            AlignSource, CGM.getTBAAInfo(T));
  }

  LValue MakeNaturalAlignPointeeAddrLValue(llvm::Value *V, QualType T);
  LValue MakeNaturalAlignAddrLValue(llvm::Value *V, QualType T);
  CharUnits getNaturalTypeAlignment(QualType T,
                                    AlignmentSource *Source = nullptr,
                                    bool forPointeeType = false);
  CharUnits getNaturalPointeeTypeAlignment(QualType T,
                                           AlignmentSource *Source = nullptr);

  Address EmitLoadOfReference(Address Ref, const ReferenceType *RefTy,
                              AlignmentSource *Source = nullptr);
  LValue EmitLoadOfReferenceLValue(Address Ref, const ReferenceType *RefTy);

  Address EmitLoadOfPointer(Address Ptr, const PointerType *PtrTy,
                            AlignmentSource *Source = nullptr);
  LValue EmitLoadOfPointerLValue(Address Ptr, const PointerType *PtrTy);

  /// CreateTempAlloca - This creates a alloca and inserts it into the entry
  /// block. The caller is responsible for setting an appropriate alignment on
  /// the alloca.
  llvm::AllocaInst *CreateTempAlloca(llvm::Type *Ty,
                                     const Twine &Name = "tmp");
  Address CreateTempAlloca(llvm::Type *Ty, CharUnits align,
                           const Twine &Name = "tmp");

  /// CreateDefaultAlignedTempAlloca - This creates an alloca with the
  /// default ABI alignment of the given LLVM type.
  ///
  /// IMPORTANT NOTE: This is *not* generally the right alignment for
  /// any given AST type that happens to have been lowered to the
  /// given IR type.  This should only ever be used for function-local,
  /// IR-driven manipulations like saving and restoring a value.  Do
  /// not hand this address off to arbitrary IRGen routines, and especially
  /// do not pass it as an argument to a function that might expect a
  /// properly ABI-aligned value.
  Address CreateDefaultAlignTempAlloca(llvm::Type *Ty,
                                       const Twine &Name = "tmp");

  /// InitTempAlloca - Provide an initial value for the given alloca which
  /// will be observable at all locations in the function.
  ///
  /// The address should be something that was returned from one of
  /// the CreateTempAlloca or CreateMemTemp routines, and the
  /// initializer must be valid in the entry block (i.e. it must
  /// either be a constant or an argument value).
  void InitTempAlloca(Address Alloca, llvm::Value *Value);

  /// CreateIRTemp - Create a temporary IR object of the given type, with
  /// appropriate alignment. This routine should only be used when an temporary
  /// value needs to be stored into an alloca (for example, to avoid explicit
  /// PHI construction), but the type is the IR type, not the type appropriate
  /// for storing in memory.
  ///
  /// That is, this is exactly equivalent to CreateMemTemp, but calling
  /// ConvertType instead of ConvertTypeForMem.
  Address CreateIRTemp(QualType T, const Twine &Name = "tmp");

  /// CreateMemTemp - Create a temporary memory object of the given type, with
  /// appropriate alignment.
  Address CreateMemTemp(QualType T, const Twine &Name = "tmp");
  Address CreateMemTemp(QualType T, CharUnits Align, const Twine &Name = "tmp");

  /// CreateAggTemp - Create a temporary memory object for the given
  /// aggregate type.
  AggValueSlot CreateAggTemp(QualType T, const Twine &Name = "tmp") {
    return AggValueSlot::forAddr(CreateMemTemp(T, Name),
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
  Address EmitVAListRef(const Expr *E);

  /// Emit a "reference" to a __builtin_ms_va_list; this is
  /// always the value of the expression, because a __builtin_ms_va_list is a
  /// pointer to a char.
  Address EmitMSVAListRef(const Expr *E);

  /// EmitAnyExprToTemp - Similary to EmitAnyExpr(), however, the result will
  /// always be accessible even if no aggregate location is provided.
  RValue EmitAnyExprToTemp(const Expr *E);

  /// EmitAnyExprToMem - Emits the code necessary to evaluate an
  /// arbitrary expression into the given memory location.
  void EmitAnyExprToMem(const Expr *E, Address Location,
                        Qualifiers Quals, bool IsInitializer);

  void EmitAnyExprToExn(const Expr *E, Address Addr);

  /// EmitExprAsInit - Emits the code necessary to initialize a
  /// location in memory with the given initializer.
  void EmitExprAsInit(const Expr *init, const ValueDecl *D, LValue lvalue,
                      bool capturedByInit);

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
  void EmitAggregateAssign(Address DestPtr, Address SrcPtr,
                           QualType EltTy) {
    bool IsVolatile = hasVolatileMember(EltTy);
    EmitAggregateCopy(DestPtr, SrcPtr, EltTy, IsVolatile, true);
  }

  void EmitAggregateCopyCtor(Address DestPtr, Address SrcPtr,
                             QualType DestTy, QualType SrcTy) {
    EmitAggregateCopy(DestPtr, SrcPtr, SrcTy, /*IsVolatile=*/false,
                      /*IsAssignment=*/false);
  }

  /// EmitAggregateCopy - Emit an aggregate copy.
  ///
  /// \param isVolatile - True iff either the source or the destination is
  /// volatile.
  /// \param isAssignment - If false, allow padding to be copied.  This often
  /// yields more efficient.
  void EmitAggregateCopy(Address DestPtr, Address SrcPtr,
                         QualType EltTy, bool isVolatile=false,
                         bool isAssignment = false);

  /// GetAddrOfLocalVar - Return the address of a local variable.
  Address GetAddrOfLocalVar(const VarDecl *VD) {
    auto it = LocalDeclMap.find(VD);
    assert(it != LocalDeclMap.end() &&
           "Invalid argument to GetAddrOfLocalVar(), no decl!");
    return it->second;
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
  void EmitNullInitialization(Address DestPtr, QualType Ty);

  /// Emits a call to an LLVM variable-argument intrinsic, either
  /// \c llvm.va_start or \c llvm.va_end.
  /// \param ArgValue A reference to the \c va_list as emitted by either
  /// \c EmitVAListRef or \c EmitMSVAListRef.
  /// \param IsStart If \c true, emits a call to \c llvm.va_start; otherwise,
  /// calls \c llvm.va_end.
  llvm::Value *EmitVAStartEnd(llvm::Value *ArgValue, bool IsStart);

  /// Generate code to get an argument from the passed in pointer
  /// and update it accordingly.
  /// \param VE The \c VAArgExpr for which to generate code.
  /// \param VAListAddr Receives a reference to the \c va_list as emitted by
  /// either \c EmitVAListRef or \c EmitMSVAListRef.
  /// \returns A pointer to the argument.
  // FIXME: We should be able to get rid of this method and use the va_arg
  // instruction in LLVM instead once it works well enough.
  Address EmitVAArg(VAArgExpr *VE, Address &VAListAddr);

  /// emitArrayLength - Compute the length of an array, even if it's a
  /// VLA, and drill down to the base element type.
  llvm::Value *emitArrayLength(const ArrayType *arrayType,
                               QualType &baseType,
                               Address &addr);

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
  Address LoadCXXThisAddress();

  /// LoadCXXVTT - Load the VTT parameter to base constructors/destructors have
  /// virtual bases.
  // FIXME: Every place that calls LoadCXXVTT is something
  // that needs to be abstracted properly.
  llvm::Value *LoadCXXVTT() {
    assert(CXXStructorImplicitParamValue && "no VTT value for this function");
    return CXXStructorImplicitParamValue;
  }

  /// GetAddressOfBaseOfCompleteClass - Convert the given pointer to a
  /// complete class to the given direct base.
  Address
  GetAddressOfDirectBaseInCompleteClass(Address Value,
                                        const CXXRecordDecl *Derived,
                                        const CXXRecordDecl *Base,
                                        bool BaseIsVirtual);

  static bool ShouldNullCheckClassCastValue(const CastExpr *Cast);

  /// GetAddressOfBaseClass - This function will add the necessary delta to the
  /// load of 'this' and returns address of the base class.
  Address GetAddressOfBaseClass(Address Value,
                                const CXXRecordDecl *Derived,
                                CastExpr::path_const_iterator PathBegin,
                                CastExpr::path_const_iterator PathEnd,
                                bool NullCheckValue, SourceLocation Loc);

  Address GetAddressOfDerivedClass(Address Value,
                                   const CXXRecordDecl *Derived,
                                   CastExpr::path_const_iterator PathBegin,
                                   CastExpr::path_const_iterator PathEnd,
                                   bool NullCheckValue);

  /// GetVTTParameter - Return the VTT parameter that should be passed to a
  /// base constructor/destructor with virtual bases.
  /// FIXME: VTTs are Itanium ABI-specific, so the definition should move
  /// to ItaniumCXXABI.cpp together with all the references to VTT.
  llvm::Value *GetVTTParameter(GlobalDecl GD, bool ForVirtualBase,
                               bool Delegating);

  void EmitDelegateCXXConstructorCall(const CXXConstructorDecl *Ctor,
                                      CXXCtorType CtorType,
                                      const FunctionArgList &Args,
                                      SourceLocation Loc);
  // It's important not to confuse this and the previous function. Delegating
  // constructors are the C++0x feature. The constructor delegate optimization
  // is used to reduce duplication in the base and complete consturctors where
  // they are substantially the same.
  void EmitDelegatingCXXConstructorCall(const CXXConstructorDecl *Ctor,
                                        const FunctionArgList &Args);

  /// Emit a call to an inheriting constructor (that is, one that invokes a
  /// constructor inherited from a base class) by inlining its definition. This
  /// is necessary if the ABI does not support forwarding the arguments to the
  /// base class constructor (because they're variadic or similar).
  void EmitInlinedInheritingCXXConstructorCall(const CXXConstructorDecl *Ctor,
                                               CXXCtorType CtorType,
                                               bool ForVirtualBase,
                                               bool Delegating,
                                               CallArgList &Args);

  /// Emit a call to a constructor inherited from a base class, passing the
  /// current constructor's arguments along unmodified (without even making
  /// a copy).
  void EmitInheritedCXXConstructorCall(const CXXConstructorDecl *D,
                                       bool ForVirtualBase, Address This,
                                       bool InheritedFromVBase,
                                       const CXXInheritedCtorInitExpr *E);

  void EmitCXXConstructorCall(const CXXConstructorDecl *D, CXXCtorType Type,
                              bool ForVirtualBase, bool Delegating,
                              Address This, const CXXConstructExpr *E);

  void EmitCXXConstructorCall(const CXXConstructorDecl *D, CXXCtorType Type,
                              bool ForVirtualBase, bool Delegating,
                              Address This, CallArgList &Args);

  /// Emit assumption load for all bases. Requires to be be called only on
  /// most-derived class and not under construction of the object.
  void EmitVTableAssumptionLoads(const CXXRecordDecl *ClassDecl, Address This);

  /// Emit assumption that vptr load == global vtable.
  void EmitVTableAssumptionLoad(const VPtr &vptr, Address This);

  void EmitSynthesizedCXXCopyCtorCall(const CXXConstructorDecl *D,
                                      Address This, Address Src,
                                      const CXXConstructExpr *E);

  void EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                  const ArrayType *ArrayTy,
                                  Address ArrayPtr,
                                  const CXXConstructExpr *E,
                                  bool ZeroInitialization = false);

  void EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                  llvm::Value *NumElements,
                                  Address ArrayPtr,
                                  const CXXConstructExpr *E,
                                  bool ZeroInitialization = false);

  static Destroyer destroyCXXObject;

  void EmitCXXDestructorCall(const CXXDestructorDecl *D, CXXDtorType Type,
                             bool ForVirtualBase, bool Delegating,
                             Address This);

  void EmitNewArrayInitializer(const CXXNewExpr *E, QualType elementType,
                               llvm::Type *ElementTy, Address NewPtr,
                               llvm::Value *NumElements,
                               llvm::Value *AllocSizeWithoutCookie);

  void EmitCXXTemporary(const CXXTemporary *Temporary, QualType TempType,
                        Address Ptr);

  llvm::Value *EmitLifetimeStart(uint64_t Size, llvm::Value *Addr);
  void EmitLifetimeEnd(llvm::Value *Size, llvm::Value *Addr);

  llvm::Value *EmitCXXNewExpr(const CXXNewExpr *E);
  void EmitCXXDeleteExpr(const CXXDeleteExpr *E);

  void EmitDeleteCall(const FunctionDecl *DeleteFD, llvm::Value *Ptr,
                      QualType DeleteTy);

  RValue EmitBuiltinNewDeleteCall(const FunctionProtoType *Type,
                                  const Expr *Arg, bool IsDelete);

  llvm::Value *EmitCXXTypeidExpr(const CXXTypeidExpr *E);
  llvm::Value *EmitDynamicCast(Address V, const CXXDynamicCastExpr *DCE);
  Address EmitCXXUuidofExpr(const CXXUuidofExpr *E);

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
    TCK_DowncastReference,
    /// Checking the operand of a cast to a base object. Must be suitably sized
    /// and aligned.
    TCK_Upcast,
    /// Checking the operand of a cast to a virtual base object. Must be an
    /// object within its lifetime.
    TCK_UpcastToVirtualBase
  };

  /// \brief Whether any type-checking sanitizers are enabled. If \c false,
  /// calls to EmitTypeCheck can be skipped.
  bool sanitizePerformTypeCheck() const;

  /// \brief Emit a check that \p V is the address of storage of the
  /// appropriate size and alignment for an object of type \p Type.
  void EmitTypeCheck(TypeCheckKind TCK, SourceLocation Loc, llvm::Value *V,
                     QualType Type, CharUnits Alignment = CharUnits::Zero(),
                     bool SkipNullCheck = false);

  /// \brief Emit a check that \p Base points into an array object, which
  /// we can access at index \p Index. \p Accessed should be \c false if we
  /// this expression is used as an lvalue, for instance in "&Arr[Idx]".
  void EmitBoundsCheck(const Expr *E, const Expr *Base, llvm::Value *Index,
                       QualType IndexType, bool Accessed);

  llvm::Value *EmitScalarPrePostIncDec(const UnaryOperator *E, LValue LV,
                                       bool isInc, bool isPre);
  ComplexPairTy EmitComplexPrePostIncDec(const UnaryOperator *E, LValue LV,
                                         bool isInc, bool isPre);

  void EmitAlignmentAssumption(llvm::Value *PtrValue, unsigned Alignment,
                               llvm::Value *OffsetValue = nullptr) {
    Builder.CreateAlignmentAssumption(CGM.getDataLayout(), PtrValue, Alignment,
                                      OffsetValue);
  }

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

  void EmitScalarInit(const Expr *init, const ValueDecl *D, LValue lvalue,
                      bool capturedByInit);
  void EmitScalarInit(llvm::Value *init, LValue lvalue);

  typedef void SpecialInitFn(CodeGenFunction &Init, const VarDecl &D,
                             llvm::Value *Address);

  /// \brief Determine whether the given initializer is trivial in the sense
  /// that it requires no code to be generated.
  bool isTrivialInitializer(const Expr *Init);

  /// EmitAutoVarDecl - Emit an auto variable declaration.
  ///
  /// This function can be called with a null (unreachable) insert point.
  void EmitAutoVarDecl(const VarDecl &D);

  class AutoVarEmission {
    friend class CodeGenFunction;

    const VarDecl *Variable;

    /// The address of the alloca.  Invalid if the variable was emitted
    /// as a global constant.
    Address Addr;

    llvm::Value *NRVOFlag;

    /// True if the variable is a __block variable.
    bool IsByRef;

    /// True if the variable is of aggregate type and has a constant
    /// initializer.
    bool IsConstantAggregate;

    /// Non-null if we should use lifetime annotations.
    llvm::Value *SizeForLifetimeMarkers;

    struct Invalid {};
    AutoVarEmission(Invalid) : Variable(nullptr), Addr(Address::invalid()) {}

    AutoVarEmission(const VarDecl &variable)
      : Variable(&variable), Addr(Address::invalid()), NRVOFlag(nullptr),
        IsByRef(false), IsConstantAggregate(false),
        SizeForLifetimeMarkers(nullptr) {}

    bool wasEmittedAsGlobal() const { return !Addr.isValid(); }

  public:
    static AutoVarEmission invalid() { return AutoVarEmission(Invalid()); }

    bool useLifetimeMarkers() const {
      return SizeForLifetimeMarkers != nullptr;
    }
    llvm::Value *getSizeForLifetimeMarkers() const {
      assert(useLifetimeMarkers());
      return SizeForLifetimeMarkers;
    }

    /// Returns the raw, allocated address, which is not necessarily
    /// the address of the object itself.
    Address getAllocatedAddress() const {
      return Addr;
    }

    /// Returns the address of the object within this declaration.
    /// Note that this does not chase the forwarding pointer for
    /// __block decls.
    Address getObjectAddress(CodeGenFunction &CGF) const {
      if (!IsByRef) return Addr;

      return CGF.emitBlockByrefAddress(Addr, Variable, /*forward*/ false);
    }
  };
  AutoVarEmission EmitAutoVarAlloca(const VarDecl &var);
  void EmitAutoVarInit(const AutoVarEmission &emission);
  void EmitAutoVarCleanups(const AutoVarEmission &emission);  
  void emitAutoVarTypeCleanup(const AutoVarEmission &emission,
                              QualType::DestructionKind dtorKind);

  void EmitStaticVarDecl(const VarDecl &D,
                         llvm::GlobalValue::LinkageTypes Linkage);

  class ParamValue {
    llvm::Value *Value;
    unsigned Alignment;
    ParamValue(llvm::Value *V, unsigned A) : Value(V), Alignment(A) {}
  public:
    static ParamValue forDirect(llvm::Value *value) {
      return ParamValue(value, 0);
    }
    static ParamValue forIndirect(Address addr) {
      assert(!addr.getAlignment().isZero());
      return ParamValue(addr.getPointer(), addr.getAlignment().getQuantity());
    }

    bool isIndirect() const { return Alignment != 0; }
    llvm::Value *getAnyValue() const { return Value; }
    
    llvm::Value *getDirectValue() const {
      assert(!isIndirect());
      return Value;
    }

    Address getIndirectAddress() const {
      assert(isIndirect());
      return Address(Value, CharUnits::fromQuantity(Alignment));
    }
  };

  /// EmitParmDecl - Emit a ParmVarDecl or an ImplicitParamDecl.
  void EmitParmDecl(const VarDecl &D, ParamValue Arg, unsigned ArgNo);

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

  Address EmitCompoundStmt(const CompoundStmt &S, bool GetLast = false,
                           AggValueSlot AVS = AggValueSlot::ignored());
  Address EmitCompoundStmtWithoutScope(const CompoundStmt &S,
                                       bool GetLast = false,
                                       AggValueSlot AVS =
                                                AggValueSlot::ignored());

  /// EmitLabel - Emit the block for the given label. It is legal to call this
  /// function even if there is no current insertion point.
  void EmitLabel(const LabelDecl *D); // helper for EmitLabelStmt.

  void EmitLabelStmt(const LabelStmt &S);
  void EmitAttributedStmt(const AttributedStmt &S);
  void EmitGotoStmt(const GotoStmt &S);
  void EmitIndirectGotoStmt(const IndirectGotoStmt &S);
  void EmitIfStmt(const IfStmt &S);

  void EmitWhileStmt(const WhileStmt &S,
                     ArrayRef<const Attr *> Attrs = None);
  void EmitDoStmt(const DoStmt &S, ArrayRef<const Attr *> Attrs = None);
  void EmitForStmt(const ForStmt &S,
                   ArrayRef<const Attr *> Attrs = None);
  void EmitReturnStmt(const ReturnStmt &S);
  void EmitDeclStmt(const DeclStmt &S);
  void EmitBreakStmt(const BreakStmt &S);
  void EmitContinueStmt(const ContinueStmt &S);
  void EmitSwitchStmt(const SwitchStmt &S);
  void EmitDefaultStmt(const DefaultStmt &S);
  void EmitCaseStmt(const CaseStmt &S);
  void EmitCaseStmtRange(const CaseStmt &S);
  void EmitAsmStmt(const AsmStmt &S);

  void EmitObjCForCollectionStmt(const ObjCForCollectionStmt &S);
  void EmitObjCAtTryStmt(const ObjCAtTryStmt &S);
  void EmitObjCAtThrowStmt(const ObjCAtThrowStmt &S);
  void EmitObjCAtSynchronizedStmt(const ObjCAtSynchronizedStmt &S);
  void EmitObjCAutoreleasePoolStmt(const ObjCAutoreleasePoolStmt &S);

  void EnterCXXTryStmt(const CXXTryStmt &S, bool IsFnTryBlock = false);
  void ExitCXXTryStmt(const CXXTryStmt &S, bool IsFnTryBlock = false);

  void EmitCXXTryStmt(const CXXTryStmt &S);
  void EmitSEHTryStmt(const SEHTryStmt &S);
  void EmitSEHLeaveStmt(const SEHLeaveStmt &S);
  void EnterSEHTryStmt(const SEHTryStmt &S);
  void ExitSEHTryStmt(const SEHTryStmt &S);

  void startOutlinedSEHHelper(CodeGenFunction &ParentCGF, bool IsFilter,
                              const Stmt *OutlinedStmt);

  llvm::Function *GenerateSEHFilterFunction(CodeGenFunction &ParentCGF,
                                            const SEHExceptStmt &Except);

  llvm::Function *GenerateSEHFinallyFunction(CodeGenFunction &ParentCGF,
                                             const SEHFinallyStmt &Finally);

  void EmitSEHExceptionCodeSave(CodeGenFunction &ParentCGF,
                                llvm::Value *ParentFP,
                                llvm::Value *EntryEBP);
  llvm::Value *EmitSEHExceptionCode();
  llvm::Value *EmitSEHExceptionInfo();
  llvm::Value *EmitSEHAbnormalTermination();

  /// Scan the outlined statement for captures from the parent function. For
  /// each capture, mark the capture as escaped and emit a call to
  /// llvm.localrecover. Insert the localrecover result into the LocalDeclMap.
  void EmitCapturedLocals(CodeGenFunction &ParentCGF, const Stmt *OutlinedStmt,
                          bool IsFilter);

  /// Recovers the address of a local in a parent function. ParentVar is the
  /// address of the variable used in the immediate parent function. It can
  /// either be an alloca or a call to llvm.localrecover if there are nested
  /// outlined functions. ParentFP is the frame pointer of the outermost parent
  /// frame.
  Address recoverAddrOfEscapedLocal(CodeGenFunction &ParentCGF,
                                    Address ParentVar,
                                    llvm::Value *ParentFP);

  void EmitCXXForRangeStmt(const CXXForRangeStmt &S,
                           ArrayRef<const Attr *> Attrs = None);

  /// Returns calculated size of the specified type.
  llvm::Value *getTypeSize(QualType Ty);
  LValue InitCapturedStruct(const CapturedStmt &S);
  llvm::Function *EmitCapturedStmt(const CapturedStmt &S, CapturedRegionKind K);
  llvm::Function *GenerateCapturedStmtFunction(const CapturedStmt &S);
  Address GenerateCapturedStmtArgument(const CapturedStmt &S);
  llvm::Function *GenerateOpenMPCapturedStmtFunction(const CapturedStmt &S);
  void GenerateOpenMPCapturedVars(const CapturedStmt &S,
                                  SmallVectorImpl<llvm::Value *> &CapturedVars);
  void emitOMPSimpleStore(LValue LVal, RValue RVal, QualType RValTy,
                          SourceLocation Loc);
  /// \brief Perform element by element copying of arrays with type \a
  /// OriginalType from \a SrcAddr to \a DestAddr using copying procedure
  /// generated by \a CopyGen.
  ///
  /// \param DestAddr Address of the destination array.
  /// \param SrcAddr Address of the source array.
  /// \param OriginalType Type of destination and source arrays.
  /// \param CopyGen Copying procedure that copies value of single array element
  /// to another single array element.
  void EmitOMPAggregateAssign(
      Address DestAddr, Address SrcAddr, QualType OriginalType,
      const llvm::function_ref<void(Address, Address)> &CopyGen);
  /// \brief Emit proper copying of data from one variable to another.
  ///
  /// \param OriginalType Original type of the copied variables.
  /// \param DestAddr Destination address.
  /// \param SrcAddr Source address.
  /// \param DestVD Destination variable used in \a CopyExpr (for arrays, has
  /// type of the base array element).
  /// \param SrcVD Source variable used in \a CopyExpr (for arrays, has type of
  /// the base array element).
  /// \param Copy Actual copygin expression for copying data from \a SrcVD to \a
  /// DestVD.
  void EmitOMPCopy(QualType OriginalType,
                   Address DestAddr, Address SrcAddr,
                   const VarDecl *DestVD, const VarDecl *SrcVD,
                   const Expr *Copy);
  /// \brief Emit atomic update code for constructs: \a X = \a X \a BO \a E or
  /// \a X = \a E \a BO \a E.
  ///
  /// \param X Value to be updated.
  /// \param E Update value.
  /// \param BO Binary operation for update operation.
  /// \param IsXLHSInRHSPart true if \a X is LHS in RHS part of the update
  /// expression, false otherwise.
  /// \param AO Atomic ordering of the generated atomic instructions.
  /// \param CommonGen Code generator for complex expressions that cannot be
  /// expressed through atomicrmw instruction.
  /// \returns <true, OldAtomicValue> if simple 'atomicrmw' instruction was
  /// generated, <false, RValue::get(nullptr)> otherwise.
  std::pair<bool, RValue> EmitOMPAtomicSimpleUpdateExpr(
      LValue X, RValue E, BinaryOperatorKind BO, bool IsXLHSInRHSPart,
      llvm::AtomicOrdering AO, SourceLocation Loc,
      const llvm::function_ref<RValue(RValue)> &CommonGen);
  bool EmitOMPFirstprivateClause(const OMPExecutableDirective &D,
                                 OMPPrivateScope &PrivateScope);
  void EmitOMPPrivateClause(const OMPExecutableDirective &D,
                            OMPPrivateScope &PrivateScope);
  void EmitOMPUseDevicePtrClause(
      const OMPClause &C, OMPPrivateScope &PrivateScope,
      const llvm::DenseMap<const ValueDecl *, Address> &CaptureDeviceAddrMap);
  /// \brief Emit code for copyin clause in \a D directive. The next code is
  /// generated at the start of outlined functions for directives:
  /// \code
  /// threadprivate_var1 = master_threadprivate_var1;
  /// operator=(threadprivate_var2, master_threadprivate_var2);
  /// ...
  /// __kmpc_barrier(&loc, global_tid);
  /// \endcode
  ///
  /// \param D OpenMP directive possibly with 'copyin' clause(s).
  /// \returns true if at least one copyin variable is found, false otherwise.
  bool EmitOMPCopyinClause(const OMPExecutableDirective &D);
  /// \brief Emit initial code for lastprivate variables. If some variable is
  /// not also firstprivate, then the default initialization is used. Otherwise
  /// initialization of this variable is performed by EmitOMPFirstprivateClause
  /// method.
  ///
  /// \param D Directive that may have 'lastprivate' directives.
  /// \param PrivateScope Private scope for capturing lastprivate variables for
  /// proper codegen in internal captured statement.
  ///
  /// \returns true if there is at least one lastprivate variable, false
  /// otherwise.
  bool EmitOMPLastprivateClauseInit(const OMPExecutableDirective &D,
                                    OMPPrivateScope &PrivateScope);
  /// \brief Emit final copying of lastprivate values to original variables at
  /// the end of the worksharing or simd directive.
  ///
  /// \param D Directive that has at least one 'lastprivate' directives.
  /// \param IsLastIterCond Boolean condition that must be set to 'i1 true' if
  /// it is the last iteration of the loop code in associated directive, or to
  /// 'i1 false' otherwise. If this item is nullptr, no final check is required.
  void EmitOMPLastprivateClauseFinal(const OMPExecutableDirective &D,
                                     bool NoFinals,
                                     llvm::Value *IsLastIterCond = nullptr);
  /// Emit initial code for linear clauses.
  void EmitOMPLinearClause(const OMPLoopDirective &D,
                           CodeGenFunction::OMPPrivateScope &PrivateScope);
  /// Emit final code for linear clauses.
  /// \param CondGen Optional conditional code for final part of codegen for
  /// linear clause.
  void EmitOMPLinearClauseFinal(
      const OMPLoopDirective &D,
      const llvm::function_ref<llvm::Value *(CodeGenFunction &)> &CondGen);
  /// \brief Emit initial code for reduction variables. Creates reduction copies
  /// and initializes them with the values according to OpenMP standard.
  ///
  /// \param D Directive (possibly) with the 'reduction' clause.
  /// \param PrivateScope Private scope for capturing reduction variables for
  /// proper codegen in internal captured statement.
  ///
  void EmitOMPReductionClauseInit(const OMPExecutableDirective &D,
                                  OMPPrivateScope &PrivateScope);
  /// \brief Emit final update of reduction values to original variables at
  /// the end of the directive.
  ///
  /// \param D Directive that has at least one 'reduction' directives.
  void EmitOMPReductionClauseFinal(const OMPExecutableDirective &D);
  /// \brief Emit initial code for linear variables. Creates private copies
  /// and initializes them with the values according to OpenMP standard.
  ///
  /// \param D Directive (possibly) with the 'linear' clause.
  void EmitOMPLinearClauseInit(const OMPLoopDirective &D);

  typedef const llvm::function_ref<void(CodeGenFunction & /*CGF*/,
                                        llvm::Value * /*OutlinedFn*/,
                                        const OMPTaskDataTy & /*Data*/)>
      TaskGenTy;
  void EmitOMPTaskBasedDirective(const OMPExecutableDirective &S,
                                 const RegionCodeGenTy &BodyGen,
                                 const TaskGenTy &TaskGen, OMPTaskDataTy &Data);

  void EmitOMPParallelDirective(const OMPParallelDirective &S);
  void EmitOMPSimdDirective(const OMPSimdDirective &S);
  void EmitOMPForDirective(const OMPForDirective &S);
  void EmitOMPForSimdDirective(const OMPForSimdDirective &S);
  void EmitOMPSectionsDirective(const OMPSectionsDirective &S);
  void EmitOMPSectionDirective(const OMPSectionDirective &S);
  void EmitOMPSingleDirective(const OMPSingleDirective &S);
  void EmitOMPMasterDirective(const OMPMasterDirective &S);
  void EmitOMPCriticalDirective(const OMPCriticalDirective &S);
  void EmitOMPParallelForDirective(const OMPParallelForDirective &S);
  void EmitOMPParallelForSimdDirective(const OMPParallelForSimdDirective &S);
  void EmitOMPParallelSectionsDirective(const OMPParallelSectionsDirective &S);
  void EmitOMPTaskDirective(const OMPTaskDirective &S);
  void EmitOMPTaskyieldDirective(const OMPTaskyieldDirective &S);
  void EmitOMPBarrierDirective(const OMPBarrierDirective &S);
  void EmitOMPTaskwaitDirective(const OMPTaskwaitDirective &S);
  void EmitOMPTaskgroupDirective(const OMPTaskgroupDirective &S);
  void EmitOMPFlushDirective(const OMPFlushDirective &S);
  void EmitOMPOrderedDirective(const OMPOrderedDirective &S);
  void EmitOMPAtomicDirective(const OMPAtomicDirective &S);
  void EmitOMPTargetDirective(const OMPTargetDirective &S);
  void EmitOMPTargetDataDirective(const OMPTargetDataDirective &S);
  void EmitOMPTargetEnterDataDirective(const OMPTargetEnterDataDirective &S);
  void EmitOMPTargetExitDataDirective(const OMPTargetExitDataDirective &S);
  void EmitOMPTargetUpdateDirective(const OMPTargetUpdateDirective &S);
  void EmitOMPTargetParallelDirective(const OMPTargetParallelDirective &S);
  void
  EmitOMPTargetParallelForDirective(const OMPTargetParallelForDirective &S);
  void EmitOMPTeamsDirective(const OMPTeamsDirective &S);
  void
  EmitOMPCancellationPointDirective(const OMPCancellationPointDirective &S);
  void EmitOMPCancelDirective(const OMPCancelDirective &S);
  void EmitOMPTaskLoopBasedDirective(const OMPLoopDirective &S);
  void EmitOMPTaskLoopDirective(const OMPTaskLoopDirective &S);
  void EmitOMPTaskLoopSimdDirective(const OMPTaskLoopSimdDirective &S);
  void EmitOMPDistributeDirective(const OMPDistributeDirective &S);
  void EmitOMPDistributeLoop(const OMPDistributeDirective &S);
  void EmitOMPDistributeParallelForDirective(
      const OMPDistributeParallelForDirective &S);
  void EmitOMPDistributeParallelForSimdDirective(
      const OMPDistributeParallelForSimdDirective &S);
  void EmitOMPDistributeSimdDirective(const OMPDistributeSimdDirective &S);
  void EmitOMPTargetParallelForSimdDirective(
      const OMPTargetParallelForSimdDirective &S);
  void EmitOMPTargetSimdDirective(const OMPTargetSimdDirective &S);
  void EmitOMPTeamsDistributeDirective(const OMPTeamsDistributeDirective &S);

  /// Emit outlined function for the target directive.
  static std::pair<llvm::Function * /*OutlinedFn*/,
                   llvm::Constant * /*OutlinedFnID*/>
  EmitOMPTargetDirectiveOutlinedFunction(CodeGenModule &CGM,
                                         const OMPTargetDirective &S,
                                         StringRef ParentName,
                                         bool IsOffloadEntry);
  /// \brief Emit inner loop of the worksharing/simd construct.
  ///
  /// \param S Directive, for which the inner loop must be emitted.
  /// \param RequiresCleanup true, if directive has some associated private
  /// variables.
  /// \param LoopCond Bollean condition for loop continuation.
  /// \param IncExpr Increment expression for loop control variable.
  /// \param BodyGen Generator for the inner body of the inner loop.
  /// \param PostIncGen Genrator for post-increment code (required for ordered
  /// loop directvies).
  void EmitOMPInnerLoop(
      const Stmt &S, bool RequiresCleanup, const Expr *LoopCond,
      const Expr *IncExpr,
      const llvm::function_ref<void(CodeGenFunction &)> &BodyGen,
      const llvm::function_ref<void(CodeGenFunction &)> &PostIncGen);

  JumpDest getOMPCancelDestination(OpenMPDirectiveKind Kind);
  /// Emit initial code for loop counters of loop-based directives.
  void EmitOMPPrivateLoopCounters(const OMPLoopDirective &S,
                                  OMPPrivateScope &LoopScope);

private:
  /// Helpers for the OpenMP loop directives.
  void EmitOMPLoopBody(const OMPLoopDirective &D, JumpDest LoopExit);
  void EmitOMPSimdInit(const OMPLoopDirective &D, bool IsMonotonic = false);
  void EmitOMPSimdFinal(
      const OMPLoopDirective &D,
      const llvm::function_ref<llvm::Value *(CodeGenFunction &)> &CondGen);
  /// \brief Emit code for the worksharing loop-based directive.
  /// \return true, if this construct has any lastprivate clause, false -
  /// otherwise.
  bool EmitOMPWorksharingLoop(const OMPLoopDirective &S);
  void EmitOMPOuterLoop(bool IsMonotonic, bool DynamicOrOrdered,
      const OMPLoopDirective &S, OMPPrivateScope &LoopScope, bool Ordered,
      Address LB, Address UB, Address ST, Address IL, llvm::Value *Chunk);
  void EmitOMPForOuterLoop(const OpenMPScheduleTy &ScheduleKind,
                           bool IsMonotonic, const OMPLoopDirective &S,
                           OMPPrivateScope &LoopScope, bool Ordered, Address LB,
                           Address UB, Address ST, Address IL,
                           llvm::Value *Chunk);
  void EmitOMPDistributeOuterLoop(
      OpenMPDistScheduleClauseKind ScheduleKind,
      const OMPDistributeDirective &S, OMPPrivateScope &LoopScope,
      Address LB, Address UB, Address ST, Address IL, llvm::Value *Chunk);
  /// \brief Emit code for sections directive.
  void EmitSections(const OMPExecutableDirective &S);

public:

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

  RValue convertTempToRValue(Address addr, QualType type,
                             SourceLocation Loc);

  void EmitAtomicInit(Expr *E, LValue lvalue);

  bool LValueIsSuitableForInlineAtomic(LValue Src);

  RValue EmitAtomicLoad(LValue LV, SourceLocation SL,
                        AggValueSlot Slot = AggValueSlot::ignored());

  RValue EmitAtomicLoad(LValue lvalue, SourceLocation loc,
                        llvm::AtomicOrdering AO, bool IsVolatile = false,
                        AggValueSlot slot = AggValueSlot::ignored());

  void EmitAtomicStore(RValue rvalue, LValue lvalue, bool isInit);

  void EmitAtomicStore(RValue rvalue, LValue lvalue, llvm::AtomicOrdering AO,
                       bool IsVolatile, bool isInit);

  std::pair<RValue, llvm::Value *> EmitAtomicCompareExchange(
      LValue Obj, RValue Expected, RValue Desired, SourceLocation Loc,
      llvm::AtomicOrdering Success =
          llvm::AtomicOrdering::SequentiallyConsistent,
      llvm::AtomicOrdering Failure =
          llvm::AtomicOrdering::SequentiallyConsistent,
      bool IsWeak = false, AggValueSlot Slot = AggValueSlot::ignored());

  void EmitAtomicUpdate(LValue LVal, llvm::AtomicOrdering AO,
                        const llvm::function_ref<RValue(RValue)> &UpdateOp,
                        bool IsVolatile);

  /// EmitToMemory - Change a scalar value from its value
  /// representation to its in-memory representation.
  llvm::Value *EmitToMemory(llvm::Value *Value, QualType Ty);

  /// EmitFromMemory - Change a scalar value from its memory
  /// representation to its value representation.
  llvm::Value *EmitFromMemory(llvm::Value *Value, QualType Ty);

  /// EmitLoadOfScalar - Load a scalar value from an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.
  llvm::Value *EmitLoadOfScalar(Address Addr, bool Volatile, QualType Ty,
                                SourceLocation Loc,
                                AlignmentSource AlignSource =
                                  AlignmentSource::Type,
                                llvm::MDNode *TBAAInfo = nullptr,
                                QualType TBAABaseTy = QualType(),
                                uint64_t TBAAOffset = 0,
                                bool isNontemporal = false);

  /// EmitLoadOfScalar - Load a scalar value from an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.  The l-value must be a simple
  /// l-value.
  llvm::Value *EmitLoadOfScalar(LValue lvalue, SourceLocation Loc);

  /// EmitStoreOfScalar - Store a scalar value to an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.
  void EmitStoreOfScalar(llvm::Value *Value, Address Addr,
                         bool Volatile, QualType Ty,
                         AlignmentSource AlignSource = AlignmentSource::Type,
                         llvm::MDNode *TBAAInfo = nullptr, bool isInit = false,
                         QualType TBAABaseTy = QualType(),
                         uint64_t TBAAOffset = 0, bool isNontemporal = false);

  /// EmitStoreOfScalar - Store a scalar value to an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.  The l-value must be a simple
  /// l-value.  The isInit flag indicates whether this is an initialization.
  /// If so, atomic qualifiers are ignored and the store is always non-atomic.
  void EmitStoreOfScalar(llvm::Value *value, LValue lvalue, bool isInit=false);

  /// EmitLoadOfLValue - Given an expression that represents a value lvalue,
  /// this method emits the address of the lvalue, then loads the result as an
  /// rvalue, returning the rvalue.
  RValue EmitLoadOfLValue(LValue V, SourceLocation Loc);
  RValue EmitLoadOfExtVectorElementLValue(LValue V);
  RValue EmitLoadOfBitfieldLValue(LValue LV);
  RValue EmitLoadOfGlobalRegLValue(LValue LV);

  /// EmitStoreThroughLValue - Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void EmitStoreThroughLValue(RValue Src, LValue Dst, bool isInit = false);
  void EmitStoreThroughExtVectorComponentLValue(RValue Src, LValue Dst);
  void EmitStoreThroughGlobalRegLValue(RValue Src, LValue Dst);

  /// EmitStoreThroughBitfieldLValue - Store Src into Dst with same constraints
  /// as EmitStoreThroughLValue.
  ///
  /// \param Result [out] - If non-null, this will be set to a Value* for the
  /// bit-field contents after the store, appropriate for use as the result of
  /// an assignment to the bit-field.
  void EmitStoreThroughBitfieldLValue(RValue Src, LValue Dst,
                                      llvm::Value **Result=nullptr);

  /// Emit an l-value for an assignment (simple or compound) of complex type.
  LValue EmitComplexAssignmentLValue(const BinaryOperator *E);
  LValue EmitComplexCompoundAssignmentLValue(const CompoundAssignOperator *E);
  LValue EmitScalarCompoundAssignWithComplex(const CompoundAssignOperator *E,
                                             llvm::Value *&Result);

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
  LValue EmitOMPArraySectionExpr(const OMPArraySectionExpr *E,
                                 bool IsLowerBound = true);
  LValue EmitExtVectorElementExpr(const ExtVectorElementExpr *E);
  LValue EmitMemberExpr(const MemberExpr *E);
  LValue EmitObjCIsaExpr(const ObjCIsaExpr *E);
  LValue EmitCompoundLiteralLValue(const CompoundLiteralExpr *E);
  LValue EmitInitListLValue(const InitListExpr *E);
  LValue EmitConditionalOperatorLValue(const AbstractConditionalOperator *E);
  LValue EmitCastLValue(const CastExpr *E);
  LValue EmitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *E);
  LValue EmitOpaqueValueLValue(const OpaqueValueExpr *e);
  
  Address EmitExtVectorElementLValue(LValue V);

  RValue EmitRValueForField(LValue LV, const FieldDecl *FD, SourceLocation Loc);

  Address EmitArrayToPointerDecay(const Expr *Array,
                                  AlignmentSource *AlignSource = nullptr);

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

    explicit operator bool() const {
      return ValueAndIsReference.getOpaqueValue() != nullptr;
    }

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
  void   EmitDeclRefExprDbgValue(const DeclRefExpr *E, const APValue &Init);

  //===--------------------------------------------------------------------===//
  //                         Scalar Expression Emission
  //===--------------------------------------------------------------------===//

  /// EmitCall - Generate a call of the given function, expecting the given
  /// result type, and using the given argument list which specifies both the
  /// LLVM arguments and the types they were derived from.
  RValue EmitCall(const CGFunctionInfo &FnInfo, llvm::Value *Callee,
                  ReturnValueSlot ReturnValue, const CallArgList &Args,
                  CGCalleeInfo CalleeInfo = CGCalleeInfo(),
                  llvm::Instruction **callOrInvoke = nullptr);

  RValue EmitCall(QualType FnType, llvm::Value *Callee, const CallExpr *E,
                  ReturnValueSlot ReturnValue,
                  CGCalleeInfo CalleeInfo = CGCalleeInfo(),
                  llvm::Value *Chain = nullptr);
  RValue EmitCallExpr(const CallExpr *E,
                      ReturnValueSlot ReturnValue = ReturnValueSlot());

  void checkTargetFeatures(const CallExpr *E, const FunctionDecl *TargetDecl);

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
  llvm::CallSite EmitRuntimeCallOrInvoke(llvm::Value *callee,
                                         ArrayRef<llvm::Value*> args,
                                         const Twine &name = "");
  llvm::CallSite EmitRuntimeCallOrInvoke(llvm::Value *callee,
                                         const Twine &name = "");
  void EmitNoreturnRuntimeCallOrInvoke(llvm::Value *callee,
                                       ArrayRef<llvm::Value*> args);

  llvm::Value *BuildAppleKextVirtualCall(const CXXMethodDecl *MD, 
                                         NestedNameSpecifier *Qual,
                                         llvm::Type *Ty);
  
  llvm::Value *BuildAppleKextVirtualDestructorCall(const CXXDestructorDecl *DD,
                                                   CXXDtorType Type, 
                                                   const CXXRecordDecl *RD);

  RValue
  EmitCXXMemberOrOperatorCall(const CXXMethodDecl *MD, llvm::Value *Callee,
                              ReturnValueSlot ReturnValue, llvm::Value *This,
                              llvm::Value *ImplicitParam,
                              QualType ImplicitParamTy, const CallExpr *E);
  RValue EmitCXXDestructorCall(const CXXDestructorDecl *DD, llvm::Value *Callee,
                               llvm::Value *This, llvm::Value *ImplicitParam,
                               QualType ImplicitParamTy, const CallExpr *E,
                               StructorType Type);
  RValue EmitCXXMemberCallExpr(const CXXMemberCallExpr *E,
                               ReturnValueSlot ReturnValue);
  RValue EmitCXXMemberOrOperatorMemberCallExpr(const CallExpr *CE,
                                               const CXXMethodDecl *MD,
                                               ReturnValueSlot ReturnValue,
                                               bool HasQualifier,
                                               NestedNameSpecifier *Qualifier,
                                               bool IsArrow, const Expr *Base);
  // Compute the object pointer.
  Address EmitCXXMemberDataPointerAddress(const Expr *E, Address base,
                                          llvm::Value *memberPtr,
                                          const MemberPointerType *memberPtrType,
                                          AlignmentSource *AlignSource = nullptr);
  RValue EmitCXXMemberPointerCallExpr(const CXXMemberCallExpr *E,
                                      ReturnValueSlot ReturnValue);

  RValue EmitCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *E,
                                       const CXXMethodDecl *MD,
                                       ReturnValueSlot ReturnValue);

  RValue EmitCUDAKernelCallExpr(const CUDAKernelCallExpr *E,
                                ReturnValueSlot ReturnValue);

  RValue EmitCUDADevicePrintfCallExpr(const CallExpr *E,
                                      ReturnValueSlot ReturnValue);

  RValue EmitBuiltinExpr(const FunctionDecl *FD,
                         unsigned BuiltinID, const CallExpr *E,
                         ReturnValueSlot ReturnValue);

  RValue EmitBlockCallExpr(const CallExpr *E, ReturnValueSlot ReturnValue);

  /// EmitTargetBuiltinExpr - Emit the given builtin call. Returns 0 if the call
  /// is unhandled by the current target.
  llvm::Value *EmitTargetBuiltinExpr(unsigned BuiltinID, const CallExpr *E);

  llvm::Value *EmitAArch64CompareBuiltinExpr(llvm::Value *Op, llvm::Type *Ty,
                                             const llvm::CmpInst::Predicate Fp,
                                             const llvm::CmpInst::Predicate Ip,
                                             const llvm::Twine &Name = "");
  llvm::Value *EmitARMBuiltinExpr(unsigned BuiltinID, const CallExpr *E);

  llvm::Value *EmitCommonNeonBuiltinExpr(unsigned BuiltinID,
                                         unsigned LLVMIntrinsic,
                                         unsigned AltLLVMIntrinsic,
                                         const char *NameHint,
                                         unsigned Modifier,
                                         const CallExpr *E,
                                         SmallVectorImpl<llvm::Value *> &Ops,
                                         Address PtrOp0, Address PtrOp1);
  llvm::Function *LookupNeonLLVMIntrinsic(unsigned IntrinsicID,
                                          unsigned Modifier, llvm::Type *ArgTy,
                                          const CallExpr *E);
  llvm::Value *EmitNeonCall(llvm::Function *F,
                            SmallVectorImpl<llvm::Value*> &O,
                            const char *name,
                            unsigned shift = 0, bool rightshift = false);
  llvm::Value *EmitNeonSplat(llvm::Value *V, llvm::Constant *Idx);
  llvm::Value *EmitNeonShiftVector(llvm::Value *V, llvm::Type *Ty,
                                   bool negateForRightShift);
  llvm::Value *EmitNeonRShiftImm(llvm::Value *Vec, llvm::Value *Amt,
                                 llvm::Type *Ty, bool usgn, const char *name);
  llvm::Value *vectorWrapScalar16(llvm::Value *Op);
  llvm::Value *EmitAArch64BuiltinExpr(unsigned BuiltinID, const CallExpr *E);

  llvm::Value *BuildVector(ArrayRef<llvm::Value*> Ops);
  llvm::Value *EmitX86BuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  llvm::Value *EmitPPCBuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  llvm::Value *EmitAMDGPUBuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  llvm::Value *EmitSystemZBuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  llvm::Value *EmitNVPTXBuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  llvm::Value *EmitWebAssemblyBuiltinExpr(unsigned BuiltinID,
                                          const CallExpr *E);

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
  void EmitARCInitWeak(Address addr, llvm::Value *value);
  void EmitARCDestroyWeak(Address addr);
  llvm::Value *EmitARCLoadWeak(Address addr);
  llvm::Value *EmitARCLoadWeakRetained(Address addr);
  llvm::Value *EmitARCStoreWeak(Address addr, llvm::Value *value, bool ignored);
  void EmitARCCopyWeak(Address dst, Address src);
  void EmitARCMoveWeak(Address dst, Address src);
  llvm::Value *EmitARCRetainAutorelease(QualType type, llvm::Value *value);
  llvm::Value *EmitARCRetainAutoreleaseNonBlock(llvm::Value *value);
  llvm::Value *EmitARCStoreStrong(LValue lvalue, llvm::Value *value,
                                  bool resultIgnored);
  llvm::Value *EmitARCStoreStrongCall(Address addr, llvm::Value *value,
                                      bool resultIgnored);
  llvm::Value *EmitARCRetain(QualType type, llvm::Value *value);
  llvm::Value *EmitARCRetainNonBlock(llvm::Value *value);
  llvm::Value *EmitARCRetainBlock(llvm::Value *value, bool mandatory);
  void EmitARCDestroyStrong(Address addr, ARCPreciseLifetime_t precise);
  void EmitARCRelease(llvm::Value *value, ARCPreciseLifetime_t precise);
  llvm::Value *EmitARCAutorelease(llvm::Value *value);
  llvm::Value *EmitARCAutoreleaseReturnValue(llvm::Value *value);
  llvm::Value *EmitARCRetainAutoreleaseReturnValue(llvm::Value *value);
  llvm::Value *EmitARCRetainAutoreleasedReturnValue(llvm::Value *value);
  llvm::Value *EmitARCUnsafeClaimAutoreleasedReturnValue(llvm::Value *value);

  std::pair<LValue,llvm::Value*>
  EmitARCStoreAutoreleasing(const BinaryOperator *e);
  std::pair<LValue,llvm::Value*>
  EmitARCStoreStrong(const BinaryOperator *e, bool ignored);
  std::pair<LValue,llvm::Value*>
  EmitARCStoreUnsafeUnretained(const BinaryOperator *e, bool ignored);

  llvm::Value *EmitObjCThrowOperand(const Expr *expr);
  llvm::Value *EmitObjCConsumeObject(QualType T, llvm::Value *Ptr);
  llvm::Value *EmitObjCExtendObjectLifetime(QualType T, llvm::Value *Ptr);

  llvm::Value *EmitARCExtendBlockObject(const Expr *expr);
  llvm::Value *EmitARCReclaimReturnedObject(const Expr *e,
                                            bool allowUnsafeClaim);
  llvm::Value *EmitARCRetainScalarExpr(const Expr *expr);
  llvm::Value *EmitARCRetainAutoreleaseScalarExpr(const Expr *expr);
  llvm::Value *EmitARCUnsafeUnretainedScalarExpr(const Expr *expr);

  void EmitARCIntrinsicUse(ArrayRef<llvm::Value*> values);

  static Destroyer destroyARCStrongImprecise;
  static Destroyer destroyARCStrongPrecise;
  static Destroyer destroyARCWeak;

  void EmitObjCAutoreleasePoolPop(llvm::Value *Ptr); 
  llvm::Value *EmitObjCAutoreleasePoolPush();
  llvm::Value *EmitObjCMRRAutoreleasePoolPush();
  void EmitObjCAutoreleasePoolCleanup(llvm::Value *Ptr);
  void EmitObjCMRRAutoreleasePoolPop(llvm::Value *Ptr); 

  /// \brief Emits a reference binding to the passed in expression.
  RValue EmitReferenceBindingToExpr(const Expr *E);

  //===--------------------------------------------------------------------===//
  //                           Expression Emission
  //===--------------------------------------------------------------------===//

  // Expressions are broken into three classes: scalar, complex, aggregate.

  /// EmitScalarExpr - Emit the computation of the specified expression of LLVM
  /// scalar type, returning the result.
  llvm::Value *EmitScalarExpr(const Expr *E , bool IgnoreResultAssign = false);

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are LLVM scalar types.
  llvm::Value *EmitScalarConversion(llvm::Value *Src, QualType SrcTy,
                                    QualType DstTy, SourceLocation Loc);

  /// Emit a conversion from the specified complex type to the specified
  /// destination type, where the destination type is an LLVM scalar type.
  llvm::Value *EmitComplexToScalarConversion(ComplexPairTy Src, QualType SrcTy,
                                             QualType DstTy,
                                             SourceLocation Loc);

  /// EmitAggExpr - Emit the computation of the specified expression
  /// of aggregate type.  The result is computed into the given slot,
  /// which may be null to indicate that the value is not needed.
  void EmitAggExpr(const Expr *E, AggValueSlot AS);

  /// EmitAggExprToLValue - Emit the computation of the specified expression of
  /// aggregate type into a temporary LValue.
  LValue EmitAggExprToLValue(const Expr *E);

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
  ComplexPairTy EmitLoadOfComplex(LValue src, SourceLocation loc);

  Address emitAddrOfRealComponent(Address complex, QualType complexType);
  Address emitAddrOfImagComponent(Address complex, QualType complexType);

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

  llvm::Constant *createAtExitStub(const VarDecl &VD, llvm::Constant *Dtor,
                                   llvm::Constant *Addr);

  /// Call atexit() with a function that passes the given argument to
  /// the given function.
  void registerGlobalDtorWithAtExit(const VarDecl &D, llvm::Constant *fn,
                                    llvm::Constant *addr);

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
                                 ArrayRef<llvm::Function *> CXXThreadLocals,
                                 Address Guard = Address::invalid());

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
  
  void EmitSynthesizedCXXCopyCtor(Address Dest, Address Src, const Expr *Exp);

  void enterFullExpression(const ExprWithCleanups *E) {
    if (E->getNumObjects() == 0) return;
    enterNonTrivialFullExpression(E);
  }
  void enterNonTrivialFullExpression(const ExprWithCleanups *E);

  void EmitCXXThrowExpr(const CXXThrowExpr *E, bool KeepInsertionPoint = true);

  void EmitLambdaExpr(const LambdaExpr *E, AggValueSlot Dest);

  RValue EmitAtomicExpr(AtomicExpr *E);

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
  Address EmitFieldAnnotations(const FieldDecl *D, Address V);

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
  bool ConstantFoldsToSimpleInteger(const Expr *Cond, bool &Result,
                                    bool AllowLabels = false);

  /// ConstantFoldsToSimpleInteger - If the specified expression does not fold
  /// to a constant, or if it does but contains a label, return false.  If it
  /// constant folds return true and set the folded value.
  bool ConstantFoldsToSimpleInteger(const Expr *Cond, llvm::APSInt &Result,
                                    bool AllowLabels = false);

  /// EmitBranchOnBoolExpr - Emit a branch on a boolean condition (e.g. for an
  /// if statement) to the specified blocks.  Based on the condition, this might
  /// try to simplify the codegen of the conditional based on the branch.
  /// TrueCount should be the number of times we expect the condition to
  /// evaluate to true based on PGO data.
  void EmitBranchOnBoolExpr(const Expr *Cond, llvm::BasicBlock *TrueBlock,
                            llvm::BasicBlock *FalseBlock, uint64_t TrueCount);

  /// \brief Emit a description of a type in a format suitable for passing to
  /// a runtime sanitizer handler.
  llvm::Constant *EmitCheckTypeDescriptor(QualType T);

  /// \brief Convert a value into a format suitable for passing to a runtime
  /// sanitizer handler.
  llvm::Value *EmitCheckValue(llvm::Value *V);

  /// \brief Emit a description of a source location in a format suitable for
  /// passing to a runtime sanitizer handler.
  llvm::Constant *EmitCheckSourceLocation(SourceLocation Loc);

  /// \brief Create a basic block that will call a handler function in a
  /// sanitizer runtime with the provided arguments, and create a conditional
  /// branch to it.
  void EmitCheck(ArrayRef<std::pair<llvm::Value *, SanitizerMask>> Checked,
                 StringRef CheckName, ArrayRef<llvm::Constant *> StaticArgs,
                 ArrayRef<llvm::Value *> DynamicArgs);

  /// \brief Emit a slow path cross-DSO CFI check which calls __cfi_slowpath
  /// if Cond if false.
  void EmitCfiSlowPathCheck(SanitizerMask Kind, llvm::Value *Cond,
                            llvm::ConstantInt *TypeId, llvm::Value *Ptr,
                            ArrayRef<llvm::Constant *> StaticArgs);

  /// \brief Create a basic block that will call the trap intrinsic, and emit a
  /// conditional branch to it, for the -ftrapv checks.
  void EmitTrapCheck(llvm::Value *Checked);

  /// \brief Emit a call to trap or debugtrap and attach function attribute
  /// "trap-func-name" if specified.
  llvm::CallInst *EmitTrapCall(llvm::Intrinsic::ID IntrID);

  /// \brief Emit a cross-DSO CFI failure handling function.
  void EmitCfiCheckFail();

  /// \brief Create a check for a function parameter that may potentially be
  /// declared as non-null.
  void EmitNonNullArgCheck(RValue RV, QualType ArgType, SourceLocation ArgLoc,
                           const FunctionDecl *FD, unsigned ParmNum);

  /// EmitCallArg - Emit a single call argument.
  void EmitCallArg(CallArgList &args, const Expr *E, QualType ArgType);

  /// EmitDelegateCallArg - We are performing a delegate call; that
  /// is, the current function is delegating to another one.  Produce
  /// a r-value suitable for passing the given parameter.
  void EmitDelegateCallArg(CallArgList &args, const VarDecl *param,
                           SourceLocation loc);

  /// SetFPAccuracy - Set the minimum required accuracy of the given floating
  /// point operation, expressed as the maximum relative error in ulp.
  void SetFPAccuracy(llvm::Value *Val, float Accuracy);

private:
  llvm::MDNode *getRangeForLoadFromType(QualType Ty);
  void EmitReturnOfRValue(RValue RV, QualType Ty);

  void deferPlaceholderReplacement(llvm::Instruction *Old, llvm::Value *New);

  llvm::SmallVector<std::pair<llvm::Instruction *, llvm::Value *>, 4>
  DeferredReplacements;

  /// Set the address of a local variable.
  void setAddrOfLocalVar(const VarDecl *VD, Address Addr) {
    assert(!LocalDeclMap.count(VD) && "Decl already exists in LocalDeclMap!");
    LocalDeclMap.insert({VD, Addr});
  }

  /// ExpandTypeFromArgs - Reconstruct a structure of type \arg Ty
  /// from function arguments into \arg Dst. See ABIArgInfo::Expand.
  ///
  /// \param AI - The first function argument of the expansion.
  void ExpandTypeFromArgs(QualType Ty, LValue Dst,
                          SmallVectorImpl<llvm::Value *>::iterator &AI);

  /// ExpandTypeToArgs - Expand an RValue \arg RV, with the LLVM type for \arg
  /// Ty, into individual arguments on the provided vector \arg IRCallArgs,
  /// starting at index \arg IRCallArgPos. See ABIArgInfo::Expand.
  void ExpandTypeToArgs(QualType Ty, RValue RV, llvm::FunctionType *IRFuncTy,
                        SmallVectorImpl<llvm::Value *> &IRCallArgs,
                        unsigned &IRCallArgPos);

  llvm::Value* EmitAsmInput(const TargetInfo::ConstraintInfo &Info,
                            const Expr *InputExpr, std::string &ConstraintStr);

  llvm::Value* EmitAsmInputLValue(const TargetInfo::ConstraintInfo &Info,
                                  LValue InputValue, QualType InputType,
                                  std::string &ConstraintStr,
                                  SourceLocation Loc);

  /// \brief Attempts to statically evaluate the object size of E. If that
  /// fails, emits code to figure the size of E out for us. This is
  /// pass_object_size aware.
  llvm::Value *evaluateOrEmitBuiltinObjectSize(const Expr *E, unsigned Type,
                                               llvm::IntegerType *ResType);

  /// \brief Emits the size of E, as required by __builtin_object_size. This
  /// function is aware of pass_object_size parameters, and will act accordingly
  /// if E is a parameter with the pass_object_size attribute.
  llvm::Value *emitBuiltinObjectSize(const Expr *E, unsigned Type,
                                     llvm::IntegerType *ResType);

public:
#ifndef NDEBUG
  // Determine whether the given argument is an Objective-C method
  // that may have type parameters in its signature.
  static bool isObjCMethodWithTypeParams(const ObjCMethodDecl *method) {
    const DeclContext *dc = method->getDeclContext();
    if (const ObjCInterfaceDecl *classDecl= dyn_cast<ObjCInterfaceDecl>(dc)) {
      return classDecl->getTypeParamListAsWritten();
    }

    if (const ObjCCategoryDecl *catDecl = dyn_cast<ObjCCategoryDecl>(dc)) {
      return catDecl->getTypeParamList();
    }

    return false;
  }

  template<typename T>
  static bool isObjCMethodWithTypeParams(const T *) { return false; }
#endif

  /// EmitCallArgs - Emit call arguments for a function.
  template <typename T>
  void EmitCallArgs(CallArgList &Args, const T *CallArgTypeInfo,
                    llvm::iterator_range<CallExpr::const_arg_iterator> ArgRange,
                    const FunctionDecl *CalleeDecl = nullptr,
                    unsigned ParamsToSkip = 0) {
    SmallVector<QualType, 16> ArgTypes;
    CallExpr::const_arg_iterator Arg = ArgRange.begin();

    assert((ParamsToSkip == 0 || CallArgTypeInfo) &&
           "Can't skip parameters if type info is not provided");
    if (CallArgTypeInfo) {
#ifndef NDEBUG
      bool isGenericMethod = isObjCMethodWithTypeParams(CallArgTypeInfo);
#endif

      // First, use the argument types that the type info knows about
      for (auto I = CallArgTypeInfo->param_type_begin() + ParamsToSkip,
                E = CallArgTypeInfo->param_type_end();
           I != E; ++I, ++Arg) {
        assert(Arg != ArgRange.end() && "Running over edge of argument list!");
        assert((isGenericMethod ||
                ((*I)->isVariablyModifiedType() ||
                 (*I).getNonReferenceType()->isObjCRetainableType() ||
                 getContext()
                         .getCanonicalType((*I).getNonReferenceType())
                         .getTypePtr() ==
                     getContext()
                         .getCanonicalType((*Arg)->getType())
                         .getTypePtr())) &&
               "type mismatch in call argument!");
        ArgTypes.push_back(*I);
      }
    }

    // Either we've emitted all the call args, or we have a call to variadic
    // function.
    assert((Arg == ArgRange.end() || !CallArgTypeInfo ||
            CallArgTypeInfo->isVariadic()) &&
           "Extra arguments in non-variadic function!");

    // If we still have any arguments, emit them using the type of the argument.
    for (auto *A : llvm::make_range(Arg, ArgRange.end()))
      ArgTypes.push_back(getVarArgType(A));

    EmitCallArgs(Args, ArgTypes, ArgRange, CalleeDecl, ParamsToSkip);
  }

  void EmitCallArgs(CallArgList &Args, ArrayRef<QualType> ArgTypes,
                    llvm::iterator_range<CallExpr::const_arg_iterator> ArgRange,
                    const FunctionDecl *CalleeDecl = nullptr,
                    unsigned ParamsToSkip = 0);

  /// EmitPointerWithAlignment - Given an expression with a pointer
  /// type, emit the value and compute our best estimate of the
  /// alignment of the pointee.
  ///
  /// Note that this function will conservatively fall back on the type
  /// when it doesn't 
  ///
  /// \param Source - If non-null, this will be initialized with
  ///   information about the source of the alignment.  Note that this
  ///   function will conservatively fall back on the type when it
  ///   doesn't recognize the expression, which means that sometimes
  ///   
  ///   a worst-case One
  ///   reasonable way to use this information is when there's a
  ///   language guarantee that the pointer must be aligned to some
  ///   stricter value, and we're simply trying to ensure that
  ///   sufficiently obvious uses of under-aligned objects don't get
  ///   miscompiled; for example, a placement new into the address of
  ///   a local variable.  In such a case, it's quite reasonable to
  ///   just ignore the returned alignment when it isn't from an
  ///   explicit source.
  Address EmitPointerWithAlignment(const Expr *Addr,
                                   AlignmentSource *Source = nullptr);

  void EmitSanitizerStatReport(llvm::SanitizerStatKind SSK);

private:
  QualType getVarArgType(const Expr *Arg);

  const TargetCodeGenInfo &getTargetHooks() const {
    return CGM.getTargetCodeGenInfo();
  }

  void EmitDeclMetadata();

  BlockByrefHelpers *buildByrefHelpers(llvm::StructType &byrefType,
                                  const AutoVarEmission &emission);

  void AddObjCARCExceptionMetadata(llvm::Instruction *Inst);

  llvm::Value *GetValueForARMHint(unsigned BuiltinID);
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

    // Otherwise, we need an alloca.
    auto align = CharUnits::fromQuantity(
              CGF.CGM.getDataLayout().getPrefTypeAlignment(value->getType()));
    Address alloca =
      CGF.CreateTempAlloca(value->getType(), align, "cond-cleanup.save");
    CGF.Builder.CreateStore(value, alloca);

    return saved_type(alloca.getPointer(), true);
  }

  static llvm::Value *restore(CodeGenFunction &CGF, saved_type value) {
    // If the value says it wasn't saved, trust that it's still dominating.
    if (!value.getInt()) return value.getPointer();

    // Otherwise, it should be an alloca instruction, as set up in save().
    auto alloca = cast<llvm::AllocaInst>(value.getPointer());
    return CGF.Builder.CreateAlignedLoad(alloca, alloca->getAlignment());
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

/// A specialization of DominatingValue for Address.
template <> struct DominatingValue<Address> {
  typedef Address type;

  struct saved_type {
    DominatingLLVMValue::saved_type SavedValue;
    CharUnits Alignment;
  };

  static bool needsSaving(type value) {
    return DominatingLLVMValue::needsSaving(value.getPointer());
  }
  static saved_type save(CodeGenFunction &CGF, type value) {
    return { DominatingLLVMValue::save(CGF, value.getPointer()),
             value.getAlignment() };
  }
  static type restore(CodeGenFunction &CGF, saved_type value) {
    return Address(DominatingLLVMValue::restore(CGF, value.SavedValue),
                   value.Alignment);
  }
};

/// A specialization of DominatingValue for RValue.
template <> struct DominatingValue<RValue> {
  typedef RValue type;
  class saved_type {
    enum Kind { ScalarLiteral, ScalarAddress, AggregateLiteral,
                AggregateAddress, ComplexAddress };

    llvm::Value *Value;
    unsigned K : 3;
    unsigned Align : 29;
    saved_type(llvm::Value *v, Kind k, unsigned a = 0)
      : Value(v), K(k), Align(a) {}

  public:
    static bool needsSaving(RValue value);
    static saved_type save(CodeGenFunction &CGF, RValue value);
    RValue restore(CodeGenFunction &CGF);

    // implementations in CGCleanup.cpp
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
