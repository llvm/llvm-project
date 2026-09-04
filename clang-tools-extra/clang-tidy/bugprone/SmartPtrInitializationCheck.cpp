//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SmartPtrInitializationCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

const auto DefaultSharedPointers = "::std::shared_ptr;::boost::shared_ptr";
const auto DefaultUniquePointers = "::std::unique_ptr";
const auto DefaultDefaultDeleters = "::std::default_delete";

} // namespace

// Remove wrappers that do not carry semantic load for classifying the value:
// brackets, implicit casts, temporary objects, cleanup nodes.
static const clang::Expr *stripWrappers(const clang::Expr *E) {
  while (E) {
    const clang::Expr *Prev = E;
    E = E->IgnoreParens();
    switch (E->getStmtClass()) {
    case clang::Stmt::ImplicitCastExprClass:
      E = cast<clang::ImplicitCastExpr>(E)->getSubExpr();
      break;
    case clang::Stmt::ExprWithCleanupsClass:
      E = cast<clang::ExprWithCleanups>(E)->getSubExpr();
      break;
    case clang::Stmt::MaterializeTemporaryExprClass:
      E = cast<clang::MaterializeTemporaryExpr>(E)->getSubExpr();
      break;
    case clang::Stmt::CXXBindTemporaryExprClass:
      E = cast<clang::CXXBindTemporaryExpr>(E)->getSubExpr();
      break;
    case clang::Stmt::ConstantExprClass:
      E = cast<clang::ConstantExpr>(E)->getSubExpr();
      break;
    default:
      break;
    }
    if (E == Prev)
      break;
  }
  return E;
}

namespace {

struct PointerLocation {
  // Root == nullptr && IsThis == true  => via `this`
  // Root != nullptr                    => regular variable/parameter
  const clang::VarDecl *Root = nullptr;
  std::vector<const clang::FieldDecl *>
      Path; // [] => the variable Root itself (для Root!=nullptr);
            // [f] => Root.f / Root->f / this->f;
            // [f1,f2] => Root.f1.f2 and etc..
  bool IsThis = false;

  bool operator==(const PointerLocation &O) const {
    return Root == O.Root && IsThis == O.IsThis && Path == O.Path;
  }
  bool operator<(const PointerLocation &O) const {
    if (Root != O.Root)
      return Root < O.Root;
    if (IsThis != O.IsThis)
      return IsThis < O.IsThis;
    return Path < O.Path; // lexicographically by pointers - sufficient for
                          // strict ordering in std::map
  }
};

enum PointerState : unsigned {
  PS_Unknown = 0,
  PS_PlainPointer = 1,
  PS_NewPointer = 2,
  PS_SmartPtrWrapper = 3
};

struct Transition {
  PointerState fromState;  // 0-3
  PointerState toState;    // 1-3
  const clang::Stmt *stmt; // instruction that caused the transition
};

class TransitionsFinder;

// The main visitor applies the "transfer function" of one block to the
// transferred state. It is created anew for each call to runBlockTransfer() and
// does not store any inter-block state itself.
class PointerStateVisitor
    : public clang::ConstStmtVisitor<PointerStateVisitor> {
  using StateMap = std::map<PointerLocation, PointerState>;
  using VariablesSet = llvm::SmallPtrSet<const clang::VarDecl *, 32>;
  using FieldsSet = llvm::SmallPtrSet<const clang::FieldDecl *, 32>;

  PointerState getState(const StateMap &M, const PointerLocation &Loc) {
    auto It = M.find(Loc);
    return It == M.end() ? PS_Unknown : It->second;
  }

public:
  // Sink == nullptr -> transitions are not recorded, only the final state is
  // calculated (used in phase 1 / fixpoint, and in the preliminary location
  // detection phase).
  PointerStateVisitor(clang::ASTContext *Context, const VariablesSet &Vars,
                      const FieldsSet &Fields,
                      llvm::ArrayRef<StringRef> SharedPointers,
                      llvm::ArrayRef<StringRef> UniquePointers, StateMap &State,
                      std::map<PointerLocation, std::vector<Transition>> *Sink)
      : Context(Context), PtrVars(Vars), PtrFields(Fields),
        SharedPointers(SharedPointers), UniquePointers(UniquePointers),
        CurrentState(State), Sink(Sink) {}

  // int* a = nullptr; / int* a = new int; / int* b = a; ...
  void VisitDeclStmt(const clang::DeclStmt *DS) {
    for (const clang::Decl *D : DS->decls()) {
      const auto *VD = llvm::dyn_cast<clang::VarDecl>(D);
      if (!VD || !PtrVars.count(VD))
        continue;
      if (const clang::Expr *Init = VD->getInit()) {
        PointerState NewState = classify(Init);
        addTransition(PointerLocation{VD, {}}, NewState, DS);
      }
    }
    // std::shared_ptr<int> sp(a); / std::shared_ptr<int> sp(a.val); ->
    // "a" / "a.val" also gets infected SmartPtrWrapper
    scanForSmartPtrWrap(DS);
  }

  // a = ...; / a.val = ...; (etc = new int; / = b; / = nullptr;)
  void VisitBinaryOperator(const clang::BinaryOperator *BO) {
    if (BO->getOpcode() != clang::BO_Assign) {
      scanForSmartPtrWrap(BO);
      return;
    }
    PointerLocation Loc;
    if (resolveLocation(BO->getLHS(), Loc)) {
      PointerState NewState = classify(BO->getRHS());
      addTransition(std::move(Loc), NewState, BO);
    }
    scanForSmartPtrWrap(BO);
  }

  // Direct meeting of the smartpointer constructor as a separate element of CFG
  void VisitCXXConstructExpr(const clang::CXXConstructExpr *CE) {
    handleSmartPtrConstruct(CE, CE);
  }

  // Any other instruction: simply search for "infection" inside it by passing a
  // pointer/field to the smartpointer constructor.
  void VisitStmt(const clang::Stmt *S) { scanForSmartPtrWrap(S); }

private:
  clang::ASTContext *Context;
  const VariablesSet &PtrVars;
  const FieldsSet &PtrFields;

  const llvm::ArrayRef<StringRef> SharedPointers;
  const llvm::ArrayRef<StringRef> UniquePointers;

  StateMap &CurrentState; // state of a specific block (outside: IN -> resulting
                          // in OUT)
  std::map<PointerLocation, std::vector<Transition>> *Sink;
  llvm::SmallPtrSet<const clang::Stmt *, 32>
      ProcessedConstructsAndResets; // dedup within ONE block processing call

  // Structural (without PtrVars/PtrFields checking) "path" recognition:
  // DeclRefExpr(var) -> {var, []}; MemberExpr(base, field) -> resolveBase(base)
  // + [field].
  // Used both for the MemberExpr base (where the base itself is a structure,
  // not a pointer, and does NOT have to be in PtrVars) and within
  // resolveLocation. Stops (returns nullopt) on anything that is not
  // DeclRefExpr or MemberExpr - array indexing, function calls, this, etc. (see
  // the file header for restrictions).
  bool resolveBase(const clang::Expr *E, PointerLocation &Out) {
    E = stripWrappers(E);
    if (!E)
      return false;

    if (const auto *DRE = llvm::dyn_cast<clang::DeclRefExpr>(E)) {
      const auto *VD = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl());
      if (!VD)
        return false;
      Out.Root = VD;
      Out.IsThis = false;
      Out.Path.clear();
      return true;
    }

    // this->val / val (imoplicit `this`) inside method of class.
    if (llvm::isa<clang::CXXThisExpr>(E)) {
      Out.Root = nullptr;
      Out.IsThis = true;
      Out.Path.clear();
      return true;
    }

    if (const auto *ME = llvm::dyn_cast<clang::MemberExpr>(E)) {
      const auto *FD = llvm::dyn_cast<clang::FieldDecl>(ME->getMemberDecl());
      if (!FD)
        return false; // a method, static field, etc. is not a path to data
      if (!resolveBase(ME->getBase(), Out))
        return false;
      Out.Path.push_back(FD);
      return true;
    }

    return false; // arr[i].val, f().val и т.п. - not supported
  }

  // Checking that a type (after expansion of typedef/using) is a specialization
  // std::shared_ptr
  bool isStdSharedPtrType(clang::QualType QT) {
    QT = QT.getCanonicalType();
    const clang::CXXRecordDecl *RD = QT->getAsCXXRecordDecl();
    if (!RD)
      return false;
    if (!RD->getDeclName().isIdentifier())
      return false;

    auto SharedPtrMatcher = cxxRecordDecl(hasAnyName(SharedPointers));

    return !match(SharedPtrMatcher, *RD, *Context).empty();
  }

  // Checking that a type (after expansion of typedef/using) is a specialization
  // std::unique_ptr
  bool isStdUniquePtrType(clang::QualType QT) {
    QT = QT.getCanonicalType();
    const clang::CXXRecordDecl *RD = QT->getAsCXXRecordDecl();
    if (!RD)
      return false;
    if (!RD->getDeclName().isIdentifier())
      return false;

    auto UniquePtrMatcher = cxxRecordDecl(hasAnyName(UniquePointers));

    return !match(UniquePtrMatcher, *RD, *Context).empty();
  }

  bool isSmartPtrType(clang::QualType QT) {
    return isStdSharedPtrType(QT) || isStdUniquePtrType(QT);
  }

  // Attempts to recognize E as a tracked location:
  // - DeclRefExpr(var), where var is in PtrVars (a regular pointer variable)
  // - MemberExpr(base, field), where field is in PtrFields (a.val / a->val),
  // and base is recognized STRUCTURALLY (without checking PtrVars for base -
  // since base is usually NOT a pointer, but a structure/object).
  bool resolveLocation(const clang::Expr *E, PointerLocation &Out) {
    const clang::Expr *S = stripWrappers(E);
    if (!S)
      return false;

    if (const auto *DRE = llvm::dyn_cast<clang::DeclRefExpr>(S)) {
      const auto *VD = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl());
      if (VD && PtrVars.count(VD)) {
        Out.Root = VD;
        Out.IsThis = false;
        Out.Path.clear();
        return true;
      }
      return false;
    }

    if (const auto *ME = llvm::dyn_cast<clang::MemberExpr>(S)) {
      const auto *FD = llvm::dyn_cast<clang::FieldDecl>(ME->getMemberDecl());
      if (!FD || !PtrFields.count(FD))
        return false;
      if (!resolveBase(ME->getBase(), Out))
        return false;
      Out.Path.push_back(FD);
      return true;
    }

    return false;
  }

  // Defines the state that a pointer goes into when it is assigned/initialized
  // with the value of expression E.
  PointerState classify(const clang::Expr *E) {
    const clang::Expr *S = stripWrappers(E);
    if (!S)
      return PS_PlainPointer;

    // a = new int;
    if (llvm::isa<clang::CXXNewExpr>(S))
      return PS_NewPointer;

    // a = nullptr; / a = NULL;
    if (llvm::isa<clang::CXXNullPtrLiteralExpr>(S) ||
        llvm::isa<clang::GNUNullExpr>(S))
      return PS_PlainPointer;

    // a = 0;
    if (const auto *IL = llvm::dyn_cast<clang::IntegerLiteral>(S)) {
      if (IL->getValue() == 0)
        return PS_PlainPointer;
    }

    // rare case: the pointer is directly assigned the result of constructing a
    // smart pointer
    if (const auto *CE = llvm::dyn_cast<clang::CXXConstructExpr>(S)) {
      if (isSmartPtrType(CE->getType()))
        return PS_SmartPtrWrapper;
    }

    // b = a;  /  b = a.val;  -> infection with the CURRENT (for this block)
    // state of the source
    PointerLocation SrcLoc;
    if (resolveLocation(S, SrcLoc))
      return getState(CurrentState, SrcLoc);

    // An unknown expression of a pointer type (function call, type cast, etc.)
    // is considered an ordinary pointer.
    if (S->getType()->isPointerType())
      return PS_PlainPointer;

    return PS_Unknown;
  }

  void addTransition(const PointerLocation &&Loc, PointerState NewState,
                     const clang::Stmt *S) {
    PointerState From = getState(CurrentState, Loc);
    if (Sink)
      (*Sink)[Loc].push_back(Transition{From, NewState, S});
    CurrentState[Loc] = NewState;
  }

  // Searches for a CXXConstructExpr of type shared_ptr or unique_ptr within the
  // subtree of S and marks as SmartPtrWrapper any tracked locations passed as
  // constructor arguments.
  //
  // The traversal is iterative (explicit stack on the heap), not recursive -
  // recursion overflows the call stack on deeply nested expressions (confirmed
  // by AddressSanitizer: stack-overflow).
  void scanForSmartPtrWrap(const clang::Stmt *Root) {
    if (!Root)
      return;

    std::vector<const clang::Stmt *> Worklist;
    Worklist.push_back(Root);

    while (!Worklist.empty()) {
      const clang::Stmt *S = Worklist.back();
      Worklist.pop_back();
      if (!S)
        continue;

      if (const auto *CE = llvm::dyn_cast<clang::CXXConstructExpr>(S))
        handleSmartPtrConstruct(CE, S);

      if (const auto *ME = llvm::dyn_cast<clang::CXXMemberCallExpr>(S))
        handleSmartPtrReset(ME, S);

      for (const clang::Stmt *Child : S->children())
        Worklist.push_back(Child);
    }
  }

  void handleSmartPtrConstruct(const clang::CXXConstructExpr *CE,
                               const clang::Stmt *EnclosingStmt) {
    if (!isSmartPtrType(CE->getType()))
      return;
    if (!ProcessedConstructsAndResets.insert(CE).second)
      return; // has already been processed within this call

    for (const clang::Expr *Arg : CE->arguments()) {
      PointerLocation Loc;
      if (resolveLocation(Arg, Loc))
        addTransition(std::move(Loc), PS_SmartPtrWrapper, EnclosingStmt);
    }
  }

  void handleSmartPtrReset(const clang::CXXMemberCallExpr *ME,
                           const clang::Stmt *EnclosingStmt) {
    assert(ME);
    if (!ME->getMethodDecl())
      return;
    if (!ME->getMethodDecl()->getDeclName().isIdentifier() ||
        ME->getMethodDecl()->getName() != "reset")
      return;
    if (!isSmartPtrType(ME->getImplicitObjectArgument()->getType()))
      return;
    if (!ProcessedConstructsAndResets.insert(ME).second)
      return; // has already been processed within this call

    for (const clang::Expr *Arg : ME->arguments()) {
      PointerLocation Loc;
      if (resolveLocation(Arg, Loc))
        addTransition(std::move(Loc), PS_SmartPtrWrapper, EnclosingStmt);
    }
  }
};

class TransitionsFinder {
  ASTContext *Context;
  const llvm::ArrayRef<StringRef> SharedPointers;
  const llvm::ArrayRef<StringRef> UniquePointers;

  using StateMap = std::map<PointerLocation, PointerState>;
  using VariablesSet = llvm::SmallPtrSet<const clang::VarDecl *, 32>;
  using FieldsSet = llvm::SmallPtrSet<const clang::FieldDecl *, 32>;

  PointerState getState(const StateMap &M, const PointerLocation &Loc) {
    auto It = M.find(Loc);
    return It == M.end() ? PS_Unknown : It->second;
  }

public:
  TransitionsFinder(ASTContext *TheContext,
                    const llvm::ArrayRef<StringRef> &SharedPointers,
                    const llvm::ArrayRef<StringRef> &UniquePointers)
      : Context(TheContext), SharedPointers(SharedPointers),
        UniquePointers(UniquePointers) {}

  std::map<PointerLocation, std::vector<Transition>>
  find(const VariablesSet &PtrVars, const FieldsSet &PtrFields,
       const FunctionDecl *Func, Stmt *Body) {
    // Settings to build CFG
    CFG::BuildOptions Options;
    Options.AddImplicitDtors = true;
    Options.AddTemporaryDtors = true;
    Options.AddInitializers = true;

    std::unique_ptr<CFG> TheCFG = CFG::buildCFG(Func, Body, Context, Options);
    if (!TheCFG)
      return {};

    return findInternally(PtrVars, PtrFields, *TheCFG);
  }

private:
  // Reverse post-order traversal of CFG blocks (iterative DFS without recursion
  // on the interpreter stack, to avoid dependence on the CFG depth).
  std::vector<const clang::CFGBlock *>
  computeReversePostOrder(const clang::CFG &Cfg) {
    std::vector<const clang::CFGBlock *> PostOrder;
    llvm::SmallPtrSet<const clang::CFGBlock *, 32> Visited;

    struct Frame {
      const clang::CFGBlock *Block;
      clang::CFGBlock::const_succ_iterator It;
      clang::CFGBlock::const_succ_iterator End;
    };

    const clang::CFGBlock *Entry = &Cfg.getEntry();
    if (!Entry)
      return PostOrder;

    std::vector<Frame> Stack;
    Visited.insert(Entry);
    Stack.push_back({Entry, Entry->succ_begin(), Entry->succ_end()});

    while (!Stack.empty()) {
      Frame &F = Stack.back();
      if (F.It == F.End) {
        PostOrder.push_back(F.Block);
        Stack.pop_back();
        continue;
      }
      const clang::CFGBlock *Succ = *F.It;
      ++F.It;
      if (!Succ || Visited.count(Succ))
        continue;
      Visited.insert(Succ);
      Stack.push_back({Succ, Succ->succ_begin(), Succ->succ_end()});
    }

    std::reverse(PostOrder.begin(), PostOrder.end());
    return PostOrder;
  }
  // The analysis domain is NOT fully known in advance: locations like a.val are
  // discovered only when traversing the function body. Therefore, before the
  // fixpoint phase, we perform a light preliminary pass through all blocks
  // INDEPENDENTLY (without propagating state between blocks—it's not needed
  // here), collecting the set of all PointerLocation s that are ever the target
  // of an assignment/initialization/argument to a smart pointer wrapper.
  std::set<PointerLocation>
  discoverLocations(const std::vector<const clang::CFGBlock *> &Order,
                    const VariablesSet &PtrVars, const FieldsSet &PtrFields) {
    std::set<PointerLocation> Domain;
    for (const clang::VarDecl *VD : PtrVars)
      Domain.insert(PointerLocation{VD, {}});

    for (const clang::CFGBlock *Block : Order) {
      if (!Block)
        continue;
      StateMap Scratch; // disposable, empty at the entrance of each block
      runBlockTransfer(*Block, PtrVars, PtrFields, Scratch, /*Sink=*/nullptr);
      for (const auto &KV : Scratch)
        Domain.insert(KV.first);
    }
    return Domain;
  }
  // Runs the "transfer function" of a single block: the input is the IN state,
  // the output is the final (OUT) state. If Sink != nullptr, it also records
  // the transitions.
  StateMap
  runBlockTransfer(const clang::CFGBlock &Block, const VariablesSet &PtrVars,
                   const FieldsSet &PtrFields, const StateMap &In,
                   std::map<PointerLocation, std::vector<Transition>> *Sink) {
    StateMap Working = In;
    PointerStateVisitor Visitor(Context, PtrVars, PtrFields, SharedPointers,
                                UniquePointers, Working, Sink);
    for (const clang::CFGElement &Elem : Block) {
      if (auto CS = Elem.getAs<clang::CFGStmt>()) {
        if (const clang::Stmt *S = CS->getStmt())
          Visitor.Visit(S);
      }
    }
    return Working;
  }
  std::map<PointerLocation, std::vector<Transition>>
  findInternally(const VariablesSet &PtrVars, const FieldsSet &PtrFields,
                 const clang::CFG &cfg) {
    std::map<PointerLocation, std::vector<Transition>> Result;

    std::vector<const clang::CFGBlock *> Order = computeReversePostOrder(cfg);
    if (Order.empty()) {
      for (const clang::VarDecl *VD : PtrVars)
        Result[PointerLocation{VD,
                               {}}]; // guarantee a key even without CFG blocks
      return Result;
    }

    std::set<PointerLocation> Domain =
        discoverLocations(Order, PtrVars, PtrFields);
    for (const PointerLocation &Loc : Domain)
      Result[Loc]; // guarantee the presence of a key even without transitions

    auto initialState = [&]() {
      StateMap S;
      for (const PointerLocation &Loc : Domain)
        S[Loc] = PS_Unknown;
      return S;
    };
    auto joinStates = [&](const StateMap &A, const StateMap &B) {
      StateMap R;
      for (const PointerLocation &Loc : Domain) {
        PointerState a = getState(A, Loc);
        PointerState b = getState(B, Loc);
        R[Loc] = (a == b) ? a : PS_Unknown;
      }
      return R;
    };
    auto statesEqual = [&](const StateMap &A, const StateMap &B) {
      for (const PointerLocation &Loc : Domain)
        if (getState(A, Loc) != getState(B, Loc))
          return false;
      return true;
    };

    std::map<const clang::CFGBlock *, StateMap> InState;
    std::map<const clang::CFGBlock *, StateMap> OutState;

    // ---- Phase 1: Fixpoint on IN/OUT states, without recording transitions
    const int MaxIters = static_cast<int>(Order.size()) * 4 + 16;
    bool Changed = true;
    int Iter = 0;
    while (Changed && Iter < MaxIters) {
      Changed = false;
      ++Iter;

      for (const clang::CFGBlock *Block : Order) {
        StateMap NewIn;
        bool HaveAny = false;
        for (auto PredIt = Block->pred_begin(); PredIt != Block->pred_end();
             ++PredIt) {
          const clang::CFGBlock *Pred = *PredIt;
          if (!Pred)
            continue; // unreachable edge (AdjacentBlock == nullptr)
          auto It = OutState.find(Pred);
          if (It == OutState.end())
            continue; // the predecessor has not yet been processed in this pass
          if (!HaveAny) {
            NewIn = It->second;
            HaveAny = true;
          } else {
            NewIn = joinStates(NewIn, It->second);
          }
        }
        if (!HaveAny)
          NewIn = initialState();

        auto InIt = InState.find(Block);
        bool InChanged =
            (InIt == InState.end()) || !statesEqual(InIt->second, NewIn);
        if (InChanged) {
          InState[Block] = NewIn;
          Changed = true;
        }

        StateMap NewOut = runBlockTransfer(*Block, PtrVars, PtrFields,
                                           InState[Block], /*Sink=*/nullptr);

        auto OutIt = OutState.find(Block);
        bool OutChanged =
            (OutIt == OutState.end()) || !statesEqual(OutIt->second, NewOut);
        if (OutChanged) {
          OutState[Block] = NewOut;
          Changed = true;
        }
      }
    }

    // ---- Phase 2: Emission - each block exactly once, with a final IN
    for (const clang::CFGBlock *Block : Order) {
      StateMap In = InState.count(Block) ? InState[Block] : initialState();
      runBlockTransfer(*Block, PtrVars, PtrFields, In, &Result);
    }

    return Result;
  }
};

} // namespace

class SmartPtrInitializationCheckImpl {
public:
  virtual ~SmartPtrInitializationCheckImpl() = default;
  virtual void registerMatchers(ast_matchers::MatchFinder *Finder) = 0;
  virtual void check(const ast_matchers::MatchFinder::MatchResult &Result) = 0;
  virtual bool isStrictMode() = 0;
};

class SmartPtrInitializationCheckPermissiveMode
    : public SmartPtrInitializationCheckImpl {
  SmartPtrInitializationCheck &Check;

public:
  explicit SmartPtrInitializationCheckPermissiveMode(
      SmartPtrInitializationCheck &Check)
      : Check(Check) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override {
    const auto IsSharedPtr = hasAnyName(Check.SharedPointers);
    const auto IsUniquePtr = hasAnyName(Check.UniquePointers);
    const auto IsSmartPtr = anyOf(IsSharedPtr, IsUniquePtr);

    const auto IsSharedPtrRecord = cxxRecordDecl(IsSharedPtr);
    const auto IsUniquePtrRecord = cxxRecordDecl(IsUniquePtr);
    const auto IsSmartPtrRecord = cxxRecordDecl(IsSmartPtr);

    auto ResetCallMatcher = cxxMemberCallExpr(
        on(hasType(hasUnqualifiedDesugaredType(recordType(
            hasDeclaration(classTemplateSpecializationDecl(IsSmartPtr)))))),
        callee(cxxMethodDecl(ofClass(IsSmartPtrRecord), hasName("reset"))));
    auto SmartPtrGetCallMatcher = cxxMemberCallExpr(
        callee(cxxMethodDecl(hasName("get"))),
        on(hasType(hasUnqualifiedDesugaredType(recordType(
            hasDeclaration(classTemplateSpecializationDecl(IsSmartPtr)))))));

    // Search for `std::shared_ptr(this);` or `std::shared_ptr(other_sp.get());`
    auto SmartPtrConstructorMatcher =
        cxxConstructExpr(
            hasDeclaration(cxxConstructorDecl(ofClass(IsSmartPtrRecord))),
            hasArgument(0, anyOf(ignoringParenCasts(cxxThisExpr()),
                                 ignoringParenCasts(SmartPtrGetCallMatcher))))
            .bind("dangerous-ctor");

    // Search for `sp.reset(this);` or `sp.reset(other_sp.get())`
    auto ResetCallWithThisMatcher =
        cxxMemberCallExpr(
            on(hasType(hasUnqualifiedDesugaredType(recordType(
                hasDeclaration(classTemplateSpecializationDecl(IsSmartPtr)))))),
            callee(cxxMethodDecl(ofClass(IsSmartPtrRecord), hasName("reset"))),
            hasArgument(0, anyOf(ignoringParenCasts(cxxThisExpr()),
                                 ignoringParenCasts(SmartPtrGetCallMatcher))))
            .bind("dangerous-reset");

    // Search a functions to perform data flow sensitive analysis.
    const auto PotentiallyDangerousFunction =
        functionDecl(
            hasAnyBody(anything()),
            anyOf(hasDescendant(cxxNewExpr()), hasDescendant(ResetCallMatcher),
                  hasDescendant(cxxConstructExpr(hasDeclaration(
                      cxxConstructorDecl(ofClass(IsSmartPtrRecord)))))))
            .bind("func");

    Finder->addMatcher(PotentiallyDangerousFunction, &Check);
    Finder->addMatcher(SmartPtrConstructorMatcher, &Check);
    Finder->addMatcher(ResetCallWithThisMatcher, &Check);
  }

  void check(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *CtorWithThisExpr =
        Result.Nodes.getNodeAs<CXXConstructExpr>("dangerous-ctor");
    const auto *ResetWithThisExpr =
        Result.Nodes.getNodeAs<CXXMemberCallExpr>("dangerous-reset");
    if (CtorWithThisExpr)
      Check.emitDiagnostic(*Result.Context, CtorWithThisExpr);
    else if (ResetWithThisExpr)
      Check.emitDiagnostic(*Result.Context, ResetWithThisExpr);
    else
      checkFlowSensitive(Result);
  }

  bool isStrictMode() override { return false; }

private:
  void
  checkFlowSensitive(const ast_matchers::MatchFinder::MatchResult &Result) {
    const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
    if (!Func || !Func->hasBody())
      return;

    Stmt *Body = Func->getBody();
    if (!Body)
      return;

    // Collect all pointer variables inside the functions
    llvm::SmallPtrSet<const VarDecl *, 32> PtrVars;
    llvm::SmallPtrSet<const FieldDecl *, 32> PtrFields;

    for (const ParmVarDecl *PVD : Func->parameters())
      if (PVD->getType()->isPointerType())
        PtrVars.insert(PVD);

    std::function<void(const Stmt *)> CollectPtrVars = [&](const Stmt *S) {
      if (!S)
        return;

      if (const auto *DS = dyn_cast<DeclStmt>(S)) {
        for (const auto *D : DS->decls()) {
          if (const auto *VD = dyn_cast<VarDecl>(D)) {
            if (VD->getType()->isPointerType())
              PtrVars.insert(VD);
          }
        }
      } else if (const auto *MS = dyn_cast<MemberExpr>(S)) {
        if (const auto *MD = MS->getMemberDecl()) {
          if (const auto *FD = dyn_cast<FieldDecl>(MD)) {
            if (FD->getType()->isPointerType())
              PtrFields.insert(FD);
          }
        }
      }

      for (const auto *Child : S->children())
        CollectPtrVars(Child);
    };

    CollectPtrVars(Body);

    TransitionsFinder Finder(Result.Context, Check.SharedPointers,
                             Check.UniquePointers);
    const auto transitions = Finder.find(PtrVars, PtrFields, Func, Body);
    for (const auto &[var, transList] : transitions) {
      for (const auto &t : transList) {
        if (t.fromState == t.toState && t.fromState == PS_SmartPtrWrapper) {
          if (const auto *E = dyn_cast<const Expr>(t.stmt))
            Check.emitDiagnostic(*Result.Context, E);
        }
      }
    }
  }
};

class SmartPtrInitializationCheckStrictMode
    : public SmartPtrInitializationCheckImpl {
  SmartPtrInitializationCheck &Check;

public:
  explicit SmartPtrInitializationCheckStrictMode(
      SmartPtrInitializationCheck &Check)
      : Check(Check) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override {
    const auto IsSharedPtr = hasAnyName(Check.SharedPointers);
    const auto IsUniquePtr = hasAnyName(Check.UniquePointers);
    const auto IsSmartPtr = anyOf(IsSharedPtr, IsUniquePtr);
    const auto IsDefaultDeleter = hasAnyName(Check.DefaultDeleters);

    const auto IsSharedPtrRecord = cxxRecordDecl(IsSharedPtr);
    const auto IsUniquePtrRecord = cxxRecordDecl(IsUniquePtr);
    const auto IsSmartPtrRecord = cxxRecordDecl(IsSmartPtr);

    auto ReleaseCallMatcher =
        cxxMemberCallExpr(callee(cxxMethodDecl(hasName("release"))));

    // Array automatically decays to pointer
    auto PointerArg = expr(anyOf(hasType(pointerType()), hasType(arrayType())));

    // Matcher for unique_ptr types with custom deleters
    auto UniquePtrWithCustomDeleter = classTemplateSpecializationDecl(
        IsUniquePtr, templateArgumentCountIs(2),
        hasTemplateArgument(
            1, refersToType(
                   unless(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                       classTemplateSpecializationDecl(IsDefaultDeleter))))))));

    // Matcher for shared_ptr with custom deleter in constructor
    // Check if the second argument is NOT std::default_delete
    auto SharedPtrWithCustomDeleter = allOf(
        hasDeclaration(cxxConstructorDecl(ofClass(IsSharedPtrRecord))),
        hasArgument(
            1, ignoringParenCasts(unless(hasType(hasUnqualifiedDesugaredType(
                   recordType(hasDeclaration(classTemplateSpecializationDecl(
                       IsDefaultDeleter)))))))));

    // Matcher for smart pointer constructors
    auto HasCustomDeleter = anyOf(
        SharedPtrWithCustomDeleter,
        allOf(hasType(hasUnqualifiedDesugaredType(
                  recordType(hasDeclaration(UniquePtrWithCustomDeleter)))),
              hasDeclaration(cxxConstructorDecl(ofClass(IsUniquePtrRecord)))));

    auto AllowedArguments = anyOf(ignoringParenCasts(cxxNewExpr()),
                                  ignoringParenCasts(ReleaseCallMatcher));

    auto OptionalCondOp =
        optionally(ignoringParenCasts(conditionalOperator().bind("cond-op")));

    auto SmartPtrConstructorMatcher =
        cxxConstructExpr(
            hasDeclaration(cxxConstructorDecl(ofClass(IsSmartPtrRecord))),
            hasArgument(0, PointerArg), unless(HasCustomDeleter),
            unless(hasArgument(0, AllowedArguments)),
            hasArgument(0, OptionalCondOp))
            .bind("ctor");

    // For reset() - we need to check the type of the smart pointer
    // If it's shared_ptr with custom deleter (2+ args in constructor)
    // or unique_ptr with custom deleter type
    auto SmartPtrWithCustomDeleterType = anyOf(
        // shared_ptr with custom deleter - check if the type has a second
        // template argument that is NOT std::default_delete
        classTemplateSpecializationDecl(
            IsSharedPtr, templateArgumentCountIs(2),
            hasTemplateArgument(
                1, refersToType(unless(hasUnqualifiedDesugaredType(recordType(
                       hasDeclaration(classTemplateSpecializationDecl(
                           IsDefaultDeleter)))))))),
        UniquePtrWithCustomDeleter);

    auto HasCustomDeleterInReset =
        anyOf(on(hasType(hasUnqualifiedDesugaredType(
                  recordType(hasDeclaration(SmartPtrWithCustomDeleterType))))),
              // Also check if reset call has 2 arguments (second is deleter)
              // but we can't easily check if it's default_delete without
              // matching the function parameters, so we'll skip this case
              hasArgument(1, anything()));

    // Actually, for simplicity, let's just check if the smart pointer type
    // has a custom deleter. If it does, we skip the warning.
    auto SmartPtrWithDefaultDeleter = classTemplateSpecializationDecl(
        IsSmartPtr,
        anyOf(
            // shared_ptr with default deleter (1 template arg or 2nd is
            // default_delete)
            allOf(IsSharedPtr,
                  anyOf(templateArgumentCountIs(1),
                        allOf(templateArgumentCountIs(2),
                              hasTemplateArgument(
                                  1, refersToType(hasUnqualifiedDesugaredType(
                                         recordType(hasDeclaration(
                                             classTemplateSpecializationDecl(
                                                 IsDefaultDeleter))))))))),
            // unique_ptr with default deleter
            allOf(IsUniquePtr,
                  anyOf(templateArgumentCountIs(1),
                        allOf(templateArgumentCountIs(2),
                              hasTemplateArgument(
                                  1, refersToType(hasUnqualifiedDesugaredType(
                                         recordType(hasDeclaration(
                                             classTemplateSpecializationDecl(
                                                 IsDefaultDeleter)))))))))));

    auto ResetCallMatcher =
        cxxMemberCallExpr(
            on(hasType(hasUnqualifiedDesugaredType(
                recordType(hasDeclaration(SmartPtrWithDefaultDeleter))))),
            callee(cxxMethodDecl(ofClass(IsSmartPtrRecord), hasName("reset"))),
            hasArgument(0, PointerArg),
            unless(hasArgument(0, AllowedArguments)),
            hasArgument(0, OptionalCondOp))
            .bind("reset");

    Finder->addMatcher(SmartPtrConstructorMatcher, &Check);
    Finder->addMatcher(ResetCallMatcher, &Check);
  }

  void check(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");
    const auto *Reset = Result.Nodes.getNodeAs<CXXMemberCallExpr>("reset");
    const auto *Cond = Result.Nodes.getNodeAs<ConditionalOperator>("cond-op");
    const Expr *ConstructorOrMember = Ctor;
    if (!ConstructorOrMember)
      ConstructorOrMember = Reset;

    if (ConstructorOrMember)
      checkInternal(*Result.Context, ConstructorOrMember, Cond);
  }

  bool isStrictMode() override { return true; }

private:
  void checkInternal(ASTContext &Context, const Expr *ConstructorOrMember,
                     const ConditionalOperator *Cond) {
    if (Cond && validateConditionalOperator(Context, Cond))
      return;
    Check.emitDiagnostic(Context, ConstructorOrMember);
  }

  bool validateConditionalOperator(ASTContext &Context,
                                   const ConditionalOperator *Cond) {
    assert(Cond);

    static const auto ReleaseCallMatcher =
        cxxMemberCallExpr(callee(cxxMethodDecl(hasName("release"))));

    static const StatementMatcher Matcher =
        anyOf(integerLiteral(equals(0)), cxxNullPtrLiteralExpr(), cxxNewExpr(),
              ReleaseCallMatcher);

    auto isValidExpr = [&](const Expr *E) -> bool {
      if (!E)
        return false;

      E = E->IgnoreParenCasts();

      // Если это вложенный тернарный оператор - проверяем рекурсивно
      if (const auto *NestedCond = dyn_cast<ConditionalOperator>(E))
        return validateConditionalOperator(Context, NestedCond);

      // Иначе проверяем через матчер
      const auto Matches = match(Matcher, *E, Context);
      return !Matches.empty();
    };

    return isValidExpr(Cond->getTrueExpr()) &&
           isValidExpr(Cond->getFalseExpr());
  }
};

static std::unique_ptr<SmartPtrInitializationCheckImpl>
makeImpl(bool StrictMode, SmartPtrInitializationCheck &Check) {
  if (StrictMode)
    return std::make_unique<SmartPtrInitializationCheckStrictMode>(Check);
  else
    return std::make_unique<SmartPtrInitializationCheckPermissiveMode>(Check);
}

SmartPtrInitializationCheck::SmartPtrInitializationCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SharedPointers(utils::options::parseStringList(
          Options.get("SharedPointers", DefaultSharedPointers))),
      UniquePointers(utils::options::parseStringList(
          Options.get("UniquePointers", DefaultUniquePointers))),
      DefaultDeleters(utils::options::parseStringList(
          Options.get("DefaultDeleters", DefaultDefaultDeleters))),
      Impl(makeImpl(Options.get("StrictMode", false), *this)) {}

SmartPtrInitializationCheck::~SmartPtrInitializationCheck() = default;

void SmartPtrInitializationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SharedPointers",
                utils::options::serializeStringList(SharedPointers));
  Options.store(Opts, "UniquePointers",
                utils::options::serializeStringList(UniquePointers));
  Options.store(Opts, "DefaultDeleters",
                utils::options::serializeStringList(DefaultDeleters));
  Options.store(Opts, "StrictMode", Impl->isStrictMode());
}

void SmartPtrInitializationCheck::registerMatchers(MatchFinder *Finder) {
  Impl->registerMatchers(Finder);
}

void SmartPtrInitializationCheck::check(
    const MatchFinder::MatchResult &Result) {
  Impl->check(Result);
}

void SmartPtrInitializationCheck::emitDiagnostic(
    ASTContext &Context, const Expr *ConstructorOrMember) {
  if (const auto *SmartPtrCtor =
          dyn_cast<const CXXConstructExpr>(ConstructorOrMember)) {
    const Expr *PointerArg = stripWrappers(SmartPtrCtor->getArg(0));
    if (!PointerArg)
      return;
    const SourceLocation Loc = PointerArg->getBeginLoc();
    if (Loc.isInvalid())
      return;
    diag(Loc, "passing a raw pointer %0 to %1 constructor may cause "
              "double deletion")
        << PointerArg->getType() << SmartPtrCtor->getType();
  } else if (const auto *ResetCall =
                 dyn_cast<const CXXMemberCallExpr>(ConstructorOrMember)) {
    const Expr *PointerArg = stripWrappers(ResetCall->getArg(0));
    if (!PointerArg)
      return;
    const SourceLocation Loc = PointerArg->getBeginLoc();
    if (Loc.isInvalid())
      return;
    diag(
        Loc,
        "passing a raw pointer %0 to %1 reset method may cause double deletion")
        << PointerArg->getType() << ResetCall->getObjectType();
  }
}

} // namespace clang::tidy::bugprone
