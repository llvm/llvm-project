//===- Consumed.cpp --------------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A intra-procedural analysis for checking consumed properties.  This is based,
// in part, on research on linear types.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/Analyses/Consumed.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

// TODO: Adjust states of args to constructors in the same way that arguments to
//       function calls are handled.
// TODO: Use information from tests in for- and while-loop conditional.
// TODO: Add notes about the actual and expected state for 
// TODO: Correctly identify unreachable blocks when chaining boolean operators.
// TODO: Adjust the parser and AttributesList class to support lists of
//       identifiers.
// TODO: Warn about unreachable code.
// TODO: Switch to using a bitmap to track unreachable blocks.
// TODO: Handle variable definitions, e.g. bool valid = x.isValid();
//       if (valid) ...; (Deferred)
// TODO: Take notes on state transitions to provide better warning messages.
//       (Deferred)
// TODO: Test nested conditionals: A) Checking the same value multiple times,
//       and 2) Checking different values. (Deferred)

using namespace clang;
using namespace consumed;

// Key method definition
ConsumedWarningsHandlerBase::~ConsumedWarningsHandlerBase() {}

static SourceLocation getFirstStmtLoc(const CFGBlock *Block) {
  // Find the source location of the first statement in the block, if the block
  // is not empty.
  for (CFGBlock::const_iterator BI = Block->begin(), BE = Block->end();
       BI != BE; ++BI) {
    if (Optional<CFGStmt> CS = BI->getAs<CFGStmt>())
      return CS->getStmt()->getLocStart();
  }

  // Block is empty.
  // If we have one successor, return the first statement in that block
  if (Block->succ_size() == 1 && *Block->succ_begin())
    return getFirstStmtLoc(*Block->succ_begin());

  return SourceLocation();
}

static SourceLocation getLastStmtLoc(const CFGBlock *Block) {
  // Find the source location of the last statement in the block, if the block
  // is not empty.
  if (const Stmt *StmtNode = Block->getTerminator()) {
    return StmtNode->getLocStart();
  } else {
    for (CFGBlock::const_reverse_iterator BI = Block->rbegin(),
         BE = Block->rend(); BI != BE; ++BI) {
      if (Optional<CFGStmt> CS = BI->getAs<CFGStmt>())
        return CS->getStmt()->getLocStart();
    }
  }

  // If we have one successor, return the first statement in that block
  SourceLocation Loc;
  if (Block->succ_size() == 1 && *Block->succ_begin())
    Loc = getFirstStmtLoc(*Block->succ_begin());
  if (Loc.isValid())
    return Loc;

  // If we have one predecessor, return the last statement in that block
  if (Block->pred_size() == 1 && *Block->pred_begin())
    return getLastStmtLoc(*Block->pred_begin());

  return Loc;
}

static ConsumedState invertConsumedUnconsumed(ConsumedState State) {
  switch (State) {
  case CS_Unconsumed:
    return CS_Consumed;
  case CS_Consumed:
    return CS_Unconsumed;
  case CS_None:
    return CS_None;
  case CS_Unknown:
    return CS_Unknown;
  }
  llvm_unreachable("invalid enum");
}

static bool isCallableInState(const CallableWhenAttr *CWAttr,
                              ConsumedState State) {
  
  CallableWhenAttr::callableState_iterator I = CWAttr->callableState_begin(),
                                           E = CWAttr->callableState_end();
  
  for (; I != E; ++I) {
    
    ConsumedState MappedAttrState = CS_None;
    
    switch (*I) {
    case CallableWhenAttr::Unknown:
      MappedAttrState = CS_Unknown;
      break;
      
    case CallableWhenAttr::Unconsumed:
      MappedAttrState = CS_Unconsumed;
      break;
      
    case CallableWhenAttr::Consumed:
      MappedAttrState = CS_Consumed;
      break;
    }
    
    if (MappedAttrState == State)
      return true;
  }
  
  return false;
}

static bool isConsumableType(const QualType &QT) {
  if (QT->isPointerType() || QT->isReferenceType())
    return false;
  
  if (const CXXRecordDecl *RD = QT->getAsCXXRecordDecl())
    return RD->hasAttr<ConsumableAttr>();
  
  return false;
}

static bool isKnownState(ConsumedState State) {
  switch (State) {
  case CS_Unconsumed:
  case CS_Consumed:
    return true;
  case CS_None:
  case CS_Unknown:
    return false;
  }
  llvm_unreachable("invalid enum");
}

static bool isRValueRefish(QualType ParamType) {
  return ParamType->isRValueReferenceType() ||
        (ParamType->isLValueReferenceType() &&
         !cast<LValueReferenceType>(
           ParamType.getCanonicalType())->isSpelledAsLValue());
}

static bool isTestingFunction(const FunctionDecl *FunDecl) {
  return FunDecl->hasAttr<TestTypestateAttr>();
}

static bool isValueType(QualType ParamType) {
  return !(ParamType->isPointerType() || ParamType->isReferenceType());
}

static ConsumedState mapConsumableAttrState(const QualType QT) {
  assert(isConsumableType(QT));

  const ConsumableAttr *CAttr =
      QT->getAsCXXRecordDecl()->getAttr<ConsumableAttr>();

  switch (CAttr->getDefaultState()) {
  case ConsumableAttr::Unknown:
    return CS_Unknown;
  case ConsumableAttr::Unconsumed:
    return CS_Unconsumed;
  case ConsumableAttr::Consumed:
    return CS_Consumed;
  }
  llvm_unreachable("invalid enum");
}

static ConsumedState
mapParamTypestateAttrState(const ParamTypestateAttr *PTAttr) {
  switch (PTAttr->getParamState()) {
  case ParamTypestateAttr::Unknown:
    return CS_Unknown;
  case ParamTypestateAttr::Unconsumed:
    return CS_Unconsumed;
  case ParamTypestateAttr::Consumed:
    return CS_Consumed;
  }
  llvm_unreachable("invalid_enum");
}

static ConsumedState
mapReturnTypestateAttrState(const ReturnTypestateAttr *RTSAttr) {
  switch (RTSAttr->getState()) {
  case ReturnTypestateAttr::Unknown:
    return CS_Unknown;
  case ReturnTypestateAttr::Unconsumed:
    return CS_Unconsumed;
  case ReturnTypestateAttr::Consumed:
    return CS_Consumed;
  }
  llvm_unreachable("invalid enum");
}

static ConsumedState mapSetTypestateAttrState(const SetTypestateAttr *STAttr) {
  switch (STAttr->getNewState()) {
  case SetTypestateAttr::Unknown:
    return CS_Unknown;
  case SetTypestateAttr::Unconsumed:
    return CS_Unconsumed;
  case SetTypestateAttr::Consumed:
    return CS_Consumed;
  }
  llvm_unreachable("invalid_enum");
}

static StringRef stateToString(ConsumedState State) {
  switch (State) {
  case consumed::CS_None:
    return "none";
  
  case consumed::CS_Unknown:
    return "unknown";
  
  case consumed::CS_Unconsumed:
    return "unconsumed";
  
  case consumed::CS_Consumed:
    return "consumed";
  }
  llvm_unreachable("invalid enum");
}

static ConsumedState testsFor(const FunctionDecl *FunDecl) {
  assert(isTestingFunction(FunDecl));
  switch (FunDecl->getAttr<TestTypestateAttr>()->getTestState()) {
  case TestTypestateAttr::Unconsumed:
    return CS_Unconsumed;
  case TestTypestateAttr::Consumed:
    return CS_Consumed;
  }
  llvm_unreachable("invalid enum");
}

namespace {
struct VarTestResult {
  const VarDecl *Var;
  ConsumedState TestsFor;
};
} // end anonymous::VarTestResult

namespace clang {
namespace consumed {

enum EffectiveOp {
  EO_And,
  EO_Or
};

class PropagationInfo {
  enum {
    IT_None,
    IT_State,
    IT_VarTest,
    IT_BinTest,
    IT_Var,
    IT_Tmp
  } InfoType;

  struct BinTestTy {
    const BinaryOperator *Source;
    EffectiveOp EOp;
    VarTestResult LTest;
    VarTestResult RTest;
  };
  
  union {
    ConsumedState State;
    VarTestResult VarTest;
    const VarDecl *Var;
    const CXXBindTemporaryExpr *Tmp;
    BinTestTy BinTest;
  };
  
public:
  PropagationInfo() : InfoType(IT_None) {}
  
  PropagationInfo(const VarTestResult &VarTest)
    : InfoType(IT_VarTest), VarTest(VarTest) {}
  
  PropagationInfo(const VarDecl *Var, ConsumedState TestsFor)
    : InfoType(IT_VarTest) {
    
    VarTest.Var      = Var;
    VarTest.TestsFor = TestsFor;
  }
  
  PropagationInfo(const BinaryOperator *Source, EffectiveOp EOp,
                  const VarTestResult &LTest, const VarTestResult &RTest)
    : InfoType(IT_BinTest) {
    
    BinTest.Source  = Source;
    BinTest.EOp     = EOp;
    BinTest.LTest   = LTest;
    BinTest.RTest   = RTest;
  }
  
  PropagationInfo(const BinaryOperator *Source, EffectiveOp EOp,
                  const VarDecl *LVar, ConsumedState LTestsFor,
                  const VarDecl *RVar, ConsumedState RTestsFor)
    : InfoType(IT_BinTest) {
    
    BinTest.Source         = Source;
    BinTest.EOp            = EOp;
    BinTest.LTest.Var      = LVar;
    BinTest.LTest.TestsFor = LTestsFor;
    BinTest.RTest.Var      = RVar;
    BinTest.RTest.TestsFor = RTestsFor;
  }
  
  PropagationInfo(ConsumedState State)
    : InfoType(IT_State), State(State) {}
  
  PropagationInfo(const VarDecl *Var) : InfoType(IT_Var), Var(Var) {}
  PropagationInfo(const CXXBindTemporaryExpr *Tmp)
    : InfoType(IT_Tmp), Tmp(Tmp) {}
  
  const ConsumedState & getState() const {
    assert(InfoType == IT_State);
    return State;
  }
  
  const VarTestResult & getVarTest() const {
    assert(InfoType == IT_VarTest);
    return VarTest;
  }
  
  const VarTestResult & getLTest() const {
    assert(InfoType == IT_BinTest);
    return BinTest.LTest;
  }
  
  const VarTestResult & getRTest() const {
    assert(InfoType == IT_BinTest);
    return BinTest.RTest;
  }
  
  const VarDecl * getVar() const {
    assert(InfoType == IT_Var);
    return Var;
  }
  
  const CXXBindTemporaryExpr * getTmp() const {
    assert(InfoType == IT_Tmp);
    return Tmp;
  }
  
  ConsumedState getAsState(const ConsumedStateMap *StateMap) const {
    assert(isVar() || isTmp() || isState());
    
    if (isVar())
      return StateMap->getState(Var);
    else if (isTmp())
      return StateMap->getState(Tmp);
    else if (isState())
      return State;
    else
      return CS_None;
  }
  
  EffectiveOp testEffectiveOp() const {
    assert(InfoType == IT_BinTest);
    return BinTest.EOp;
  }
  
  const BinaryOperator * testSourceNode() const {
    assert(InfoType == IT_BinTest);
    return BinTest.Source;
  }
  
  inline bool isValid()   const { return InfoType != IT_None;    }
  inline bool isState()   const { return InfoType == IT_State;   }
  inline bool isVarTest() const { return InfoType == IT_VarTest; }
  inline bool isBinTest() const { return InfoType == IT_BinTest; }
  inline bool isVar()     const { return InfoType == IT_Var;     }
  inline bool isTmp()     const { return InfoType == IT_Tmp;     }
  
  bool isTest() const {
    return InfoType == IT_VarTest || InfoType == IT_BinTest;
  }
  
  bool isPointerToValue() const {
    return InfoType == IT_Var || InfoType == IT_Tmp;
  }
  
  PropagationInfo invertTest() const {
    assert(InfoType == IT_VarTest || InfoType == IT_BinTest);
    
    if (InfoType == IT_VarTest) {
      return PropagationInfo(VarTest.Var,
                             invertConsumedUnconsumed(VarTest.TestsFor));
    
    } else if (InfoType == IT_BinTest) {
      return PropagationInfo(BinTest.Source,
        BinTest.EOp == EO_And ? EO_Or : EO_And,
        BinTest.LTest.Var, invertConsumedUnconsumed(BinTest.LTest.TestsFor),
        BinTest.RTest.Var, invertConsumedUnconsumed(BinTest.RTest.TestsFor));
    } else {
      return PropagationInfo();
    }
  }
};

static inline void
setStateForVarOrTmp(ConsumedStateMap *StateMap, const PropagationInfo &PInfo,
                    ConsumedState State) {

  assert(PInfo.isVar() || PInfo.isTmp());
  
  if (PInfo.isVar())
    StateMap->setState(PInfo.getVar(), State);
  else
    StateMap->setState(PInfo.getTmp(), State);
}

class ConsumedStmtVisitor : public ConstStmtVisitor<ConsumedStmtVisitor> {
  
  typedef llvm::DenseMap<const Stmt *, PropagationInfo> MapType;
  typedef std::pair<const Stmt *, PropagationInfo> PairType;
  typedef MapType::iterator InfoEntry;
  typedef MapType::const_iterator ConstInfoEntry;
  
  AnalysisDeclContext &AC;
  ConsumedAnalyzer &Analyzer;
  ConsumedStateMap *StateMap;
  MapType PropagationMap;
  void forwardInfo(const Stmt *From, const Stmt *To);
  bool isLikeMoveAssignment(const CXXMethodDecl *MethodDecl);
  void propagateReturnType(const Stmt *Call, const FunctionDecl *Fun,
                           QualType ReturnType);

public:
  void checkCallability(const PropagationInfo &PInfo,
                        const FunctionDecl *FunDecl,
                        SourceLocation BlameLoc);
  
  void VisitBinaryOperator(const BinaryOperator *BinOp);
  void VisitCallExpr(const CallExpr *Call);
  void VisitCastExpr(const CastExpr *Cast);
  void VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *Temp);
  void VisitCXXConstructExpr(const CXXConstructExpr *Call);
  void VisitCXXMemberCallExpr(const CXXMemberCallExpr *Call);
  void VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *Call);
  void VisitDeclRefExpr(const DeclRefExpr *DeclRef);
  void VisitDeclStmt(const DeclStmt *DelcS);
  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *Temp);
  void VisitMemberExpr(const MemberExpr *MExpr);
  void VisitParmVarDecl(const ParmVarDecl *Param);
  void VisitReturnStmt(const ReturnStmt *Ret);
  void VisitUnaryOperator(const UnaryOperator *UOp);
  void VisitVarDecl(const VarDecl *Var);

  ConsumedStmtVisitor(AnalysisDeclContext &AC, ConsumedAnalyzer &Analyzer,
                      ConsumedStateMap *StateMap)
      : AC(AC), Analyzer(Analyzer), StateMap(StateMap) {}
  
  PropagationInfo getInfo(const Stmt *StmtNode) const {
    ConstInfoEntry Entry = PropagationMap.find(StmtNode);
    
    if (Entry != PropagationMap.end())
      return Entry->second;
    else
      return PropagationInfo();
  }
  
  void reset(ConsumedStateMap *NewStateMap) {
    StateMap = NewStateMap;
  }
};

void ConsumedStmtVisitor::checkCallability(const PropagationInfo &PInfo,
                                           const FunctionDecl *FunDecl,
                                           SourceLocation BlameLoc) {
  assert(!PInfo.isTest());
  
  if (!FunDecl->hasAttr<CallableWhenAttr>())
    return;
  
  const CallableWhenAttr *CWAttr = FunDecl->getAttr<CallableWhenAttr>();
  
  if (PInfo.isVar()) {
    ConsumedState VarState = StateMap->getState(PInfo.getVar());
    
    if (VarState == CS_None || isCallableInState(CWAttr, VarState))
      return;
    
    Analyzer.WarningsHandler.warnUseInInvalidState(
      FunDecl->getNameAsString(), PInfo.getVar()->getNameAsString(),
      stateToString(VarState), BlameLoc);
    
  } else {
    ConsumedState TmpState = PInfo.getAsState(StateMap);
    
    if (TmpState == CS_None || isCallableInState(CWAttr, TmpState))
      return;
    
    Analyzer.WarningsHandler.warnUseOfTempInInvalidState(
      FunDecl->getNameAsString(), stateToString(TmpState), BlameLoc);
  }
}

void ConsumedStmtVisitor::forwardInfo(const Stmt *From, const Stmt *To) {
  InfoEntry Entry = PropagationMap.find(From);
  
  if (Entry != PropagationMap.end())
    PropagationMap.insert(PairType(To, Entry->second));
}

bool ConsumedStmtVisitor::isLikeMoveAssignment(
  const CXXMethodDecl *MethodDecl) {
  
  return MethodDecl->isMoveAssignmentOperator() ||
         (MethodDecl->getOverloadedOperator() == OO_Equal &&
          MethodDecl->getNumParams() == 1 &&
          MethodDecl->getParamDecl(0)->getType()->isRValueReferenceType());
}

void ConsumedStmtVisitor::propagateReturnType(const Stmt *Call,
                                              const FunctionDecl *Fun,
                                              QualType ReturnType) {
  if (isConsumableType(ReturnType)) {
    
    ConsumedState ReturnState;
    
    if (Fun->hasAttr<ReturnTypestateAttr>())
      ReturnState = mapReturnTypestateAttrState(
        Fun->getAttr<ReturnTypestateAttr>());
    else
      ReturnState = mapConsumableAttrState(ReturnType);
    
    PropagationMap.insert(PairType(Call, PropagationInfo(ReturnState)));
  }
}

void ConsumedStmtVisitor::VisitBinaryOperator(const BinaryOperator *BinOp) {
  switch (BinOp->getOpcode()) {
  case BO_LAnd:
  case BO_LOr : {
    InfoEntry LEntry = PropagationMap.find(BinOp->getLHS()),
              REntry = PropagationMap.find(BinOp->getRHS());
    
    VarTestResult LTest, RTest;
    
    if (LEntry != PropagationMap.end() && LEntry->second.isVarTest()) {
      LTest = LEntry->second.getVarTest();
      
    } else {
      LTest.Var      = NULL;
      LTest.TestsFor = CS_None;
    }
    
    if (REntry != PropagationMap.end() && REntry->second.isVarTest()) {
      RTest = REntry->second.getVarTest();
      
    } else {
      RTest.Var      = NULL;
      RTest.TestsFor = CS_None;
    }
    
    if (!(LTest.Var == NULL && RTest.Var == NULL))
      PropagationMap.insert(PairType(BinOp, PropagationInfo(BinOp,
        static_cast<EffectiveOp>(BinOp->getOpcode() == BO_LOr), LTest, RTest)));
    
    break;
  }
    
  case BO_PtrMemD:
  case BO_PtrMemI:
    forwardInfo(BinOp->getLHS(), BinOp);
    break;
    
  default:
    break;
  }
}

static bool isStdNamespace(const DeclContext *DC) {
  if (!DC->isNamespace()) return false;
  while (DC->getParent()->isNamespace())
    DC = DC->getParent();
  const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(DC);

  return ND && ND->getName() == "std" &&
         ND->getDeclContext()->isTranslationUnit();
}

void ConsumedStmtVisitor::VisitCallExpr(const CallExpr *Call) {
  if (const FunctionDecl *FunDecl =
    dyn_cast_or_null<FunctionDecl>(Call->getDirectCallee())) {
    
    // Special case for the std::move function.
    // TODO: Make this more specific. (Deferred)
    if (Call->getNumArgs() == 1 &&
        FunDecl->getNameAsString() == "move" &&
        isStdNamespace(FunDecl->getDeclContext())) {
      forwardInfo(Call->getArg(0), Call);
      return;
    }
    
    unsigned Offset = Call->getNumArgs() - FunDecl->getNumParams();
    
    for (unsigned Index = Offset; Index < Call->getNumArgs(); ++Index) {
      const ParmVarDecl *Param = FunDecl->getParamDecl(Index - Offset);
      QualType ParamType = Param->getType();
      
      InfoEntry Entry = PropagationMap.find(Call->getArg(Index));
      
      if (Entry == PropagationMap.end() || Entry->second.isTest())
        continue;
      
      PropagationInfo PInfo = Entry->second;
      
      // Check that the parameter is in the correct state.
      
      if (Param->hasAttr<ParamTypestateAttr>()) {
        ConsumedState ParamState = PInfo.getAsState(StateMap);
        
        ConsumedState ExpectedState =
          mapParamTypestateAttrState(Param->getAttr<ParamTypestateAttr>());
        
        if (ParamState != ExpectedState)
          Analyzer.WarningsHandler.warnParamTypestateMismatch(
            Call->getArg(Index - Offset)->getExprLoc(),
            stateToString(ExpectedState), stateToString(ParamState));
      }
      
      if (!(Entry->second.isVar() || Entry->second.isTmp()))
        continue;
      
      // Adjust state on the caller side.
      
      if (isRValueRefish(ParamType)) {
        setStateForVarOrTmp(StateMap, PInfo, consumed::CS_Consumed);
        
      } else if (Param->hasAttr<ReturnTypestateAttr>()) {
        setStateForVarOrTmp(StateMap, PInfo,
          mapReturnTypestateAttrState(Param->getAttr<ReturnTypestateAttr>()));
        
      } else if (!isValueType(ParamType) &&
                 !ParamType->getPointeeType().isConstQualified()) {
        
        setStateForVarOrTmp(StateMap, PInfo, consumed::CS_Unknown);
      }
    }
    
    QualType RetType = FunDecl->getCallResultType();
    if (RetType->isReferenceType())
      RetType = RetType->getPointeeType();
    
    propagateReturnType(Call, FunDecl, RetType);
  }
}

void ConsumedStmtVisitor::VisitCastExpr(const CastExpr *Cast) {
  forwardInfo(Cast->getSubExpr(), Cast);
}

void ConsumedStmtVisitor::VisitCXXBindTemporaryExpr(
  const CXXBindTemporaryExpr *Temp) {
  
  InfoEntry Entry = PropagationMap.find(Temp->getSubExpr());
  
  if (Entry != PropagationMap.end() && !Entry->second.isTest()) {
    StateMap->setState(Temp, Entry->second.getAsState(StateMap));
    PropagationMap.insert(PairType(Temp, PropagationInfo(Temp)));
  }
}

void ConsumedStmtVisitor::VisitCXXConstructExpr(const CXXConstructExpr *Call) {
  CXXConstructorDecl *Constructor = Call->getConstructor();

  ASTContext &CurrContext = AC.getASTContext();
  QualType ThisType = Constructor->getThisType(CurrContext)->getPointeeType();
  
  if (!isConsumableType(ThisType))
    return;
  
  // FIXME: What should happen if someone annotates the move constructor?
  if (Constructor->hasAttr<ReturnTypestateAttr>()) {
    // TODO: Adjust state of args appropriately.
    
    ReturnTypestateAttr *RTAttr = Constructor->getAttr<ReturnTypestateAttr>();
    ConsumedState RetState = mapReturnTypestateAttrState(RTAttr);
    PropagationMap.insert(PairType(Call, PropagationInfo(RetState)));
    
  } else if (Constructor->isDefaultConstructor()) {
    
    PropagationMap.insert(PairType(Call,
      PropagationInfo(consumed::CS_Consumed)));
    
  } else if (Constructor->isMoveConstructor()) {
    
    InfoEntry Entry = PropagationMap.find(Call->getArg(0));
    
    if (Entry != PropagationMap.end()) {
      PropagationInfo PInfo = Entry->second;
      
      if (PInfo.isVar()) {
        const VarDecl* Var = PInfo.getVar();
        
        PropagationMap.insert(PairType(Call,
          PropagationInfo(StateMap->getState(Var))));
        
        StateMap->setState(Var, consumed::CS_Consumed);
        
      } else if (PInfo.isTmp()) {
        const CXXBindTemporaryExpr *Tmp = PInfo.getTmp();
        
        PropagationMap.insert(PairType(Call,
          PropagationInfo(StateMap->getState(Tmp))));
        
        StateMap->setState(Tmp, consumed::CS_Consumed);
        
      } else {
        PropagationMap.insert(PairType(Call, PInfo));
      }
    }
  } else if (Constructor->isCopyConstructor()) {
    forwardInfo(Call->getArg(0), Call);
    
  } else {
    // TODO: Adjust state of args appropriately.
    
    ConsumedState RetState = mapConsumableAttrState(ThisType);
    PropagationMap.insert(PairType(Call, PropagationInfo(RetState)));
  }
}

void ConsumedStmtVisitor::VisitCXXMemberCallExpr(
  const CXXMemberCallExpr *Call) {
  
  VisitCallExpr(Call);
  
  InfoEntry Entry = PropagationMap.find(Call->getCallee()->IgnoreParens());
  
  if (Entry != PropagationMap.end()) {
    PropagationInfo PInfo = Entry->second;
    const CXXMethodDecl *MethodDecl = Call->getMethodDecl();
    
    checkCallability(PInfo, MethodDecl, Call->getExprLoc());
    
    if (PInfo.isVar()) {
      if (isTestingFunction(MethodDecl))
        PropagationMap.insert(PairType(Call,
          PropagationInfo(PInfo.getVar(), testsFor(MethodDecl))));
      else if (MethodDecl->hasAttr<SetTypestateAttr>())
        StateMap->setState(PInfo.getVar(),
          mapSetTypestateAttrState(MethodDecl->getAttr<SetTypestateAttr>()));
    } else if (PInfo.isTmp() && MethodDecl->hasAttr<SetTypestateAttr>()) {
      StateMap->setState(PInfo.getTmp(),
        mapSetTypestateAttrState(MethodDecl->getAttr<SetTypestateAttr>()));
    }
  }
}

void ConsumedStmtVisitor::VisitCXXOperatorCallExpr(
  const CXXOperatorCallExpr *Call) {
  
  const FunctionDecl *FunDecl =
    dyn_cast_or_null<FunctionDecl>(Call->getDirectCallee());
  
  if (!FunDecl) return;
    
  if (isa<CXXMethodDecl>(FunDecl) &&
      isLikeMoveAssignment(cast<CXXMethodDecl>(FunDecl))) {
    
    InfoEntry LEntry = PropagationMap.find(Call->getArg(0));
    InfoEntry REntry = PropagationMap.find(Call->getArg(1));
    
    PropagationInfo LPInfo, RPInfo;
    
    if (LEntry != PropagationMap.end() &&
        REntry != PropagationMap.end()) {
      
      LPInfo = LEntry->second;
      RPInfo = REntry->second;
      
      if (LPInfo.isPointerToValue() && RPInfo.isPointerToValue()) {
        setStateForVarOrTmp(StateMap, LPInfo, RPInfo.getAsState(StateMap));
        PropagationMap.insert(PairType(Call, LPInfo));
        setStateForVarOrTmp(StateMap, RPInfo, consumed::CS_Consumed);
        
      } else if (RPInfo.isState()) {
        setStateForVarOrTmp(StateMap, LPInfo, RPInfo.getState());
        PropagationMap.insert(PairType(Call, LPInfo));
        
      } else {
        setStateForVarOrTmp(StateMap, RPInfo, consumed::CS_Consumed);
      }
      
    } else if (LEntry != PropagationMap.end() &&
               REntry == PropagationMap.end()) {
      
      LPInfo = LEntry->second;
      
      assert(!LPInfo.isTest());
      
      if (LPInfo.isPointerToValue()) {
        setStateForVarOrTmp(StateMap, LPInfo, consumed::CS_Unknown);
        PropagationMap.insert(PairType(Call, LPInfo));
        
      } else {
        PropagationMap.insert(PairType(Call,
          PropagationInfo(consumed::CS_Unknown)));
      }
      
    } else if (LEntry == PropagationMap.end() &&
               REntry != PropagationMap.end()) {
      
      RPInfo = REntry->second;
      
      if (RPInfo.isPointerToValue())
        setStateForVarOrTmp(StateMap, RPInfo, consumed::CS_Consumed);
    }
    
  } else {
    
    VisitCallExpr(Call);
    
    InfoEntry Entry = PropagationMap.find(Call->getArg(0));
    
    if (Entry != PropagationMap.end()) {
      PropagationInfo PInfo = Entry->second;
      
      checkCallability(PInfo, FunDecl, Call->getExprLoc());
      
      if (PInfo.isVar()) {
        if (isTestingFunction(FunDecl))
          PropagationMap.insert(PairType(Call,
            PropagationInfo(PInfo.getVar(), testsFor(FunDecl))));
        else if (FunDecl->hasAttr<SetTypestateAttr>())
          StateMap->setState(PInfo.getVar(),
            mapSetTypestateAttrState(FunDecl->getAttr<SetTypestateAttr>()));
        
      } else if (PInfo.isTmp() && FunDecl->hasAttr<SetTypestateAttr>()) {
        StateMap->setState(PInfo.getTmp(),
          mapSetTypestateAttrState(FunDecl->getAttr<SetTypestateAttr>()));
    }
    }
  }
}

void ConsumedStmtVisitor::VisitDeclRefExpr(const DeclRefExpr *DeclRef) {
  if (const VarDecl *Var = dyn_cast_or_null<VarDecl>(DeclRef->getDecl()))
    if (StateMap->getState(Var) != consumed::CS_None)
      PropagationMap.insert(PairType(DeclRef, PropagationInfo(Var)));
}

void ConsumedStmtVisitor::VisitDeclStmt(const DeclStmt *DeclS) {
  for (DeclStmt::const_decl_iterator DI = DeclS->decl_begin(),
       DE = DeclS->decl_end(); DI != DE; ++DI) {
    
    if (isa<VarDecl>(*DI)) VisitVarDecl(cast<VarDecl>(*DI));
  }
  
  if (DeclS->isSingleDecl())
    if (const VarDecl *Var = dyn_cast_or_null<VarDecl>(DeclS->getSingleDecl()))
      PropagationMap.insert(PairType(DeclS, PropagationInfo(Var)));
}

void ConsumedStmtVisitor::VisitMaterializeTemporaryExpr(
  const MaterializeTemporaryExpr *Temp) {
  
  forwardInfo(Temp->GetTemporaryExpr(), Temp);
}

void ConsumedStmtVisitor::VisitMemberExpr(const MemberExpr *MExpr) {
  forwardInfo(MExpr->getBase(), MExpr);
}


void ConsumedStmtVisitor::VisitParmVarDecl(const ParmVarDecl *Param) {
  QualType ParamType = Param->getType();
  ConsumedState ParamState = consumed::CS_None;
  
  if (Param->hasAttr<ParamTypestateAttr>()) {
    const ParamTypestateAttr *PTAttr = Param->getAttr<ParamTypestateAttr>();
    ParamState = mapParamTypestateAttrState(PTAttr);
    
  } else if (isConsumableType(ParamType)) {
    ParamState = mapConsumableAttrState(ParamType);
    
  } else if (isRValueRefish(ParamType) &&
             isConsumableType(ParamType->getPointeeType())) {
    
    ParamState = mapConsumableAttrState(ParamType->getPointeeType());
    
  } else if (ParamType->isReferenceType() &&
             isConsumableType(ParamType->getPointeeType())) {
    ParamState = consumed::CS_Unknown;
  }
  
  if (ParamState != CS_None)
    StateMap->setState(Param, ParamState);
}

void ConsumedStmtVisitor::VisitReturnStmt(const ReturnStmt *Ret) {
  ConsumedState ExpectedState = Analyzer.getExpectedReturnState();
  
  if (ExpectedState != CS_None) {
    InfoEntry Entry = PropagationMap.find(Ret->getRetValue());
    
    if (Entry != PropagationMap.end()) {
      ConsumedState RetState = Entry->second.getAsState(StateMap);
        
      if (RetState != ExpectedState)
        Analyzer.WarningsHandler.warnReturnTypestateMismatch(
          Ret->getReturnLoc(), stateToString(ExpectedState),
          stateToString(RetState));
    }
  }
  
  StateMap->checkParamsForReturnTypestate(Ret->getLocStart(),
                                          Analyzer.WarningsHandler);
}

void ConsumedStmtVisitor::VisitUnaryOperator(const UnaryOperator *UOp) {
  InfoEntry Entry = PropagationMap.find(UOp->getSubExpr()->IgnoreParens());
  if (Entry == PropagationMap.end()) return;
  
  switch (UOp->getOpcode()) {
  case UO_AddrOf:
    PropagationMap.insert(PairType(UOp, Entry->second));
    break;
  
  case UO_LNot:
    if (Entry->second.isTest())
      PropagationMap.insert(PairType(UOp, Entry->second.invertTest()));
    break;
  
  default:
    break;
  }
}

// TODO: See if I need to check for reference types here.
void ConsumedStmtVisitor::VisitVarDecl(const VarDecl *Var) {
  if (isConsumableType(Var->getType())) {
    if (Var->hasInit()) {
      MapType::iterator VIT = PropagationMap.find(
        Var->getInit()->IgnoreImplicit());
      if (VIT != PropagationMap.end()) {
        PropagationInfo PInfo = VIT->second;
        ConsumedState St = PInfo.getAsState(StateMap);
        
        if (St != consumed::CS_None) {
          StateMap->setState(Var, St);
          return;
        }
      }
    }
    // Otherwise
    StateMap->setState(Var, consumed::CS_Unknown);
  }
}
}} // end clang::consumed::ConsumedStmtVisitor

namespace clang {
namespace consumed {

void splitVarStateForIf(const IfStmt * IfNode, const VarTestResult &Test,
                        ConsumedStateMap *ThenStates,
                        ConsumedStateMap *ElseStates) {

  ConsumedState VarState = ThenStates->getState(Test.Var);
  
  if (VarState == CS_Unknown) {
    ThenStates->setState(Test.Var, Test.TestsFor);
    ElseStates->setState(Test.Var, invertConsumedUnconsumed(Test.TestsFor));
  
  } else if (VarState == invertConsumedUnconsumed(Test.TestsFor)) {
    ThenStates->markUnreachable();
    
  } else if (VarState == Test.TestsFor) {
    ElseStates->markUnreachable();
  }
}

void splitVarStateForIfBinOp(const PropagationInfo &PInfo,
  ConsumedStateMap *ThenStates, ConsumedStateMap *ElseStates) {
  
  const VarTestResult &LTest = PInfo.getLTest(),
                      &RTest = PInfo.getRTest();
  
  ConsumedState LState = LTest.Var ? ThenStates->getState(LTest.Var) : CS_None,
                RState = RTest.Var ? ThenStates->getState(RTest.Var) : CS_None;
  
  if (LTest.Var) {
    if (PInfo.testEffectiveOp() == EO_And) {
      if (LState == CS_Unknown) {
        ThenStates->setState(LTest.Var, LTest.TestsFor);
        
      } else if (LState == invertConsumedUnconsumed(LTest.TestsFor)) {
        ThenStates->markUnreachable();
        
      } else if (LState == LTest.TestsFor && isKnownState(RState)) {
        if (RState == RTest.TestsFor)
          ElseStates->markUnreachable();
        else
          ThenStates->markUnreachable();
      }
      
    } else {
      if (LState == CS_Unknown) {
        ElseStates->setState(LTest.Var,
                             invertConsumedUnconsumed(LTest.TestsFor));
      
      } else if (LState == LTest.TestsFor) {
        ElseStates->markUnreachable();
        
      } else if (LState == invertConsumedUnconsumed(LTest.TestsFor) &&
                 isKnownState(RState)) {
        
        if (RState == RTest.TestsFor)
          ElseStates->markUnreachable();
        else
          ThenStates->markUnreachable();
      }
    }
  }
  
  if (RTest.Var) {
    if (PInfo.testEffectiveOp() == EO_And) {
      if (RState == CS_Unknown)
        ThenStates->setState(RTest.Var, RTest.TestsFor);
      else if (RState == invertConsumedUnconsumed(RTest.TestsFor))
        ThenStates->markUnreachable();
      
    } else {
      if (RState == CS_Unknown)
        ElseStates->setState(RTest.Var,
                             invertConsumedUnconsumed(RTest.TestsFor));
      else if (RState == RTest.TestsFor)
        ElseStates->markUnreachable();
    }
  }
}

bool ConsumedBlockInfo::allBackEdgesVisited(const CFGBlock *CurrBlock,
                                            const CFGBlock *TargetBlock) {
  
  assert(CurrBlock && "Block pointer must not be NULL");
  assert(TargetBlock && "TargetBlock pointer must not be NULL");
  
  unsigned int CurrBlockOrder = VisitOrder[CurrBlock->getBlockID()];
  for (CFGBlock::const_pred_iterator PI = TargetBlock->pred_begin(),
       PE = TargetBlock->pred_end(); PI != PE; ++PI) {
    if (*PI && CurrBlockOrder < VisitOrder[(*PI)->getBlockID()] )
      return false;
  }
  return true;
}

void ConsumedBlockInfo::addInfo(const CFGBlock *Block,
                                ConsumedStateMap *StateMap,
                                bool &AlreadyOwned) {
  
  assert(Block && "Block pointer must not be NULL");
  
  ConsumedStateMap *Entry = StateMapsArray[Block->getBlockID()];
    
  if (Entry) {
    Entry->intersect(StateMap);
    
  } else if (AlreadyOwned) {
    StateMapsArray[Block->getBlockID()] = new ConsumedStateMap(*StateMap);
    
  } else {
    StateMapsArray[Block->getBlockID()] = StateMap;
    AlreadyOwned = true;
  }
}

void ConsumedBlockInfo::addInfo(const CFGBlock *Block,
                                ConsumedStateMap *StateMap) {
  
  assert(Block != NULL && "Block pointer must not be NULL");
  
  ConsumedStateMap *Entry = StateMapsArray[Block->getBlockID()];
    
  if (Entry) {
    Entry->intersect(StateMap);
    delete StateMap;
    
  } else {
    StateMapsArray[Block->getBlockID()] = StateMap;
  }
}

ConsumedStateMap* ConsumedBlockInfo::borrowInfo(const CFGBlock *Block) {
  assert(Block && "Block pointer must not be NULL");
  assert(StateMapsArray[Block->getBlockID()] && "Block has no block info");
  
  return StateMapsArray[Block->getBlockID()];
}

void ConsumedBlockInfo::discardInfo(const CFGBlock *Block) {
  unsigned int BlockID = Block->getBlockID();
  delete StateMapsArray[BlockID];
  StateMapsArray[BlockID] = NULL;
}

ConsumedStateMap* ConsumedBlockInfo::getInfo(const CFGBlock *Block) {
  assert(Block && "Block pointer must not be NULL");
  
  ConsumedStateMap *StateMap = StateMapsArray[Block->getBlockID()];
  if (isBackEdgeTarget(Block)) {
    return new ConsumedStateMap(*StateMap);
  } else {
    StateMapsArray[Block->getBlockID()] = NULL;
    return StateMap;
  }
}

bool ConsumedBlockInfo::isBackEdge(const CFGBlock *From, const CFGBlock *To) {
  assert(From && "From block must not be NULL");
  assert(To   && "From block must not be NULL");
  
  return VisitOrder[From->getBlockID()] > VisitOrder[To->getBlockID()];
}

bool ConsumedBlockInfo::isBackEdgeTarget(const CFGBlock *Block) {
  assert(Block != NULL && "Block pointer must not be NULL");
  
  // Anything with less than two predecessors can't be the target of a back
  // edge.
  if (Block->pred_size() < 2)
    return false;
  
  unsigned int BlockVisitOrder = VisitOrder[Block->getBlockID()];
  for (CFGBlock::const_pred_iterator PI = Block->pred_begin(),
       PE = Block->pred_end(); PI != PE; ++PI) {
    if (*PI && BlockVisitOrder < VisitOrder[(*PI)->getBlockID()])
      return true;
  }
  return false;
}

void ConsumedStateMap::checkParamsForReturnTypestate(SourceLocation BlameLoc,
  ConsumedWarningsHandlerBase &WarningsHandler) const {
  
  ConsumedState ExpectedState;
  
  for (VarMapType::const_iterator DMI = VarMap.begin(), DME = VarMap.end();
       DMI != DME; ++DMI) {
    
    if (isa<ParmVarDecl>(DMI->first)) {
      const ParmVarDecl *Param = cast<ParmVarDecl>(DMI->first);
      
      if (!Param->hasAttr<ReturnTypestateAttr>()) continue;
      
      ExpectedState =
        mapReturnTypestateAttrState(Param->getAttr<ReturnTypestateAttr>());
      
      if (DMI->second != ExpectedState) {
        WarningsHandler.warnParamReturnTypestateMismatch(BlameLoc,
          Param->getNameAsString(), stateToString(ExpectedState),
          stateToString(DMI->second));
      }
    }
  }
}

void ConsumedStateMap::clearTemporaries() {
  TmpMap.clear();
}

ConsumedState ConsumedStateMap::getState(const VarDecl *Var) const {
  VarMapType::const_iterator Entry = VarMap.find(Var);
  
  if (Entry != VarMap.end())
    return Entry->second;
    
  return CS_None;
}

ConsumedState
ConsumedStateMap::getState(const CXXBindTemporaryExpr *Tmp) const {
  TmpMapType::const_iterator Entry = TmpMap.find(Tmp);
  
  if (Entry != TmpMap.end())
    return Entry->second;
  
  return CS_None;
}

void ConsumedStateMap::intersect(const ConsumedStateMap *Other) {
  ConsumedState LocalState;
  
  if (this->From && this->From == Other->From && !Other->Reachable) {
    this->markUnreachable();
    return;
  }
  
  for (VarMapType::const_iterator DMI = Other->VarMap.begin(),
       DME = Other->VarMap.end(); DMI != DME; ++DMI) {
    
    LocalState = this->getState(DMI->first);
    
    if (LocalState == CS_None)
      continue;
    
    if (LocalState != DMI->second)
       VarMap[DMI->first] = CS_Unknown;
  }
}

void ConsumedStateMap::intersectAtLoopHead(const CFGBlock *LoopHead,
  const CFGBlock *LoopBack, const ConsumedStateMap *LoopBackStates,
  ConsumedWarningsHandlerBase &WarningsHandler) {
  
  ConsumedState LocalState;
  SourceLocation BlameLoc = getLastStmtLoc(LoopBack);
  
  for (VarMapType::const_iterator DMI = LoopBackStates->VarMap.begin(),
       DME = LoopBackStates->VarMap.end(); DMI != DME; ++DMI) {
    
    LocalState = this->getState(DMI->first);
    
    if (LocalState == CS_None)
      continue;
    
    if (LocalState != DMI->second) {
      VarMap[DMI->first] = CS_Unknown;
      WarningsHandler.warnLoopStateMismatch(
        BlameLoc, DMI->first->getNameAsString());
    }
  }
}

void ConsumedStateMap::markUnreachable() {
  this->Reachable = false;
  VarMap.clear();
  TmpMap.clear();
}

void ConsumedStateMap::setState(const VarDecl *Var, ConsumedState State) {
  VarMap[Var] = State;
}

void ConsumedStateMap::setState(const CXXBindTemporaryExpr *Tmp,
                                ConsumedState State) {
  TmpMap[Tmp] = State;
}

void ConsumedStateMap::remove(const VarDecl *Var) {
  VarMap.erase(Var);
}

bool ConsumedStateMap::operator!=(const ConsumedStateMap *Other) const {
  for (VarMapType::const_iterator DMI = Other->VarMap.begin(),
       DME = Other->VarMap.end(); DMI != DME; ++DMI) {
    
    if (this->getState(DMI->first) != DMI->second)
      return true;
  }
  
  return false;
}

void ConsumedAnalyzer::determineExpectedReturnState(AnalysisDeclContext &AC,
                                                    const FunctionDecl *D) {
  QualType ReturnType;
  if (const CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(D)) {
    ASTContext &CurrContext = AC.getASTContext();
    ReturnType = Constructor->getThisType(CurrContext)->getPointeeType();
  } else
    ReturnType = D->getCallResultType();

  if (D->hasAttr<ReturnTypestateAttr>()) {
    const ReturnTypestateAttr *RTSAttr = D->getAttr<ReturnTypestateAttr>();

    const CXXRecordDecl *RD = ReturnType->getAsCXXRecordDecl();
    if (!RD || !RD->hasAttr<ConsumableAttr>()) {
      // FIXME: This should be removed when template instantiation propagates
      //        attributes at template specialization definition, not
      //        declaration. When it is removed the test needs to be enabled
      //        in SemaDeclAttr.cpp.
      WarningsHandler.warnReturnTypestateForUnconsumableType(
          RTSAttr->getLocation(), ReturnType.getAsString());
      ExpectedReturnState = CS_None;
    } else
      ExpectedReturnState = mapReturnTypestateAttrState(RTSAttr);
  } else if (isConsumableType(ReturnType))
    ExpectedReturnState = mapConsumableAttrState(ReturnType);
  else
    ExpectedReturnState = CS_None;
}

bool ConsumedAnalyzer::splitState(const CFGBlock *CurrBlock,
                                  const ConsumedStmtVisitor &Visitor) {
  
  OwningPtr<ConsumedStateMap> FalseStates(new ConsumedStateMap(*CurrStates));
  PropagationInfo PInfo;
  
  if (const IfStmt *IfNode =
    dyn_cast_or_null<IfStmt>(CurrBlock->getTerminator().getStmt())) {
    
    const Stmt *Cond = IfNode->getCond();
    
    PInfo = Visitor.getInfo(Cond);
    if (!PInfo.isValid() && isa<BinaryOperator>(Cond))
      PInfo = Visitor.getInfo(cast<BinaryOperator>(Cond)->getRHS());
    
    if (PInfo.isVarTest()) {
      CurrStates->setSource(Cond);
      FalseStates->setSource(Cond);
      splitVarStateForIf(IfNode, PInfo.getVarTest(), CurrStates,
                         FalseStates.get());
      
    } else if (PInfo.isBinTest()) {
      CurrStates->setSource(PInfo.testSourceNode());
      FalseStates->setSource(PInfo.testSourceNode());
      splitVarStateForIfBinOp(PInfo, CurrStates, FalseStates.get());
      
    } else {
      return false;
    }
    
  } else if (const BinaryOperator *BinOp =
    dyn_cast_or_null<BinaryOperator>(CurrBlock->getTerminator().getStmt())) {
    
    PInfo = Visitor.getInfo(BinOp->getLHS());
    if (!PInfo.isVarTest()) {
      if ((BinOp = dyn_cast_or_null<BinaryOperator>(BinOp->getLHS()))) {
        PInfo = Visitor.getInfo(BinOp->getRHS());
        
        if (!PInfo.isVarTest())
          return false;
        
      } else {
        return false;
      }
    }
    
    CurrStates->setSource(BinOp);
    FalseStates->setSource(BinOp);
    
    const VarTestResult &Test = PInfo.getVarTest();
    ConsumedState VarState = CurrStates->getState(Test.Var);
    
    if (BinOp->getOpcode() == BO_LAnd) {
      if (VarState == CS_Unknown)
        CurrStates->setState(Test.Var, Test.TestsFor);
      else if (VarState == invertConsumedUnconsumed(Test.TestsFor))
        CurrStates->markUnreachable();
      
    } else if (BinOp->getOpcode() == BO_LOr) {
      if (VarState == CS_Unknown)
        FalseStates->setState(Test.Var,
                              invertConsumedUnconsumed(Test.TestsFor));
      else if (VarState == Test.TestsFor)
        FalseStates->markUnreachable();
    }
    
  } else {
    return false;
  }
  
  CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin();
  
  if (*SI)
    BlockInfo.addInfo(*SI, CurrStates);
  else
    delete CurrStates;
    
  if (*++SI)
    BlockInfo.addInfo(*SI, FalseStates.take());
  
  CurrStates = NULL;
  return true;
}

void ConsumedAnalyzer::run(AnalysisDeclContext &AC) {
  const FunctionDecl *D = dyn_cast_or_null<FunctionDecl>(AC.getDecl());
  if (!D)
    return;
  
  CFG *CFGraph = AC.getCFG();
  if (!CFGraph)
    return;

  determineExpectedReturnState(AC, D);

  PostOrderCFGView *SortedGraph = AC.getAnalysis<PostOrderCFGView>();
  // AC.getCFG()->viewCFG(LangOptions());
  
  BlockInfo = ConsumedBlockInfo(CFGraph->getNumBlockIDs(), SortedGraph);
  
  CurrStates = new ConsumedStateMap();
  ConsumedStmtVisitor Visitor(AC, *this, CurrStates);
  
  // Add all trackable parameters to the state map.
  for (FunctionDecl::param_const_iterator PI = D->param_begin(),
       PE = D->param_end(); PI != PE; ++PI) {
    Visitor.VisitParmVarDecl(*PI);
  }
  
  // Visit all of the function's basic blocks.
  for (PostOrderCFGView::iterator I = SortedGraph->begin(),
       E = SortedGraph->end(); I != E; ++I) {
    
    const CFGBlock *CurrBlock = *I;
    
    if (CurrStates == NULL)
      CurrStates = BlockInfo.getInfo(CurrBlock);
    
    if (!CurrStates) {
      continue;
      
    } else if (!CurrStates->isReachable()) {
      delete CurrStates;
      CurrStates = NULL;
      continue;
    }
    
    Visitor.reset(CurrStates);
    
    // Visit all of the basic block's statements.
    for (CFGBlock::const_iterator BI = CurrBlock->begin(),
         BE = CurrBlock->end(); BI != BE; ++BI) {
      
      switch (BI->getKind()) {
      case CFGElement::Statement:
        Visitor.Visit(BI->castAs<CFGStmt>().getStmt());
        break;
        
      case CFGElement::TemporaryDtor: {
        const CFGTemporaryDtor DTor = BI->castAs<CFGTemporaryDtor>();
        const CXXBindTemporaryExpr *BTE = DTor.getBindTemporaryExpr();
        
        Visitor.checkCallability(PropagationInfo(BTE),
                                 DTor.getDestructorDecl(AC.getASTContext()),
                                 BTE->getExprLoc());
        break;
      }
      
      case CFGElement::AutomaticObjectDtor: {
        const CFGAutomaticObjDtor DTor = BI->castAs<CFGAutomaticObjDtor>();
        SourceLocation Loc = DTor.getTriggerStmt()->getLocEnd();
        const VarDecl *Var = DTor.getVarDecl();
        
        Visitor.checkCallability(PropagationInfo(Var),
                                 DTor.getDestructorDecl(AC.getASTContext()),
                                 Loc);
        break;
      }
      
      default:
        break;
      }
    }
    
    CurrStates->clearTemporaries();
    
    // TODO: Handle other forms of branching with precision, including while-
    //       and for-loops. (Deferred)
    if (!splitState(CurrBlock, Visitor)) {
      CurrStates->setSource(NULL);
      
      if (CurrBlock->succ_size() > 1 ||
          (CurrBlock->succ_size() == 1 &&
           (*CurrBlock->succ_begin())->pred_size() > 1)) {
        
        bool OwnershipTaken = false;
        
        for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
             SE = CurrBlock->succ_end(); SI != SE; ++SI) {
          
          if (*SI == NULL) continue;
          
          if (BlockInfo.isBackEdge(CurrBlock, *SI)) {
            BlockInfo.borrowInfo(*SI)->intersectAtLoopHead(*SI, CurrBlock,
                                                           CurrStates,
                                                           WarningsHandler);
            
            if (BlockInfo.allBackEdgesVisited(*SI, CurrBlock))
              BlockInfo.discardInfo(*SI);
          } else {
            BlockInfo.addInfo(*SI, CurrStates, OwnershipTaken);
          }
        }
        
        if (!OwnershipTaken)
          delete CurrStates;
        
        CurrStates = NULL;
      }
    }
    
    if (CurrBlock == &AC.getCFG()->getExit() &&
        D->getCallResultType()->isVoidType())
      CurrStates->checkParamsForReturnTypestate(D->getLocation(),
                                                WarningsHandler);
  } // End of block iterator.
  
  // Delete the last existing state map.
  delete CurrStates;
  
  WarningsHandler.emitDiagnostics();
}
}} // end namespace clang::consumed
