//== AnalysisDeclContext.cpp - Analysis context for Path Sens analysis -*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines AnalysisDeclContext, a class that manages the analysis context
// data for path sensitive analysis.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/AnalysisContext.h"
#include "BodyFarm.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Analyses/CFGReachabilityAnalysis.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/Analyses/PseudoConstantAnalysis.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CFGStmtMap.h"
#include "clang/Analysis/Support/BumpVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

typedef llvm::DenseMap<const void *, ManagedAnalysis *> ManagedAnalysisMap;

AnalysisDeclContext::AnalysisDeclContext(AnalysisDeclContextManager *Mgr,
                                         const Decl *d,
                                         const CFG::BuildOptions &buildOptions)
  : Manager(Mgr),
    D(d),
    cfgBuildOptions(buildOptions),
    forcedBlkExprs(nullptr),
    builtCFG(false),
    builtCompleteCFG(false),
    ReferencedBlockVars(nullptr),
    ManagedAnalyses(nullptr)
{  
  cfgBuildOptions.forcedBlkExprs = &forcedBlkExprs;
}

AnalysisDeclContext::AnalysisDeclContext(AnalysisDeclContextManager *Mgr,
                                         const Decl *d)
: Manager(Mgr),
  D(d),
  forcedBlkExprs(nullptr),
  builtCFG(false),
  builtCompleteCFG(false),
  ReferencedBlockVars(nullptr),
  ManagedAnalyses(nullptr)
{  
  cfgBuildOptions.forcedBlkExprs = &forcedBlkExprs;
}

AnalysisDeclContextManager::AnalysisDeclContextManager(bool useUnoptimizedCFG,
                                                       bool addImplicitDtors,
                                                       bool addInitializers,
                                                       bool addTemporaryDtors,
                                                       bool synthesizeBodies,
                                                       bool addStaticInitBranch,
                                                       bool addCXXNewAllocator,
                                                       CodeInjector *injector)
  : Injector(injector), SynthesizeBodies(synthesizeBodies)
{
  cfgBuildOptions.PruneTriviallyFalseEdges = !useUnoptimizedCFG;
  cfgBuildOptions.AddImplicitDtors = addImplicitDtors;
  cfgBuildOptions.AddInitializers = addInitializers;
  cfgBuildOptions.AddTemporaryDtors = addTemporaryDtors;
  cfgBuildOptions.AddStaticInitBranches = addStaticInitBranch;
  cfgBuildOptions.AddCXXNewAllocator = addCXXNewAllocator;
}

void AnalysisDeclContextManager::clear() {
  llvm::DeleteContainerSeconds(Contexts);
}

static BodyFarm &getBodyFarm(ASTContext &C, CodeInjector *injector = nullptr) {
  static BodyFarm *BF = new BodyFarm(C, injector);
  return *BF;
}

Stmt *AnalysisDeclContext::getBody(bool &IsAutosynthesized) const {
  IsAutosynthesized = false;
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    Stmt *Body = FD->getBody();
    if (!Body && Manager && Manager->synthesizeBodies()) {
      Body = getBodyFarm(getASTContext(), Manager->Injector.get()).getBody(FD);
      if (Body)
        IsAutosynthesized = true;
    }
    return Body;
  }
  else if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    Stmt *Body = MD->getBody();
    if (!Body && Manager && Manager->synthesizeBodies()) {
      Body = getBodyFarm(getASTContext(), Manager->Injector.get()).getBody(MD);
      if (Body)
        IsAutosynthesized = true;
    }
    return Body;
  } else if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->getBody();
  else if (const FunctionTemplateDecl *FunTmpl
           = dyn_cast_or_null<FunctionTemplateDecl>(D))
    return FunTmpl->getTemplatedDecl()->getBody();

  llvm_unreachable("unknown code decl");
}

Stmt *AnalysisDeclContext::getBody() const {
  bool Tmp;
  return getBody(Tmp);
}

bool AnalysisDeclContext::isBodyAutosynthesized() const {
  bool Tmp;
  getBody(Tmp);
  return Tmp;
}

bool AnalysisDeclContext::isBodyAutosynthesizedFromModelFile() const {
  bool Tmp;
  Stmt *Body = getBody(Tmp);
  return Tmp && Body->getLocStart().isValid();
}


const ImplicitParamDecl *AnalysisDeclContext::getSelfDecl() const {
  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->getSelfDecl();
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D)) {
    // See if 'self' was captured by the block.
    for (const auto &I : BD->captures()) {
      const VarDecl *VD = I.getVariable();
      if (VD->getName() == "self")
        return dyn_cast<ImplicitParamDecl>(VD);
    }    
  }

  auto *CXXMethod = dyn_cast<CXXMethodDecl>(D);
  if (!CXXMethod)
    return nullptr;

  const CXXRecordDecl *parent = CXXMethod->getParent();
  if (!parent->isLambda())
    return nullptr;

  for (const LambdaCapture &LC : parent->captures()) {
    if (!LC.capturesVariable())
      continue;

    VarDecl *VD = LC.getCapturedVar();
    if (VD->getName() == "self")
      return dyn_cast<ImplicitParamDecl>(VD);
  }

  return nullptr;
}

void AnalysisDeclContext::registerForcedBlockExpression(const Stmt *stmt) {
  if (!forcedBlkExprs)
    forcedBlkExprs = new CFG::BuildOptions::ForcedBlkExprs();
  // Default construct an entry for 'stmt'.
  if (const Expr *e = dyn_cast<Expr>(stmt))
    stmt = e->IgnoreParens();
  (void) (*forcedBlkExprs)[stmt];
}

const CFGBlock *
AnalysisDeclContext::getBlockForRegisteredExpression(const Stmt *stmt) {
  assert(forcedBlkExprs);
  if (const Expr *e = dyn_cast<Expr>(stmt))
    stmt = e->IgnoreParens();
  CFG::BuildOptions::ForcedBlkExprs::const_iterator itr = 
    forcedBlkExprs->find(stmt);
  assert(itr != forcedBlkExprs->end());
  return itr->second;
}

/// Add each synthetic statement in the CFG to the parent map, using the
/// source statement's parent.
static void addParentsForSyntheticStmts(const CFG *TheCFG, ParentMap &PM) {
  if (!TheCFG)
    return;

  for (CFG::synthetic_stmt_iterator I = TheCFG->synthetic_stmt_begin(),
                                    E = TheCFG->synthetic_stmt_end();
       I != E; ++I) {
    PM.setParent(I->first, PM.getParent(I->second));
  }
}

CFG *AnalysisDeclContext::getCFG() {
  if (!cfgBuildOptions.PruneTriviallyFalseEdges)
    return getUnoptimizedCFG();

  if (!builtCFG) {
    cfg = CFG::buildCFG(D, getBody(), &D->getASTContext(), cfgBuildOptions);
    // Even when the cfg is not successfully built, we don't
    // want to try building it again.
    builtCFG = true;

    if (PM)
      addParentsForSyntheticStmts(cfg.get(), *PM);

    // The Observer should only observe one build of the CFG.
    getCFGBuildOptions().Observer = nullptr;
  }
  return cfg.get();
}

CFG *AnalysisDeclContext::getUnoptimizedCFG() {
  if (!builtCompleteCFG) {
    SaveAndRestore<bool> NotPrune(cfgBuildOptions.PruneTriviallyFalseEdges,
                                  false);
    completeCFG =
        CFG::buildCFG(D, getBody(), &D->getASTContext(), cfgBuildOptions);
    // Even when the cfg is not successfully built, we don't
    // want to try building it again.
    builtCompleteCFG = true;

    if (PM)
      addParentsForSyntheticStmts(completeCFG.get(), *PM);

    // The Observer should only observe one build of the CFG.
    getCFGBuildOptions().Observer = nullptr;
  }
  return completeCFG.get();
}

CFGStmtMap *AnalysisDeclContext::getCFGStmtMap() {
  if (cfgStmtMap)
    return cfgStmtMap.get();
  
  if (CFG *c = getCFG()) {
    cfgStmtMap.reset(CFGStmtMap::Build(c, &getParentMap()));
    return cfgStmtMap.get();
  }

  return nullptr;
}

CFGReverseBlockReachabilityAnalysis *AnalysisDeclContext::getCFGReachablityAnalysis() {
  if (CFA)
    return CFA.get();
  
  if (CFG *c = getCFG()) {
    CFA.reset(new CFGReverseBlockReachabilityAnalysis(*c));
    return CFA.get();
  }

  return nullptr;
}

void AnalysisDeclContext::dumpCFG(bool ShowColors) {
    getCFG()->dump(getASTContext().getLangOpts(), ShowColors);
}

ParentMap &AnalysisDeclContext::getParentMap() {
  if (!PM) {
    PM.reset(new ParentMap(getBody()));
    if (const CXXConstructorDecl *C = dyn_cast<CXXConstructorDecl>(getDecl())) {
      for (const auto *I : C->inits()) {
        PM->addStmt(I->getInit());
      }
    }
    if (builtCFG)
      addParentsForSyntheticStmts(getCFG(), *PM);
    if (builtCompleteCFG)
      addParentsForSyntheticStmts(getUnoptimizedCFG(), *PM);
  }
  return *PM;
}

PseudoConstantAnalysis *AnalysisDeclContext::getPseudoConstantAnalysis() {
  if (!PCA)
    PCA.reset(new PseudoConstantAnalysis(getBody()));
  return PCA.get();
}

AnalysisDeclContext *AnalysisDeclContextManager::getContext(const Decl *D) {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // Calling 'hasBody' replaces 'FD' in place with the FunctionDecl
    // that has the body.
    FD->hasBody(FD);
    D = FD;
  }

  AnalysisDeclContext *&AC = Contexts[D];
  if (!AC)
    AC = new AnalysisDeclContext(this, D, cfgBuildOptions);
  return AC;
}

const StackFrameContext *
AnalysisDeclContext::getStackFrame(LocationContext const *Parent, const Stmt *S,
                               const CFGBlock *Blk, unsigned Idx) {
  return getLocationContextManager().getStackFrame(this, Parent, S, Blk, Idx);
}

const BlockInvocationContext *
AnalysisDeclContext::getBlockInvocationContext(const LocationContext *parent,
                                               const clang::BlockDecl *BD,
                                               const void *ContextData) {
  return getLocationContextManager().getBlockInvocationContext(this, parent,
                                                               BD, ContextData);
}

bool AnalysisDeclContext::isInStdNamespace(const Decl *D) {
  const DeclContext *DC = D->getDeclContext()->getEnclosingNamespaceContext();
  const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(DC);
  if (!ND)
    return false;

  while (const DeclContext *Parent = ND->getParent()) {
    if (!isa<NamespaceDecl>(Parent))
      break;
    ND = cast<NamespaceDecl>(Parent);
  }

  return ND->isStdNamespace();
}

LocationContextManager & AnalysisDeclContext::getLocationContextManager() {
  assert(Manager &&
         "Cannot create LocationContexts without an AnalysisDeclContextManager!");
  return Manager->getLocationContextManager();  
}

//===----------------------------------------------------------------------===//
// FoldingSet profiling.
//===----------------------------------------------------------------------===//

void LocationContext::ProfileCommon(llvm::FoldingSetNodeID &ID,
                                    ContextKind ck,
                                    AnalysisDeclContext *ctx,
                                    const LocationContext *parent,
                                    const void *data) {
  ID.AddInteger(ck);
  ID.AddPointer(ctx);
  ID.AddPointer(parent);
  ID.AddPointer(data);
}

void StackFrameContext::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getAnalysisDeclContext(), getParent(), CallSite, Block, Index);
}

void ScopeContext::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getAnalysisDeclContext(), getParent(), Enter);
}

void BlockInvocationContext::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getAnalysisDeclContext(), getParent(), BD, ContextData);
}

//===----------------------------------------------------------------------===//
// LocationContext creation.
//===----------------------------------------------------------------------===//

template <typename LOC, typename DATA>
const LOC*
LocationContextManager::getLocationContext(AnalysisDeclContext *ctx,
                                           const LocationContext *parent,
                                           const DATA *d) {
  llvm::FoldingSetNodeID ID;
  LOC::Profile(ID, ctx, parent, d);
  void *InsertPos;

  LOC *L = cast_or_null<LOC>(Contexts.FindNodeOrInsertPos(ID, InsertPos));

  if (!L) {
    L = new LOC(ctx, parent, d);
    Contexts.InsertNode(L, InsertPos);
  }
  return L;
}

const StackFrameContext*
LocationContextManager::getStackFrame(AnalysisDeclContext *ctx,
                                      const LocationContext *parent,
                                      const Stmt *s,
                                      const CFGBlock *blk, unsigned idx) {
  llvm::FoldingSetNodeID ID;
  StackFrameContext::Profile(ID, ctx, parent, s, blk, idx);
  void *InsertPos;
  StackFrameContext *L =
   cast_or_null<StackFrameContext>(Contexts.FindNodeOrInsertPos(ID, InsertPos));
  if (!L) {
    L = new StackFrameContext(ctx, parent, s, blk, idx);
    Contexts.InsertNode(L, InsertPos);
  }
  return L;
}

const ScopeContext *
LocationContextManager::getScope(AnalysisDeclContext *ctx,
                                 const LocationContext *parent,
                                 const Stmt *s) {
  return getLocationContext<ScopeContext, Stmt>(ctx, parent, s);
}

const BlockInvocationContext *
LocationContextManager::getBlockInvocationContext(AnalysisDeclContext *ctx,
                                                  const LocationContext *parent,
                                                  const BlockDecl *BD,
                                                  const void *ContextData) {
  llvm::FoldingSetNodeID ID;
  BlockInvocationContext::Profile(ID, ctx, parent, BD, ContextData);
  void *InsertPos;
  BlockInvocationContext *L =
    cast_or_null<BlockInvocationContext>(Contexts.FindNodeOrInsertPos(ID,
                                                                    InsertPos));
  if (!L) {
    L = new BlockInvocationContext(ctx, parent, BD, ContextData);
    Contexts.InsertNode(L, InsertPos);
  }
  return L;
}

//===----------------------------------------------------------------------===//
// LocationContext methods.
//===----------------------------------------------------------------------===//

const StackFrameContext *LocationContext::getCurrentStackFrame() const {
  const LocationContext *LC = this;
  while (LC) {
    if (const StackFrameContext *SFC = dyn_cast<StackFrameContext>(LC))
      return SFC;
    LC = LC->getParent();
  }
  return nullptr;
}

bool LocationContext::inTopFrame() const {
  return getCurrentStackFrame()->inTopFrame();
}

bool LocationContext::isParentOf(const LocationContext *LC) const {
  do {
    const LocationContext *Parent = LC->getParent();
    if (Parent == this)
      return true;
    else
      LC = Parent;
  } while (LC);

  return false;
}

void LocationContext::dumpStack(raw_ostream &OS, StringRef Indent) const {
  ASTContext &Ctx = getAnalysisDeclContext()->getASTContext();
  PrintingPolicy PP(Ctx.getLangOpts());
  PP.TerseOutput = 1;

  unsigned Frame = 0;
  for (const LocationContext *LCtx = this; LCtx; LCtx = LCtx->getParent()) {
    switch (LCtx->getKind()) {
    case StackFrame:
      OS << Indent << '#' << Frame++ << ' ';
      cast<StackFrameContext>(LCtx)->getDecl()->print(OS, PP);
      OS << '\n';
      break;
    case Scope:
      OS << Indent << "    (scope)\n";
      break;
    case Block:
      OS << Indent << "    (block context: "
                   << cast<BlockInvocationContext>(LCtx)->getContextData()
                   << ")\n";
      break;
    }
  }
}

LLVM_DUMP_METHOD void LocationContext::dumpStack() const {
  dumpStack(llvm::errs());
}

//===----------------------------------------------------------------------===//
// Lazily generated map to query the external variables referenced by a Block.
//===----------------------------------------------------------------------===//

namespace {
class FindBlockDeclRefExprsVals : public StmtVisitor<FindBlockDeclRefExprsVals>{
  BumpVector<const VarDecl*> &BEVals;
  BumpVectorContext &BC;
  llvm::SmallPtrSet<const VarDecl*, 4> Visited;
  llvm::SmallPtrSet<const DeclContext*, 4> IgnoredContexts;
public:
  FindBlockDeclRefExprsVals(BumpVector<const VarDecl*> &bevals,
                            BumpVectorContext &bc)
  : BEVals(bevals), BC(bc) {}

  void VisitStmt(Stmt *S) {
    for (Stmt *Child : S->children())
      if (Child)
        Visit(Child);
  }

  void VisitDeclRefExpr(DeclRefExpr *DR) {
    // Non-local variables are also directly modified.
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
      if (!VD->hasLocalStorage()) {
        if (Visited.insert(VD).second)
          BEVals.push_back(VD, BC);
      }
    }
  }

  void VisitBlockExpr(BlockExpr *BR) {
    // Blocks containing blocks can transitively capture more variables.
    IgnoredContexts.insert(BR->getBlockDecl());
    Visit(BR->getBlockDecl()->getBody());
  }
  
  void VisitPseudoObjectExpr(PseudoObjectExpr *PE) {
    for (PseudoObjectExpr::semantics_iterator it = PE->semantics_begin(), 
         et = PE->semantics_end(); it != et; ++it) {
      Expr *Semantic = *it;
      if (OpaqueValueExpr *OVE = dyn_cast<OpaqueValueExpr>(Semantic))
        Semantic = OVE->getSourceExpr();
      Visit(Semantic);
    }
  }
};
} // end anonymous namespace

typedef BumpVector<const VarDecl*> DeclVec;

static DeclVec* LazyInitializeReferencedDecls(const BlockDecl *BD,
                                              void *&Vec,
                                              llvm::BumpPtrAllocator &A) {
  if (Vec)
    return (DeclVec*) Vec;

  BumpVectorContext BC(A);
  DeclVec *BV = (DeclVec*) A.Allocate<DeclVec>();
  new (BV) DeclVec(BC, 10);

  // Go through the capture list.
  for (const auto &CI : BD->captures()) {
    BV->push_back(CI.getVariable(), BC);
  }

  // Find the referenced global/static variables.
  FindBlockDeclRefExprsVals F(*BV, BC);
  F.Visit(BD->getBody());

  Vec = BV;
  return BV;
}

llvm::iterator_range<AnalysisDeclContext::referenced_decls_iterator>
AnalysisDeclContext::getReferencedBlockVars(const BlockDecl *BD) {
  if (!ReferencedBlockVars)
    ReferencedBlockVars = new llvm::DenseMap<const BlockDecl*,void*>();

  const DeclVec *V =
      LazyInitializeReferencedDecls(BD, (*ReferencedBlockVars)[BD], A);
  return llvm::make_range(V->begin(), V->end());
}

ManagedAnalysis *&AnalysisDeclContext::getAnalysisImpl(const void *tag) {
  if (!ManagedAnalyses)
    ManagedAnalyses = new ManagedAnalysisMap();
  ManagedAnalysisMap *M = (ManagedAnalysisMap*) ManagedAnalyses;
  return (*M)[tag];
}

//===----------------------------------------------------------------------===//
// Cleanup.
//===----------------------------------------------------------------------===//

ManagedAnalysis::~ManagedAnalysis() {}

AnalysisDeclContext::~AnalysisDeclContext() {
  delete forcedBlkExprs;
  delete ReferencedBlockVars;
  // Release the managed analyses.
  if (ManagedAnalyses) {
    ManagedAnalysisMap *M = (ManagedAnalysisMap*) ManagedAnalyses;
    llvm::DeleteContainerSeconds(*M);
    delete M;
  }
}

AnalysisDeclContextManager::~AnalysisDeclContextManager() {
  llvm::DeleteContainerSeconds(Contexts);
}

LocationContext::~LocationContext() {}

LocationContextManager::~LocationContextManager() {
  clear();
}

void LocationContextManager::clear() {
  for (llvm::FoldingSet<LocationContext>::iterator I = Contexts.begin(),
       E = Contexts.end(); I != E; ) {
    LocationContext *LC = &*I;
    ++I;
    delete LC;
  }

  Contexts.clear();
}

