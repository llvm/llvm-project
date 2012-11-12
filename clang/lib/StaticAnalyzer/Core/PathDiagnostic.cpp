//===--- PathDiagnostic.cpp - Path-Specific Diagnostic Handling -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PathDiagnostic-related interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtCXX.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace ento;

bool PathDiagnosticMacroPiece::containsEvent() const {
  for (PathPieces::const_iterator I = subPieces.begin(), E = subPieces.end();
       I!=E; ++I) {
    if (isa<PathDiagnosticEventPiece>(*I))
      return true;
    if (PathDiagnosticMacroPiece *MP = dyn_cast<PathDiagnosticMacroPiece>(*I))
      if (MP->containsEvent())
        return true;
  }
  return false;
}

static StringRef StripTrailingDots(StringRef s) {
  for (StringRef::size_type i = s.size(); i != 0; --i)
    if (s[i - 1] != '.')
      return s.substr(0, i);
  return "";
}

PathDiagnosticPiece::PathDiagnosticPiece(StringRef s,
                                         Kind k, DisplayHint hint)
  : str(StripTrailingDots(s)), kind(k), Hint(hint) {}

PathDiagnosticPiece::PathDiagnosticPiece(Kind k, DisplayHint hint)
  : kind(k), Hint(hint) {}

PathDiagnosticPiece::~PathDiagnosticPiece() {}
PathDiagnosticEventPiece::~PathDiagnosticEventPiece() {}
PathDiagnosticCallPiece::~PathDiagnosticCallPiece() {}
PathDiagnosticControlFlowPiece::~PathDiagnosticControlFlowPiece() {}
PathDiagnosticMacroPiece::~PathDiagnosticMacroPiece() {}


PathPieces::~PathPieces() {}

void PathPieces::flattenTo(PathPieces &Primary, PathPieces &Current,
                           bool ShouldFlattenMacros) const {
  for (PathPieces::const_iterator I = begin(), E = end(); I != E; ++I) {
    PathDiagnosticPiece *Piece = I->getPtr();

    switch (Piece->getKind()) {
    case PathDiagnosticPiece::Call: {
      PathDiagnosticCallPiece *Call = cast<PathDiagnosticCallPiece>(Piece);
      IntrusiveRefCntPtr<PathDiagnosticEventPiece> CallEnter =
        Call->getCallEnterEvent();
      if (CallEnter)
        Current.push_back(CallEnter);
      Call->path.flattenTo(Primary, Primary, ShouldFlattenMacros);
      IntrusiveRefCntPtr<PathDiagnosticEventPiece> callExit =
        Call->getCallExitEvent();
      if (callExit)
        Current.push_back(callExit);
      break;
    }
    case PathDiagnosticPiece::Macro: {
      PathDiagnosticMacroPiece *Macro = cast<PathDiagnosticMacroPiece>(Piece);
      if (ShouldFlattenMacros) {
        Macro->subPieces.flattenTo(Primary, Primary, ShouldFlattenMacros);
      } else {
        Current.push_back(Piece);
        PathPieces NewPath;
        Macro->subPieces.flattenTo(Primary, NewPath, ShouldFlattenMacros);
        // FIXME: This probably shouldn't mutate the original path piece.
        Macro->subPieces = NewPath;
      }
      break;
    }
    case PathDiagnosticPiece::Event:
    case PathDiagnosticPiece::ControlFlow:
      Current.push_back(Piece);
      break;
    }
  }
}


PathDiagnostic::~PathDiagnostic() {}

PathDiagnostic::PathDiagnostic(const Decl *declWithIssue,
                               StringRef bugtype, StringRef verboseDesc,
                               StringRef shortDesc, StringRef category)
  : DeclWithIssue(declWithIssue),
    BugType(StripTrailingDots(bugtype)),
    VerboseDesc(StripTrailingDots(verboseDesc)),
    ShortDesc(StripTrailingDots(shortDesc)),
    Category(StripTrailingDots(category)),
    path(pathImpl) {}

void PathDiagnosticConsumer::anchor() { }

PathDiagnosticConsumer::~PathDiagnosticConsumer() {
  // Delete the contents of the FoldingSet if it isn't empty already.
  for (llvm::FoldingSet<PathDiagnostic>::iterator it =
       Diags.begin(), et = Diags.end() ; it != et ; ++it) {
    delete &*it;
  }
}

void PathDiagnosticConsumer::HandlePathDiagnostic(PathDiagnostic *D) {
  llvm::OwningPtr<PathDiagnostic> OwningD(D);
  
  if (!D || D->path.empty())
    return;
  
  // We need to flatten the locations (convert Stmt* to locations) because
  // the referenced statements may be freed by the time the diagnostics
  // are emitted.
  D->flattenLocations();

  // If the PathDiagnosticConsumer does not support diagnostics that
  // cross file boundaries, prune out such diagnostics now.
  if (!supportsCrossFileDiagnostics()) {
    // Verify that the entire path is from the same FileID.
    FileID FID;
    const SourceManager &SMgr = (*D->path.begin())->getLocation().getManager();
    llvm::SmallVector<const PathPieces *, 5> WorkList;
    WorkList.push_back(&D->path);

    while (!WorkList.empty()) {
      const PathPieces &path = *WorkList.back();
      WorkList.pop_back();

      for (PathPieces::const_iterator I = path.begin(), E = path.end();
           I != E; ++I) {
        const PathDiagnosticPiece *piece = I->getPtr();
        FullSourceLoc L = piece->getLocation().asLocation().getExpansionLoc();
      
        if (FID.isInvalid()) {
          FID = SMgr.getFileID(L);
        } else if (SMgr.getFileID(L) != FID)
          return; // FIXME: Emit a warning?
      
        // Check the source ranges.
        ArrayRef<SourceRange> Ranges = piece->getRanges();
        for (ArrayRef<SourceRange>::iterator I = Ranges.begin(),
                                             E = Ranges.end(); I != E; ++I) {
          SourceLocation L = SMgr.getExpansionLoc(I->getBegin());
          if (!L.isFileID() || SMgr.getFileID(L) != FID)
            return; // FIXME: Emit a warning?
          L = SMgr.getExpansionLoc(I->getEnd());
          if (!L.isFileID() || SMgr.getFileID(L) != FID)
            return; // FIXME: Emit a warning?
        }
        
        if (const PathDiagnosticCallPiece *call =
            dyn_cast<PathDiagnosticCallPiece>(piece)) {
          WorkList.push_back(&call->path);
        }
        else if (const PathDiagnosticMacroPiece *macro =
                 dyn_cast<PathDiagnosticMacroPiece>(piece)) {
          WorkList.push_back(&macro->subPieces);
        }
      }
    }
    
    if (FID.isInvalid())
      return; // FIXME: Emit a warning?
  }  

  // Profile the node to see if we already have something matching it
  llvm::FoldingSetNodeID profile;
  D->Profile(profile);
  void *InsertPos = 0;

  if (PathDiagnostic *orig = Diags.FindNodeOrInsertPos(profile, InsertPos)) {
    // Keep the PathDiagnostic with the shorter path.
    // Note, the enclosing routine is called in deterministic order, so the
    // results will be consistent between runs (no reason to break ties if the
    // size is the same).
    const unsigned orig_size = orig->full_size();
    const unsigned new_size = D->full_size();
    if (orig_size <= new_size)
      return;

    assert(orig != D);
    Diags.RemoveNode(orig);
    delete orig;
  }
  
  Diags.InsertNode(OwningD.take());
}

static llvm::Optional<bool> comparePath(const PathPieces &X,
                                        const PathPieces &Y);
static llvm::Optional<bool>
compareControlFlow(const PathDiagnosticControlFlowPiece &X,
                   const PathDiagnosticControlFlowPiece &Y) {
  FullSourceLoc XSL = X.getStartLocation().asLocation();
  FullSourceLoc YSL = Y.getStartLocation().asLocation();
  if (XSL != YSL)
    return XSL.isBeforeInTranslationUnitThan(YSL);
  FullSourceLoc XEL = X.getEndLocation().asLocation();
  FullSourceLoc YEL = Y.getEndLocation().asLocation();
  if (XEL != YEL)
    return XEL.isBeforeInTranslationUnitThan(YEL);
  return llvm::Optional<bool>();
}

static llvm::Optional<bool>
compareMacro(const PathDiagnosticMacroPiece &X,
             const PathDiagnosticMacroPiece &Y) {
  return comparePath(X.subPieces, Y.subPieces);
}

static llvm::Optional<bool>
compareCall(const PathDiagnosticCallPiece &X,
            const PathDiagnosticCallPiece &Y) {
  FullSourceLoc X_CEL = X.callEnter.asLocation();
  FullSourceLoc Y_CEL = Y.callEnter.asLocation();
  if (X_CEL != Y_CEL)
    return X_CEL.isBeforeInTranslationUnitThan(Y_CEL);
  FullSourceLoc X_CEWL = X.callEnterWithin.asLocation();
  FullSourceLoc Y_CEWL = Y.callEnterWithin.asLocation();
  if (X_CEWL != Y_CEWL)
    return X_CEWL.isBeforeInTranslationUnitThan(Y_CEWL);
  FullSourceLoc X_CRL = X.callReturn.asLocation();
  FullSourceLoc Y_CRL = Y.callReturn.asLocation();
  if (X_CRL != Y_CRL)
    return X_CRL.isBeforeInTranslationUnitThan(Y_CRL);
  return comparePath(X.path, Y.path);
}

static llvm::Optional<bool> comparePiece(const PathDiagnosticPiece &X,
                                         const PathDiagnosticPiece &Y) {
  if (X.getKind() != Y.getKind())
    return X.getKind() < Y.getKind();
  
  FullSourceLoc XL = X.getLocation().asLocation();
  FullSourceLoc YL = Y.getLocation().asLocation();
  if (XL != YL)
    return XL.isBeforeInTranslationUnitThan(YL);

  if (X.getString() != Y.getString())
    return X.getString() < Y.getString();

  if (X.getRanges().size() != Y.getRanges().size())
    return X.getRanges().size() < Y.getRanges().size();

  const SourceManager &SM = XL.getManager();
  
  for (unsigned i = 0, n = X.getRanges().size(); i < n; ++i) {
    SourceRange XR = X.getRanges()[i];
    SourceRange YR = Y.getRanges()[i];
    if (XR != YR) {
      if (XR.getBegin() != YR.getBegin())
        return SM.isBeforeInTranslationUnit(XR.getBegin(), YR.getBegin());
      return SM.isBeforeInTranslationUnit(XR.getEnd(), YR.getEnd());
    }
  }
  
  switch (X.getKind()) {
    case clang::ento::PathDiagnosticPiece::ControlFlow:
      return compareControlFlow(cast<PathDiagnosticControlFlowPiece>(X),
                                cast<PathDiagnosticControlFlowPiece>(Y));
    case clang::ento::PathDiagnosticPiece::Event:
      return llvm::Optional<bool>();
    case clang::ento::PathDiagnosticPiece::Macro:
      return compareMacro(cast<PathDiagnosticMacroPiece>(X),
                          cast<PathDiagnosticMacroPiece>(Y));
    case clang::ento::PathDiagnosticPiece::Call:
      return compareCall(cast<PathDiagnosticCallPiece>(X),
                         cast<PathDiagnosticCallPiece>(Y));
  }
  llvm_unreachable("all cases handled");
}

static llvm::Optional<bool> comparePath(const PathPieces &X,
                                        const PathPieces &Y) {
  if (X.size() != Y.size())
    return X.size() < Y.size();
  for (unsigned i = 0, n = X.size(); i != n; ++i) {
    llvm::Optional<bool> b = comparePiece(*X[i], *Y[i]);
    if (b.hasValue())
      return b.getValue();
  }
  return llvm::Optional<bool>();
}

static bool compare(const PathDiagnostic &X, const PathDiagnostic &Y) {
  FullSourceLoc XL = X.getLocation().asLocation();
  FullSourceLoc YL = Y.getLocation().asLocation();
  if (XL != YL)
    return XL.isBeforeInTranslationUnitThan(YL);
  if (X.getBugType() != Y.getBugType())
    return X.getBugType() < Y.getBugType();
  if (X.getCategory() != Y.getCategory())
    return X.getCategory() < Y.getCategory();
  if (X.getVerboseDescription() != Y.getVerboseDescription())
    return X.getVerboseDescription() < Y.getVerboseDescription();
  if (X.getShortDescription() != Y.getShortDescription())
    return X.getShortDescription() < Y.getShortDescription();
  if (X.getDeclWithIssue() != Y.getDeclWithIssue()) {
    const Decl *XD = X.getDeclWithIssue();
    if (!XD)
      return true;
    const Decl *YD = Y.getDeclWithIssue();
    if (!YD)
      return false;
    SourceLocation XDL = XD->getLocation();
    SourceLocation YDL = YD->getLocation();
    if (XDL != YDL) {
      const SourceManager &SM = XL.getManager();
      return SM.isBeforeInTranslationUnit(XDL, YDL);
    }
  }
  PathDiagnostic::meta_iterator XI = X.meta_begin(), XE = X.meta_end();
  PathDiagnostic::meta_iterator YI = Y.meta_begin(), YE = Y.meta_end();
  if (XE - XI != YE - YI)
    return (XE - XI) < (YE - YI);
  for ( ; XI != XE ; ++XI, ++YI) {
    if (*XI != *YI)
      return (*XI) < (*YI);
  }
  llvm::Optional<bool> b = comparePath(X.path, Y.path);
  assert(b.hasValue());
  return b.getValue();
}

namespace {
struct CompareDiagnostics {
  // Compare if 'X' is "<" than 'Y'.
  bool operator()(const PathDiagnostic *X, const PathDiagnostic *Y) const {
    if (X == Y)
      return false;
    return compare(*X, *Y);
  }
};
}

void PathDiagnosticConsumer::FlushDiagnostics(
                                     PathDiagnosticConsumer::FilesMade *Files) {
  if (flushed)
    return;
  
  flushed = true;
  
  std::vector<const PathDiagnostic *> BatchDiags;
  for (llvm::FoldingSet<PathDiagnostic>::iterator it = Diags.begin(),
       et = Diags.end(); it != et; ++it) {
    const PathDiagnostic *D = &*it;
    BatchDiags.push_back(D);
  }

  // Sort the diagnostics so that they are always emitted in a deterministic
  // order.
  if (!BatchDiags.empty())
    std::sort(BatchDiags.begin(), BatchDiags.end(), CompareDiagnostics());
  
  FlushDiagnosticsImpl(BatchDiags, Files);

  // Delete the flushed diagnostics.
  for (std::vector<const PathDiagnostic *>::iterator it = BatchDiags.begin(),
       et = BatchDiags.end(); it != et; ++it) {
    const PathDiagnostic *D = *it;
    delete D;
  }
  
  // Clear out the FoldingSet.
  Diags.clear();
}

void PathDiagnosticConsumer::FilesMade::addDiagnostic(const PathDiagnostic &PD,
                                                      StringRef ConsumerName,
                                                      StringRef FileName) {
  llvm::FoldingSetNodeID NodeID;
  NodeID.Add(PD);
  void *InsertPos;
  PDFileEntry *Entry = FindNodeOrInsertPos(NodeID, InsertPos);
  if (!Entry) {
    Entry = Alloc.Allocate<PDFileEntry>();
    Entry = new (Entry) PDFileEntry(NodeID);
    InsertNode(Entry, InsertPos);
  }
  
  // Allocate persistent storage for the file name.
  char *FileName_cstr = (char*) Alloc.Allocate(FileName.size(), 1);
  memcpy(FileName_cstr, FileName.data(), FileName.size());

  Entry->files.push_back(std::make_pair(ConsumerName,
                                        StringRef(FileName_cstr,
                                                  FileName.size())));
}

PathDiagnosticConsumer::PDFileEntry::ConsumerFiles *
PathDiagnosticConsumer::FilesMade::getFiles(const PathDiagnostic &PD) {
  llvm::FoldingSetNodeID NodeID;
  NodeID.Add(PD);
  void *InsertPos;
  PDFileEntry *Entry = FindNodeOrInsertPos(NodeID, InsertPos);
  if (!Entry)
    return 0;
  return &Entry->files;
}

//===----------------------------------------------------------------------===//
// PathDiagnosticLocation methods.
//===----------------------------------------------------------------------===//

static SourceLocation getValidSourceLocation(const Stmt* S,
                                             LocationOrAnalysisDeclContext LAC,
                                             bool UseEnd = false) {
  SourceLocation L = UseEnd ? S->getLocEnd() : S->getLocStart();
  assert(!LAC.isNull() && "A valid LocationContext or AnalysisDeclContext should "
                          "be passed to PathDiagnosticLocation upon creation.");

  // S might be a temporary statement that does not have a location in the
  // source code, so find an enclosing statement and use its location.
  if (!L.isValid()) {

    AnalysisDeclContext *ADC;
    if (LAC.is<const LocationContext*>())
      ADC = LAC.get<const LocationContext*>()->getAnalysisDeclContext();
    else
      ADC = LAC.get<AnalysisDeclContext*>();

    ParentMap &PM = ADC->getParentMap();

    const Stmt *Parent = S;
    do {
      Parent = PM.getParent(Parent);

      // In rare cases, we have implicit top-level expressions,
      // such as arguments for implicit member initializers.
      // In this case, fall back to the start of the body (even if we were
      // asked for the statement end location).
      if (!Parent) {
        const Stmt *Body = ADC->getBody();
        if (Body)
          L = Body->getLocStart();
        else
          L = ADC->getDecl()->getLocEnd();
        break;
      }

      L = UseEnd ? Parent->getLocEnd() : Parent->getLocStart();
    } while (!L.isValid());
  }

  return L;
}

static PathDiagnosticLocation
getLocationForCaller(const StackFrameContext *SFC,
                     const LocationContext *CallerCtx,
                     const SourceManager &SM) {
  const CFGBlock &Block = *SFC->getCallSiteBlock();
  CFGElement Source = Block[SFC->getIndex()];

  switch (Source.getKind()) {
  case CFGElement::Invalid:
    llvm_unreachable("Invalid CFGElement");
  case CFGElement::Statement:
    return PathDiagnosticLocation(cast<CFGStmt>(Source).getStmt(),
                                  SM, CallerCtx);
  case CFGElement::Initializer: {
    const CFGInitializer &Init = cast<CFGInitializer>(Source);
    return PathDiagnosticLocation(Init.getInitializer()->getInit(),
                                  SM, CallerCtx);
  }
  case CFGElement::AutomaticObjectDtor: {
    const CFGAutomaticObjDtor &Dtor = cast<CFGAutomaticObjDtor>(Source);
    return PathDiagnosticLocation::createEnd(Dtor.getTriggerStmt(),
                                             SM, CallerCtx);
  }
  case CFGElement::BaseDtor:
  case CFGElement::MemberDtor: {
    const AnalysisDeclContext *CallerInfo = CallerCtx->getAnalysisDeclContext();
    if (const Stmt *CallerBody = CallerInfo->getBody())
      return PathDiagnosticLocation::createEnd(CallerBody, SM, CallerCtx);
    return PathDiagnosticLocation::create(CallerInfo->getDecl(), SM);
  }
  case CFGElement::TemporaryDtor:
    llvm_unreachable("not yet implemented!");
  }

  llvm_unreachable("Unknown CFGElement kind");
}


PathDiagnosticLocation
  PathDiagnosticLocation::createBegin(const Decl *D,
                                      const SourceManager &SM) {
  return PathDiagnosticLocation(D->getLocStart(), SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createBegin(const Stmt *S,
                                      const SourceManager &SM,
                                      LocationOrAnalysisDeclContext LAC) {
  return PathDiagnosticLocation(getValidSourceLocation(S, LAC),
                                SM, SingleLocK);
}


PathDiagnosticLocation
PathDiagnosticLocation::createEnd(const Stmt *S,
                                  const SourceManager &SM,
                                  LocationOrAnalysisDeclContext LAC) {
  if (const CompoundStmt *CS = dyn_cast<CompoundStmt>(S))
    return createEndBrace(CS, SM);
  return PathDiagnosticLocation(getValidSourceLocation(S, LAC, /*End=*/true),
                                SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createOperatorLoc(const BinaryOperator *BO,
                                            const SourceManager &SM) {
  return PathDiagnosticLocation(BO->getOperatorLoc(), SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createMemberLoc(const MemberExpr *ME,
                                          const SourceManager &SM) {
  return PathDiagnosticLocation(ME->getMemberLoc(), SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createBeginBrace(const CompoundStmt *CS,
                                           const SourceManager &SM) {
  SourceLocation L = CS->getLBracLoc();
  return PathDiagnosticLocation(L, SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createEndBrace(const CompoundStmt *CS,
                                         const SourceManager &SM) {
  SourceLocation L = CS->getRBracLoc();
  return PathDiagnosticLocation(L, SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createDeclBegin(const LocationContext *LC,
                                          const SourceManager &SM) {
  // FIXME: Should handle CXXTryStmt if analyser starts supporting C++.
  if (const CompoundStmt *CS =
        dyn_cast_or_null<CompoundStmt>(LC->getDecl()->getBody()))
    if (!CS->body_empty()) {
      SourceLocation Loc = (*CS->body_begin())->getLocStart();
      return PathDiagnosticLocation(Loc, SM, SingleLocK);
    }

  return PathDiagnosticLocation();
}

PathDiagnosticLocation
  PathDiagnosticLocation::createDeclEnd(const LocationContext *LC,
                                        const SourceManager &SM) {
  SourceLocation L = LC->getDecl()->getBodyRBrace();
  return PathDiagnosticLocation(L, SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::create(const ProgramPoint& P,
                                 const SourceManager &SMng) {

  const Stmt* S = 0;
  if (const BlockEdge *BE = dyn_cast<BlockEdge>(&P)) {
    const CFGBlock *BSrc = BE->getSrc();
    S = BSrc->getTerminatorCondition();
  }
  else if (const StmtPoint *SP = dyn_cast<StmtPoint>(&P)) {
    S = SP->getStmt();
  }
  else if (const PostImplicitCall *PIE = dyn_cast<PostImplicitCall>(&P)) {
    return PathDiagnosticLocation(PIE->getLocation(), SMng);
  }
  else if (const CallEnter *CE = dyn_cast<CallEnter>(&P)) {
    return getLocationForCaller(CE->getCalleeContext(),
                                CE->getLocationContext(),
                                SMng);
  }
  else if (const CallExitEnd *CEE = dyn_cast<CallExitEnd>(&P)) {
    return getLocationForCaller(CEE->getCalleeContext(),
                                CEE->getLocationContext(),
                                SMng);
  }
  else {
    llvm_unreachable("Unexpected ProgramPoint");
  }

  return PathDiagnosticLocation(S, SMng, P.getLocationContext());
}

PathDiagnosticLocation
  PathDiagnosticLocation::createEndOfPath(const ExplodedNode* N,
                                          const SourceManager &SM) {
  assert(N && "Cannot create a location with a null node.");

  const ExplodedNode *NI = N;
  const Stmt *S = 0;

  while (NI) {
    ProgramPoint P = NI->getLocation();
    if (const StmtPoint *PS = dyn_cast<StmtPoint>(&P))
      S = PS->getStmt();
    else if (const BlockEdge *BE = dyn_cast<BlockEdge>(&P))
      S = BE->getSrc()->getTerminator();
    if (S)
      break;
    NI = NI->succ_empty() ? 0 : *(NI->succ_begin());
  }

  if (S) {
    const LocationContext *LC = NI->getLocationContext();
    if (S->getLocStart().isValid())
      return PathDiagnosticLocation(S, SM, LC);
    return PathDiagnosticLocation(getValidSourceLocation(S, LC), SM);
  }

  return createDeclEnd(N->getLocationContext(), SM);
}

PathDiagnosticLocation PathDiagnosticLocation::createSingleLocation(
                                           const PathDiagnosticLocation &PDL) {
  FullSourceLoc L = PDL.asLocation();
  return PathDiagnosticLocation(L, L.getManager(), SingleLocK);
}

FullSourceLoc
  PathDiagnosticLocation::genLocation(SourceLocation L,
                                      LocationOrAnalysisDeclContext LAC) const {
  assert(isValid());
  // Note that we want a 'switch' here so that the compiler can warn us in
  // case we add more cases.
  switch (K) {
    case SingleLocK:
    case RangeK:
      break;
    case StmtK:
      // Defensive checking.
      if (!S)
        break;
      return FullSourceLoc(getValidSourceLocation(S, LAC),
                           const_cast<SourceManager&>(*SM));
    case DeclK:
      // Defensive checking.
      if (!D)
        break;
      return FullSourceLoc(D->getLocation(), const_cast<SourceManager&>(*SM));
  }

  return FullSourceLoc(L, const_cast<SourceManager&>(*SM));
}

PathDiagnosticRange
  PathDiagnosticLocation::genRange(LocationOrAnalysisDeclContext LAC) const {
  assert(isValid());
  // Note that we want a 'switch' here so that the compiler can warn us in
  // case we add more cases.
  switch (K) {
    case SingleLocK:
      return PathDiagnosticRange(SourceRange(Loc,Loc), true);
    case RangeK:
      break;
    case StmtK: {
      const Stmt *S = asStmt();
      switch (S->getStmtClass()) {
        default:
          break;
        case Stmt::DeclStmtClass: {
          const DeclStmt *DS = cast<DeclStmt>(S);
          if (DS->isSingleDecl()) {
            // Should always be the case, but we'll be defensive.
            return SourceRange(DS->getLocStart(),
                               DS->getSingleDecl()->getLocation());
          }
          break;
        }
          // FIXME: Provide better range information for different
          //  terminators.
        case Stmt::IfStmtClass:
        case Stmt::WhileStmtClass:
        case Stmt::DoStmtClass:
        case Stmt::ForStmtClass:
        case Stmt::ChooseExprClass:
        case Stmt::IndirectGotoStmtClass:
        case Stmt::SwitchStmtClass:
        case Stmt::BinaryConditionalOperatorClass:
        case Stmt::ConditionalOperatorClass:
        case Stmt::ObjCForCollectionStmtClass: {
          SourceLocation L = getValidSourceLocation(S, LAC);
          return SourceRange(L, L);
        }
      }
      SourceRange R = S->getSourceRange();
      if (R.isValid())
        return R;
      break;  
    }
    case DeclK:
      if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
        return MD->getSourceRange();
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        if (Stmt *Body = FD->getBody())
          return Body->getSourceRange();
      }
      else {
        SourceLocation L = D->getLocation();
        return PathDiagnosticRange(SourceRange(L, L), true);
      }
  }

  return SourceRange(Loc,Loc);
}

void PathDiagnosticLocation::flatten() {
  if (K == StmtK) {
    K = RangeK;
    S = 0;
    D = 0;
  }
  else if (K == DeclK) {
    K = SingleLocK;
    S = 0;
    D = 0;
  }
}

//===----------------------------------------------------------------------===//
// Manipulation of PathDiagnosticCallPieces.
//===----------------------------------------------------------------------===//

PathDiagnosticCallPiece *
PathDiagnosticCallPiece::construct(const ExplodedNode *N,
                                   const CallExitEnd &CE,
                                   const SourceManager &SM) {
  const Decl *caller = CE.getLocationContext()->getDecl();
  PathDiagnosticLocation pos = getLocationForCaller(CE.getCalleeContext(),
                                                    CE.getLocationContext(),
                                                    SM);
  return new PathDiagnosticCallPiece(caller, pos);
}

PathDiagnosticCallPiece *
PathDiagnosticCallPiece::construct(PathPieces &path,
                                   const Decl *caller) {
  PathDiagnosticCallPiece *C = new PathDiagnosticCallPiece(path, caller);
  path.clear();
  path.push_front(C);
  return C;
}

void PathDiagnosticCallPiece::setCallee(const CallEnter &CE,
                                        const SourceManager &SM) {
  const StackFrameContext *CalleeCtx = CE.getCalleeContext();
  Callee = CalleeCtx->getDecl();

  callEnterWithin = PathDiagnosticLocation::createBegin(Callee, SM);
  callEnter = getLocationForCaller(CalleeCtx, CE.getLocationContext(), SM);
}

IntrusiveRefCntPtr<PathDiagnosticEventPiece>
PathDiagnosticCallPiece::getCallEnterEvent() const {
  if (!Callee)
    return 0;  
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  if (isa<BlockDecl>(Callee))
    Out << "Calling anonymous block";
  else if (const NamedDecl *ND = dyn_cast<NamedDecl>(Callee))
    Out << "Calling '" << *ND << "'";
  StringRef msg = Out.str();
  if (msg.empty())
    return 0;
  return new PathDiagnosticEventPiece(callEnter, msg);
}

IntrusiveRefCntPtr<PathDiagnosticEventPiece>
PathDiagnosticCallPiece::getCallEnterWithinCallerEvent() const {
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  if (const NamedDecl *ND = dyn_cast_or_null<NamedDecl>(Caller))
    Out << "Entered call from '" << *ND << "'";
  else
    Out << "Entered call";
  StringRef msg = Out.str();
  if (msg.empty())
    return 0;
  return new PathDiagnosticEventPiece(callEnterWithin, msg);
}

IntrusiveRefCntPtr<PathDiagnosticEventPiece>
PathDiagnosticCallPiece::getCallExitEvent() const {
  if (NoExit)
    return 0;
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  if (!CallStackMessage.empty())
    Out << CallStackMessage;
  else if (const NamedDecl *ND = dyn_cast_or_null<NamedDecl>(Callee))
    Out << "Returning from '" << *ND << "'";
  else
    Out << "Returning to caller";
  return new PathDiagnosticEventPiece(callReturn, Out.str());
}

static void compute_path_size(const PathPieces &pieces, unsigned &size) {
  for (PathPieces::const_iterator it = pieces.begin(),
                                  et = pieces.end(); it != et; ++it) {
    const PathDiagnosticPiece *piece = it->getPtr();
    if (const PathDiagnosticCallPiece *cp = 
        dyn_cast<PathDiagnosticCallPiece>(piece)) {
      compute_path_size(cp->path, size);
    }
    else
      ++size;
  }
}

unsigned PathDiagnostic::full_size() {
  unsigned size = 0;
  compute_path_size(path, size);
  return size;
}

//===----------------------------------------------------------------------===//
// FoldingSet profiling methods.
//===----------------------------------------------------------------------===//

void PathDiagnosticLocation::Profile(llvm::FoldingSetNodeID &ID) const {
  ID.AddInteger(Range.getBegin().getRawEncoding());
  ID.AddInteger(Range.getEnd().getRawEncoding());
  ID.AddInteger(Loc.getRawEncoding());
  return;
}

void PathDiagnosticPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  ID.AddInteger((unsigned) getKind());
  ID.AddString(str);
  // FIXME: Add profiling support for code hints.
  ID.AddInteger((unsigned) getDisplayHint());
  ArrayRef<SourceRange> Ranges = getRanges();
  for (ArrayRef<SourceRange>::iterator I = Ranges.begin(), E = Ranges.end();
                                        I != E; ++I) {
    ID.AddInteger(I->getBegin().getRawEncoding());
    ID.AddInteger(I->getEnd().getRawEncoding());
  }  
}

void PathDiagnosticCallPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  PathDiagnosticPiece::Profile(ID);
  for (PathPieces::const_iterator it = path.begin(), 
       et = path.end(); it != et; ++it) {
    ID.Add(**it);
  }
}

void PathDiagnosticSpotPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  PathDiagnosticPiece::Profile(ID);
  ID.Add(Pos);
}

void PathDiagnosticControlFlowPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  PathDiagnosticPiece::Profile(ID);
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    ID.Add(*I);
}

void PathDiagnosticMacroPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  PathDiagnosticSpotPiece::Profile(ID);
  for (PathPieces::const_iterator I = subPieces.begin(), E = subPieces.end();
       I != E; ++I)
    ID.Add(**I);
}

void PathDiagnostic::Profile(llvm::FoldingSetNodeID &ID) const {
  ID.Add(getLocation());
  ID.AddString(BugType);
  ID.AddString(VerboseDesc);
  ID.AddString(Category);
}

void PathDiagnostic::FullProfile(llvm::FoldingSetNodeID &ID) const {
  Profile(ID);
  for (PathPieces::const_iterator I = path.begin(), E = path.end(); I != E; ++I)
    ID.Add(**I);
  for (meta_iterator I = meta_begin(), E = meta_end(); I != E; ++I)
    ID.AddString(*I);
}

StackHintGenerator::~StackHintGenerator() {}

std::string StackHintGeneratorForSymbol::getMessage(const ExplodedNode *N){
  ProgramPoint P = N->getLocation();
  const CallExitEnd *CExit = dyn_cast<CallExitEnd>(&P);
  assert(CExit && "Stack Hints should be constructed at CallExitEnd points.");

  // FIXME: Use CallEvent to abstract this over all calls.
  const Stmt *CallSite = CExit->getCalleeContext()->getCallSite();
  const CallExpr *CE = dyn_cast_or_null<CallExpr>(CallSite);
  if (!CE)
    return "";

  if (!N)
    return getMessageForSymbolNotFound();

  // Check if one of the parameters are set to the interesting symbol.
  ProgramStateRef State = N->getState();
  const LocationContext *LCtx = N->getLocationContext();
  unsigned ArgIndex = 0;
  for (CallExpr::const_arg_iterator I = CE->arg_begin(),
                                    E = CE->arg_end(); I != E; ++I, ++ArgIndex){
    SVal SV = State->getSVal(*I, LCtx);

    // Check if the variable corresponding to the symbol is passed by value.
    SymbolRef AS = SV.getAsLocSymbol();
    if (AS == Sym) {
      return getMessageForArg(*I, ArgIndex);
    }

    // Check if the parameter is a pointer to the symbol.
    if (const loc::MemRegionVal *Reg = dyn_cast<loc::MemRegionVal>(&SV)) {
      SVal PSV = State->getSVal(Reg->getRegion());
      SymbolRef AS = PSV.getAsLocSymbol();
      if (AS == Sym) {
        return getMessageForArg(*I, ArgIndex);
      }
    }
  }

  // Check if we are returning the interesting symbol.
  SVal SV = State->getSVal(CE, LCtx);
  SymbolRef RetSym = SV.getAsLocSymbol();
  if (RetSym == Sym) {
    return getMessageForReturn(CE);
  }

  return getMessageForSymbolNotFound();
}

std::string StackHintGeneratorForSymbol::getMessageForArg(const Expr *ArgE,
                                                          unsigned ArgIndex) {
  // Printed parameters start at 1, not 0.
  ++ArgIndex;

  SmallString<200> buf;
  llvm::raw_svector_ostream os(buf);

  os << Msg << " via " << ArgIndex << llvm::getOrdinalSuffix(ArgIndex)
     << " parameter";

  return os.str();
}
