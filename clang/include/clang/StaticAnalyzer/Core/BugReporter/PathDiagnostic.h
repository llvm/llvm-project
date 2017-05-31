//===--- PathDiagnostic.h - Path-Specific Diagnostic Handling ---*- C++ -*-===//
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

#ifndef LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_PATHDIAGNOSTIC_H
#define LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_PATHDIAGNOSTIC_H

#include "clang/Analysis/ProgramPoint.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerUnion.h"
#include <deque>
#include <iterator>
#include <list>
#include <string>
#include <vector>

namespace clang {
class ConditionalOperator;
class AnalysisDeclContext;
class BinaryOperator;
class CompoundStmt;
class Decl;
class LocationContext;
class MemberExpr;
class ParentMap;
class ProgramPoint;
class SourceManager;
class Stmt;
class CallExpr;

namespace ento {

class ExplodedNode;
class SymExpr;
typedef const SymExpr* SymbolRef;

//===----------------------------------------------------------------------===//
// High-level interface for handlers of path-sensitive diagnostics.
//===----------------------------------------------------------------------===//

class PathDiagnostic;

class PathDiagnosticConsumer {
public:
  class PDFileEntry : public llvm::FoldingSetNode {
  public:
    PDFileEntry(llvm::FoldingSetNodeID &NodeID) : NodeID(NodeID) {}

    typedef std::vector<std::pair<StringRef, StringRef> > ConsumerFiles;
    
    /// \brief A vector of <consumer,file> pairs.
    ConsumerFiles files;
    
    /// \brief A precomputed hash tag used for uniquing PDFileEntry objects.
    const llvm::FoldingSetNodeID NodeID;

    /// \brief Used for profiling in the FoldingSet.
    void Profile(llvm::FoldingSetNodeID &ID) { ID = NodeID; }
  };
  
  class FilesMade {
    llvm::BumpPtrAllocator Alloc;
    llvm::FoldingSet<PDFileEntry> Set;

  public:
    ~FilesMade();

    bool empty() const { return Set.empty(); }

    void addDiagnostic(const PathDiagnostic &PD,
                       StringRef ConsumerName,
                       StringRef fileName);
    
    PDFileEntry::ConsumerFiles *getFiles(const PathDiagnostic &PD);
  };

private:
  virtual void anchor();
public:
  PathDiagnosticConsumer() : flushed(false) {}
  virtual ~PathDiagnosticConsumer();

  void FlushDiagnostics(FilesMade *FilesMade);

  virtual void FlushDiagnosticsImpl(std::vector<const PathDiagnostic *> &Diags,
                                    FilesMade *filesMade) = 0;

  virtual StringRef getName() const = 0;

  void HandlePathDiagnostic(std::unique_ptr<PathDiagnostic> D);

  enum PathGenerationScheme { None, Minimal, Extensive, AlternateExtensive };
  virtual PathGenerationScheme getGenerationScheme() const { return Minimal; }
  virtual bool supportsLogicalOpControlFlow() const { return false; }
  
  /// Return true if the PathDiagnosticConsumer supports individual
  /// PathDiagnostics that span multiple files.
  virtual bool supportsCrossFileDiagnostics() const { return false; }

protected:
  bool flushed;
  llvm::FoldingSet<PathDiagnostic> Diags;
};

//===----------------------------------------------------------------------===//
// Path-sensitive diagnostics.
//===----------------------------------------------------------------------===//

class PathDiagnosticRange : public SourceRange {
public:
  bool isPoint;

  PathDiagnosticRange(SourceRange R, bool isP = false)
    : SourceRange(R), isPoint(isP) {}

  PathDiagnosticRange() : isPoint(false) {}
};

typedef llvm::PointerUnion<const LocationContext*, AnalysisDeclContext*>
                                                   LocationOrAnalysisDeclContext;

class PathDiagnosticLocation {
private:
  enum Kind { RangeK, SingleLocK, StmtK, DeclK } K;
  const Stmt *S;
  const Decl *D;
  const SourceManager *SM;
  FullSourceLoc Loc;
  PathDiagnosticRange Range;

  PathDiagnosticLocation(SourceLocation L, const SourceManager &sm,
                         Kind kind)
    : K(kind), S(nullptr), D(nullptr), SM(&sm),
      Loc(genLocation(L)), Range(genRange()) {
  }

  FullSourceLoc genLocation(
      SourceLocation L = SourceLocation(),
      LocationOrAnalysisDeclContext LAC = (AnalysisDeclContext *)nullptr) const;

  PathDiagnosticRange genRange(
      LocationOrAnalysisDeclContext LAC = (AnalysisDeclContext *)nullptr) const;

public:
  /// Create an invalid location.
  PathDiagnosticLocation()
    : K(SingleLocK), S(nullptr), D(nullptr), SM(nullptr) {}

  /// Create a location corresponding to the given statement.
  PathDiagnosticLocation(const Stmt *s,
                         const SourceManager &sm,
                         LocationOrAnalysisDeclContext lac)
    : K(s->getLocStart().isValid() ? StmtK : SingleLocK),
      S(K == StmtK ? s : nullptr),
      D(nullptr), SM(&sm),
      Loc(genLocation(SourceLocation(), lac)),
      Range(genRange(lac)) {
    assert(K == SingleLocK || S);
    assert(K == SingleLocK || Loc.isValid());
    assert(K == SingleLocK || Range.isValid());
  }

  /// Create a location corresponding to the given declaration.
  PathDiagnosticLocation(const Decl *d, const SourceManager &sm)
    : K(DeclK), S(nullptr), D(d), SM(&sm),
      Loc(genLocation()), Range(genRange()) {
    assert(D);
    assert(Loc.isValid());
    assert(Range.isValid());
  }

  /// Create a location at an explicit offset in the source.
  ///
  /// This should only be used if there are no more appropriate constructors.
  PathDiagnosticLocation(SourceLocation loc, const SourceManager &sm)
    : K(SingleLocK), S(nullptr), D(nullptr), SM(&sm), Loc(loc, sm),
      Range(genRange()) {
    assert(Loc.isValid());
    assert(Range.isValid());
  }

  /// Create a location corresponding to the given declaration.
  static PathDiagnosticLocation create(const Decl *D,
                                       const SourceManager &SM) {
    return PathDiagnosticLocation(D, SM);
  }

  /// Create a location for the beginning of the declaration.
  static PathDiagnosticLocation createBegin(const Decl *D,
                                            const SourceManager &SM);

  /// Create a location for the beginning of the statement.
  static PathDiagnosticLocation createBegin(const Stmt *S,
                                            const SourceManager &SM,
                                            const LocationOrAnalysisDeclContext LAC);

  /// Create a location for the end of the statement.
  ///
  /// If the statement is a CompoundStatement, the location will point to the
  /// closing brace instead of following it.
  static PathDiagnosticLocation createEnd(const Stmt *S,
                                          const SourceManager &SM,
                                       const LocationOrAnalysisDeclContext LAC);

  /// Create the location for the operator of the binary expression.
  /// Assumes the statement has a valid location.
  static PathDiagnosticLocation createOperatorLoc(const BinaryOperator *BO,
                                                  const SourceManager &SM);
  static PathDiagnosticLocation createConditionalColonLoc(
                                                  const ConditionalOperator *CO,
                                                  const SourceManager &SM);

  /// For member expressions, return the location of the '.' or '->'.
  /// Assumes the statement has a valid location.
  static PathDiagnosticLocation createMemberLoc(const MemberExpr *ME,
                                                const SourceManager &SM);

  /// Create a location for the beginning of the compound statement.
  /// Assumes the statement has a valid location.
  static PathDiagnosticLocation createBeginBrace(const CompoundStmt *CS,
                                                 const SourceManager &SM);

  /// Create a location for the end of the compound statement.
  /// Assumes the statement has a valid location.
  static PathDiagnosticLocation createEndBrace(const CompoundStmt *CS,
                                               const SourceManager &SM);

  /// Create a location for the beginning of the enclosing declaration body.
  /// Defaults to the beginning of the first statement in the declaration body.
  static PathDiagnosticLocation createDeclBegin(const LocationContext *LC,
                                                const SourceManager &SM);

  /// Constructs a location for the end of the enclosing declaration body.
  /// Defaults to the end of brace.
  static PathDiagnosticLocation createDeclEnd(const LocationContext *LC,
                                                   const SourceManager &SM);

  /// Create a location corresponding to the given valid ExplodedNode.
  static PathDiagnosticLocation create(const ProgramPoint& P,
                                       const SourceManager &SMng);

  /// Create a location corresponding to the next valid ExplodedNode as end
  /// of path location.
  static PathDiagnosticLocation createEndOfPath(const ExplodedNode* N,
                                                const SourceManager &SM);

  /// Convert the given location into a single kind location.
  static PathDiagnosticLocation createSingleLocation(
                                             const PathDiagnosticLocation &PDL);

  bool operator==(const PathDiagnosticLocation &X) const {
    return K == X.K && Loc == X.Loc && Range == X.Range;
  }

  bool operator!=(const PathDiagnosticLocation &X) const {
    return !(*this == X);
  }

  bool isValid() const {
    return SM != nullptr;
  }

  FullSourceLoc asLocation() const {
    return Loc;
  }

  PathDiagnosticRange asRange() const {
    return Range;
  }

  const Stmt *asStmt() const { assert(isValid()); return S; }
  const Decl *asDecl() const { assert(isValid()); return D; }

  bool hasRange() const { return K == StmtK || K == RangeK || K == DeclK; }

  void invalidate() {
    *this = PathDiagnosticLocation();
  }

  void flatten();

  const SourceManager& getManager() const { assert(isValid()); return *SM; }
  
  void Profile(llvm::FoldingSetNodeID &ID) const;

  void dump() const;

  /// \brief Given an exploded node, retrieve the statement that should be used 
  /// for the diagnostic location.
  static const Stmt *getStmt(const ExplodedNode *N);

  /// \brief Retrieve the statement corresponding to the successor node.
  static const Stmt *getNextStmt(const ExplodedNode *N);
};

class PathDiagnosticLocationPair {
private:
  PathDiagnosticLocation Start, End;
public:
  PathDiagnosticLocationPair(const PathDiagnosticLocation &start,
                             const PathDiagnosticLocation &end)
    : Start(start), End(end) {}

  const PathDiagnosticLocation &getStart() const { return Start; }
  const PathDiagnosticLocation &getEnd() const { return End; }

  void setStart(const PathDiagnosticLocation &L) { Start = L; }
  void setEnd(const PathDiagnosticLocation &L) { End = L; }

  void flatten() {
    Start.flatten();
    End.flatten();
  }
  
  void Profile(llvm::FoldingSetNodeID &ID) const {
    Start.Profile(ID);
    End.Profile(ID);
  }
};

//===----------------------------------------------------------------------===//
// Path "pieces" for path-sensitive diagnostics.
//===----------------------------------------------------------------------===//

class PathDiagnosticPiece : public RefCountedBaseVPTR {
public:
  enum Kind { ControlFlow, Event, Macro, Call };
  enum DisplayHint { Above, Below };

private:
  const std::string str;
  const Kind kind;
  const DisplayHint Hint;

  /// \brief In the containing bug report, this piece is the last piece from
  /// the main source file.
  bool LastInMainSourceFile;
  
  /// A constant string that can be used to tag the PathDiagnosticPiece,
  /// typically with the identification of the creator.  The actual pointer
  /// value is meant to be an identifier; the string itself is useful for
  /// debugging.
  StringRef Tag;

  std::vector<SourceRange> ranges;

  PathDiagnosticPiece() = delete;
  PathDiagnosticPiece(const PathDiagnosticPiece &P) = delete;
  void operator=(const PathDiagnosticPiece &P) = delete;

protected:
  PathDiagnosticPiece(StringRef s, Kind k, DisplayHint hint = Below);

  PathDiagnosticPiece(Kind k, DisplayHint hint = Below);

public:
  ~PathDiagnosticPiece() override;

  StringRef getString() const { return str; }

  /// Tag this PathDiagnosticPiece with the given C-string.
  void setTag(const char *tag) { Tag = tag; }
  
  /// Return the opaque tag (if any) on the PathDiagnosticPiece.
  const void *getTag() const { return Tag.data(); }
  
  /// Return the string representation of the tag.  This is useful
  /// for debugging.
  StringRef getTagStr() const { return Tag; }
  
  /// getDisplayHint - Return a hint indicating where the diagnostic should
  ///  be displayed by the PathDiagnosticConsumer.
  DisplayHint getDisplayHint() const { return Hint; }

  virtual PathDiagnosticLocation getLocation() const = 0;
  virtual void flattenLocations() = 0;

  Kind getKind() const { return kind; }

  void addRange(SourceRange R) {
    if (!R.isValid())
      return;
    ranges.push_back(R);
  }

  void addRange(SourceLocation B, SourceLocation E) {
    if (!B.isValid() || !E.isValid())
      return;
    ranges.push_back(SourceRange(B,E));
  }

  /// Return the SourceRanges associated with this PathDiagnosticPiece.
  ArrayRef<SourceRange> getRanges() const { return ranges; }

  virtual void Profile(llvm::FoldingSetNodeID &ID) const;

  void setAsLastInMainSourceFile() {
    LastInMainSourceFile = true;
  }

  bool isLastInMainSourceFile() const {
    return LastInMainSourceFile;
  }

  virtual void dump() const = 0;
};
  
  
class PathPieces : public std::list<IntrusiveRefCntPtr<PathDiagnosticPiece> > {
  void flattenTo(PathPieces &Primary, PathPieces &Current,
                 bool ShouldFlattenMacros) const;
public:

  PathPieces flatten(bool ShouldFlattenMacros) const {
    PathPieces Result;
    flattenTo(Result, Result, ShouldFlattenMacros);
    return Result;
  }

  void dump() const;
};

class PathDiagnosticSpotPiece : public PathDiagnosticPiece {
private:
  PathDiagnosticLocation Pos;
public:
  PathDiagnosticSpotPiece(const PathDiagnosticLocation &pos,
                          StringRef s,
                          PathDiagnosticPiece::Kind k,
                          bool addPosRange = true)
  : PathDiagnosticPiece(s, k), Pos(pos) {
    assert(Pos.isValid() && Pos.asLocation().isValid() &&
           "PathDiagnosticSpotPiece's must have a valid location.");
    if (addPosRange && Pos.hasRange()) addRange(Pos.asRange());
  }

  PathDiagnosticLocation getLocation() const override { return Pos; }
  void flattenLocations() override { Pos.flatten(); }

  void Profile(llvm::FoldingSetNodeID &ID) const override;

  static bool classof(const PathDiagnosticPiece *P) {
    return P->getKind() == Event || P->getKind() == Macro;
  }
};

/// \brief Interface for classes constructing Stack hints.
///
/// If a PathDiagnosticEvent occurs in a different frame than the final 
/// diagnostic the hints can be used to summarize the effect of the call.
class StackHintGenerator {
public:
  virtual ~StackHintGenerator() = 0;

  /// \brief Construct the Diagnostic message for the given ExplodedNode.
  virtual std::string getMessage(const ExplodedNode *N) = 0;
};

/// \brief Constructs a Stack hint for the given symbol.
///
/// The class knows how to construct the stack hint message based on
/// traversing the CallExpr associated with the call and checking if the given
/// symbol is returned or is one of the arguments.
/// The hint can be customized by redefining 'getMessageForX()' methods.
class StackHintGeneratorForSymbol : public StackHintGenerator {
private:
  SymbolRef Sym;
  std::string Msg;

public:
  StackHintGeneratorForSymbol(SymbolRef S, StringRef M) : Sym(S), Msg(M) {}
  ~StackHintGeneratorForSymbol() override {}

  /// \brief Search the call expression for the symbol Sym and dispatch the
  /// 'getMessageForX()' methods to construct a specific message.
  std::string getMessage(const ExplodedNode *N) override;

  /// Produces the message of the following form:
  ///   'Msg via Nth parameter'
  virtual std::string getMessageForArg(const Expr *ArgE, unsigned ArgIndex);
  virtual std::string getMessageForReturn(const CallExpr *CallExpr) {
    return Msg;
  }
  virtual std::string getMessageForSymbolNotFound() {
    return Msg;
  }
};

class PathDiagnosticEventPiece : public PathDiagnosticSpotPiece {
  Optional<bool> IsPrunable;

  /// If the event occurs in a different frame than the final diagnostic,
  /// supply a message that will be used to construct an extra hint on the
  /// returns from all the calls on the stack from this event to the final
  /// diagnostic.
  std::unique_ptr<StackHintGenerator> CallStackHint;

public:
  PathDiagnosticEventPiece(const PathDiagnosticLocation &pos,
                           StringRef s, bool addPosRange = true,
                           StackHintGenerator *stackHint = nullptr)
    : PathDiagnosticSpotPiece(pos, s, Event, addPosRange),
      CallStackHint(stackHint) {}

  ~PathDiagnosticEventPiece() override;

  /// Mark the diagnostic piece as being potentially prunable.  This
  /// flag may have been previously set, at which point it will not
  /// be reset unless one specifies to do so.
  void setPrunable(bool isPrunable, bool override = false) {
    if (IsPrunable.hasValue() && !override)
     return;
    IsPrunable = isPrunable;
  }

  /// Return true if the diagnostic piece is prunable.
  bool isPrunable() const {
    return IsPrunable.hasValue() ? IsPrunable.getValue() : false;
  }

  bool hasCallStackHint() { return (bool)CallStackHint; }

  /// Produce the hint for the given node. The node contains 
  /// information about the call for which the diagnostic can be generated.
  std::string getCallStackMessage(const ExplodedNode *N) {
    if (CallStackHint)
      return CallStackHint->getMessage(N);
    return "";  
  }

  void dump() const override;

  static inline bool classof(const PathDiagnosticPiece *P) {
    return P->getKind() == Event;
  }
};

class PathDiagnosticCallPiece : public PathDiagnosticPiece {
  PathDiagnosticCallPiece(const Decl *callerD,
                          const PathDiagnosticLocation &callReturnPos)
    : PathDiagnosticPiece(Call), Caller(callerD), Callee(nullptr),
      NoExit(false), callReturn(callReturnPos) {}

  PathDiagnosticCallPiece(PathPieces &oldPath, const Decl *caller)
    : PathDiagnosticPiece(Call), Caller(caller), Callee(nullptr),
      NoExit(true), path(oldPath) {}
  
  const Decl *Caller;
  const Decl *Callee;

  // Flag signifying that this diagnostic has only call enter and no matching
  // call exit.
  bool NoExit;

  // The custom string, which should appear after the call Return Diagnostic.
  // TODO: Should we allow multiple diagnostics?
  std::string CallStackMessage;

public:
  PathDiagnosticLocation callEnter;
  PathDiagnosticLocation callEnterWithin;
  PathDiagnosticLocation callReturn;  
  PathPieces path;

  ~PathDiagnosticCallPiece() override;

  const Decl *getCaller() const { return Caller; }
  
  const Decl *getCallee() const { return Callee; }
  void setCallee(const CallEnter &CE, const SourceManager &SM);
  
  bool hasCallStackMessage() { return !CallStackMessage.empty(); }
  void setCallStackMessage(StringRef st) {
    CallStackMessage = st;
  }

  PathDiagnosticLocation getLocation() const override {
    return callEnter;
  }
  
  IntrusiveRefCntPtr<PathDiagnosticEventPiece> getCallEnterEvent() const;
  IntrusiveRefCntPtr<PathDiagnosticEventPiece>
    getCallEnterWithinCallerEvent() const;
  IntrusiveRefCntPtr<PathDiagnosticEventPiece> getCallExitEvent() const;

  void flattenLocations() override {
    callEnter.flatten();
    callReturn.flatten();
    for (PathPieces::iterator I = path.begin(), 
         E = path.end(); I != E; ++I) (*I)->flattenLocations();
  }
  
  static PathDiagnosticCallPiece *construct(const ExplodedNode *N,
                                            const CallExitEnd &CE,
                                            const SourceManager &SM);
  
  static PathDiagnosticCallPiece *construct(PathPieces &pieces,
                                            const Decl *caller);

  void dump() const override;

  void Profile(llvm::FoldingSetNodeID &ID) const override;

  static inline bool classof(const PathDiagnosticPiece *P) {
    return P->getKind() == Call;
  }
};

class PathDiagnosticControlFlowPiece : public PathDiagnosticPiece {
  std::vector<PathDiagnosticLocationPair> LPairs;
public:
  PathDiagnosticControlFlowPiece(const PathDiagnosticLocation &startPos,
                                 const PathDiagnosticLocation &endPos,
                                 StringRef s)
    : PathDiagnosticPiece(s, ControlFlow) {
      LPairs.push_back(PathDiagnosticLocationPair(startPos, endPos));
    }

  PathDiagnosticControlFlowPiece(const PathDiagnosticLocation &startPos,
                                 const PathDiagnosticLocation &endPos)
    : PathDiagnosticPiece(ControlFlow) {
      LPairs.push_back(PathDiagnosticLocationPair(startPos, endPos));
    }

    ~PathDiagnosticControlFlowPiece() override;

  PathDiagnosticLocation getStartLocation() const {
    assert(!LPairs.empty() &&
           "PathDiagnosticControlFlowPiece needs at least one location.");
    return LPairs[0].getStart();
  }

  PathDiagnosticLocation getEndLocation() const {
    assert(!LPairs.empty() &&
           "PathDiagnosticControlFlowPiece needs at least one location.");
    return LPairs[0].getEnd();
  }

  void setStartLocation(const PathDiagnosticLocation &L) {
    LPairs[0].setStart(L);
  }

  void setEndLocation(const PathDiagnosticLocation &L) {
    LPairs[0].setEnd(L);
  }

  void push_back(const PathDiagnosticLocationPair &X) { LPairs.push_back(X); }

  PathDiagnosticLocation getLocation() const override {
    return getStartLocation();
  }

  typedef std::vector<PathDiagnosticLocationPair>::iterator iterator;
  iterator begin() { return LPairs.begin(); }
  iterator end()   { return LPairs.end(); }

  void flattenLocations() override {
    for (iterator I=begin(), E=end(); I!=E; ++I) I->flatten();
  }

  typedef std::vector<PathDiagnosticLocationPair>::const_iterator
          const_iterator;
  const_iterator begin() const { return LPairs.begin(); }
  const_iterator end() const   { return LPairs.end(); }

  static inline bool classof(const PathDiagnosticPiece *P) {
    return P->getKind() == ControlFlow;
  }

  void dump() const override;

  void Profile(llvm::FoldingSetNodeID &ID) const override;
};

class PathDiagnosticMacroPiece : public PathDiagnosticSpotPiece {
public:
  PathDiagnosticMacroPiece(const PathDiagnosticLocation &pos)
    : PathDiagnosticSpotPiece(pos, "", Macro) {}

  ~PathDiagnosticMacroPiece() override;

  PathPieces subPieces;
  
  bool containsEvent() const;

  void flattenLocations() override {
    PathDiagnosticSpotPiece::flattenLocations();
    for (PathPieces::iterator I = subPieces.begin(), 
         E = subPieces.end(); I != E; ++I) (*I)->flattenLocations();
  }

  static inline bool classof(const PathDiagnosticPiece *P) {
    return P->getKind() == Macro;
  }

  void dump() const override;

  void Profile(llvm::FoldingSetNodeID &ID) const override;
};

/// PathDiagnostic - PathDiagnostic objects represent a single path-sensitive
///  diagnostic.  It represents an ordered-collection of PathDiagnosticPieces,
///  each which represent the pieces of the path.
class PathDiagnostic : public llvm::FoldingSetNode {
  std::string CheckName;
  const Decl *DeclWithIssue;
  std::string BugType;
  std::string VerboseDesc;
  std::string ShortDesc;
  std::string Category;
  std::deque<std::string> OtherDesc;

  /// \brief Loc The location of the path diagnostic report.
  PathDiagnosticLocation Loc;

  PathPieces pathImpl;
  SmallVector<PathPieces *, 3> pathStack;
  
  /// \brief Important bug uniqueing location.
  /// The location info is useful to differentiate between bugs.
  PathDiagnosticLocation UniqueingLoc;
  const Decl *UniqueingDecl;

  PathDiagnostic() = delete;
public:
  PathDiagnostic(StringRef CheckName, const Decl *DeclWithIssue,
                 StringRef bugtype, StringRef verboseDesc, StringRef shortDesc,
                 StringRef category, PathDiagnosticLocation LocationToUnique,
                 const Decl *DeclToUnique);

  ~PathDiagnostic();
  
  const PathPieces &path;

  /// Return the path currently used by builders for constructing the 
  /// PathDiagnostic.
  PathPieces &getActivePath() {
    if (pathStack.empty())
      return pathImpl;
    return *pathStack.back();
  }
  
  /// Return a mutable version of 'path'.
  PathPieces &getMutablePieces() {
    return pathImpl;
  }
    
  /// Return the unrolled size of the path.
  unsigned full_size();

  void pushActivePath(PathPieces *p) { pathStack.push_back(p); }
  void popActivePath() { if (!pathStack.empty()) pathStack.pop_back(); }

  bool isWithinCall() const { return !pathStack.empty(); }

  void setEndOfPath(std::unique_ptr<PathDiagnosticPiece> EndPiece) {
    assert(!Loc.isValid() && "End location already set!");
    Loc = EndPiece->getLocation();
    assert(Loc.isValid() && "Invalid location for end-of-path piece");
    getActivePath().push_back(EndPiece.release());
  }

  void appendToDesc(StringRef S) {
    if (!ShortDesc.empty())
      ShortDesc += S;
    VerboseDesc += S;
  }

  void resetPath() {
    pathStack.clear();
    pathImpl.clear();
    Loc = PathDiagnosticLocation();
  }

  /// \brief If the last piece of the report point to the header file, resets
  /// the location of the report to be the last location in the main source
  /// file.
  void resetDiagnosticLocationToMainFile();

  StringRef getVerboseDescription() const { return VerboseDesc; }
  StringRef getShortDescription() const {
    return ShortDesc.empty() ? VerboseDesc : ShortDesc;
  }
  StringRef getCheckName() const { return CheckName; }
  StringRef getBugType() const { return BugType; }
  StringRef getCategory() const { return Category; }

  /// Return the semantic context where an issue occurred.  If the
  /// issue occurs along a path, this represents the "central" area
  /// where the bug manifests.
  const Decl *getDeclWithIssue() const { return DeclWithIssue; }

  typedef std::deque<std::string>::const_iterator meta_iterator;
  meta_iterator meta_begin() const { return OtherDesc.begin(); }
  meta_iterator meta_end() const { return OtherDesc.end(); }
  void addMeta(StringRef s) { OtherDesc.push_back(s); }

  PathDiagnosticLocation getLocation() const {
    assert(Loc.isValid() && "No report location set yet!");
    return Loc;
  }

  /// \brief Get the location on which the report should be uniqued.
  PathDiagnosticLocation getUniqueingLoc() const {
    return UniqueingLoc;
  }

  /// \brief Get the declaration containing the uniqueing location.
  const Decl *getUniqueingDecl() const {
    return UniqueingDecl;
  }

  void flattenLocations() {
    Loc.flatten();
    for (PathPieces::iterator I = pathImpl.begin(), E = pathImpl.end(); 
         I != E; ++I) (*I)->flattenLocations();
  }

  /// Profiles the diagnostic, independent of the path it references.
  ///
  /// This can be used to merge diagnostics that refer to the same issue
  /// along different paths.
  void Profile(llvm::FoldingSetNodeID &ID) const;

  /// Profiles the diagnostic, including its path.
  ///
  /// Two diagnostics with the same issue along different paths will generate
  /// different profiles.
  void FullProfile(llvm::FoldingSetNodeID &ID) const;
};  

} // end GR namespace

} //end clang namespace

#endif
