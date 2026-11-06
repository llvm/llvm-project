//== CheckerContext.h - Context info for path-sensitive checkers--*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines CheckerContext that provides contextual info for
//  path-sensitive checkers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_CHECKERCONTEXT_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_CHECKERCONTEXT_H

#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include <optional>

namespace clang {
namespace ento {

class CheckerContext {
  ExprEngine &Eng;
  /// The current exploded(symbolic execution) graph node.
  ExplodedNode *Pred;
  /// The flag is true if the (state of the execution) has been modified
  /// by the checker using this context. For example, a new transition has been
  /// added or a bug report issued.
  bool Changed;
  /// The tagged location, which is used to generate all new nodes.
  const ProgramPoint Location;
  /// At the end of the checker evaluation, the analysis will continue from the
  /// nodes in this set. When the checker adds a transition, freshly created
  /// non-sink nodes are added to the `Frontier` and the node that was the
  /// source of the transition is unconditionally removed from the `Frontier`
  /// (it is superseded, even if the node creation fails or produces a sink).
  /// At the beginning, the `Frontier` usually contains `Pred`.
  ExplodedNodeSet &Frontier;

public:
  /// If we are post visiting a call, this flag will be set if the
  /// call was inlined.  In all other cases it will be false.
  const bool wasInlined;

  CheckerContext(ExprEngine &Eng, ExplodedNode *Pred, ExplodedNodeSet &Dst,
                 const ProgramPoint &Loc, bool WasInlined = false)
      : Eng(Eng), Pred(Pred), Changed(false), Location(Loc), Frontier(Dst),
        wasInlined(WasInlined) {
    assert(Pred->getState() &&
           "We should not call the checkers on an empty state.");
    assert(Loc.getTag() && "The ProgramPoint associated with CheckerContext "
                           "must be tagged with the active checker.");
  }

  AnalysisManager &getAnalysisManager() {
    return Eng.getAnalysisManager();
  }
  const AnalysisManager &getAnalysisManager() const {
    return Eng.getAnalysisManager();
  }

  ConstraintManager &getConstraintManager() {
    return Eng.getConstraintManager();
  }
  const ConstraintManager &getConstraintManager() const {
    return Eng.getConstraintManager();
  }

  StoreManager &getStoreManager() {
    return Eng.getStoreManager();
  }
  const StoreManager &getStoreManager() const { return Eng.getStoreManager(); }

  /// Returns the previous node in the exploded graph, which includes
  /// the state of the program before the checker ran. Note, checkers should
  /// not retain the node in their state since the nodes might get invalidated.
  ExplodedNode *getPredecessor() const { return Pred; }
  const ProgramPoint getLocation() const { return Location; }
  const ProgramStateRef &getState() const { return Pred->getState(); }

  /// Check if the checker changed the state of the execution; ex: added
  /// a new transition or a bug report.
  bool isDifferent() const { return Changed; }

  /// Returns the number of times the current block has been visited
  /// along the analyzed path.
  unsigned blockCount() const { return Eng.getNumVisitedCurrent(); }

  ASTContext &getASTContext() {
    return Eng.getContext();
  }

  const ASTContext &getASTContext() const { return Eng.getContext(); }

  const LangOptions &getLangOpts() const {
    return Eng.getContext().getLangOpts();
  }

  const StackFrame *getStackFrame() const { return Pred->getStackFrame(); }

  /// Iterates over the current stack frame and all of its ancestors.
  llvm::iterator_range<StackFrame::parent_iterator> stackframes() const {
    return getStackFrame()->parentsIncludingSelf();
  }

  /// Return true if the current StackFrame has no caller context.
  bool inTopFrame() const { return getStackFrame()->inTopFrame(); }

  BugReporter &getBugReporter() {
    return Eng.getBugReporter();
  }
  const BugReporter &getBugReporter() const { return Eng.getBugReporter(); }

  const SourceManager &getSourceManager() {
    return getBugReporter().getSourceManager();
  }
  const SourceManager &getSourceManager() const {
    return getBugReporter().getSourceManager();
  }

  Preprocessor &getPreprocessor() { return getBugReporter().getPreprocessor(); }
  const Preprocessor &getPreprocessor() const {
    return getBugReporter().getPreprocessor();
  }

  SValBuilder &getSValBuilder() {
    return Eng.getSValBuilder();
  }
  const SValBuilder &getSValBuilder() const { return Eng.getSValBuilder(); }

  SymbolManager &getSymbolManager() {
    return getSValBuilder().getSymbolManager();
  }
  const SymbolManager &getSymbolManager() const {
    return getSValBuilder().getSymbolManager();
  }

  ProgramStateManager &getStateManager() {
    return Eng.getStateManager();
  }
  const ProgramStateManager &getStateManager() const {
    return Eng.getStateManager();
  }

  AnalysisDeclContext *getCurrentAnalysisDeclContext() const {
    return Pred->getStackFrame()->getAnalysisDeclContext();
  }

  /// Get the blockID.
  unsigned getBlockID() const { return Eng.getCurrBlock()->getBlockID(); }

  /// If the given node corresponds to a PostStore program point,
  /// retrieve the location region as it was uttered in the code.
  ///
  /// This utility can be useful for generating extensive diagnostics, for
  /// example, for finding variables that the given symbol was assigned to.
  static const MemRegion *getLocationRegionIfPostStore(const ExplodedNode *N) {
    ProgramPoint L = N->getLocation();
    if (std::optional<PostStore> PSL = L.getAs<PostStore>())
      return reinterpret_cast<const MemRegion*>(PSL->getLocationValue());
    return nullptr;
  }

  /// Get the value of arbitrary expressions at this point in the path.
  SVal getSVal(const Expr *E) const { return Pred->getSVal(E); }

  ConstCFGElementRef getCFGElementRef() const { return Eng.getCFGElementRef(); }

  /// Returns true if the value of \p E is greater than or equal to \p
  /// Val under unsigned comparison.
  bool isGreaterOrEqual(const Expr *E, unsigned long long Val);

  /// Returns true if the value of \p E is negative.
  bool isNegative(const Expr *E);

  /// Creates a new ExplodedNode and adds it to the ExplodedGraph as a child of
  /// 'getPredecessor()', the node that was passed to the constructor of
  /// CheckerContext.
  /// Can return nullptr if the node creation fails (when it would recreate
  /// an existing node or when the state is posteriorly overconstrained).
  /// Can return 'getPredecessor()' instead of creating a node when the state
  /// would remain unchanged and there is no explicitly specified tag.
  ///
  /// @param State The state of the generated node. Passing nullptr is handled
  ///        as passing 'getState()' (the state of 'getPredecessor()').
  /// @param Tag A ProgramPointTag that will be attached to the node. This is
  ///        primarily useful for attaching a NoteTag; but another tag could be
  ///        used as a workaround when the node wouldn't be unique otherwise.
  ///        When \p Tag is nullptr, node creation will default to using the
  ///        active checker as the tag -- but if there is no state change,
  ///        the predecessor node may be returned even if it isn't tagged so.
  ExplodedNode *addTransition(ProgramStateRef State = nullptr,
                              const ProgramPointTag *Tag = nullptr) {
    if ((!State || State == getState()) && !Tag)
      return getPredecessor();
    return addTransitionImpl(State, false, nullptr, Tag);
  }

  /// Creates a new ExplodedNode and adds it to the ExplodedGraph as a child of
  /// the parameter \p Pred. Allows checkers to make a chain of nodes.
  /// Can return nullptr if the node creation fails (when it would recreate
  /// an existing node or when the state is posteriorly overconstrained).
  ///
  /// Can return getPredecessor(), the node that was passed to the constructor
  /// of CheckerContext (often the ancestor of \p Pred) in some corner cases.
  /// FIXME: This is bad logic and should be fixed in the next non-NFC change.
  ///
  /// @param State The state of the generated node.
  /// @param Pred The node that will become the predecessor of the new node.
  /// @param Tag A ProgramPointTag that will be attached to the node. This is
  ///        primarily useful for attaching a NoteTag; but another tag could be
  ///        used as a workaround when the node wouldn't be unique otherwise.
  ///        When \p Tag is nullptr, node creation will default to using the
  ///        active checker as the tag.
  ExplodedNode *addTransition(ProgramStateRef State, ExplodedNode *Pred,
                              const ProgramPointTag *Tag = nullptr) {
    // FIXME: In these two early return statements we should use 'Pred' and
    // 'Pred->getState()' (the node received as an argument and its state)
    // instead of 'getPredecessor()' and 'getState()' (the older node received
    // by the constructor of CheckerContext, and its state).
    // Alternatively, we should perhaps return null if the state is null.
    if (!State)
      return getPredecessor();
    if (State == getState() && !Tag)
      return getPredecessor();
    return addTransitionImpl(State, false, Pred, Tag);
  }

  /// Creates a new ExplodedNode which is marked as a sink and adds it to the
  /// ExplodedGraph.
  /// A sink node terminates the execution path: it cannot have children and
  /// prevents the default behavior that the path "passes through" the checker
  /// callback without changes if the checker doesn't create any node.
  /// Can return nullptr if the node creation fails (when it would recreate
  /// an existing node or when the state is posteriorly overconstrained).
  ///
  /// @param State The state of the generated sink node; nullptr (the default)
  ///        means keeping the state of the parent node.
  /// @param Pred The node that will become the predecessor of the new node;
  ///        nullptr (the default) means the node that was passed to the
  ///        constructor of CheckerContext.
  /// @param Tag A ProgramPointTag that will be attached to the node; nullptr
  ///        (the default) means using the active checker as the tag. Using
  ///        another tag has no compelling use case (apart from debugging).
  ExplodedNode *addSink(ProgramStateRef State = nullptr,
                        ExplodedNode *Pred = nullptr,
                        const ProgramPointTag *Tag = nullptr) {
    return addTransitionImpl(State, true, Pred, Tag);
  }

  /// Creates a new ExplodedNode and adds it to the ExplodedGraph as a child of
  /// 'getPredecessor()', the node that was passed to the constructor of
  /// CheckerContext. The difference between this and 'addTransition' is that
  /// this always attempts to create a new node, even if there is no state
  /// change and no explicitly speficified tag.
  /// Can return nullptr if the node creation fails (when it would recreate
  /// an existing node or when the state is posteriorly overconstrained).
  ///
  /// @param State The state of the generated node. Passing nullptr is handled
  ///        as passing 'getState()' (the state of 'getPredecessor()').
  /// @param Tag A ProgramPointTag that will be attached to the node. This is
  ///        primarily useful for attaching a NoteTag; but another tag could be
  ///        used as a workaround when the node wouldn't be unique otherwise.
  ///        When \p Tag is nullptr, node creation will default to using the
  ///        active checker as the tag.
  ExplodedNode *
  generateNonFatalErrorNode(ProgramStateRef State = nullptr,
                            const ProgramPointTag *Tag = nullptr) {
    return addTransitionImpl(State, false, nullptr, Tag);
  }

  /// Creates a new ExplodedNode and adds it to the ExplodedGraph as a child of
  /// the parameter \p Pred. Allows checkers to make a chain of nodes.
  /// The difference between this and 'addTransition()' should be that this
  /// always attempts to create a new node (even if there is no state change
  /// and no explicitly specified tag). Currently this logic is incorrect in
  /// addTransition and the difference is that this function lacks one of the
  /// two problematic early return statements.
  /// Can return nullptr if the node creation fails (when it would recreate
  /// an existing node or when the state is posteriorly overconstrained).
  ///
  /// @param State The state of the generated node.
  /// @param Pred The node that will become the predecessor of the new node.
  /// @param Tag A ProgramPointTag that will be attached to the node. This is
  ///        primarily useful for attaching a NoteTag; but another tag could be
  ///        used as a workaround when the node wouldn't be unique otherwise.
  ///        When \p Tag is nullptr, node creation will default to using the
  ///        active checker as the tag.
  ExplodedNode *
  generateNonFatalErrorNode(ProgramStateRef State,
                            ExplodedNode *Pred,
                            const ProgramPointTag *Tag = nullptr) {
    // FIXME: This can "jump back" to an ancestor of 'Pred'. We should return
    // either Pred or nullptr in this case.
    if (!State)
      return getPredecessor();
    return addTransitionImpl(State, false, Pred, Tag);
  }

  /// Emit the diagnostics report.
  void emitReport(std::unique_ptr<BugReport> R) {
    Changed = true;
    Eng.getBugReporter().emitReport(std::move(R));
  }

  /// Produce a program point tag that displays an additional path note
  /// to the user. This is a lightweight alternative to the
  /// BugReporterVisitor mechanism: instead of visiting the bug report
  /// node-by-node to restore the sequence of events that led to discovering
  /// a bug, you can add notes as you add your transitions.
  ///
  /// @param Cb Callback with 'BugReporterContext &, BugReport &' parameters.
  /// @param IsPrunable Whether the note is prunable. It allows BugReporter
  ///        to omit the note from the report if it would make the displayed
  ///        bug path significantly shorter.
  LLVM_ATTRIBUTE_RETURNS_NONNULL
  const NoteTag *getNoteTag(NoteTag::Callback &&Cb, bool IsPrunable = false) {
    return Eng.getDataTags().make<NoteTag>(std::move(Cb), IsPrunable);
  }

  /// A shorthand version of getNoteTag that doesn't require you to accept
  /// the 'BugReporterContext' argument when you don't need it.
  ///
  /// @param Cb Callback only with 'BugReport &' parameter.
  /// @param IsPrunable Whether the note is prunable. It allows BugReporter
  ///        to omit the note from the report if it would make the displayed
  ///        bug path significantly shorter.
  const NoteTag
  *getNoteTag(std::function<std::string(PathSensitiveBugReport &)> &&Cb,
              bool IsPrunable = false) {
    return getNoteTag(
        [Cb](BugReporterContext &,
             PathSensitiveBugReport &BR) { return Cb(BR); },
        IsPrunable);
  }

  /// A shorthand version of getNoteTag that doesn't require you to accept
  /// the arguments when you don't need it.
  ///
  /// @param Cb Callback without parameters.
  /// @param IsPrunable Whether the note is prunable. It allows BugReporter
  ///        to omit the note from the report if it would make the displayed
  ///        bug path significantly shorter.
  const NoteTag *getNoteTag(std::function<std::string()> &&Cb,
                            bool IsPrunable = false) {
    return getNoteTag([Cb](BugReporterContext &,
                           PathSensitiveBugReport &) { return Cb(); },
                      IsPrunable);
  }

  /// A shorthand version of getNoteTag that accepts a plain note.
  ///
  /// @param Note The note.
  /// @param IsPrunable Whether the note is prunable. It allows BugReporter
  ///        to omit the note from the report if it would make the displayed
  ///        bug path significantly shorter.
  const NoteTag *getNoteTag(StringRef Note, bool IsPrunable = false) {
    return getNoteTag(
        [Note = std::string(Note)](BugReporterContext &,
               PathSensitiveBugReport &) { return Note; },
        IsPrunable);
  }

  /// A shorthand version of getNoteTag that accepts a lambda with stream for
  /// note.
  ///
  /// @param Cb Callback with 'BugReport &' and 'llvm::raw_ostream &'.
  /// @param IsPrunable Whether the note is prunable. It allows BugReporter
  ///        to omit the note from the report if it would make the displayed
  ///        bug path significantly shorter.
  const NoteTag *getNoteTag(
      std::function<void(PathSensitiveBugReport &BR, llvm::raw_ostream &OS)> &&Cb,
      bool IsPrunable = false) {
    return getNoteTag(
        [Cb](PathSensitiveBugReport &BR) -> std::string {
          llvm::SmallString<128> Str;
          llvm::raw_svector_ostream OS(Str);
          Cb(BR, OS);
          return std::string(OS.str());
        },
        IsPrunable);
  }

  /// Returns the word that should be used to refer to the declaration
  /// in the report.
  StringRef getDeclDescription(const Decl *D);

  /// Get the declaration of the called function (path-sensitive).
  const FunctionDecl *getCalleeDecl(const CallExpr *CE) const;

  /// Get the name of the called function (path-sensitive).
  StringRef getCalleeName(const FunctionDecl *FunDecl) const;

  /// Get the identifier of the called function (path-sensitive).
  const IdentifierInfo *getCalleeIdentifier(const CallExpr *CE) const {
    const FunctionDecl *FunDecl = getCalleeDecl(CE);
    if (FunDecl)
      return FunDecl->getIdentifier();
    else
      return nullptr;
  }

  /// Get the name of the called function (path-sensitive).
  StringRef getCalleeName(const CallExpr *CE) const {
    const FunctionDecl *FunDecl = getCalleeDecl(CE);
    return getCalleeName(FunDecl);
  }

  /// Returns true if the given function is an externally-visible function in
  /// the top-level namespace, such as \c malloc.
  ///
  /// If a name is provided, the function must additionally match the given
  /// name.
  ///
  /// Note that this also accepts functions from the \c std namespace (because
  /// headers like <cstdlib> declare them there) and does not check if the
  /// function is declared as 'extern "C"' or if it uses C++ name mangling.
  static bool isCLibraryFunction(const FunctionDecl *FD,
                                 StringRef Name = StringRef());

  /// In builds that use source hardening (-D_FORTIFY_SOURCE), many standard
  /// functions are implemented as macros that expand to calls of hardened
  /// functions that take additional arguments compared to the "usual"
  /// variant and perform additional input validation. For example, a `memcpy`
  /// call may expand to `__memcpy_chk()` or `__builtin___memcpy_chk()`.
  ///
  /// This method returns true if `FD` declares a fortified variant of the
  /// standard library function `Name`.
  ///
  /// NOTE: This method relies on heuristics; extend it if you need to handle a
  /// hardened variant that's not yet covered by it.
  static bool isHardenedVariantOf(const FunctionDecl *FD, StringRef Name);

  /// Depending on whether the location corresponds to a macro, return
  /// either the macro name or the token spelling.
  ///
  /// This could be useful when checkers' logic depends on whether a function
  /// is called with a given macro argument. For example:
  ///   s = socket(AF_INET,..)
  /// If AF_INET is a macro, the result should be treated as a source of taint.
  ///
  /// \sa clang::Lexer::getSpelling(), clang::Lexer::getImmediateMacroName().
  std::string getMacroNameOrSpelling(SourceLocation &Loc);

private:
  ExplodedNode *addTransitionImpl(ProgramStateRef State, bool MarkAsSink,
                                  ExplodedNode *P, const ProgramPointTag *Tag) {
    Changed = true;
    const ProgramPoint &LocalLoc = (Tag ? Location.withTag(Tag) : Location);

    if (!P)
      P = getPredecessor();
    if (!State)
      State = P->getState();

    Frontier.erase(P);
    ExplodedNode *N =
        Eng.getCoreEngine().makeNode(LocalLoc, State, P, MarkAsSink);

    Frontier.insert(N);
    return N;
  }
};

} // end GR namespace

} // end clang namespace

#endif
