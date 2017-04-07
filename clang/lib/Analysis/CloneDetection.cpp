//===--- CloneDetection.cpp - Finds code clones in an AST -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  This file implements classes for searching and anlyzing source code clones.
///
//===----------------------------------------------------------------------===//

#include "clang/Analysis/CloneDetection.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

StmtSequence::StmtSequence(const CompoundStmt *Stmt, const Decl *D,
                           unsigned StartIndex, unsigned EndIndex)
    : S(Stmt), D(D), StartIndex(StartIndex), EndIndex(EndIndex) {
  assert(Stmt && "Stmt must not be a nullptr");
  assert(StartIndex < EndIndex && "Given array should not be empty");
  assert(EndIndex <= Stmt->size() && "Given array too big for this Stmt");
}

StmtSequence::StmtSequence(const Stmt *Stmt, const Decl *D)
    : S(Stmt), D(D), StartIndex(0), EndIndex(0) {}

StmtSequence::StmtSequence()
    : S(nullptr), D(nullptr), StartIndex(0), EndIndex(0) {}

bool StmtSequence::contains(const StmtSequence &Other) const {
  // If both sequences reside in different declarations, they can never contain
  // each other.
  if (D != Other.D)
    return false;

  const SourceManager &SM = getASTContext().getSourceManager();

  // Otherwise check if the start and end locations of the current sequence
  // surround the other sequence.
  bool StartIsInBounds =
      SM.isBeforeInTranslationUnit(getStartLoc(), Other.getStartLoc()) ||
      getStartLoc() == Other.getStartLoc();
  if (!StartIsInBounds)
    return false;

  bool EndIsInBounds =
      SM.isBeforeInTranslationUnit(Other.getEndLoc(), getEndLoc()) ||
      Other.getEndLoc() == getEndLoc();
  return EndIsInBounds;
}

StmtSequence::iterator StmtSequence::begin() const {
  if (!holdsSequence()) {
    return &S;
  }
  auto CS = cast<CompoundStmt>(S);
  return CS->body_begin() + StartIndex;
}

StmtSequence::iterator StmtSequence::end() const {
  if (!holdsSequence()) {
    return reinterpret_cast<StmtSequence::iterator>(&S) + 1;
  }
  auto CS = cast<CompoundStmt>(S);
  return CS->body_begin() + EndIndex;
}

ASTContext &StmtSequence::getASTContext() const {
  assert(D);
  return D->getASTContext();
}

SourceLocation StmtSequence::getStartLoc() const {
  return front()->getLocStart();
}

SourceLocation StmtSequence::getEndLoc() const { return back()->getLocEnd(); }

SourceRange StmtSequence::getSourceRange() const {
  return SourceRange(getStartLoc(), getEndLoc());
}

/// Prints the macro name that contains the given SourceLocation into the given
/// raw_string_ostream.
static void printMacroName(llvm::raw_string_ostream &MacroStack,
                           ASTContext &Context, SourceLocation Loc) {
  MacroStack << Lexer::getImmediateMacroName(Loc, Context.getSourceManager(),
                                             Context.getLangOpts());

  // Add an empty space at the end as a padding to prevent
  // that macro names concatenate to the names of other macros.
  MacroStack << " ";
}

/// Returns a string that represents all macro expansions that expanded into the
/// given SourceLocation.
///
/// If 'getMacroStack(A) == getMacroStack(B)' is true, then the SourceLocations
/// A and B are expanded from the same macros in the same order.
static std::string getMacroStack(SourceLocation Loc, ASTContext &Context) {
  std::string MacroStack;
  llvm::raw_string_ostream MacroStackStream(MacroStack);
  SourceManager &SM = Context.getSourceManager();

  // Iterate over all macros that expanded into the given SourceLocation.
  while (Loc.isMacroID()) {
    // Add the macro name to the stream.
    printMacroName(MacroStackStream, Context, Loc);
    Loc = SM.getImmediateMacroCallerLoc(Loc);
  }
  MacroStackStream.flush();
  return MacroStack;
}

namespace {
typedef unsigned DataPiece;

/// Collects the data of a single Stmt.
///
/// This class defines what a code clone is: If it collects for two statements
/// the same data, then those two statements are considered to be clones of each
/// other.
///
/// All collected data is forwarded to the given data consumer of the type T.
/// The data consumer class needs to provide a member method with the signature:
///   update(StringRef Str)
template <typename T>
class StmtDataCollector : public ConstStmtVisitor<StmtDataCollector<T>> {

  ASTContext &Context;
  /// The data sink to which all data is forwarded.
  T &DataConsumer;

public:
  /// Collects data of the given Stmt.
  /// \param S The given statement.
  /// \param Context The ASTContext of S.
  /// \param DataConsumer The data sink to which all data is forwarded.
  StmtDataCollector(const Stmt *S, ASTContext &Context, T &DataConsumer)
      : Context(Context), DataConsumer(DataConsumer) {
    this->Visit(S);
  }

  // Below are utility methods for appending different data to the vector.

  void addData(DataPiece Integer) {
    DataConsumer.update(
        StringRef(reinterpret_cast<char *>(&Integer), sizeof(Integer)));
  }

  void addData(llvm::StringRef Str) { DataConsumer.update(Str); }

  void addData(const QualType &QT) { addData(QT.getAsString()); }

// The functions below collect the class specific data of each Stmt subclass.

// Utility macro for defining a visit method for a given class. This method
// calls back to the ConstStmtVisitor to visit all parent classes.
#define DEF_ADD_DATA(CLASS, CODE)                                              \
  void Visit##CLASS(const CLASS *S) {                                          \
    CODE;                                                                      \
    ConstStmtVisitor<StmtDataCollector>::Visit##CLASS(S);                      \
  }

  DEF_ADD_DATA(Stmt, {
    addData(S->getStmtClass());
    // This ensures that macro generated code isn't identical to macro-generated
    // code.
    addData(getMacroStack(S->getLocStart(), Context));
    addData(getMacroStack(S->getLocEnd(), Context));
  })
  DEF_ADD_DATA(Expr, { addData(S->getType()); })

  //--- Builtin functionality ----------------------------------------------//
  DEF_ADD_DATA(ArrayTypeTraitExpr, { addData(S->getTrait()); })
  DEF_ADD_DATA(ExpressionTraitExpr, { addData(S->getTrait()); })
  DEF_ADD_DATA(PredefinedExpr, { addData(S->getIdentType()); })
  DEF_ADD_DATA(TypeTraitExpr, {
    addData(S->getTrait());
    for (unsigned i = 0; i < S->getNumArgs(); ++i)
      addData(S->getArg(i)->getType());
  })

  //--- Calls --------------------------------------------------------------//
  DEF_ADD_DATA(CallExpr, {
    // Function pointers don't have a callee and we just skip hashing it.
    if (const FunctionDecl *D = S->getDirectCallee()) {
      // If the function is a template specialization, we also need to handle
      // the template arguments as they are not included in the qualified name.
      if (auto Args = D->getTemplateSpecializationArgs()) {
        std::string ArgString;

        // Print all template arguments into ArgString
        llvm::raw_string_ostream OS(ArgString);
        for (unsigned i = 0; i < Args->size(); ++i) {
          Args->get(i).print(Context.getLangOpts(), OS);
          // Add a padding character so that 'foo<X, XX>()' != 'foo<XX, X>()'.
          OS << '\n';
        }
        OS.flush();

        addData(ArgString);
      }
      addData(D->getQualifiedNameAsString());
    }
  })

  //--- Exceptions ---------------------------------------------------------//
  DEF_ADD_DATA(CXXCatchStmt, { addData(S->getCaughtType()); })

  //--- C++ OOP Stmts ------------------------------------------------------//
  DEF_ADD_DATA(CXXDeleteExpr, {
    addData(S->isArrayFormAsWritten());
    addData(S->isGlobalDelete());
  })

  //--- Casts --------------------------------------------------------------//
  DEF_ADD_DATA(ObjCBridgedCastExpr, { addData(S->getBridgeKind()); })

  //--- Miscellaneous Exprs ------------------------------------------------//
  DEF_ADD_DATA(BinaryOperator, { addData(S->getOpcode()); })
  DEF_ADD_DATA(UnaryOperator, { addData(S->getOpcode()); })

  //--- Control flow -------------------------------------------------------//
  DEF_ADD_DATA(GotoStmt, { addData(S->getLabel()->getName()); })
  DEF_ADD_DATA(IndirectGotoStmt, {
    if (S->getConstantTarget())
      addData(S->getConstantTarget()->getName());
  })
  DEF_ADD_DATA(LabelStmt, { addData(S->getDecl()->getName()); })
  DEF_ADD_DATA(MSDependentExistsStmt, { addData(S->isIfExists()); })
  DEF_ADD_DATA(AddrLabelExpr, { addData(S->getLabel()->getName()); })

  //--- Objective-C --------------------------------------------------------//
  DEF_ADD_DATA(ObjCIndirectCopyRestoreExpr, { addData(S->shouldCopy()); })
  DEF_ADD_DATA(ObjCPropertyRefExpr, {
    addData(S->isSuperReceiver());
    addData(S->isImplicitProperty());
  })
  DEF_ADD_DATA(ObjCAtCatchStmt, { addData(S->hasEllipsis()); })

  //--- Miscellaneous Stmts ------------------------------------------------//
  DEF_ADD_DATA(CXXFoldExpr, {
    addData(S->isRightFold());
    addData(S->getOperator());
  })
  DEF_ADD_DATA(GenericSelectionExpr, {
    for (unsigned i = 0; i < S->getNumAssocs(); ++i) {
      addData(S->getAssocType(i));
    }
  })
  DEF_ADD_DATA(LambdaExpr, {
    for (const LambdaCapture &C : S->captures()) {
      addData(C.isPackExpansion());
      addData(C.getCaptureKind());
      if (C.capturesVariable())
        addData(C.getCapturedVar()->getType());
    }
    addData(S->isGenericLambda());
    addData(S->isMutable());
  })
  DEF_ADD_DATA(DeclStmt, {
    auto numDecls = std::distance(S->decl_begin(), S->decl_end());
    addData(static_cast<DataPiece>(numDecls));
    for (const Decl *D : S->decls()) {
      if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
        addData(VD->getType());
      }
    }
  })
  DEF_ADD_DATA(AsmStmt, {
    addData(S->isSimple());
    addData(S->isVolatile());
    addData(S->generateAsmString(Context));
    for (unsigned i = 0; i < S->getNumInputs(); ++i) {
      addData(S->getInputConstraint(i));
    }
    for (unsigned i = 0; i < S->getNumOutputs(); ++i) {
      addData(S->getOutputConstraint(i));
    }
    for (unsigned i = 0; i < S->getNumClobbers(); ++i) {
      addData(S->getClobber(i));
    }
  })
  DEF_ADD_DATA(AttributedStmt, {
    for (const Attr *A : S->getAttrs()) {
      addData(std::string(A->getSpelling()));
    }
  })
};
} // end anonymous namespace

void CloneDetector::analyzeCodeBody(const Decl *D) {
  assert(D);
  assert(D->hasBody());

  Sequences.push_back(StmtSequence(D->getBody(), D));
}

/// Returns true if and only if \p Stmt contains at least one other
/// sequence in the \p Group.
static bool containsAnyInGroup(StmtSequence &Seq,
                               CloneDetector::CloneGroup &Group) {
  for (StmtSequence &GroupSeq : Group) {
    if (Seq.contains(GroupSeq))
      return true;
  }
  return false;
}

/// Returns true if and only if all sequences in \p OtherGroup are
/// contained by a sequence in \p Group.
static bool containsGroup(CloneDetector::CloneGroup &Group,
                          CloneDetector::CloneGroup &OtherGroup) {
  // We have less sequences in the current group than we have in the other,
  // so we will never fulfill the requirement for returning true. This is only
  // possible because we know that a sequence in Group can contain at most
  // one sequence in OtherGroup.
  if (Group.size() < OtherGroup.size())
    return false;

  for (StmtSequence &Stmt : Group) {
    if (!containsAnyInGroup(Stmt, OtherGroup))
      return false;
  }
  return true;
}

void OnlyLargestCloneConstraint::constrain(
    std::vector<CloneDetector::CloneGroup> &Result) {
  std::vector<unsigned> IndexesToRemove;

  // Compare every group in the result with the rest. If one groups contains
  // another group, we only need to return the bigger group.
  // Note: This doesn't scale well, so if possible avoid calling any heavy
  // function from this loop to minimize the performance impact.
  for (unsigned i = 0; i < Result.size(); ++i) {
    for (unsigned j = 0; j < Result.size(); ++j) {
      // Don't compare a group with itself.
      if (i == j)
        continue;

      if (containsGroup(Result[j], Result[i])) {
        IndexesToRemove.push_back(i);
        break;
      }
    }
  }

  // Erasing a list of indexes from the vector should be done with decreasing
  // indexes. As IndexesToRemove is constructed with increasing values, we just
  // reverse iterate over it to get the desired order.
  for (auto I = IndexesToRemove.rbegin(); I != IndexesToRemove.rend(); ++I) {
    Result.erase(Result.begin() + *I);
  }
}

static size_t createHash(llvm::MD5 &Hash) {
  size_t HashCode;

  // Create the final hash code for the current Stmt.
  llvm::MD5::MD5Result HashResult;
  Hash.final(HashResult);

  // Copy as much as possible of the generated hash code to the Stmt's hash
  // code.
  std::memcpy(&HashCode, &HashResult,
              std::min(sizeof(HashCode), sizeof(HashResult)));

  return HashCode;
}

size_t RecursiveCloneTypeIIConstraint::saveHash(
    const Stmt *S, const Decl *D,
    std::vector<std::pair<size_t, StmtSequence>> &StmtsByHash) {
  llvm::MD5 Hash;
  ASTContext &Context = D->getASTContext();

  StmtDataCollector<llvm::MD5>(S, Context, Hash);

  auto CS = dyn_cast<CompoundStmt>(S);
  SmallVector<size_t, 8> ChildHashes;

  for (const Stmt *Child : S->children()) {
    if (Child == nullptr) {
      ChildHashes.push_back(0);
      continue;
    }
    size_t ChildHash = saveHash(Child, D, StmtsByHash);
    Hash.update(
        StringRef(reinterpret_cast<char *>(&ChildHash), sizeof(ChildHash)));
    ChildHashes.push_back(ChildHash);
  }

  if (CS) {
    for (unsigned Length = 2; Length <= CS->size(); ++Length) {
      for (unsigned Pos = 0; Pos <= CS->size() - Length; ++Pos) {
        llvm::MD5 Hash;
        for (unsigned i = Pos; i < Pos + Length; ++i) {
          size_t ChildHash = ChildHashes[i];
          Hash.update(StringRef(reinterpret_cast<char *>(&ChildHash),
                                sizeof(ChildHash)));
        }
        StmtsByHash.push_back(std::make_pair(
            createHash(Hash), StmtSequence(CS, D, Pos, Pos + Length)));
      }
    }
  }

  size_t HashCode = createHash(Hash);
  StmtsByHash.push_back(std::make_pair(HashCode, StmtSequence(S, D)));
  return HashCode;
}

namespace {
/// Wrapper around FoldingSetNodeID that it can be used as the template
/// argument of the StmtDataCollector.
class FoldingSetNodeIDWrapper {

  llvm::FoldingSetNodeID &FS;

public:
  FoldingSetNodeIDWrapper(llvm::FoldingSetNodeID &FS) : FS(FS) {}

  void update(StringRef Str) { FS.AddString(Str); }
};
} // end anonymous namespace

/// Writes the relevant data from all statements and child statements
/// in the given StmtSequence into the given FoldingSetNodeID.
static void CollectStmtSequenceData(const StmtSequence &Sequence,
                                    FoldingSetNodeIDWrapper &OutputData) {
  for (const Stmt *S : Sequence) {
    StmtDataCollector<FoldingSetNodeIDWrapper>(S, Sequence.getASTContext(),
                                               OutputData);

    for (const Stmt *Child : S->children()) {
      if (!Child)
        continue;

      CollectStmtSequenceData(StmtSequence(Child, Sequence.getContainingDecl()),
                              OutputData);
    }
  }
}

/// Returns true if both sequences are clones of each other.
static bool areSequencesClones(const StmtSequence &LHS,
                               const StmtSequence &RHS) {
  // We collect the data from all statements in the sequence as we did before
  // when generating a hash value for each sequence. But this time we don't
  // hash the collected data and compare the whole data set instead. This
  // prevents any false-positives due to hash code collisions.
  llvm::FoldingSetNodeID DataLHS, DataRHS;
  FoldingSetNodeIDWrapper LHSWrapper(DataLHS);
  FoldingSetNodeIDWrapper RHSWrapper(DataRHS);

  CollectStmtSequenceData(LHS, LHSWrapper);
  CollectStmtSequenceData(RHS, RHSWrapper);

  return DataLHS == DataRHS;
}

void RecursiveCloneTypeIIConstraint::constrain(
    std::vector<CloneDetector::CloneGroup> &Sequences) {
  // FIXME: Maybe we can do this in-place and don't need this additional vector.
  std::vector<CloneDetector::CloneGroup> Result;

  for (CloneDetector::CloneGroup &Group : Sequences) {
    // We assume in the following code that the Group is non-empty, so we
    // skip all empty groups.
    if (Group.empty())
      continue;

    std::vector<std::pair<size_t, StmtSequence>> StmtsByHash;

    // Generate hash codes for all children of S and save them in StmtsByHash.
    for (const StmtSequence &S : Group) {
      saveHash(S.front(), S.getContainingDecl(), StmtsByHash);
    }

    // Sort hash_codes in StmtsByHash.
    std::stable_sort(StmtsByHash.begin(), StmtsByHash.end(),
                     [](std::pair<size_t, StmtSequence> LHS,
                            std::pair<size_t, StmtSequence> RHS) {
                       return LHS.first < RHS.first;
                     });

    // Check for each StmtSequence if its successor has the same hash value.
    // We don't check the last StmtSequence as it has no successor.
    // Note: The 'size - 1 ' in the condition is safe because we check for an
    // empty Group vector at the beginning of this function.
    for (unsigned i = 0; i < StmtsByHash.size() - 1; ++i) {
      const auto Current = StmtsByHash[i];

      // It's likely that we just found an sequence of StmtSequences that
      // represent a CloneGroup, so we create a new group and start checking and
      // adding the StmtSequences in this sequence.
      CloneDetector::CloneGroup NewGroup;

      size_t PrototypeHash = Current.first;

      for (; i < StmtsByHash.size(); ++i) {
        // A different hash value means we have reached the end of the sequence.
        if (PrototypeHash != StmtsByHash[i].first ||
            !areSequencesClones(StmtsByHash[i].second, Current.second)) {
          // The current sequence could be the start of a new CloneGroup. So we
          // decrement i so that we visit it again in the outer loop.
          // Note: i can never be 0 at this point because we are just comparing
          // the hash of the Current StmtSequence with itself in the 'if' above.
          assert(i != 0);
          --i;
          break;
        }
        // Same hash value means we should add the StmtSequence to the current
        // group.
        NewGroup.push_back(StmtsByHash[i].second);
      }

      // We created a new clone group with matching hash codes and move it to
      // the result vector.
      Result.push_back(NewGroup);
    }
  }
  // Sequences is the output parameter, so we copy our result into it.
  Sequences = Result;
}

size_t MinComplexityConstraint::calculateStmtComplexity(
    const StmtSequence &Seq, const std::string &ParentMacroStack) {
  if (Seq.empty())
    return 0;

  size_t Complexity = 1;

  ASTContext &Context = Seq.getASTContext();

  // Look up what macros expanded into the current statement.
  std::string StartMacroStack = getMacroStack(Seq.getStartLoc(), Context);
  std::string EndMacroStack = getMacroStack(Seq.getEndLoc(), Context);

  // First, check if ParentMacroStack is not empty which means we are currently
  // dealing with a parent statement which was expanded from a macro.
  // If this parent statement was expanded from the same macros as this
  // statement, we reduce the initial complexity of this statement to zero.
  // This causes that a group of statements that were generated by a single
  // macro expansion will only increase the total complexity by one.
  // Note: This is not the final complexity of this statement as we still
  // add the complexity of the child statements to the complexity value.
  if (!ParentMacroStack.empty() && (StartMacroStack == ParentMacroStack &&
                                    EndMacroStack == ParentMacroStack)) {
    Complexity = 0;
  }

  // Iterate over the Stmts in the StmtSequence and add their complexity values
  // to the current complexity value.
  if (Seq.holdsSequence()) {
    for (const Stmt *S : Seq) {
      Complexity += calculateStmtComplexity(
          StmtSequence(S, Seq.getContainingDecl()), StartMacroStack);
    }
  } else {
    for (const Stmt *S : Seq.front()->children()) {
      Complexity += calculateStmtComplexity(
          StmtSequence(S, Seq.getContainingDecl()), StartMacroStack);
    }
  }
  return Complexity;
}

void MatchingVariablePatternConstraint::constrain(
    std::vector<CloneDetector::CloneGroup> &CloneGroups) {
  CloneConstraint::splitCloneGroups(
      CloneGroups, [](const StmtSequence &A, const StmtSequence &B) {
        VariablePattern PatternA(A);
        VariablePattern PatternB(B);
        return PatternA.countPatternDifferences(PatternB) == 0;
      });
}

void CloneConstraint::splitCloneGroups(
    std::vector<CloneDetector::CloneGroup> &CloneGroups,
    std::function<bool(const StmtSequence &, const StmtSequence &)> Compare) {
  std::vector<CloneDetector::CloneGroup> Result;
  for (auto &HashGroup : CloneGroups) {
    // Contains all indexes in HashGroup that were already added to a
    // CloneGroup.
    std::vector<char> Indexes;
    Indexes.resize(HashGroup.size());

    for (unsigned i = 0; i < HashGroup.size(); ++i) {
      // Skip indexes that are already part of a CloneGroup.
      if (Indexes[i])
        continue;

      // Pick the first unhandled StmtSequence and consider it as the
      // beginning
      // of a new CloneGroup for now.
      // We don't add i to Indexes because we never iterate back.
      StmtSequence Prototype = HashGroup[i];
      CloneDetector::CloneGroup PotentialGroup = {Prototype};
      ++Indexes[i];

      // Check all following StmtSequences for clones.
      for (unsigned j = i + 1; j < HashGroup.size(); ++j) {
        // Skip indexes that are already part of a CloneGroup.
        if (Indexes[j])
          continue;

        // If a following StmtSequence belongs to our CloneGroup, we add it to
        // it.
        const StmtSequence &Candidate = HashGroup[j];

        if (!Compare(Prototype, Candidate))
          continue;

        PotentialGroup.push_back(Candidate);
        // Make sure we never visit this StmtSequence again.
        ++Indexes[j];
      }

      // Otherwise, add it to the result and continue searching for more
      // groups.
      Result.push_back(PotentialGroup);
    }

    assert(std::all_of(Indexes.begin(), Indexes.end(),
                       [](char c) { return c == 1; }));
  }
  CloneGroups = Result;
}

void VariablePattern::addVariableOccurence(const VarDecl *VarDecl,
                                           const Stmt *Mention) {
  // First check if we already reference this variable
  for (size_t KindIndex = 0; KindIndex < Variables.size(); ++KindIndex) {
    if (Variables[KindIndex] == VarDecl) {
      // If yes, add a new occurence that points to the existing entry in
      // the Variables vector.
      Occurences.emplace_back(KindIndex, Mention);
      return;
    }
  }
  // If this variable wasn't already referenced, add it to the list of
  // referenced variables and add a occurence that points to this new entry.
  Occurences.emplace_back(Variables.size(), Mention);
  Variables.push_back(VarDecl);
}

void VariablePattern::addVariables(const Stmt *S) {
  // Sometimes we get a nullptr (such as from IfStmts which often have nullptr
  // children). We skip such statements as they don't reference any
  // variables.
  if (!S)
    return;

  // Check if S is a reference to a variable. If yes, add it to the pattern.
  if (auto D = dyn_cast<DeclRefExpr>(S)) {
    if (auto VD = dyn_cast<VarDecl>(D->getDecl()->getCanonicalDecl()))
      addVariableOccurence(VD, D);
  }

  // Recursively check all children of the given statement.
  for (const Stmt *Child : S->children()) {
    addVariables(Child);
  }
}

unsigned VariablePattern::countPatternDifferences(
    const VariablePattern &Other,
    VariablePattern::SuspiciousClonePair *FirstMismatch) {
  unsigned NumberOfDifferences = 0;

  assert(Other.Occurences.size() == Occurences.size());
  for (unsigned i = 0; i < Occurences.size(); ++i) {
    auto ThisOccurence = Occurences[i];
    auto OtherOccurence = Other.Occurences[i];
    if (ThisOccurence.KindID == OtherOccurence.KindID)
      continue;

    ++NumberOfDifferences;

    // If FirstMismatch is not a nullptr, we need to store information about
    // the first difference between the two patterns.
    if (FirstMismatch == nullptr)
      continue;

    // Only proceed if we just found the first difference as we only store
    // information about the first difference.
    if (NumberOfDifferences != 1)
      continue;

    const VarDecl *FirstSuggestion = nullptr;
    // If there is a variable available in the list of referenced variables
    // which wouldn't break the pattern if it is used in place of the
    // current variable, we provide this variable as the suggested fix.
    if (OtherOccurence.KindID < Variables.size())
      FirstSuggestion = Variables[OtherOccurence.KindID];

    // Store information about the first clone.
    FirstMismatch->FirstCloneInfo =
        VariablePattern::SuspiciousClonePair::SuspiciousCloneInfo(
            Variables[ThisOccurence.KindID], ThisOccurence.Mention,
            FirstSuggestion);

    // Same as above but with the other clone. We do this for both clones as
    // we don't know which clone is the one containing the unintended
    // pattern error.
    const VarDecl *SecondSuggestion = nullptr;
    if (ThisOccurence.KindID < Other.Variables.size())
      SecondSuggestion = Other.Variables[ThisOccurence.KindID];

    // Store information about the second clone.
    FirstMismatch->SecondCloneInfo =
        VariablePattern::SuspiciousClonePair::SuspiciousCloneInfo(
            Other.Variables[OtherOccurence.KindID], OtherOccurence.Mention,
            SecondSuggestion);

    // SuspiciousClonePair guarantees that the first clone always has a
    // suggested variable associated with it. As we know that one of the two
    // clones in the pair always has suggestion, we swap the two clones
    // in case the first clone has no suggested variable which means that
    // the second clone has a suggested variable and should be first.
    if (!FirstMismatch->FirstCloneInfo.Suggestion)
      std::swap(FirstMismatch->FirstCloneInfo, FirstMismatch->SecondCloneInfo);

    // This ensures that we always have at least one suggestion in a pair.
    assert(FirstMismatch->FirstCloneInfo.Suggestion);
  }

  return NumberOfDifferences;
}
