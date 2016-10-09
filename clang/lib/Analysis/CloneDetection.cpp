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

StmtSequence::StmtSequence(const CompoundStmt *Stmt, ASTContext &Context,
                           unsigned StartIndex, unsigned EndIndex)
    : S(Stmt), Context(&Context), StartIndex(StartIndex), EndIndex(EndIndex) {
  assert(Stmt && "Stmt must not be a nullptr");
  assert(StartIndex < EndIndex && "Given array should not be empty");
  assert(EndIndex <= Stmt->size() && "Given array too big for this Stmt");
}

StmtSequence::StmtSequence(const Stmt *Stmt, ASTContext &Context)
    : S(Stmt), Context(&Context), StartIndex(0), EndIndex(0) {}

StmtSequence::StmtSequence()
    : S(nullptr), Context(nullptr), StartIndex(0), EndIndex(0) {}

bool StmtSequence::contains(const StmtSequence &Other) const {
  // If both sequences reside in different translation units, they can never
  // contain each other.
  if (Context != Other.Context)
    return false;

  const SourceManager &SM = Context->getSourceManager();

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

SourceLocation StmtSequence::getStartLoc() const {
  return front()->getLocStart();
}

SourceLocation StmtSequence::getEndLoc() const { return back()->getLocEnd(); }

SourceRange StmtSequence::getSourceRange() const {
  return SourceRange(getStartLoc(), getEndLoc());
}

namespace {

/// \brief Analyzes the pattern of the referenced variables in a statement.
class VariablePattern {

  /// \brief Describes an occurence of a variable reference in a statement.
  struct VariableOccurence {
    /// The index of the associated VarDecl in the Variables vector.
    size_t KindID;
    /// The statement in the code where the variable was referenced.
    const Stmt *Mention;

    VariableOccurence(size_t KindID, const Stmt *Mention)
        : KindID(KindID), Mention(Mention) {}
  };

  /// All occurences of referenced variables in the order of appearance.
  std::vector<VariableOccurence> Occurences;
  /// List of referenced variables in the order of appearance.
  /// Every item in this list is unique.
  std::vector<const VarDecl *> Variables;

  /// \brief Adds a new variable referenced to this pattern.
  /// \param VarDecl The declaration of the variable that is referenced.
  /// \param Mention The SourceRange where this variable is referenced.
  void addVariableOccurence(const VarDecl *VarDecl, const Stmt *Mention) {
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

  /// \brief Adds each referenced variable from the given statement.
  void addVariables(const Stmt *S) {
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

public:
  /// \brief Creates an VariablePattern object with information about the given
  ///        StmtSequence.
  VariablePattern(const StmtSequence &Sequence) {
    for (const Stmt *S : Sequence)
      addVariables(S);
  }

  /// \brief Counts the differences between this pattern and the given one.
  /// \param Other The given VariablePattern to compare with.
  /// \param FirstMismatch Output parameter that will be filled with information
  ///        about the first difference between the two patterns. This parameter
  ///        can be a nullptr, in which case it will be ignored.
  /// \return Returns the number of differences between the pattern this object
  ///         is following and the given VariablePattern.
  ///
  /// For example, the following statements all have the same pattern and this
  /// function would return zero:
  ///
  ///   if (a < b) return a; return b;
  ///   if (x < y) return x; return y;
  ///   if (u2 < u1) return u2; return u1;
  ///
  /// But the following statement has a different pattern (note the changed
  /// variables in the return statements) and would have two differences when
  /// compared with one of the statements above.
  ///
  ///   if (a < b) return b; return a;
  ///
  /// This function should only be called if the related statements of the given
  /// pattern and the statements of this objects are clones of each other.
  unsigned countPatternDifferences(
      const VariablePattern &Other,
      CloneDetector::SuspiciousClonePair *FirstMismatch = nullptr) {
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
          CloneDetector::SuspiciousClonePair::SuspiciousCloneInfo(
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
          CloneDetector::SuspiciousClonePair::SuspiciousCloneInfo(
              Other.Variables[OtherOccurence.KindID], OtherOccurence.Mention,
              SecondSuggestion);

      // SuspiciousClonePair guarantees that the first clone always has a
      // suggested variable associated with it. As we know that one of the two
      // clones in the pair always has suggestion, we swap the two clones
      // in case the first clone has no suggested variable which means that
      // the second clone has a suggested variable and should be first.
      if (!FirstMismatch->FirstCloneInfo.Suggestion)
        std::swap(FirstMismatch->FirstCloneInfo,
                  FirstMismatch->SecondCloneInfo);

      // This ensures that we always have at least one suggestion in a pair.
      assert(FirstMismatch->FirstCloneInfo.Suggestion);
    }

    return NumberOfDifferences;
  }
};
}

/// \brief Prints the macro name that contains the given SourceLocation into
///        the given raw_string_ostream.
static void printMacroName(llvm::raw_string_ostream &MacroStack,
                           ASTContext &Context, SourceLocation Loc) {
  MacroStack << Lexer::getImmediateMacroName(Loc, Context.getSourceManager(),
                                             Context.getLangOpts());

  // Add an empty space at the end as a padding to prevent
  // that macro names concatenate to the names of other macros.
  MacroStack << " ";
}

/// \brief Returns a string that represents all macro expansions that
///        expanded into the given SourceLocation.
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
/// \brief Collects the data of a single Stmt.
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
  /// \brief The data sink to which all data is forwarded.
  T &DataConsumer;

public:
  /// \brief Collects data of the given Stmt.
  /// \param S The given statement.
  /// \param Context The ASTContext of S.
  /// \param DataConsumer The data sink to which all data is forwarded.
  StmtDataCollector(const Stmt *S, ASTContext &Context, T &DataConsumer)
      : Context(Context), DataConsumer(DataConsumer) {
    this->Visit(S);
  }

  // Below are utility methods for appending different data to the vector.

  void addData(CloneDetector::DataPiece Integer) {
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
    addData(static_cast<CloneDetector::DataPiece>(numDecls));
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

namespace {
/// Generates CloneSignatures for a set of statements and stores the results in
/// a CloneDetector object.
class CloneSignatureGenerator {

  CloneDetector &CD;
  ASTContext &Context;

  /// \brief Generates CloneSignatures for all statements in the given statement
  /// tree and stores them in the CloneDetector.
  ///
  /// \param S The root of the given statement tree.
  /// \param ParentMacroStack A string representing the macros that generated
  ///                         the parent statement or an empty string if no
  ///                         macros generated the parent statement.
  ///                         See getMacroStack() for generating such a string.
  /// \return The CloneSignature of the root statement.
  CloneDetector::CloneSignature
  generateSignatures(const Stmt *S, const std::string &ParentMacroStack) {
    // Create an empty signature that will be filled in this method.
    CloneDetector::CloneSignature Signature;

    llvm::MD5 Hash;

    // Collect all relevant data from S and hash it.
    StmtDataCollector<llvm::MD5>(S, Context, Hash);

    // Look up what macros expanded into the current statement.
    std::string StartMacroStack = getMacroStack(S->getLocStart(), Context);
    std::string EndMacroStack = getMacroStack(S->getLocEnd(), Context);

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
      Signature.Complexity = 0;
    }

    // Storage for the signatures of the direct child statements. This is only
    // needed if the current statement is a CompoundStmt.
    std::vector<CloneDetector::CloneSignature> ChildSignatures;
    const CompoundStmt *CS = dyn_cast<const CompoundStmt>(S);

    // The signature of a statement includes the signatures of its children.
    // Therefore we create the signatures for every child and add them to the
    // current signature.
    for (const Stmt *Child : S->children()) {
      // Some statements like 'if' can have nullptr children that we will skip.
      if (!Child)
        continue;

      // Recursive call to create the signature of the child statement. This
      // will also create and store all clone groups in this child statement.
      // We pass only the StartMacroStack along to keep things simple.
      auto ChildSignature = generateSignatures(Child, StartMacroStack);

      // Add the collected data to the signature of the current statement.
      Signature.Complexity += ChildSignature.Complexity;
      Hash.update(StringRef(reinterpret_cast<char *>(&ChildSignature.Hash),
                            sizeof(ChildSignature.Hash)));

      // If the current statement is a CompoundStatement, we need to store the
      // signature for the generation of the sub-sequences.
      if (CS)
        ChildSignatures.push_back(ChildSignature);
    }

    // If the current statement is a CompoundStmt, we also need to create the
    // clone groups from the sub-sequences inside the children.
    if (CS)
      handleSubSequences(CS, ChildSignatures);

    // Create the final hash code for the current signature.
    llvm::MD5::MD5Result HashResult;
    Hash.final(HashResult);

    // Copy as much of the generated hash code to the signature's hash code.
    std::memcpy(&Signature.Hash, &HashResult,
                std::min(sizeof(Signature.Hash), sizeof(HashResult)));

    // Save the signature for the current statement in the CloneDetector object.
    CD.add(StmtSequence(S, Context), Signature);

    return Signature;
  }

  /// \brief Adds all possible sub-sequences in the child array of the given
  ///        CompoundStmt to the CloneDetector.
  /// \param CS The given CompoundStmt.
  /// \param ChildSignatures A list of calculated signatures for each child in
  ///                        the given CompoundStmt.
  void handleSubSequences(
      const CompoundStmt *CS,
      const std::vector<CloneDetector::CloneSignature> &ChildSignatures) {

    // FIXME: This function has quadratic runtime right now. Check if skipping
    // this function for too long CompoundStmts is an option.

    // The length of the sub-sequence. We don't need to handle sequences with
    // the length 1 as they are already handled in CollectData().
    for (unsigned Length = 2; Length <= CS->size(); ++Length) {
      // The start index in the body of the CompoundStmt. We increase the
      // position until the end of the sub-sequence reaches the end of the
      // CompoundStmt body.
      for (unsigned Pos = 0; Pos <= CS->size() - Length; ++Pos) {
        // Create an empty signature and add the signatures of all selected
        // child statements to it.
        CloneDetector::CloneSignature SubSignature;
        llvm::MD5 SubHash;

        for (unsigned i = Pos; i < Pos + Length; ++i) {
          SubSignature.Complexity += ChildSignatures[i].Complexity;
          size_t ChildHash = ChildSignatures[i].Hash;

          SubHash.update(StringRef(reinterpret_cast<char *>(&ChildHash),
                                sizeof(ChildHash)));
        }

        // Create the final hash code for the current signature.
        llvm::MD5::MD5Result HashResult;
        SubHash.final(HashResult);

        // Copy as much of the generated hash code to the signature's hash code.
        std::memcpy(&SubSignature.Hash, &HashResult,
                    std::min(sizeof(SubSignature.Hash), sizeof(HashResult)));

        // Save the signature together with the information about what children
        // sequence we selected.
        CD.add(StmtSequence(CS, Context, Pos, Pos + Length), SubSignature);
      }
    }
  }

public:
  explicit CloneSignatureGenerator(CloneDetector &CD, ASTContext &Context)
      : CD(CD), Context(Context) {}

  /// \brief Generates signatures for all statements in the given function body.
  void consumeCodeBody(const Stmt *S) { generateSignatures(S, ""); }
};
} // end anonymous namespace

void CloneDetector::analyzeCodeBody(const Decl *D) {
  assert(D);
  assert(D->hasBody());
  CloneSignatureGenerator Generator(*this, D->getASTContext());
  Generator.consumeCodeBody(D->getBody());
}

void CloneDetector::add(const StmtSequence &S,
                        const CloneSignature &Signature) {
  Sequences.push_back(std::make_pair(Signature, S));
}

namespace {
/// \brief Returns true if and only if \p Stmt contains at least one other
/// sequence in the \p Group.
bool containsAnyInGroup(StmtSequence &Stmt, CloneDetector::CloneGroup &Group) {
  for (StmtSequence &GroupStmt : Group.Sequences) {
    if (Stmt.contains(GroupStmt))
      return true;
  }
  return false;
}

/// \brief Returns true if and only if all sequences in \p OtherGroup are
/// contained by a sequence in \p Group.
bool containsGroup(CloneDetector::CloneGroup &Group,
                   CloneDetector::CloneGroup &OtherGroup) {
  // We have less sequences in the current group than we have in the other,
  // so we will never fulfill the requirement for returning true. This is only
  // possible because we know that a sequence in Group can contain at most
  // one sequence in OtherGroup.
  if (Group.Sequences.size() < OtherGroup.Sequences.size())
    return false;

  for (StmtSequence &Stmt : Group.Sequences) {
    if (!containsAnyInGroup(Stmt, OtherGroup))
      return false;
  }
  return true;
}
} // end anonymous namespace

namespace {
/// \brief Wrapper around FoldingSetNodeID that it can be used as the template
///        argument of the StmtDataCollector.
class FoldingSetNodeIDWrapper {

  llvm::FoldingSetNodeID &FS;

public:
  FoldingSetNodeIDWrapper(llvm::FoldingSetNodeID &FS) : FS(FS) {}

  void update(StringRef Str) { FS.AddString(Str); }
};
} // end anonymous namespace

/// \brief Writes the relevant data from all statements and child statements
///        in the given StmtSequence into the given FoldingSetNodeID.
static void CollectStmtSequenceData(const StmtSequence &Sequence,
                                    FoldingSetNodeIDWrapper &OutputData) {
  for (const Stmt *S : Sequence) {
    StmtDataCollector<FoldingSetNodeIDWrapper>(S, Sequence.getASTContext(),
                                               OutputData);

    for (const Stmt *Child : S->children()) {
      if (!Child)
        continue;

      CollectStmtSequenceData(StmtSequence(Child, Sequence.getASTContext()),
                              OutputData);
    }
  }
}

/// \brief Returns true if both sequences are clones of each other.
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

/// \brief Finds all actual clone groups in a single group of presumed clones.
/// \param Result Output parameter to which all found groups are added.
/// \param Group A group of presumed clones. The clones are allowed to have a
///              different variable pattern and may not be actual clones of each
///              other.
/// \param CheckVariablePattern If true, every clone in a group that was added
///              to the output follows the same variable pattern as the other
///              clones in its group.
static void createCloneGroups(std::vector<CloneDetector::CloneGroup> &Result,
                              const CloneDetector::CloneGroup &Group,
                              bool CheckVariablePattern) {
  // We remove the Sequences one by one, so a list is more appropriate.
  std::list<StmtSequence> UnassignedSequences(Group.Sequences.begin(),
                                              Group.Sequences.end());

  // Search for clones as long as there could be clones in UnassignedSequences.
  while (UnassignedSequences.size() > 1) {

    // Pick the first Sequence as a protoype for a new clone group.
    StmtSequence Prototype = UnassignedSequences.front();
    UnassignedSequences.pop_front();

    CloneDetector::CloneGroup FilteredGroup(Prototype, Group.Signature);

    // Analyze the variable pattern of the prototype. Every other StmtSequence
    // needs to have the same pattern to get into the new clone group.
    VariablePattern PrototypeFeatures(Prototype);

    // Search all remaining StmtSequences for an identical variable pattern
    // and assign them to our new clone group.
    auto I = UnassignedSequences.begin(), E = UnassignedSequences.end();
    while (I != E) {
      // If the sequence doesn't fit to the prototype, we have encountered
      // an unintended hash code collision and we skip it.
      if (!areSequencesClones(Prototype, *I)) {
        ++I;
        continue;
      }

      // If we weren't asked to check for a matching variable pattern in clone
      // groups we can add the sequence now to the new clone group.
      // If we were asked to check for matching variable pattern, we first have
      // to check that there are no differences between the two patterns and
      // only proceed if they match.
      if (!CheckVariablePattern ||
          VariablePattern(*I).countPatternDifferences(PrototypeFeatures) == 0) {
        FilteredGroup.Sequences.push_back(*I);
        I = UnassignedSequences.erase(I);
        continue;
      }

      // We didn't found a matching variable pattern, so we continue with the
      // next sequence.
      ++I;
    }

    // Add a valid clone group to the list of found clone groups.
    if (!FilteredGroup.isValid())
      continue;

    Result.push_back(FilteredGroup);
  }
}

void CloneDetector::findClones(std::vector<CloneGroup> &Result,
                               unsigned MinGroupComplexity,
                               bool CheckPatterns) {
  // A shortcut (and necessary for the for-loop later in this function).
  if (Sequences.empty())
    return;

  // We need to search for groups of StmtSequences with the same hash code to
  // create our initial clone groups. By sorting all known StmtSequences by
  // their hash value we make sure that StmtSequences with the same hash code
  // are grouped together in the Sequences vector.
  // Note: We stable sort here because the StmtSequences are added in the order
  // in which they appear in the source file. We want to preserve that order
  // because we also want to report them in that order in the CloneChecker.
  std::stable_sort(Sequences.begin(), Sequences.end(),
                   [](std::pair<CloneSignature, StmtSequence> LHS,
                      std::pair<CloneSignature, StmtSequence> RHS) {
                     return LHS.first.Hash < RHS.first.Hash;
                   });

  std::vector<CloneGroup> CloneGroups;

  // Check for each CloneSignature if its successor has the same hash value.
  // We don't check the last CloneSignature as it has no successor.
  // Note: The 'size - 1' in the condition is safe because we check for an empty
  // Sequences vector at the beginning of this function.
  for (unsigned i = 0; i < Sequences.size() - 1; ++i) {
    const auto Current = Sequences[i];
    const auto Next = Sequences[i + 1];

    if (Current.first.Hash != Next.first.Hash)
      continue;

    // It's likely that we just found an sequence of CloneSignatures that
    // represent a CloneGroup, so we create a new group and start checking and
    // adding the CloneSignatures in this sequence.
    CloneGroup Group;
    Group.Signature = Current.first;

    for (; i < Sequences.size(); ++i) {
      const auto &Signature = Sequences[i];

      // A different hash value means we have reached the end of the sequence.
      if (Current.first.Hash != Signature.first.Hash) {
        // The current Signature could be the start of a new CloneGroup. So we
        // decrement i so that we visit it again in the outer loop.
        // Note: i can never be 0 at this point because we are just comparing
        // the hash of the Current CloneSignature with itself in the 'if' above.
        assert(i != 0);
        --i;
        break;
      }

      // Skip CloneSignatures that won't pass the complexity requirement.
      if (Signature.first.Complexity < MinGroupComplexity)
        continue;

      Group.Sequences.push_back(Signature.second);
    }

    // There is a chance that we haven't found more than two fitting
    // CloneSignature because not enough CloneSignatures passed the complexity
    // requirement. As a CloneGroup with less than two members makes no sense,
    // we ignore this CloneGroup and won't add it to the result.
    if (!Group.isValid())
      continue;

    CloneGroups.push_back(Group);
  }

  // Add every valid clone group that fulfills the complexity requirement.
  for (const CloneGroup &Group : CloneGroups) {
    createCloneGroups(Result, Group, CheckPatterns);
  }

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

void CloneDetector::findSuspiciousClones(
    std::vector<CloneDetector::SuspiciousClonePair> &Result,
    unsigned MinGroupComplexity) {
  std::vector<CloneGroup> Clones;
  // Reuse the normal search for clones but specify that the clone groups don't
  // need to have a common referenced variable pattern so that we can manually
  // search for the kind of pattern errors this function is supposed to find.
  findClones(Clones, MinGroupComplexity, false);

  for (const CloneGroup &Group : Clones) {
    for (unsigned i = 0; i < Group.Sequences.size(); ++i) {
      VariablePattern PatternA(Group.Sequences[i]);

      for (unsigned j = i + 1; j < Group.Sequences.size(); ++j) {
        VariablePattern PatternB(Group.Sequences[j]);

        CloneDetector::SuspiciousClonePair ClonePair;
        // For now, we only report clones which break the variable pattern just
        // once because multiple differences in a pattern are an indicator that
        // those differences are maybe intended (e.g. because it's actually
        // a different algorithm).
        // TODO: In very big clones even multiple variables can be unintended,
        // so replacing this number with a percentage could better handle such
        // cases. On the other hand it could increase the false-positive rate
        // for all clones if the percentage is too high.
        if (PatternA.countPatternDifferences(PatternB, &ClonePair) == 1) {
          Result.push_back(ClonePair);
          break;
        }
      }
    }
  }
}
