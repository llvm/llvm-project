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
#include "llvm/ADT/StringRef.h"

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

namespace {

/// \brief Analyzes the pattern of the referenced variables in a statement.
class VariablePattern {

  /// \brief Describes an occurence of a variable reference in a statement.
  struct VariableOccurence {
    /// The index of the associated VarDecl in the Variables vector.
    size_t KindID;

    VariableOccurence(size_t KindID) : KindID(KindID) {}
  };

  /// All occurences of referenced variables in the order of appearance.
  std::vector<VariableOccurence> Occurences;
  /// List of referenced variables in the order of appearance.
  /// Every item in this list is unique.
  std::vector<const VarDecl *> Variables;

  /// \brief Adds a new variable referenced to this pattern.
  /// \param VarDecl The declaration of the variable that is referenced.
  void addVariableOccurence(const VarDecl *VarDecl) {
    // First check if we already reference this variable
    for (size_t KindIndex = 0; KindIndex < Variables.size(); ++KindIndex) {
      if (Variables[KindIndex] == VarDecl) {
        // If yes, add a new occurence that points to the existing entry in
        // the Variables vector.
        Occurences.emplace_back(KindIndex);
        return;
      }
    }
    // If this variable wasn't already referenced, add it to the list of
    // referenced variables and add a occurence that points to this new entry.
    Occurences.emplace_back(Variables.size());
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
        addVariableOccurence(VD);
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

  /// \brief Compares this pattern with the given one.
  /// \param Other The given VariablePattern to compare with.
  /// \return Returns true if and only if the references variables in this
  ///         object follow the same pattern than the ones in the given
  ///         VariablePattern.
  ///
  /// For example, the following statements all have the same pattern:
  ///
  ///   if (a < b) return a; return b;
  ///   if (x < y) return x; return y;
  ///   if (u2 < u1) return u2; return u1;
  ///
  /// but the following statement has a different pattern (note the changed
  /// variables in the return statements).
  ///
  ///   if (a < b) return b; return a;
  ///
  /// This function should only be called if the related statements of the given
  /// pattern and the statements of this objects are clones of each other.
  bool comparePattern(const VariablePattern &Other) {
    assert(Other.Occurences.size() == Occurences.size());
    for (unsigned i = 0; i < Occurences.size(); ++i) {
      if (Occurences[i].KindID != Other.Occurences[i].KindID) {
        return false;
      }
    }
    return true;
  }
};
}

namespace {
/// \brief Collects the data of a single Stmt.
///
/// This class defines what a code clone is: If it collects for two statements
/// the same data, then those two statements are considered to be clones of each
/// other.
class StmtDataCollector : public ConstStmtVisitor<StmtDataCollector> {

  ASTContext &Context;
  std::vector<CloneDetector::DataPiece> &CollectedData;

public:
  /// \brief Collects data of the given Stmt.
  /// \param S The given statement.
  /// \param Context The ASTContext of S.
  /// \param D The given data vector to which all collected data is appended.
  StmtDataCollector(const Stmt *S, ASTContext &Context,
                    std::vector<CloneDetector::DataPiece> &D)
      : Context(Context), CollectedData(D) {
    Visit(S);
  }

  // Below are utility methods for appending different data to the vector.

  void addData(CloneDetector::DataPiece Integer) {
    CollectedData.push_back(Integer);
  }

  // FIXME: The functions below add long strings to the data vector which are
  // probably not good for performance. Replace the strings with pointer values
  // or a some other unique integer.

  void addData(llvm::StringRef Str) {
    if (Str.empty())
      return;

    const size_t OldSize = CollectedData.size();

    const size_t PieceSize = sizeof(CloneDetector::DataPiece);
    // Calculate how many vector units we need to accomodate all string bytes.
    size_t RoundedUpPieceNumber = (Str.size() + PieceSize - 1) / PieceSize;
    // Allocate space for the string in the data vector.
    CollectedData.resize(CollectedData.size() + RoundedUpPieceNumber);

    // Copy the string to the allocated space at the end of the vector.
    std::memcpy(CollectedData.data() + OldSize, Str.data(), Str.size());
  }

  void addData(const QualType &QT) { addData(QT.getAsString()); }

// The functions below collect the class specific data of each Stmt subclass.

// Utility macro for defining a visit method for a given class. This method
// calls back to the ConstStmtVisitor to visit all parent classes.
#define DEF_ADD_DATA(CLASS, CODE)                                              \
  void Visit##CLASS(const CLASS *S) {                                          \
    CODE;                                                                      \
    ConstStmtVisitor<StmtDataCollector>::Visit##CLASS(S);                      \
  }

  DEF_ADD_DATA(Stmt, { addData(S->getStmtClass()); })
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
    if (S->getDirectCallee())
      addData(S->getDirectCallee()->getQualifiedNameAsString());
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
  /// \return The CloneSignature of the root statement.
  CloneDetector::CloneSignature generateSignatures(const Stmt *S) {
    // Create an empty signature that will be filled in this method.
    CloneDetector::CloneSignature Signature;

    // Collect all relevant data from S and put it into the empty signature.
    StmtDataCollector(S, Context, Signature.Data);

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
      auto ChildSignature = generateSignatures(Child);

      // Add the collected data to the signature of the current statement.
      Signature.add(ChildSignature);

      // If the current statement is a CompoundStatement, we need to store the
      // signature for the generation of the sub-sequences.
      if (CS)
        ChildSignatures.push_back(ChildSignature);
    }

    // If the current statement is a CompoundStmt, we also need to create the
    // clone groups from the sub-sequences inside the children.
    if (CS)
      handleSubSequences(CS, ChildSignatures);

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

        for (unsigned i = Pos; i < Pos + Length; ++i) {
          SubSignature.add(ChildSignatures[i]);
        }

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
  void consumeCodeBody(const Stmt *S) { generateSignatures(S); }
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
  // StringMap only works with StringRefs, so we create one for our data vector.
  auto &Data = Signature.Data;
  StringRef DataRef = StringRef(reinterpret_cast<const char *>(Data.data()),
                                Data.size() * sizeof(unsigned));

  // Search with the help of the signature if we already have encountered a
  // clone of the given StmtSequence.
  auto I = CloneGroupIndexes.find(DataRef);
  if (I == CloneGroupIndexes.end()) {
    // We haven't found an existing clone group, so we create a new clone group
    // for this StmtSequence and store the index of it in our search map.
    CloneGroupIndexes[DataRef] = CloneGroups.size();
    CloneGroups.emplace_back(S, Signature.Complexity);
    return;
  }

  // We have found an existing clone group and can expand it with the given
  // StmtSequence.
  CloneGroups[I->getValue()].Sequences.push_back(S);
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

/// \brief Finds all actual clone groups in a single group of presumed clones.
/// \param Result Output parameter to which all found groups are added. Every
///               clone in a group that was added this way follows the same
///               variable pattern as the other clones in its group.
/// \param Group A group of clones. The clones are allowed to have a different
///              variable pattern.
static void createCloneGroups(std::vector<CloneDetector::CloneGroup> &Result,
                              const CloneDetector::CloneGroup &Group) {
  // We remove the Sequences one by one, so a list is more appropriate.
  std::list<StmtSequence> UnassignedSequences(Group.Sequences.begin(),
                                              Group.Sequences.end());

  // Search for clones as long as there could be clones in UnassignedSequences.
  while (UnassignedSequences.size() > 1) {

    // Pick the first Sequence as a protoype for a new clone group.
    StmtSequence Prototype = UnassignedSequences.front();
    UnassignedSequences.pop_front();

    CloneDetector::CloneGroup FilteredGroup(Prototype, Group.Complexity);

    // Analyze the variable pattern of the prototype. Every other StmtSequence
    // needs to have the same pattern to get into the new clone group.
    VariablePattern PrototypeFeatures(Prototype);

    // Search all remaining StmtSequences for an identical variable pattern
    // and assign them to our new clone group.
    auto I = UnassignedSequences.begin(), E = UnassignedSequences.end();
    while (I != E) {
      if (VariablePattern(*I).comparePattern(PrototypeFeatures)) {
        FilteredGroup.Sequences.push_back(*I);
        I = UnassignedSequences.erase(I);
        continue;
      }
      ++I;
    }

    // Add a valid clone group to the list of found clone groups.
    if (!FilteredGroup.isValid())
      continue;

    Result.push_back(FilteredGroup);
  }
}

void CloneDetector::findClones(std::vector<CloneGroup> &Result,
                               unsigned MinGroupComplexity) {
  // Add every valid clone group that fulfills the complexity requirement.
  for (const CloneGroup &Group : CloneGroups) {
    if (Group.isValid() && Group.Complexity >= MinGroupComplexity) {
      createCloneGroups(Result, Group);
    }
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
