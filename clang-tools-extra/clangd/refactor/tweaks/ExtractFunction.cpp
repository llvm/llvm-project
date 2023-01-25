//===--- ExtractFunction.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Extracts statements to a new function and replaces the statements with a
// call to the new function.
// Before:
//   void f(int a) {
//     [[if(a < 5)
//       a = 5;]]
//   }
// After:
//   void extracted(int &a) {
//     if(a < 5)
//       a = 5;
//   }
//   void f(int a) {
//     extracted(a);
//   }
//
// - Only extract statements
// - Extracts from non-templated free functions only.
// - Parameters are const only if the declaration was const
//   - Always passed by l-value reference
// - Void return type
// - Cannot extract declarations that will be needed in the original function
//   after extraction.
// - Checks for broken control flow (break/continue without loop/switch)
//
// 1. ExtractFunction is the tweak subclass
//    - Prepare does basic analysis of the selection and is therefore fast.
//      Successful prepare doesn't always mean we can apply the tweak.
//    - Apply does a more detailed analysis and can be slower. In case of
//      failure, we let the user know that we are unable to perform extraction.
// 2. ExtractionZone store information about the range being extracted and the
//    enclosing function.
// 3. NewFunction stores properties of the extracted function and provides
//    methods for rendering it.
// 4. CapturedZoneInfo uses a RecursiveASTVisitor to capture information about
//    the extraction like declarations, existing return statements, etc.
// 5. getExtractedFunction is responsible for analyzing the CapturedZoneInfo and
//    creating a NewFunction.
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "FindTarget.h"
#include "ParsedAST.h"
#include "Selection.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Refactoring/Extract/SourceExtraction.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"
#include <optional>

namespace clang {
namespace clangd {
namespace {

using Node = SelectionTree::Node;

// ExtractionZone is the part of code that is being extracted.
// EnclosingFunction is the function/method inside which the zone lies.
// We split the file into 4 parts relative to extraction zone.
enum class ZoneRelative {
  Before,     // Before Zone and inside EnclosingFunction.
  Inside,     // Inside Zone.
  After,      // After Zone and inside EnclosingFunction.
  OutsideFunc // Outside EnclosingFunction.
};

enum FunctionDeclKind {
  InlineDefinition,
  ForwardDeclaration,
  OutOfLineDefinition
};

// A RootStmt is a statement that's fully selected including all it's children
// and it's parent is unselected.
// Check if a node is a root statement.
bool isRootStmt(const Node *N) {
  if (!N->ASTNode.get<Stmt>())
    return false;
  // Root statement cannot be partially selected.
  if (N->Selected == SelectionTree::Partial)
    return false;
  // Only DeclStmt can be an unselected RootStmt since VarDecls claim the entire
  // selection range in selectionTree.
  if (N->Selected == SelectionTree::Unselected && !N->ASTNode.get<DeclStmt>())
    return false;
  return true;
}

// Returns the (unselected) parent of all RootStmts given the commonAncestor.
// Returns null if:
// 1. any node is partially selected
// 2. If all completely selected nodes don't have the same common parent
// 3. Any child of Parent isn't a RootStmt.
// Returns null if any child is not a RootStmt.
// We only support extraction of RootStmts since it allows us to extract without
// having to change the selection range. Also, this means that any scope that
// begins in selection range, ends in selection range and any scope that begins
// outside the selection range, ends outside as well.
const Node *getParentOfRootStmts(const Node *CommonAnc) {
  if (!CommonAnc)
    return nullptr;
  const Node *Parent = nullptr;
  switch (CommonAnc->Selected) {
  case SelectionTree::Selection::Unselected:
    // Typically a block, with the { and } unselected, could also be ForStmt etc
    // Ensure all Children are RootStmts.
    Parent = CommonAnc;
    break;
  case SelectionTree::Selection::Partial:
    // Only a fully-selected single statement can be selected.
    return nullptr;
  case SelectionTree::Selection::Complete:
    // If the Common Ancestor is completely selected, then it's a root statement
    // and its parent will be unselected.
    Parent = CommonAnc->Parent;
    // If parent is a DeclStmt, even though it's unselected, we consider it a
    // root statement and return its parent. This is done because the VarDecls
    // claim the entire selection range of the Declaration and DeclStmt is
    // always unselected.
    if (Parent->ASTNode.get<DeclStmt>())
      Parent = Parent->Parent;
    break;
  }
  // Ensure all Children are RootStmts.
  return llvm::all_of(Parent->Children, isRootStmt) ? Parent : nullptr;
}

// The ExtractionZone class forms a view of the code wrt Zone.
struct ExtractionZone {
  // Parent of RootStatements being extracted.
  const Node *Parent = nullptr;
  // The half-open file range of the code being extracted.
  SourceRange ZoneRange;
  // The function inside which our zone resides.
  const FunctionDecl *EnclosingFunction = nullptr;
  // The half-open file range of the enclosing function.
  SourceRange EnclosingFuncRange;
  // Set of statements that form the ExtractionZone.
  llvm::DenseSet<const Stmt *> RootStmts;

  SourceLocation getInsertionPoint() const {
    return EnclosingFuncRange.getBegin();
  }
  bool isRootStmt(const Stmt *S) const;
  // The last root statement is important to decide where we need to insert a
  // semicolon after the extraction.
  const Node *getLastRootStmt() const { return Parent->Children.back(); }

  // Checks if declarations inside extraction zone are accessed afterwards.
  //
  // This performs a partial AST traversal proportional to the size of the
  // enclosing function, so it is possibly expensive.
  bool requiresHoisting(const SourceManager &SM,
                        const HeuristicResolver *Resolver) const {
    // First find all the declarations that happened inside extraction zone.
    llvm::SmallSet<const Decl *, 1> DeclsInExtZone;
    for (auto *RootStmt : RootStmts) {
      findExplicitReferences(
          RootStmt,
          [&DeclsInExtZone](const ReferenceLoc &Loc) {
            if (!Loc.IsDecl)
              return;
            DeclsInExtZone.insert(Loc.Targets.front());
          },
          Resolver);
    }
    // Early exit without performing expensive traversal below.
    if (DeclsInExtZone.empty())
      return false;
    // Then make sure they are not used outside the zone.
    for (const auto *S : EnclosingFunction->getBody()->children()) {
      if (SM.isBeforeInTranslationUnit(S->getSourceRange().getEnd(),
                                       ZoneRange.getEnd()))
        continue;
      bool HasPostUse = false;
      findExplicitReferences(
          S,
          [&](const ReferenceLoc &Loc) {
            if (HasPostUse ||
                SM.isBeforeInTranslationUnit(Loc.NameLoc, ZoneRange.getEnd()))
              return;
            HasPostUse = llvm::any_of(Loc.Targets,
                                      [&DeclsInExtZone](const Decl *Target) {
                                        return DeclsInExtZone.contains(Target);
                                      });
          },
          Resolver);
      if (HasPostUse)
        return true;
    }
    return false;
  }
};

// Whether the code in the extraction zone is guaranteed to return, assuming
// no broken control flow (unbound break/continue).
// This is a very naive check (does it end with a return stmt).
// Doing some rudimentary control flow analysis would cover more cases.
bool alwaysReturns(const ExtractionZone &EZ) {
  const Stmt *Last = EZ.getLastRootStmt()->ASTNode.get<Stmt>();
  // Unwrap enclosing (unconditional) compound statement.
  while (const auto *CS = llvm::dyn_cast<CompoundStmt>(Last)) {
    if (CS->body_empty())
      return false;
    Last = CS->body_back();
  }
  return llvm::isa<ReturnStmt>(Last);
}

bool ExtractionZone::isRootStmt(const Stmt *S) const {
  return RootStmts.contains(S);
}

// Finds the function in which the zone lies.
const FunctionDecl *findEnclosingFunction(const Node *CommonAnc) {
  // Walk up the SelectionTree until we find a function Decl
  for (const Node *CurNode = CommonAnc; CurNode; CurNode = CurNode->Parent) {
    // Don't extract from lambdas
    if (CurNode->ASTNode.get<LambdaExpr>())
      return nullptr;
    if (const FunctionDecl *Func = CurNode->ASTNode.get<FunctionDecl>()) {
      // FIXME: Support extraction from templated functions.
      if (Func->isTemplated())
        return nullptr;
      if (!Func->getBody())
        return nullptr;
      for (const auto *S : Func->getBody()->children()) {
        // During apply phase, we perform semantic analysis (e.g. figure out
        // what variables requires hoisting). We cannot perform those when the
        // body has invalid statements, so fail up front.
        if (!S)
          return nullptr;
      }
      return Func;
    }
  }
  return nullptr;
}

// Zone Range is the union of SourceRanges of all child Nodes in Parent since
// all child Nodes are RootStmts
std::optional<SourceRange> findZoneRange(const Node *Parent,
                                         const SourceManager &SM,
                                         const LangOptions &LangOpts) {
  SourceRange SR;
  if (auto BeginFileRange = toHalfOpenFileRange(
          SM, LangOpts, Parent->Children.front()->ASTNode.getSourceRange()))
    SR.setBegin(BeginFileRange->getBegin());
  else
    return std::nullopt;
  if (auto EndFileRange = toHalfOpenFileRange(
          SM, LangOpts, Parent->Children.back()->ASTNode.getSourceRange()))
    SR.setEnd(EndFileRange->getEnd());
  else
    return std::nullopt;
  return SR;
}

// Compute the range spanned by the enclosing function.
// FIXME: check if EnclosingFunction has any attributes as the AST doesn't
// always store the source range of the attributes and thus we end up extracting
// between the attributes and the EnclosingFunction.
std::optional<SourceRange>
computeEnclosingFuncRange(const FunctionDecl *EnclosingFunction,
                          const SourceManager &SM,
                          const LangOptions &LangOpts) {
  return toHalfOpenFileRange(SM, LangOpts, EnclosingFunction->getSourceRange());
}

// returns true if Child can be a single RootStmt being extracted from
// EnclosingFunc.
bool validSingleChild(const Node *Child, const FunctionDecl *EnclosingFunc) {
  // Don't extract expressions.
  // FIXME: We should extract expressions that are "statements" i.e. not
  // subexpressions
  if (Child->ASTNode.get<Expr>())
    return false;
  // Extracting the body of EnclosingFunc would remove it's definition.
  assert(EnclosingFunc->hasBody() &&
         "We should always be extracting from a function body.");
  if (Child->ASTNode.get<Stmt>() == EnclosingFunc->getBody())
    return false;
  return true;
}

// FIXME: Check we're not extracting from the initializer/condition of a control
// flow structure.
std::optional<ExtractionZone> findExtractionZone(const Node *CommonAnc,
                                                 const SourceManager &SM,
                                                 const LangOptions &LangOpts) {
  ExtractionZone ExtZone;
  ExtZone.Parent = getParentOfRootStmts(CommonAnc);
  if (!ExtZone.Parent || ExtZone.Parent->Children.empty())
    return std::nullopt;
  ExtZone.EnclosingFunction = findEnclosingFunction(ExtZone.Parent);
  if (!ExtZone.EnclosingFunction)
    return std::nullopt;
  // When there is a single RootStmt, we must check if it's valid for
  // extraction.
  if (ExtZone.Parent->Children.size() == 1 &&
      !validSingleChild(ExtZone.getLastRootStmt(), ExtZone.EnclosingFunction))
    return std::nullopt;
  if (auto FuncRange =
          computeEnclosingFuncRange(ExtZone.EnclosingFunction, SM, LangOpts))
    ExtZone.EnclosingFuncRange = *FuncRange;
  if (auto ZoneRange = findZoneRange(ExtZone.Parent, SM, LangOpts))
    ExtZone.ZoneRange = *ZoneRange;
  if (ExtZone.EnclosingFuncRange.isInvalid() || ExtZone.ZoneRange.isInvalid())
    return std::nullopt;

  for (const Node *Child : ExtZone.Parent->Children)
    ExtZone.RootStmts.insert(Child->ASTNode.get<Stmt>());

  return ExtZone;
}

// Stores information about the extracted function and provides methods for
// rendering it.
struct NewFunction {
  struct Parameter {
    std::string Name;
    QualType TypeInfo;
    bool PassByReference;
    unsigned OrderPriority; // Lower value parameters are preferred first.
    std::string render(const DeclContext *Context) const;
    bool operator<(const Parameter &Other) const {
      return OrderPriority < Other.OrderPriority;
    }
  };
  std::string Name = "extracted";
  QualType ReturnType;
  std::vector<Parameter> Parameters;
  SourceRange BodyRange;
  SourceLocation DefinitionPoint;
  std::optional<SourceLocation> ForwardDeclarationPoint;
  const CXXRecordDecl *EnclosingClass = nullptr;
  const NestedNameSpecifier *DefinitionQualifier = nullptr;
  const DeclContext *SemanticDC = nullptr;
  const DeclContext *SyntacticDC = nullptr;
  const DeclContext *ForwardDeclarationSyntacticDC = nullptr;
  bool CallerReturnsValue = false;
  bool Static = false;
  ConstexprSpecKind Constexpr = ConstexprSpecKind::Unspecified;
  bool Const = false;

  // Decides whether the extracted function body and the function call need a
  // semicolon after extraction.
  tooling::ExtractionSemicolonPolicy SemicolonPolicy;
  const LangOptions *LangOpts;
  NewFunction(tooling::ExtractionSemicolonPolicy SemicolonPolicy,
              const LangOptions *LangOpts)
      : SemicolonPolicy(SemicolonPolicy), LangOpts(LangOpts) {}
  // Render the call for this function.
  std::string renderCall() const;
  // Render the definition for this function.
  std::string renderDeclaration(FunctionDeclKind K,
                                const DeclContext &SemanticDC,
                                const DeclContext &SyntacticDC,
                                const SourceManager &SM) const;

private:
  std::string
  renderParametersForDeclaration(const DeclContext &Enclosing) const;
  std::string renderParametersForCall() const;
  std::string renderSpecifiers(FunctionDeclKind K) const;
  std::string renderQualifiers() const;
  std::string renderDeclarationName(FunctionDeclKind K) const;
  // Generate the function body.
  std::string getFuncBody(const SourceManager &SM) const;
};

std::string NewFunction::renderParametersForDeclaration(
    const DeclContext &Enclosing) const {
  std::string Result;
  bool NeedCommaBefore = false;
  for (const Parameter &P : Parameters) {
    if (NeedCommaBefore)
      Result += ", ";
    NeedCommaBefore = true;
    Result += P.render(&Enclosing);
  }
  return Result;
}

std::string NewFunction::renderParametersForCall() const {
  std::string Result;
  bool NeedCommaBefore = false;
  for (const Parameter &P : Parameters) {
    if (NeedCommaBefore)
      Result += ", ";
    NeedCommaBefore = true;
    Result += P.Name;
  }
  return Result;
}

std::string NewFunction::renderSpecifiers(FunctionDeclKind K) const {
  std::string Attributes;

  if (Static && K != FunctionDeclKind::OutOfLineDefinition) {
    Attributes += "static ";
  }

  switch (Constexpr) {
  case ConstexprSpecKind::Unspecified:
  case ConstexprSpecKind::Constinit:
    break;
  case ConstexprSpecKind::Constexpr:
    Attributes += "constexpr ";
    break;
  case ConstexprSpecKind::Consteval:
    Attributes += "consteval ";
    break;
  }

  return Attributes;
}

std::string NewFunction::renderQualifiers() const {
  std::string Attributes;

  if (Const) {
    Attributes += " const";
  }

  return Attributes;
}

std::string NewFunction::renderDeclarationName(FunctionDeclKind K) const {
  if (DefinitionQualifier == nullptr || K != OutOfLineDefinition) {
    return Name;
  }

  std::string QualifierName;
  llvm::raw_string_ostream Oss(QualifierName);
  DefinitionQualifier->print(Oss, *LangOpts);
  return llvm::formatv("{0}{1}", QualifierName, Name);
}

std::string NewFunction::renderCall() const {
  return std::string(
      llvm::formatv("{0}{1}({2}){3}", CallerReturnsValue ? "return " : "", Name,
                    renderParametersForCall(),
                    (SemicolonPolicy.isNeededInOriginalFunction() ? ";" : "")));
}

std::string NewFunction::renderDeclaration(FunctionDeclKind K,
                                           const DeclContext &SemanticDC,
                                           const DeclContext &SyntacticDC,
                                           const SourceManager &SM) const {
  std::string Declaration = std::string(llvm::formatv(
      "{0}{1} {2}({3}){4}", renderSpecifiers(K),
      printType(ReturnType, SyntacticDC), renderDeclarationName(K),
      renderParametersForDeclaration(SemanticDC), renderQualifiers()));

  switch (K) {
  case ForwardDeclaration:
    return std::string(llvm::formatv("{0};\n", Declaration));
  case OutOfLineDefinition:
  case InlineDefinition:
    return std::string(
        llvm::formatv("{0} {\n{1}\n}\n", Declaration, getFuncBody(SM)));
    break;
  }
  llvm_unreachable("Unsupported FunctionDeclKind enum");
}

std::string NewFunction::getFuncBody(const SourceManager &SM) const {
  // FIXME: Generate tooling::Replacements instead of std::string to
  // - hoist decls
  // - add return statement
  // - Add semicolon
  return toSourceCode(SM, BodyRange).str() +
         (SemicolonPolicy.isNeededInExtractedFunction() ? ";" : "");
}

std::string NewFunction::Parameter::render(const DeclContext *Context) const {
  return printType(TypeInfo, *Context) + (PassByReference ? " &" : " ") + Name;
}

// Stores captured information about Extraction Zone.
struct CapturedZoneInfo {
  struct DeclInformation {
    const Decl *TheDecl;
    ZoneRelative DeclaredIn;
    // index of the declaration or first reference.
    unsigned DeclIndex;
    bool IsReferencedInZone = false;
    bool IsReferencedInPostZone = false;
    // FIXME: Capture mutation information
    DeclInformation(const Decl *TheDecl, ZoneRelative DeclaredIn,
                    unsigned DeclIndex)
        : TheDecl(TheDecl), DeclaredIn(DeclaredIn), DeclIndex(DeclIndex){};
    // Marks the occurence of a reference for this declaration
    void markOccurence(ZoneRelative ReferenceLoc);
  };
  // Maps Decls to their DeclInfo
  llvm::DenseMap<const Decl *, DeclInformation> DeclInfoMap;
  bool HasReturnStmt = false; // Are there any return statements in the zone?
  bool AlwaysReturns = false; // Does the zone always return?
  // Control flow is broken if we are extracting a break/continue without a
  // corresponding parent loop/switch
  bool BrokenControlFlow = false;
  // FIXME: capture TypeAliasDecl and UsingDirectiveDecl
  // FIXME: Capture type information as well.
  DeclInformation *createDeclInfo(const Decl *D, ZoneRelative RelativeLoc);
  DeclInformation *getDeclInfoFor(const Decl *D);
};

CapturedZoneInfo::DeclInformation *
CapturedZoneInfo::createDeclInfo(const Decl *D, ZoneRelative RelativeLoc) {
  // The new Decl's index is the size of the map so far.
  auto InsertionResult = DeclInfoMap.insert(
      {D, DeclInformation(D, RelativeLoc, DeclInfoMap.size())});
  // Return the newly created DeclInfo
  return &InsertionResult.first->second;
}

CapturedZoneInfo::DeclInformation *
CapturedZoneInfo::getDeclInfoFor(const Decl *D) {
  // If the Decl doesn't exist, we
  auto Iter = DeclInfoMap.find(D);
  if (Iter == DeclInfoMap.end())
    return nullptr;
  return &Iter->second;
}

void CapturedZoneInfo::DeclInformation::markOccurence(
    ZoneRelative ReferenceLoc) {
  switch (ReferenceLoc) {
  case ZoneRelative::Inside:
    IsReferencedInZone = true;
    break;
  case ZoneRelative::After:
    IsReferencedInPostZone = true;
    break;
  default:
    break;
  }
}

bool isLoop(const Stmt *S) {
  return isa<ForStmt>(S) || isa<DoStmt>(S) || isa<WhileStmt>(S) ||
         isa<CXXForRangeStmt>(S);
}

// Captures information from Extraction Zone
CapturedZoneInfo captureZoneInfo(const ExtractionZone &ExtZone) {
  // We use the ASTVisitor instead of using the selection tree since we need to
  // find references in the PostZone as well.
  // FIXME: Check which statements we don't allow to extract.
  class ExtractionZoneVisitor
      : public clang::RecursiveASTVisitor<ExtractionZoneVisitor> {
  public:
    ExtractionZoneVisitor(const ExtractionZone &ExtZone) : ExtZone(ExtZone) {
      TraverseDecl(const_cast<FunctionDecl *>(ExtZone.EnclosingFunction));
    }

    bool TraverseStmt(Stmt *S) {
      if (!S)
        return true;
      bool IsRootStmt = ExtZone.isRootStmt(const_cast<const Stmt *>(S));
      // If we are starting traversal of a RootStmt, we are somewhere inside
      // ExtractionZone
      if (IsRootStmt)
        CurrentLocation = ZoneRelative::Inside;
      addToLoopSwitchCounters(S, 1);
      // Traverse using base class's TraverseStmt
      RecursiveASTVisitor::TraverseStmt(S);
      addToLoopSwitchCounters(S, -1);
      // We set the current location as after since next stmt will either be a
      // RootStmt (handled at the beginning) or after extractionZone
      if (IsRootStmt)
        CurrentLocation = ZoneRelative::After;
      return true;
    }

    // Add Increment to CurNumberOf{Loops,Switch} if statement is
    // {Loop,Switch} and inside Extraction Zone.
    void addToLoopSwitchCounters(Stmt *S, int Increment) {
      if (CurrentLocation != ZoneRelative::Inside)
        return;
      if (isLoop(S))
        CurNumberOfNestedLoops += Increment;
      else if (isa<SwitchStmt>(S))
        CurNumberOfSwitch += Increment;
    }

    bool VisitDecl(Decl *D) {
      Info.createDeclInfo(D, CurrentLocation);
      return true;
    }

    bool VisitDeclRefExpr(DeclRefExpr *DRE) {
      // Find the corresponding Decl and mark it's occurrence.
      const Decl *D = DRE->getDecl();
      auto *DeclInfo = Info.getDeclInfoFor(D);
      // If no Decl was found, the Decl must be outside the enclosingFunc.
      if (!DeclInfo)
        DeclInfo = Info.createDeclInfo(D, ZoneRelative::OutsideFunc);
      DeclInfo->markOccurence(CurrentLocation);
      // FIXME: check if reference mutates the Decl being referred.
      return true;
    }

    bool VisitReturnStmt(ReturnStmt *Return) {
      if (CurrentLocation == ZoneRelative::Inside)
        Info.HasReturnStmt = true;
      return true;
    }

    bool VisitBreakStmt(BreakStmt *Break) {
      // Control flow is broken if break statement is selected without any
      // parent loop or switch statement.
      if (CurrentLocation == ZoneRelative::Inside &&
          !(CurNumberOfNestedLoops || CurNumberOfSwitch))
        Info.BrokenControlFlow = true;
      return true;
    }

    bool VisitContinueStmt(ContinueStmt *Continue) {
      // Control flow is broken if Continue statement is selected without any
      // parent loop
      if (CurrentLocation == ZoneRelative::Inside && !CurNumberOfNestedLoops)
        Info.BrokenControlFlow = true;
      return true;
    }
    CapturedZoneInfo Info;
    const ExtractionZone &ExtZone;
    ZoneRelative CurrentLocation = ZoneRelative::Before;
    // Number of {loop,switch} statements that are currently in the traversal
    // stack inside Extraction Zone. Used to check for broken control flow.
    unsigned CurNumberOfNestedLoops = 0;
    unsigned CurNumberOfSwitch = 0;
  };
  ExtractionZoneVisitor Visitor(ExtZone);
  CapturedZoneInfo Result = std::move(Visitor.Info);
  Result.AlwaysReturns = alwaysReturns(ExtZone);
  return Result;
}

// Adds parameters to ExtractedFunc.
// Returns true if able to find the parameters successfully and no hoisting
// needed.
// FIXME: Check if the declaration has a local/anonymous type
bool createParameters(NewFunction &ExtractedFunc,
                      const CapturedZoneInfo &CapturedInfo) {
  for (const auto &KeyVal : CapturedInfo.DeclInfoMap) {
    const auto &DeclInfo = KeyVal.second;
    // If a Decl was Declared in zone and referenced in post zone, it
    // needs to be hoisted (we bail out in that case).
    // FIXME: Support Decl Hoisting.
    if (DeclInfo.DeclaredIn == ZoneRelative::Inside &&
        DeclInfo.IsReferencedInPostZone)
      return false;
    if (!DeclInfo.IsReferencedInZone)
      continue; // no need to pass as parameter, not referenced
    if (DeclInfo.DeclaredIn == ZoneRelative::Inside ||
        DeclInfo.DeclaredIn == ZoneRelative::OutsideFunc)
      continue; // no need to pass as parameter, still accessible.
    // Parameter specific checks.
    const ValueDecl *VD = dyn_cast_or_null<ValueDecl>(DeclInfo.TheDecl);
    // Can't parameterise if the Decl isn't a ValueDecl or is a FunctionDecl
    // (this includes the case of recursive call to EnclosingFunc in Zone).
    if (!VD || isa<FunctionDecl>(DeclInfo.TheDecl))
      return false;
    // Parameter qualifiers are same as the Decl's qualifiers.
    QualType TypeInfo = VD->getType().getNonReferenceType();
    // FIXME: Need better qualifier checks: check mutated status for
    // Decl(e.g. was it assigned, passed as nonconst argument, etc)
    // FIXME: check if parameter will be a non l-value reference.
    // FIXME: We don't want to always pass variables of types like int,
    // pointers, etc by reference.
    bool IsPassedByReference = true;
    // We use the index of declaration as the ordering priority for parameters.
    ExtractedFunc.Parameters.push_back({std::string(VD->getName()), TypeInfo,
                                        IsPassedByReference,
                                        DeclInfo.DeclIndex});
  }
  llvm::sort(ExtractedFunc.Parameters);
  return true;
}

// Clangd uses open ranges while ExtractionSemicolonPolicy (in Clang Tooling)
// uses closed ranges. Generates the semicolon policy for the extraction and
// extends the ZoneRange if necessary.
tooling::ExtractionSemicolonPolicy
getSemicolonPolicy(ExtractionZone &ExtZone, const SourceManager &SM,
                   const LangOptions &LangOpts) {
  // Get closed ZoneRange.
  SourceRange FuncBodyRange = {ExtZone.ZoneRange.getBegin(),
                               ExtZone.ZoneRange.getEnd().getLocWithOffset(-1)};
  auto SemicolonPolicy = tooling::ExtractionSemicolonPolicy::compute(
      ExtZone.getLastRootStmt()->ASTNode.get<Stmt>(), FuncBodyRange, SM,
      LangOpts);
  // Update ZoneRange.
  ExtZone.ZoneRange.setEnd(FuncBodyRange.getEnd().getLocWithOffset(1));
  return SemicolonPolicy;
}

// Generate return type for ExtractedFunc. Return false if unable to do so.
bool generateReturnProperties(NewFunction &ExtractedFunc,
                              const FunctionDecl &EnclosingFunc,
                              const CapturedZoneInfo &CapturedInfo) {
  // If the selected code always returns, we preserve those return statements.
  // The return type should be the same as the enclosing function.
  // (Others are possible if there are conversions, but this seems clearest).
  if (CapturedInfo.HasReturnStmt) {
    // If the return is conditional, neither replacing the code with
    // `extracted()` nor `return extracted()` is correct.
    if (!CapturedInfo.AlwaysReturns)
      return false;
    QualType Ret = EnclosingFunc.getReturnType();
    // Once we support members, it'd be nice to support e.g. extracting a method
    // of Foo<T> that returns T. But it's not clear when that's safe.
    if (Ret->isDependentType())
      return false;
    ExtractedFunc.ReturnType = Ret;
    return true;
  }
  // FIXME: Generate new return statement if needed.
  ExtractedFunc.ReturnType = EnclosingFunc.getParentASTContext().VoidTy;
  return true;
}

void captureMethodInfo(NewFunction &ExtractedFunc,
                       const CXXMethodDecl *Method) {
  ExtractedFunc.Static = Method->isStatic();
  ExtractedFunc.Const = Method->isConst();
  ExtractedFunc.EnclosingClass = Method->getParent();
}

// FIXME: add support for adding other function return types besides void.
// FIXME: assign the value returned by non void extracted function.
llvm::Expected<NewFunction> getExtractedFunction(ExtractionZone &ExtZone,
                                                 const SourceManager &SM,
                                                 const LangOptions &LangOpts) {
  CapturedZoneInfo CapturedInfo = captureZoneInfo(ExtZone);
  // Bail out if any break of continue exists
  if (CapturedInfo.BrokenControlFlow)
    return error("Cannot extract break/continue without corresponding "
                 "loop/switch statement.");
  NewFunction ExtractedFunc(getSemicolonPolicy(ExtZone, SM, LangOpts),
                            &LangOpts);

  ExtractedFunc.SyntacticDC =
      ExtZone.EnclosingFunction->getLexicalDeclContext();
  ExtractedFunc.SemanticDC = ExtZone.EnclosingFunction->getDeclContext();
  ExtractedFunc.DefinitionQualifier = ExtZone.EnclosingFunction->getQualifier();
  ExtractedFunc.Constexpr = ExtZone.EnclosingFunction->getConstexprKind();

  if (const auto *Method =
          llvm::dyn_cast<CXXMethodDecl>(ExtZone.EnclosingFunction))
    captureMethodInfo(ExtractedFunc, Method);

  if (ExtZone.EnclosingFunction->isOutOfLine()) {
    // FIXME: Put the extracted method in a private section if it's a class or
    // maybe in an anonymous namespace
    const auto *FirstOriginalDecl =
        ExtZone.EnclosingFunction->getCanonicalDecl();
    auto DeclPos =
        toHalfOpenFileRange(SM, LangOpts, FirstOriginalDecl->getSourceRange());
    if (!DeclPos)
      return error("Declaration is inside a macro");
    ExtractedFunc.ForwardDeclarationPoint = DeclPos->getBegin();
    ExtractedFunc.ForwardDeclarationSyntacticDC = ExtractedFunc.SemanticDC;
  }

  ExtractedFunc.BodyRange = ExtZone.ZoneRange;
  ExtractedFunc.DefinitionPoint = ExtZone.getInsertionPoint();

  ExtractedFunc.CallerReturnsValue = CapturedInfo.AlwaysReturns;
  if (!createParameters(ExtractedFunc, CapturedInfo) ||
      !generateReturnProperties(ExtractedFunc, *ExtZone.EnclosingFunction,
                                CapturedInfo))
    return error("Too complex to extract.");
  return ExtractedFunc;
}

class ExtractFunction : public Tweak {
public:
  const char *id() const final;
  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override { return "Extract to function"; }
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

private:
  ExtractionZone ExtZone;
};

REGISTER_TWEAK(ExtractFunction)
tooling::Replacement replaceWithFuncCall(const NewFunction &ExtractedFunc,
                                         const SourceManager &SM,
                                         const LangOptions &LangOpts) {
  std::string FuncCall = ExtractedFunc.renderCall();
  return tooling::Replacement(
      SM, CharSourceRange(ExtractedFunc.BodyRange, false), FuncCall, LangOpts);
}

tooling::Replacement createFunctionDefinition(const NewFunction &ExtractedFunc,
                                              const SourceManager &SM) {
  FunctionDeclKind DeclKind = InlineDefinition;
  if (ExtractedFunc.ForwardDeclarationPoint)
    DeclKind = OutOfLineDefinition;
  std::string FunctionDef = ExtractedFunc.renderDeclaration(
      DeclKind, *ExtractedFunc.SemanticDC, *ExtractedFunc.SyntacticDC, SM);

  return tooling::Replacement(SM, ExtractedFunc.DefinitionPoint, 0,
                              FunctionDef);
}

tooling::Replacement createForwardDeclaration(const NewFunction &ExtractedFunc,
                                              const SourceManager &SM) {
  std::string FunctionDecl = ExtractedFunc.renderDeclaration(
      ForwardDeclaration, *ExtractedFunc.SemanticDC,
      *ExtractedFunc.ForwardDeclarationSyntacticDC, SM);
  SourceLocation DeclPoint = *ExtractedFunc.ForwardDeclarationPoint;

  return tooling::Replacement(SM, DeclPoint, 0, FunctionDecl);
}

// Returns true if ExtZone contains any ReturnStmts.
bool hasReturnStmt(const ExtractionZone &ExtZone) {
  class ReturnStmtVisitor
      : public clang::RecursiveASTVisitor<ReturnStmtVisitor> {
  public:
    bool VisitReturnStmt(ReturnStmt *Return) {
      Found = true;
      return false; // We found the answer, abort the scan.
    }
    bool Found = false;
  };

  ReturnStmtVisitor V;
  for (const Stmt *RootStmt : ExtZone.RootStmts) {
    V.TraverseStmt(const_cast<Stmt *>(RootStmt));
    if (V.Found)
      break;
  }
  return V.Found;
}

bool ExtractFunction::prepare(const Selection &Inputs) {
  const LangOptions &LangOpts = Inputs.AST->getLangOpts();
  if (!LangOpts.CPlusPlus)
    return false;
  const Node *CommonAnc = Inputs.ASTSelection.commonAncestor();
  const SourceManager &SM = Inputs.AST->getSourceManager();
  auto MaybeExtZone = findExtractionZone(CommonAnc, SM, LangOpts);
  if (!MaybeExtZone ||
      (hasReturnStmt(*MaybeExtZone) && !alwaysReturns(*MaybeExtZone)))
    return false;

  // FIXME: Get rid of this check once we support hoisting.
  if (MaybeExtZone->requiresHoisting(SM, Inputs.AST->getHeuristicResolver()))
    return false;

  ExtZone = std::move(*MaybeExtZone);
  return true;
}

Expected<Tweak::Effect> ExtractFunction::apply(const Selection &Inputs) {
  const SourceManager &SM = Inputs.AST->getSourceManager();
  const LangOptions &LangOpts = Inputs.AST->getLangOpts();
  auto ExtractedFunc = getExtractedFunction(ExtZone, SM, LangOpts);
  // FIXME: Add more types of errors.
  if (!ExtractedFunc)
    return ExtractedFunc.takeError();
  tooling::Replacements Edit;
  if (auto Err = Edit.add(createFunctionDefinition(*ExtractedFunc, SM)))
    return std::move(Err);
  if (auto Err = Edit.add(replaceWithFuncCall(*ExtractedFunc, SM, LangOpts)))
    return std::move(Err);

  if (auto FwdLoc = ExtractedFunc->ForwardDeclarationPoint) {
    // If the fwd-declaration goes in the same file, merge into Replacements.
    // Otherwise it needs to be a separate file edit.
    if (SM.isWrittenInSameFile(ExtractedFunc->DefinitionPoint, *FwdLoc)) {
      if (auto Err = Edit.add(createForwardDeclaration(*ExtractedFunc, SM)))
        return std::move(Err);
    } else {
      auto MultiFileEffect = Effect::mainFileEdit(SM, std::move(Edit));
      if (!MultiFileEffect)
        return MultiFileEffect.takeError();

      tooling::Replacements OtherEdit(
          createForwardDeclaration(*ExtractedFunc, SM));
      if (auto PathAndEdit = Tweak::Effect::fileEdit(SM, SM.getFileID(*FwdLoc),
                                                 OtherEdit))
        MultiFileEffect->ApplyEdits.try_emplace(PathAndEdit->first,
                                                PathAndEdit->second);
      else
        return PathAndEdit.takeError();
      return MultiFileEffect;
    }
  }
  return Effect::mainFileEdit(SM, std::move(Edit));
}

} // namespace
} // namespace clangd
} // namespace clang
