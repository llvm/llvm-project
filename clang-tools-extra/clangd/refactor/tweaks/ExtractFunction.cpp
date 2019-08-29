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
// - Doesn't check for broken control flow (break/continue without loop/switch)
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
#include "ClangdUnit.h"
#include "Logger.h"
#include "Selection.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Refactoring/Extract/SourceExtraction.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

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
// We only support extraction of RootStmts since it allows us to extract without
// having to change the selection range. Also, this means that any scope that
// begins in selection range, ends in selection range and any scope that begins
// outside the selection range, ends outside as well.
const Node *getParentOfRootStmts(const Node *CommonAnc) {
  if (!CommonAnc)
    return nullptr;
  switch (CommonAnc->Selected) {
  case SelectionTree::Selection::Unselected:
    // Ensure all Children are RootStmts.
    return llvm::all_of(CommonAnc->Children, isRootStmt) ? CommonAnc : nullptr;
  case SelectionTree::Selection::Partial:
    // Treat Partially selected VarDecl as completely selected since
    // SelectionTree doesn't always select VarDecls correctly.
    // FIXME: Remove this after D66872 is upstream)
    if (!CommonAnc->ASTNode.get<VarDecl>())
      return nullptr;
    LLVM_FALLTHROUGH;
  case SelectionTree::Selection::Complete:
    // If the Common Ancestor is completely selected, then it's a root statement
    // and its parent will be unselected.
    const Node *Parent = CommonAnc->Parent;
    // If parent is a DeclStmt, even though it's unselected, we consider it a
    // root statement and return its parent. This is done because the VarDecls
    // claim the entire selection range of the Declaration and DeclStmt is
    // always unselected.
    return Parent->ASTNode.get<DeclStmt>() ? Parent->Parent : Parent;
  }
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
  SourceLocation getInsertionPoint() const {
    return EnclosingFuncRange.getBegin();
  }
  bool isRootStmt(const Stmt *S) const;
  // The last root statement is important to decide where we need to insert a
  // semicolon after the extraction.
  const Node *getLastRootStmt() const { return Parent->Children.back(); }
  void generateRootStmts();
private:
  llvm::DenseSet<const Stmt *> RootStmts;
};

bool ExtractionZone::isRootStmt(const Stmt *S) const {
  return RootStmts.find(S) != RootStmts.end();
}

// Generate RootStmts set
void ExtractionZone::generateRootStmts() {
  for(const Node *Child : Parent->Children)
    RootStmts.insert(Child->ASTNode.get<Stmt>());
}

// Finds the function in which the zone lies.
const FunctionDecl *findEnclosingFunction(const Node *CommonAnc) {
  // Walk up the SelectionTree until we find a function Decl
  for (const Node *CurNode = CommonAnc; CurNode; CurNode = CurNode->Parent) {
    // Don't extract from lambdas
    if (CurNode->ASTNode.get<LambdaExpr>())
      return nullptr;
    if (const FunctionDecl *Func = CurNode->ASTNode.get<FunctionDecl>()) {
      // FIXME: Support extraction from methods.
      if (isa<CXXMethodDecl>(Func))
        return nullptr;
      // FIXME: Support extraction from templated functions.
      if(Func->isTemplated())
        return nullptr;
      return Func;
    }
  }
  return nullptr;
}

// Zone Range is the union of SourceRanges of all child Nodes in Parent since
// all child Nodes are RootStmts
llvm::Optional<SourceRange> findZoneRange(const Node *Parent,
                                          const SourceManager &SM,
                                          const LangOptions &LangOpts) {
  SourceRange SR;
  if (auto BeginFileRange = toHalfOpenFileRange(
          SM, LangOpts, Parent->Children.front()->ASTNode.getSourceRange()))
    SR.setBegin(BeginFileRange->getBegin());
  else
    return llvm::None;
  if (auto EndFileRange = toHalfOpenFileRange(
          SM, LangOpts, Parent->Children.back()->ASTNode.getSourceRange()))
    SR.setEnd(EndFileRange->getEnd());
  else
    return llvm::None;
  return SR;
}

// Compute the range spanned by the enclosing function.
// FIXME: check if EnclosingFunction has any attributes as the AST doesn't
// always store the source range of the attributes and thus we end up extracting
// between the attributes and the EnclosingFunction.
llvm::Optional<SourceRange>
computeEnclosingFuncRange(const FunctionDecl *EnclosingFunction,
                          const SourceManager &SM,
                          const LangOptions &LangOpts) {
  return toHalfOpenFileRange(SM, LangOpts, EnclosingFunction->getSourceRange());
}

// FIXME: Check we're not extracting from the initializer/condition of a control
// flow structure.
// FIXME: Check that we don't extract the compound statement of the
// enclosingFunction.
llvm::Optional<ExtractionZone> findExtractionZone(const Node *CommonAnc,
                                                  const SourceManager &SM,
                                                  const LangOptions &LangOpts) {
  ExtractionZone ExtZone;
  ExtZone.Parent = getParentOfRootStmts(CommonAnc);
  if (!ExtZone.Parent || ExtZone.Parent->Children.empty())
    return llvm::None;
  // Don't extract expressions.
  // FIXME: We should extract expressions that are "statements" i.e. not
  // subexpressions
  if (ExtZone.Parent->Children.size() == 1 &&
      ExtZone.getLastRootStmt()->ASTNode.get<Expr>())
    return llvm::None;
  ExtZone.EnclosingFunction = findEnclosingFunction(ExtZone.Parent);
  if (!ExtZone.EnclosingFunction)
    return llvm::None;
  if (auto FuncRange =
          computeEnclosingFuncRange(ExtZone.EnclosingFunction, SM, LangOpts))
    ExtZone.EnclosingFuncRange = *FuncRange;
  if (auto ZoneRange = findZoneRange(ExtZone.Parent, SM, LangOpts))
    ExtZone.ZoneRange = *ZoneRange;
  if (ExtZone.EnclosingFuncRange.isInvalid() || ExtZone.ZoneRange.isInvalid())
    return llvm::None;
  ExtZone.generateRootStmts();
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
  std::string ReturnType;
  std::vector<Parameter> Parameters;
  SourceRange BodyRange;
  SourceLocation InsertionPoint;
  const DeclContext *EnclosingFuncContext;
  // Decides whether the extracted function body and the function call need a
  // semicolon after extraction.
  tooling::ExtractionSemicolonPolicy SemicolonPolicy;
  NewFunction(tooling::ExtractionSemicolonPolicy SemicolonPolicy)
      : SemicolonPolicy(SemicolonPolicy) {}
  // Render the call for this function.
  std::string renderCall() const;
  // Render the definition for this function.
  std::string renderDefinition(const SourceManager &SM) const;

private:
  std::string renderParametersForDefinition() const;
  std::string renderParametersForCall() const;
  // Generate the function body.
  std::string getFuncBody(const SourceManager &SM) const;
};

std::string NewFunction::renderParametersForDefinition() const {
  std::string Result;
  bool NeedCommaBefore = false;
  for (const Parameter &P : Parameters) {
    if (NeedCommaBefore)
      Result += ", ";
    NeedCommaBefore = true;
    Result += P.render(EnclosingFuncContext);
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

std::string NewFunction::renderCall() const {
  return Name + "(" + renderParametersForCall() + ")" +
         (SemicolonPolicy.isNeededInOriginalFunction() ? ";" : "");
}

std::string NewFunction::renderDefinition(const SourceManager &SM) const {
  return ReturnType + " " + Name + "(" + renderParametersForDefinition() + ")" +
         " {\n" + getFuncBody(SM) + "\n}\n";
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
  // True if there is a return statement in zone.
  bool HasReturnStmt = false;
  // For now we just care whether there exists a break/continue in zone.
  bool HasBreakOrContinue = false;
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
      bool IsRootStmt = ExtZone.isRootStmt(const_cast<const Stmt *>(S));
      // If we are starting traversal of a RootStmt, we are somewhere inside
      // ExtractionZone
      if (IsRootStmt)
        CurrentLocation = ZoneRelative::Inside;
      // Traverse using base class's TraverseStmt
      RecursiveASTVisitor::TraverseStmt(S);
      // We set the current location as after since next stmt will either be a
      // RootStmt (handled at the beginning) or after extractionZone
      if (IsRootStmt)
        CurrentLocation = ZoneRelative::After;
      return true;
    }
    bool VisitDecl(Decl *D) {
      Info.createDeclInfo(D, CurrentLocation);
      return true;
    }
    bool VisitDeclRefExpr(DeclRefExpr *DRE) {
      // Find the corresponding Decl and mark it's occurence.
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

    // FIXME: check for broken break/continue only.
    bool VisitBreakStmt(BreakStmt *Break) {
      if (CurrentLocation == ZoneRelative::Inside)
        Info.HasBreakOrContinue = true;
      return true;
    }
    bool VisitContinueStmt(ContinueStmt *Continue) {
      if (CurrentLocation == ZoneRelative::Inside)
        Info.HasBreakOrContinue = true;
      return true;
    }
    CapturedZoneInfo Info;
    const ExtractionZone &ExtZone;
    ZoneRelative CurrentLocation = ZoneRelative::Before;
  };
  ExtractionZoneVisitor Visitor(ExtZone);
  return std::move(Visitor.Info);
}

// Adds parameters to ExtractedFunc.
// Returns true if able to find the parameters successfully and no hoisting
// needed.
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
    ExtractedFunc.Parameters.push_back(
        {VD->getName(), TypeInfo, IsPassedByReference, DeclInfo.DeclIndex});
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
                              const CapturedZoneInfo &CapturedInfo) {

  // FIXME: Use Existing Return statements (if present)
  // FIXME: Generate new return statement if needed.
  if (CapturedInfo.HasReturnStmt)
    return false;
  ExtractedFunc.ReturnType = "void";
  return true;
}

// FIXME: add support for adding other function return types besides void.
// FIXME: assign the value returned by non void extracted function.
llvm::Expected<NewFunction> getExtractedFunction(ExtractionZone &ExtZone,
                                                 const SourceManager &SM,
                                                 const LangOptions &LangOpts) {
  CapturedZoneInfo CapturedInfo = captureZoneInfo(ExtZone);
  // Bail out if any break of continue exists
  // FIXME: check for broken control flow only
  if (CapturedInfo.HasBreakOrContinue)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   +"Cannot extract break or continue.");
  NewFunction ExtractedFunc(getSemicolonPolicy(ExtZone, SM, LangOpts));
  ExtractedFunc.BodyRange = ExtZone.ZoneRange;
  ExtractedFunc.InsertionPoint = ExtZone.getInsertionPoint();
  ExtractedFunc.EnclosingFuncContext =
      ExtZone.EnclosingFunction->getDeclContext();
  if (!createParameters(ExtractedFunc, CapturedInfo) ||
      !generateReturnProperties(ExtractedFunc, CapturedInfo))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   +"Too complex to extract.");
  return ExtractedFunc;
}

class ExtractFunction : public Tweak {
public:
  const char *id() const override final;
  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override { return "Extract to function"; }
  Intent intent() const override { return Refactor; }

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
  std::string FunctionDef = ExtractedFunc.renderDefinition(SM);
  return tooling::Replacement(SM, ExtractedFunc.InsertionPoint, 0, FunctionDef);
}

bool ExtractFunction::prepare(const Selection &Inputs) {
  const Node *CommonAnc = Inputs.ASTSelection.commonAncestor();
  const SourceManager &SM = Inputs.AST.getSourceManager();
  const LangOptions &LangOpts = Inputs.AST.getASTContext().getLangOpts();
  if (auto MaybeExtZone = findExtractionZone(CommonAnc, SM, LangOpts)) {
    ExtZone = std::move(*MaybeExtZone);
    return true;
  }
  return false;
}

Expected<Tweak::Effect> ExtractFunction::apply(const Selection &Inputs) {
  const SourceManager &SM = Inputs.AST.getSourceManager();
  const LangOptions &LangOpts = Inputs.AST.getASTContext().getLangOpts();
  auto ExtractedFunc = getExtractedFunction(ExtZone, SM, LangOpts);
  // FIXME: Add more types of errors.
  if (!ExtractedFunc)
    return ExtractedFunc.takeError();
  tooling::Replacements Result;
  if (auto Err = Result.add(createFunctionDefinition(*ExtractedFunc, SM)))
    return std::move(Err);
  if (auto Err = Result.add(replaceWithFuncCall(*ExtractedFunc, SM, LangOpts)))
    return std::move(Err);
  return Effect::applyEdit(Result);
}

} // namespace
} // namespace clangd
} // namespace clang
