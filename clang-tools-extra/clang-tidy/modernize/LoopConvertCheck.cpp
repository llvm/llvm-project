//===--- LoopConvertCheck.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoopConvertCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstring>
#include <utility>

using namespace clang::ast_matchers;
using namespace llvm;

namespace clang {
namespace tidy {
namespace modernize {

static const char LoopNameArray[] = "forLoopArray";
static const char LoopNameIterator[] = "forLoopIterator";
static const char LoopNamePseudoArray[] = "forLoopPseudoArray";
static const char ConditionBoundName[] = "conditionBound";
static const char ConditionVarName[] = "conditionVar";
static const char IncrementVarName[] = "incrementVar";
static const char InitVarName[] = "initVar";
static const char BeginCallName[] = "beginCall";
static const char EndCallName[] = "endCall";
static const char ConditionEndVarName[] = "conditionEndVar";
static const char EndVarName[] = "endVar";
static const char DerefByValueResultName[] = "derefByValueResult";
static const char DerefByRefResultName[] = "derefByRefResult";

static ArrayRef<std::pair<StringRef, Confidence::Level>>
getConfidenceMapping() {
  static constexpr std::pair<StringRef, Confidence::Level> Mapping[] = {
      {"reasonable", Confidence::CL_Reasonable},
      {"safe", Confidence::CL_Safe},
      {"risky", Confidence::CL_Risky}};
  return makeArrayRef(Mapping);
}

static ArrayRef<std::pair<StringRef, VariableNamer::NamingStyle>>
getStyleMapping() {
  static constexpr std::pair<StringRef, VariableNamer::NamingStyle> Mapping[] =
      {{"CamelCase", VariableNamer::NS_CamelCase},
       {"camelBack", VariableNamer::NS_CamelBack},
       {"lower_case", VariableNamer::NS_LowerCase},
       {"UPPER_CASE", VariableNamer::NS_UpperCase}};
  return makeArrayRef(Mapping);
}

// shared matchers
static const TypeMatcher AnyType() { return anything(); }

static const StatementMatcher IntegerComparisonMatcher() {
  return expr(ignoringParenImpCasts(
      declRefExpr(to(varDecl(hasType(isInteger())).bind(ConditionVarName)))));
}

static const DeclarationMatcher InitToZeroMatcher() {
  return varDecl(
             hasInitializer(ignoringParenImpCasts(integerLiteral(equals(0)))))
      .bind(InitVarName);
}

static const StatementMatcher IncrementVarMatcher() {
  return declRefExpr(to(varDecl(hasType(isInteger())).bind(IncrementVarName)));
}

/// The matcher for loops over arrays.
///
/// In this general example, assuming 'j' and 'k' are of integral type:
/// \code
///   for (int i = 0; j < 3 + 2; ++k) { ... }
/// \endcode
/// The following string identifiers are bound to these parts of the AST:
///   ConditionVarName: 'j' (as a VarDecl)
///   ConditionBoundName: '3 + 2' (as an Expr)
///   InitVarName: 'i' (as a VarDecl)
///   IncrementVarName: 'k' (as a VarDecl)
///   LoopName: The entire for loop (as a ForStmt)
///
/// Client code will need to make sure that:
///   - The three index variables identified by the matcher are the same
///     VarDecl.
///   - The index variable is only used as an array index.
///   - All arrays indexed by the loop are the same.
StatementMatcher makeArrayLoopMatcher() {
  StatementMatcher ArrayBoundMatcher =
      expr(hasType(isInteger())).bind(ConditionBoundName);

  return forStmt(
             unless(isInTemplateInstantiation()),
             hasLoopInit(declStmt(hasSingleDecl(InitToZeroMatcher()))),
             hasCondition(anyOf(
                 binaryOperator(hasOperatorName("<"),
                                hasLHS(IntegerComparisonMatcher()),
                                hasRHS(ArrayBoundMatcher)),
                 binaryOperator(hasOperatorName(">"), hasLHS(ArrayBoundMatcher),
                                hasRHS(IntegerComparisonMatcher())))),
             hasIncrement(unaryOperator(hasOperatorName("++"),
                                        hasUnaryOperand(IncrementVarMatcher()))))
      .bind(LoopNameArray);
}

/// The matcher used for iterator-based for loops.
///
/// This matcher is more flexible than array-based loops. It will match
/// catch loops of the following textual forms (regardless of whether the
/// iterator type is actually a pointer type or a class type):
///
/// Assuming f, g, and h are of type containerType::iterator,
/// \code
///   for (containerType::iterator it = container.begin(),
///        e = createIterator(); f != g; ++h) { ... }
///   for (containerType::iterator it = container.begin();
///        f != anotherContainer.end(); ++h) { ... }
/// \endcode
/// The following string identifiers are bound to the parts of the AST:
///   InitVarName: 'it' (as a VarDecl)
///   ConditionVarName: 'f' (as a VarDecl)
///   LoopName: The entire for loop (as a ForStmt)
///   In the first example only:
///     EndVarName: 'e' (as a VarDecl)
///     ConditionEndVarName: 'g' (as a VarDecl)
///   In the second example only:
///     EndCallName: 'container.end()' (as a CXXMemberCallExpr)
///
/// Client code will need to make sure that:
///   - The iterator variables 'it', 'f', and 'h' are the same.
///   - The two containers on which 'begin' and 'end' are called are the same.
///   - If the end iterator variable 'g' is defined, it is the same as 'f'.
StatementMatcher makeIteratorLoopMatcher() {
  StatementMatcher BeginCallMatcher =
      cxxMemberCallExpr(
          argumentCountIs(0),
          callee(cxxMethodDecl(anyOf(hasName("begin"), hasName("cbegin")))))
          .bind(BeginCallName);

  DeclarationMatcher InitDeclMatcher =
      varDecl(hasInitializer(anyOf(ignoringParenImpCasts(BeginCallMatcher),
                                   materializeTemporaryExpr(
                                       ignoringParenImpCasts(BeginCallMatcher)),
                                   hasDescendant(BeginCallMatcher))))
          .bind(InitVarName);

  DeclarationMatcher EndDeclMatcher =
      varDecl(hasInitializer(anything())).bind(EndVarName);

  StatementMatcher EndCallMatcher = cxxMemberCallExpr(
      argumentCountIs(0),
      callee(cxxMethodDecl(anyOf(hasName("end"), hasName("cend")))));

  StatementMatcher IteratorBoundMatcher =
      expr(anyOf(ignoringParenImpCasts(
                     declRefExpr(to(varDecl().bind(ConditionEndVarName)))),
                 ignoringParenImpCasts(expr(EndCallMatcher).bind(EndCallName)),
                 materializeTemporaryExpr(ignoringParenImpCasts(
                     expr(EndCallMatcher).bind(EndCallName)))));

  StatementMatcher IteratorComparisonMatcher = expr(
      ignoringParenImpCasts(declRefExpr(to(varDecl().bind(ConditionVarName)))));

  auto OverloadedNEQMatcher = ignoringImplicit(
      cxxOperatorCallExpr(hasOverloadedOperatorName("!="), argumentCountIs(2),
                          hasArgument(0, IteratorComparisonMatcher),
                          hasArgument(1, IteratorBoundMatcher)));

  // This matcher tests that a declaration is a CXXRecordDecl that has an
  // overloaded operator*(). If the operator*() returns by value instead of by
  // reference then the return type is tagged with DerefByValueResultName.
  internal::Matcher<VarDecl> TestDerefReturnsByValue =
      hasType(hasUnqualifiedDesugaredType(
          recordType(hasDeclaration(cxxRecordDecl(hasMethod(cxxMethodDecl(
              hasOverloadedOperatorName("*"),
              anyOf(
                  // Tag the return type if it's by value.
                  returns(qualType(unless(hasCanonicalType(referenceType())))
                              .bind(DerefByValueResultName)),
                  returns(
                      // Skip loops where the iterator's operator* returns an
                      // rvalue reference. This is just weird.
                      qualType(unless(hasCanonicalType(rValueReferenceType())))
                          .bind(DerefByRefResultName))))))))));

  return forStmt(
             unless(isInTemplateInstantiation()),
             hasLoopInit(anyOf(declStmt(declCountIs(2),
                                        containsDeclaration(0, InitDeclMatcher),
                                        containsDeclaration(1, EndDeclMatcher)),
                               declStmt(hasSingleDecl(InitDeclMatcher)))),
             hasCondition(
                 anyOf(binaryOperator(hasOperatorName("!="),
                                      hasLHS(IteratorComparisonMatcher),
                                      hasRHS(IteratorBoundMatcher)),
                       binaryOperator(hasOperatorName("!="),
                                      hasLHS(IteratorBoundMatcher),
                                      hasRHS(IteratorComparisonMatcher)),
                       OverloadedNEQMatcher)),
             hasIncrement(anyOf(
                 unaryOperator(hasOperatorName("++"),
                               hasUnaryOperand(declRefExpr(
                                   to(varDecl(hasType(pointsTo(AnyType())))
                                          .bind(IncrementVarName))))),
                 cxxOperatorCallExpr(
                     hasOverloadedOperatorName("++"),
                     hasArgument(
                         0, declRefExpr(to(varDecl(TestDerefReturnsByValue)
                                               .bind(IncrementVarName))))))))
      .bind(LoopNameIterator);
}

/// The matcher used for array-like containers (pseudoarrays).
///
/// This matcher is more flexible than array-based loops. It will match
/// loops of the following textual forms (regardless of whether the
/// iterator type is actually a pointer type or a class type):
///
/// Assuming f, g, and h are of type containerType::iterator,
/// \code
///   for (int i = 0, j = container.size(); f < g; ++h) { ... }
///   for (int i = 0; f < container.size(); ++h) { ... }
/// \endcode
/// The following string identifiers are bound to the parts of the AST:
///   InitVarName: 'i' (as a VarDecl)
///   ConditionVarName: 'f' (as a VarDecl)
///   LoopName: The entire for loop (as a ForStmt)
///   In the first example only:
///     EndVarName: 'j' (as a VarDecl)
///     ConditionEndVarName: 'g' (as a VarDecl)
///   In the second example only:
///     EndCallName: 'container.size()' (as a CXXMemberCallExpr)
///
/// Client code will need to make sure that:
///   - The index variables 'i', 'f', and 'h' are the same.
///   - The containers on which 'size()' is called is the container indexed.
///   - The index variable is only used in overloaded operator[] or
///     container.at().
///   - If the end iterator variable 'g' is defined, it is the same as 'j'.
///   - The container's iterators would not be invalidated during the loop.
StatementMatcher makePseudoArrayLoopMatcher() {
  // Test that the incoming type has a record declaration that has methods
  // called 'begin' and 'end'. If the incoming type is const, then make sure
  // these methods are also marked const.
  //
  // FIXME: To be completely thorough this matcher should also ensure the
  // return type of begin/end is an iterator that dereferences to the same as
  // what operator[] or at() returns. Such a test isn't likely to fail except
  // for pathological cases.
  //
  // FIXME: Also, a record doesn't necessarily need begin() and end(). Free
  // functions called begin() and end() taking the container as an argument
  // are also allowed.
  TypeMatcher RecordWithBeginEnd = qualType(anyOf(
      qualType(
          isConstQualified(),
          hasUnqualifiedDesugaredType(recordType(hasDeclaration(cxxRecordDecl(
              hasMethod(cxxMethodDecl(hasName("begin"), isConst())),
              hasMethod(cxxMethodDecl(hasName("end"),
                                      isConst()))))   // hasDeclaration
                                                 ))), // qualType
      qualType(unless(isConstQualified()),
               hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                   cxxRecordDecl(hasMethod(hasName("begin")),
                                 hasMethod(hasName("end"))))))) // qualType
      ));

  StatementMatcher SizeCallMatcher = cxxMemberCallExpr(
      argumentCountIs(0),
      callee(cxxMethodDecl(anyOf(hasName("size"), hasName("length")))),
      on(anyOf(hasType(pointsTo(RecordWithBeginEnd)),
               hasType(RecordWithBeginEnd))));

  StatementMatcher EndInitMatcher =
      expr(anyOf(ignoringParenImpCasts(expr(SizeCallMatcher).bind(EndCallName)),
                 explicitCastExpr(hasSourceExpression(ignoringParenImpCasts(
                     expr(SizeCallMatcher).bind(EndCallName))))));

  DeclarationMatcher EndDeclMatcher =
      varDecl(hasInitializer(EndInitMatcher)).bind(EndVarName);

  StatementMatcher IndexBoundMatcher =
      expr(anyOf(ignoringParenImpCasts(declRefExpr(to(
                     varDecl(hasType(isInteger())).bind(ConditionEndVarName)))),
                 EndInitMatcher));

  return forStmt(
             unless(isInTemplateInstantiation()),
             hasLoopInit(
                 anyOf(declStmt(declCountIs(2),
                                containsDeclaration(0, InitToZeroMatcher()),
                                containsDeclaration(1, EndDeclMatcher)),
                       declStmt(hasSingleDecl(InitToZeroMatcher())))),
             hasCondition(anyOf(
                 binaryOperator(hasOperatorName("<"),
                                hasLHS(IntegerComparisonMatcher()),
                                hasRHS(IndexBoundMatcher)),
                 binaryOperator(hasOperatorName(">"), hasLHS(IndexBoundMatcher),
                                hasRHS(IntegerComparisonMatcher())))),
             hasIncrement(unaryOperator(hasOperatorName("++"),
                                        hasUnaryOperand(IncrementVarMatcher()))))
      .bind(LoopNamePseudoArray);
}

/// Determine whether Init appears to be an initializing an iterator.
///
/// If it is, returns the object whose begin() or end() method is called, and
/// the output parameter isArrow is set to indicate whether the initialization
/// is called via . or ->.
static const Expr *getContainerFromBeginEndCall(const Expr *Init, bool IsBegin,
                                                bool *IsArrow) {
  // FIXME: Maybe allow declaration/initialization outside of the for loop.
  const auto *TheCall =
      dyn_cast_or_null<CXXMemberCallExpr>(digThroughConstructors(Init));
  if (!TheCall || TheCall->getNumArgs() != 0)
    return nullptr;

  const auto *Member = dyn_cast<MemberExpr>(TheCall->getCallee());
  if (!Member)
    return nullptr;
  StringRef Name = Member->getMemberDecl()->getName();
  StringRef TargetName = IsBegin ? "begin" : "end";
  StringRef ConstTargetName = IsBegin ? "cbegin" : "cend";
  if (Name != TargetName && Name != ConstTargetName)
    return nullptr;

  const Expr *SourceExpr = Member->getBase();
  if (!SourceExpr)
    return nullptr;

  *IsArrow = Member->isArrow();
  return SourceExpr;
}

/// Determines the container whose begin() and end() functions are called
/// for an iterator-based loop.
///
/// BeginExpr must be a member call to a function named "begin()", and EndExpr
/// must be a member.
static const Expr *findContainer(ASTContext *Context, const Expr *BeginExpr,
                                 const Expr *EndExpr,
                                 bool *ContainerNeedsDereference) {
  // Now that we know the loop variable and test expression, make sure they are
  // valid.
  bool BeginIsArrow = false;
  bool EndIsArrow = false;
  const Expr *BeginContainerExpr =
      getContainerFromBeginEndCall(BeginExpr, /*IsBegin=*/true, &BeginIsArrow);
  if (!BeginContainerExpr)
    return nullptr;

  const Expr *EndContainerExpr =
      getContainerFromBeginEndCall(EndExpr, /*IsBegin=*/false, &EndIsArrow);
  // Disallow loops that try evil things like this (note the dot and arrow):
  //  for (IteratorType It = Obj.begin(), E = Obj->end(); It != E; ++It) { }
  if (!EndContainerExpr || BeginIsArrow != EndIsArrow ||
      !areSameExpr(Context, EndContainerExpr, BeginContainerExpr))
    return nullptr;

  *ContainerNeedsDereference = BeginIsArrow;
  return BeginContainerExpr;
}

/// Obtain the original source code text from a SourceRange.
static StringRef getStringFromRange(SourceManager &SourceMgr,
                                    const LangOptions &LangOpts,
                                    SourceRange Range) {
  if (SourceMgr.getFileID(Range.getBegin()) !=
      SourceMgr.getFileID(Range.getEnd())) {
    return StringRef(); // Empty string.
  }

  return Lexer::getSourceText(CharSourceRange(Range, true), SourceMgr,
                              LangOpts);
}

/// If the given expression is actually a DeclRefExpr or a MemberExpr,
/// find and return the underlying ValueDecl; otherwise, return NULL.
static const ValueDecl *getReferencedVariable(const Expr *E) {
  if (const DeclRefExpr *DRE = getDeclRef(E))
    return dyn_cast<VarDecl>(DRE->getDecl());
  if (const auto *Mem = dyn_cast<MemberExpr>(E->IgnoreParenImpCasts()))
    return dyn_cast<FieldDecl>(Mem->getMemberDecl());
  return nullptr;
}

/// Returns true when the given expression is a member expression
/// whose base is `this` (implicitly or not).
static bool isDirectMemberExpr(const Expr *E) {
  if (const auto *Member = dyn_cast<MemberExpr>(E->IgnoreParenImpCasts()))
    return isa<CXXThisExpr>(Member->getBase()->IgnoreParenImpCasts());
  return false;
}

/// Given an expression that represents an usage of an element from the
/// containter that we are iterating over, returns false when it can be
/// guaranteed this element cannot be modified as a result of this usage.
static bool canBeModified(ASTContext *Context, const Expr *E) {
  if (E->getType().isConstQualified())
    return false;
  auto Parents = Context->getParents(*E);
  if (Parents.size() != 1)
    return true;
  if (const auto *Cast = Parents[0].get<ImplicitCastExpr>()) {
    if ((Cast->getCastKind() == CK_NoOp &&
         Cast->getType() == E->getType().withConst()) ||
        (Cast->getCastKind() == CK_LValueToRValue &&
         !Cast->getType().isNull() && Cast->getType()->isFundamentalType()))
      return false;
  }
  // FIXME: Make this function more generic.
  return true;
}

/// Returns true when it can be guaranteed that the elements of the
/// container are not being modified.
static bool usagesAreConst(ASTContext *Context, const UsageResult &Usages) {
  for (const Usage &U : Usages) {
    // Lambda captures are just redeclarations (VarDecl) of the same variable,
    // not expressions. If we want to know if a variable that is captured by
    // reference can be modified in an usage inside the lambda's body, we need
    // to find the expression corresponding to that particular usage, later in
    // this loop.
    if (U.Kind != Usage::UK_CaptureByCopy && U.Kind != Usage::UK_CaptureByRef &&
        canBeModified(Context, U.Expression))
      return false;
  }
  return true;
}

/// Returns true if the elements of the container are never accessed
/// by reference.
static bool usagesReturnRValues(const UsageResult &Usages) {
  for (const auto &U : Usages) {
    if (U.Expression && !U.Expression->isRValue())
      return false;
  }
  return true;
}

/// Returns true if the container is const-qualified.
static bool containerIsConst(const Expr *ContainerExpr, bool Dereference) {
  if (const auto *VDec = getReferencedVariable(ContainerExpr)) {
    QualType CType = VDec->getType();
    if (Dereference) {
      if (!CType->isPointerType())
        return false;
      CType = CType->getPointeeType();
    }
    // If VDec is a reference to a container, Dereference is false,
    // but we still need to check the const-ness of the underlying container
    // type.
    CType = CType.getNonReferenceType();
    return CType.isConstQualified();
  }
  return false;
}

LoopConvertCheck::RangeDescriptor::RangeDescriptor()
    : ContainerNeedsDereference(false), DerefByConstRef(false),
      DerefByValue(false) {}

LoopConvertCheck::LoopConvertCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), TUInfo(new TUTrackingInfo),
      MaxCopySize(Options.get("MaxCopySize", 16ULL)),
      MinConfidence(Options.get("MinConfidence", getConfidenceMapping(),
                                Confidence::CL_Reasonable)),
      NamingStyle(Options.get("NamingStyle", getStyleMapping(),
                              VariableNamer::NS_CamelCase)) {}

void LoopConvertCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MaxCopySize", std::to_string(MaxCopySize));
  Options.store(Opts, "MinConfidence", MinConfidence, getConfidenceMapping());
  Options.store(Opts, "NamingStyle", NamingStyle, getStyleMapping());
}

void LoopConvertCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(traverse(ast_type_traits::TK_AsIs, makeArrayLoopMatcher()),
                     this);
  Finder->addMatcher(
      traverse(ast_type_traits::TK_AsIs, makeIteratorLoopMatcher()), this);
  Finder->addMatcher(
      traverse(ast_type_traits::TK_AsIs, makePseudoArrayLoopMatcher()), this);
}

/// Given the range of a single declaration, such as:
/// \code
///   unsigned &ThisIsADeclarationThatCanSpanSeveralLinesOfCode =
///       InitializationValues[I];
///   next_instruction;
/// \endcode
/// Finds the range that has to be erased to remove this declaration without
/// leaving empty lines, by extending the range until the beginning of the
/// next instruction.
///
/// We need to delete a potential newline after the deleted alias, as
/// clang-format will leave empty lines untouched. For all other formatting we
/// rely on clang-format to fix it.
void LoopConvertCheck::getAliasRange(SourceManager &SM, SourceRange &Range) {
  bool Invalid = false;
  const char *TextAfter =
      SM.getCharacterData(Range.getEnd().getLocWithOffset(1), &Invalid);
  if (Invalid)
    return;
  unsigned Offset = std::strspn(TextAfter, " \t\r\n");
  Range =
      SourceRange(Range.getBegin(), Range.getEnd().getLocWithOffset(Offset));
}

/// Computes the changes needed to convert a given for loop, and
/// applies them.
void LoopConvertCheck::doConversion(
    ASTContext *Context, const VarDecl *IndexVar,
    const ValueDecl *MaybeContainer, const UsageResult &Usages,
    const DeclStmt *AliasDecl, bool AliasUseRequired, bool AliasFromForInit,
    const ForStmt *Loop, RangeDescriptor Descriptor) {
  std::string VarName;
  bool VarNameFromAlias = (Usages.size() == 1) && AliasDecl;
  bool AliasVarIsRef = false;
  bool CanCopy = true;
  std::vector<FixItHint> FixIts;
  if (VarNameFromAlias) {
    const auto *AliasVar = cast<VarDecl>(AliasDecl->getSingleDecl());
    VarName = AliasVar->getName().str();

    // Use the type of the alias if it's not the same
    QualType AliasVarType = AliasVar->getType();
    assert(!AliasVarType.isNull() && "Type in VarDecl is null");
    if (AliasVarType->isReferenceType()) {
      AliasVarType = AliasVarType.getNonReferenceType();
      AliasVarIsRef = true;
    }
    if (Descriptor.ElemType.isNull() ||
        !Context->hasSameUnqualifiedType(AliasVarType, Descriptor.ElemType))
      Descriptor.ElemType = AliasVarType;

    // We keep along the entire DeclStmt to keep the correct range here.
    SourceRange ReplaceRange = AliasDecl->getSourceRange();

    std::string ReplacementText;
    if (AliasUseRequired) {
      ReplacementText = VarName;
    } else if (AliasFromForInit) {
      // FIXME: Clang includes the location of the ';' but only for DeclStmt's
      // in a for loop's init clause. Need to put this ';' back while removing
      // the declaration of the alias variable. This is probably a bug.
      ReplacementText = ";";
    } else {
      // Avoid leaving empty lines or trailing whitespaces.
      getAliasRange(Context->getSourceManager(), ReplaceRange);
    }

    FixIts.push_back(FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(ReplaceRange), ReplacementText));
    // No further replacements are made to the loop, since the iterator or index
    // was used exactly once - in the initialization of AliasVar.
  } else {
    VariableNamer Namer(&TUInfo->getGeneratedDecls(),
                        &TUInfo->getParentFinder().getStmtToParentStmtMap(),
                        Loop, IndexVar, MaybeContainer, Context, NamingStyle);
    VarName = Namer.createIndexName();
    // First, replace all usages of the array subscript expression with our new
    // variable.
    for (const auto &Usage : Usages) {
      std::string ReplaceText;
      SourceRange Range = Usage.Range;
      if (Usage.Expression) {
        // If this is an access to a member through the arrow operator, after
        // the replacement it must be accessed through the '.' operator.
        ReplaceText = Usage.Kind == Usage::UK_MemberThroughArrow ? VarName + "."
                                                                 : VarName;
        auto Parents = Context->getParents(*Usage.Expression);
        if (Parents.size() == 1) {
          if (const auto *Paren = Parents[0].get<ParenExpr>()) {
            // Usage.Expression will be replaced with the new index variable,
            // and parenthesis around a simple DeclRefExpr can always be
            // removed.
            Range = Paren->getSourceRange();
          } else if (const auto *UOP = Parents[0].get<UnaryOperator>()) {
            // If we are taking the address of the loop variable, then we must
            // not use a copy, as it would mean taking the address of the loop's
            // local index instead.
            // FIXME: This won't catch cases where the address is taken outside
            // of the loop's body (for instance, in a function that got the
            // loop's index as a const reference parameter), or where we take
            // the address of a member (like "&Arr[i].A.B.C").
            if (UOP->getOpcode() == UO_AddrOf)
              CanCopy = false;
          }
        }
      } else {
        // The Usage expression is only null in case of lambda captures (which
        // are VarDecl). If the index is captured by value, add '&' to capture
        // by reference instead.
        ReplaceText =
            Usage.Kind == Usage::UK_CaptureByCopy ? "&" + VarName : VarName;
      }
      TUInfo->getReplacedVars().insert(std::make_pair(Loop, IndexVar));
      FixIts.push_back(FixItHint::CreateReplacement(
          CharSourceRange::getTokenRange(Range), ReplaceText));
    }
  }

  // Now, we need to construct the new range expression.
  SourceRange ParenRange(Loop->getLParenLoc(), Loop->getRParenLoc());

  QualType Type = Context->getAutoDeductType();
  if (!Descriptor.ElemType.isNull() && Descriptor.ElemType->isFundamentalType())
    Type = Descriptor.ElemType.getUnqualifiedType();
  Type = Type.getDesugaredType(*Context);

  // If the new variable name is from the aliased variable, then the reference
  // type for the new variable should only be used if the aliased variable was
  // declared as a reference.
  bool IsCheapToCopy =
      !Descriptor.ElemType.isNull() &&
      Descriptor.ElemType.isTriviallyCopyableType(*Context) &&
      // TypeInfo::Width is in bits.
      Context->getTypeInfo(Descriptor.ElemType).Width <= 8 * MaxCopySize;
  bool UseCopy = CanCopy && ((VarNameFromAlias && !AliasVarIsRef) ||
                             (Descriptor.DerefByConstRef && IsCheapToCopy));

  if (!UseCopy) {
    if (Descriptor.DerefByConstRef) {
      Type = Context->getLValueReferenceType(Context->getConstType(Type));
    } else if (Descriptor.DerefByValue) {
      if (!IsCheapToCopy)
        Type = Context->getRValueReferenceType(Type);
    } else {
      Type = Context->getLValueReferenceType(Type);
    }
  }

  StringRef MaybeDereference = Descriptor.ContainerNeedsDereference ? "*" : "";
  std::string TypeString = Type.getAsString(getLangOpts());
  std::string Range = ("(" + TypeString + " " + VarName + " : " +
                       MaybeDereference + Descriptor.ContainerString + ")")
                          .str();
  FixIts.push_back(FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(ParenRange), Range));
  diag(Loop->getForLoc(), "use range-based for loop instead") << FixIts;
  TUInfo->getGeneratedDecls().insert(make_pair(Loop, VarName));
}

/// Returns a string which refers to the container iterated over.
StringRef LoopConvertCheck::getContainerString(ASTContext *Context,
                                               const ForStmt *Loop,
                                               const Expr *ContainerExpr) {
  StringRef ContainerString;
  ContainerExpr = ContainerExpr->IgnoreParenImpCasts();
  if (isa<CXXThisExpr>(ContainerExpr)) {
    ContainerString = "this";
  } else {
    // For CXXOperatorCallExpr (e.g. vector_ptr->size()), its first argument is
    // the class object (vector_ptr) we are targeting.
    if (const auto* E = dyn_cast<CXXOperatorCallExpr>(ContainerExpr))
      ContainerExpr = E->getArg(0);
    ContainerString =
        getStringFromRange(Context->getSourceManager(), Context->getLangOpts(),
                           ContainerExpr->getSourceRange());
  }

  return ContainerString;
}

/// Determines what kind of 'auto' must be used after converting a for
/// loop that iterates over an array or pseudoarray.
void LoopConvertCheck::getArrayLoopQualifiers(ASTContext *Context,
                                              const BoundNodes &Nodes,
                                              const Expr *ContainerExpr,
                                              const UsageResult &Usages,
                                              RangeDescriptor &Descriptor) {
  // On arrays and pseudoarrays, we must figure out the qualifiers from the
  // usages.
  if (usagesAreConst(Context, Usages) ||
      containerIsConst(ContainerExpr, Descriptor.ContainerNeedsDereference)) {
    Descriptor.DerefByConstRef = true;
  }
  if (usagesReturnRValues(Usages)) {
    // If the index usages (dereference, subscript, at, ...) return rvalues,
    // then we should not use a reference, because we need to keep the code
    // correct if it mutates the returned objects.
    Descriptor.DerefByValue = true;
  }
  // Try to find the type of the elements on the container, to check if
  // they are trivially copyable.
  for (const Usage &U : Usages) {
    if (!U.Expression || U.Expression->getType().isNull())
      continue;
    QualType Type = U.Expression->getType().getCanonicalType();
    if (U.Kind == Usage::UK_MemberThroughArrow) {
      if (!Type->isPointerType()) {
        continue;
      }
      Type = Type->getPointeeType();
    }
    Descriptor.ElemType = Type;
  }
}

/// Determines what kind of 'auto' must be used after converting an
/// iterator based for loop.
void LoopConvertCheck::getIteratorLoopQualifiers(ASTContext *Context,
                                                 const BoundNodes &Nodes,
                                                 RangeDescriptor &Descriptor) {
  // The matchers for iterator loops provide bound nodes to obtain this
  // information.
  const auto *InitVar = Nodes.getNodeAs<VarDecl>(InitVarName);
  QualType CanonicalInitVarType = InitVar->getType().getCanonicalType();
  const auto *DerefByValueType =
      Nodes.getNodeAs<QualType>(DerefByValueResultName);
  Descriptor.DerefByValue = DerefByValueType;

  if (Descriptor.DerefByValue) {
    // If the dereference operator returns by value then test for the
    // canonical const qualification of the init variable type.
    Descriptor.DerefByConstRef = CanonicalInitVarType.isConstQualified();
    Descriptor.ElemType = *DerefByValueType;
  } else {
    if (const auto *DerefType =
            Nodes.getNodeAs<QualType>(DerefByRefResultName)) {
      // A node will only be bound with DerefByRefResultName if we're dealing
      // with a user-defined iterator type. Test the const qualification of
      // the reference type.
      auto ValueType = DerefType->getNonReferenceType();

      Descriptor.DerefByConstRef = ValueType.isConstQualified();
      Descriptor.ElemType = ValueType;
    } else {
      // By nature of the matcher this case is triggered only for built-in
      // iterator types (i.e. pointers).
      assert(isa<PointerType>(CanonicalInitVarType) &&
             "Non-class iterator type is not a pointer type");

      // We test for const qualification of the pointed-at type.
      Descriptor.DerefByConstRef =
          CanonicalInitVarType->getPointeeType().isConstQualified();
      Descriptor.ElemType = CanonicalInitVarType->getPointeeType();
    }
  }
}

/// Determines the parameters needed to build the range replacement.
void LoopConvertCheck::determineRangeDescriptor(
    ASTContext *Context, const BoundNodes &Nodes, const ForStmt *Loop,
    LoopFixerKind FixerKind, const Expr *ContainerExpr,
    const UsageResult &Usages, RangeDescriptor &Descriptor) {
  Descriptor.ContainerString =
      std::string(getContainerString(Context, Loop, ContainerExpr));

  if (FixerKind == LFK_Iterator)
    getIteratorLoopQualifiers(Context, Nodes, Descriptor);
  else
    getArrayLoopQualifiers(Context, Nodes, ContainerExpr, Usages, Descriptor);
}

/// Check some of the conditions that must be met for the loop to be
/// convertible.
bool LoopConvertCheck::isConvertible(ASTContext *Context,
                                     const ast_matchers::BoundNodes &Nodes,
                                     const ForStmt *Loop,
                                     LoopFixerKind FixerKind) {
  // If we already modified the range of this for loop, don't do any further
  // updates on this iteration.
  if (TUInfo->getReplacedVars().count(Loop))
    return false;

  // Check that we have exactly one index variable and at most one end variable.
  const auto *LoopVar = Nodes.getNodeAs<VarDecl>(IncrementVarName);
  const auto *CondVar = Nodes.getNodeAs<VarDecl>(ConditionVarName);
  const auto *InitVar = Nodes.getNodeAs<VarDecl>(InitVarName);
  if (!areSameVariable(LoopVar, CondVar) || !areSameVariable(LoopVar, InitVar))
    return false;
  const auto *EndVar = Nodes.getNodeAs<VarDecl>(EndVarName);
  const auto *ConditionEndVar = Nodes.getNodeAs<VarDecl>(ConditionEndVarName);
  if (EndVar && !areSameVariable(EndVar, ConditionEndVar))
    return false;

  // FIXME: Try to put most of this logic inside a matcher.
  if (FixerKind == LFK_Iterator) {
    QualType InitVarType = InitVar->getType();
    QualType CanonicalInitVarType = InitVarType.getCanonicalType();

    const auto *BeginCall = Nodes.getNodeAs<CXXMemberCallExpr>(BeginCallName);
    assert(BeginCall && "Bad Callback. No begin call expression");
    QualType CanonicalBeginType =
        BeginCall->getMethodDecl()->getReturnType().getCanonicalType();
    if (CanonicalBeginType->isPointerType() &&
        CanonicalInitVarType->isPointerType()) {
      // If the initializer and the variable are both pointers check if the
      // un-qualified pointee types match, otherwise we don't use auto.
      if (!Context->hasSameUnqualifiedType(
              CanonicalBeginType->getPointeeType(),
              CanonicalInitVarType->getPointeeType()))
        return false;
    }
  } else if (FixerKind == LFK_PseudoArray) {
    // This call is required to obtain the container.
    const auto *EndCall = Nodes.getNodeAs<CXXMemberCallExpr>(EndCallName);
    if (!EndCall || !dyn_cast<MemberExpr>(EndCall->getCallee()))
      return false;
  }
  return true;
}

void LoopConvertCheck::check(const MatchFinder::MatchResult &Result) {
  const BoundNodes &Nodes = Result.Nodes;
  Confidence ConfidenceLevel(Confidence::CL_Safe);
  ASTContext *Context = Result.Context;

  const ForStmt *Loop;
  LoopFixerKind FixerKind;
  RangeDescriptor Descriptor;

  if ((Loop = Nodes.getNodeAs<ForStmt>(LoopNameArray))) {
    FixerKind = LFK_Array;
  } else if ((Loop = Nodes.getNodeAs<ForStmt>(LoopNameIterator))) {
    FixerKind = LFK_Iterator;
  } else {
    Loop = Nodes.getNodeAs<ForStmt>(LoopNamePseudoArray);
    assert(Loop && "Bad Callback. No for statement");
    FixerKind = LFK_PseudoArray;
  }

  if (!isConvertible(Context, Nodes, Loop, FixerKind))
    return;

  const auto *LoopVar = Nodes.getNodeAs<VarDecl>(IncrementVarName);
  const auto *EndVar = Nodes.getNodeAs<VarDecl>(EndVarName);

  // If the loop calls end()/size() after each iteration, lower our confidence
  // level.
  if (FixerKind != LFK_Array && !EndVar)
    ConfidenceLevel.lowerTo(Confidence::CL_Reasonable);

  // If the end comparison isn't a variable, we can try to work with the
  // expression the loop variable is being tested against instead.
  const auto *EndCall = Nodes.getNodeAs<CXXMemberCallExpr>(EndCallName);
  const auto *BoundExpr = Nodes.getNodeAs<Expr>(ConditionBoundName);

  // Find container expression of iterators and pseudoarrays, and determine if
  // this expression needs to be dereferenced to obtain the container.
  // With array loops, the container is often discovered during the
  // ForLoopIndexUseVisitor traversal.
  const Expr *ContainerExpr = nullptr;
  if (FixerKind == LFK_Iterator) {
    ContainerExpr = findContainer(Context, LoopVar->getInit(),
                                  EndVar ? EndVar->getInit() : EndCall,
                                  &Descriptor.ContainerNeedsDereference);
  } else if (FixerKind == LFK_PseudoArray) {
    ContainerExpr = EndCall->getImplicitObjectArgument();
    Descriptor.ContainerNeedsDereference =
        dyn_cast<MemberExpr>(EndCall->getCallee())->isArrow();
  }

  // We must know the container or an array length bound.
  if (!ContainerExpr && !BoundExpr)
    return;

  ForLoopIndexUseVisitor Finder(Context, LoopVar, EndVar, ContainerExpr,
                                BoundExpr,
                                Descriptor.ContainerNeedsDereference);

  // Find expressions and variables on which the container depends.
  if (ContainerExpr) {
    ComponentFinderASTVisitor ComponentFinder;
    ComponentFinder.findExprComponents(ContainerExpr->IgnoreParenImpCasts());
    Finder.addComponents(ComponentFinder.getComponents());
  }

  // Find usages of the loop index. If they are not used in a convertible way,
  // stop here.
  if (!Finder.findAndVerifyUsages(Loop->getBody()))
    return;
  ConfidenceLevel.lowerTo(Finder.getConfidenceLevel());

  // Obtain the container expression, if we don't have it yet.
  if (FixerKind == LFK_Array) {
    ContainerExpr = Finder.getContainerIndexed()->IgnoreParenImpCasts();

    // Very few loops are over expressions that generate arrays rather than
    // array variables. Consider loops over arrays that aren't just represented
    // by a variable to be risky conversions.
    if (!getReferencedVariable(ContainerExpr) &&
        !isDirectMemberExpr(ContainerExpr))
      ConfidenceLevel.lowerTo(Confidence::CL_Risky);
  }

  // Find out which qualifiers we have to use in the loop range.
  TraversalKindScope RAII(*Context, ast_type_traits::TK_AsIs);
  const UsageResult &Usages = Finder.getUsages();
  determineRangeDescriptor(Context, Nodes, Loop, FixerKind, ContainerExpr,
                           Usages, Descriptor);

  // Ensure that we do not try to move an expression dependent on a local
  // variable declared inside the loop outside of it.
  // FIXME: Determine when the external dependency isn't an expression converted
  // by another loop.
  TUInfo->getParentFinder().gatherAncestors(*Context);
  DependencyFinderASTVisitor DependencyFinder(
      &TUInfo->getParentFinder().getStmtToParentStmtMap(),
      &TUInfo->getParentFinder().getDeclToParentStmtMap(),
      &TUInfo->getReplacedVars(), Loop);

  if (DependencyFinder.dependsOnInsideVariable(ContainerExpr) ||
      Descriptor.ContainerString.empty() || Usages.empty() ||
      ConfidenceLevel.getLevel() < MinConfidence)
    return;

  doConversion(Context, LoopVar, getReferencedVariable(ContainerExpr), Usages,
               Finder.getAliasDecl(), Finder.aliasUseRequired(),
               Finder.aliasFromForInit(), Loop, Descriptor);
}

} // namespace modernize
} // namespace tidy
} // namespace clang
