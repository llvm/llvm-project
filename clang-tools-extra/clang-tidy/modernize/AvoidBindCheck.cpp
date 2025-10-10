//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidBindCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {

enum BindArgumentKind { BK_Temporary, BK_Placeholder, BK_CallExpr, BK_Other };
enum CaptureMode { CM_None, CM_ByRef, CM_ByValue };
enum CaptureExpr { CE_None, CE_Var, CE_InitExpression };

enum CallableType {
  CT_Other,          // unknown
  CT_Function,       // global or static function
  CT_MemberFunction, // member function with implicit this
  CT_Object,         // object with operator()
};

enum CallableMaterializationKind {
  CMK_Other,       // unknown
  CMK_Function,    // callable is the name of a member or non-member function.
  CMK_VariableRef, // callable is a simple expression involving a global or
                   // local variable.
  CMK_CallExpression, // callable is obtained as the result of a call expression
};

struct BindArgument {
  // A rough classification of the type of expression this argument was.
  BindArgumentKind Kind = BK_Other;

  // If this argument required a capture, a value indicating how it was
  // captured.
  CaptureMode CM = CM_None;

  // Whether the argument is a simple variable (we can capture it directly),
  // or an expression (we must introduce a capture variable).
  CaptureExpr CE = CE_None;

  // The exact spelling of this argument in the source code.
  StringRef SourceTokens;

  // The identifier of the variable within the capture list.  This may be
  // different from UsageIdentifier for example in the expression *d, where the
  // variable is captured as d, but referred to as *d.
  std::string CaptureIdentifier;

  // If this is a placeholder or capture init expression, contains the tokens
  // used to refer to this parameter from within the body of the lambda.
  std::string UsageIdentifier;

  // If Kind == BK_Placeholder, the index of the placeholder.
  size_t PlaceHolderIndex = 0;

  // True if the argument is used inside the lambda, false otherwise.
  bool IsUsed = false;

  // The actual Expr object representing this expression.
  const Expr *E = nullptr;
};

struct CallableInfo {
  CallableType Type = CT_Other;
  CallableMaterializationKind Materialization = CMK_Other;
  CaptureMode CM = CM_None;
  CaptureExpr CE = CE_None;
  StringRef SourceTokens;
  std::string CaptureIdentifier;
  std::string UsageIdentifier;
  StringRef CaptureInitializer;
  const FunctionDecl *Decl = nullptr;
  bool DoesReturn = false;
};

struct LambdaProperties {
  CallableInfo Callable;
  SmallVector<BindArgument, 4> BindArguments;
  StringRef BindNamespace;
  bool IsFixitSupported = false;
};

} // end namespace

static bool tryCaptureAsLocalVariable(const MatchFinder::MatchResult &Result,
                                      BindArgument &B, const Expr *E);

static bool tryCaptureAsMemberVariable(const MatchFinder::MatchResult &Result,
                                       BindArgument &B, const Expr *E);

static const Expr *ignoreTemporariesAndPointers(const Expr *E) {
  if (const auto *T = dyn_cast<UnaryOperator>(E))
    return ignoreTemporariesAndPointers(T->getSubExpr());

  const Expr *F = E->IgnoreImplicit();
  if (E != F)
    return ignoreTemporariesAndPointers(F);

  return E;
}

static const Expr *ignoreTemporariesAndConstructors(const Expr *E) {
  if (const auto *T = dyn_cast<CXXConstructExpr>(E))
    return ignoreTemporariesAndConstructors(T->getArg(0));

  const Expr *F = E->IgnoreImplicit();
  if (E != F)
    return ignoreTemporariesAndPointers(F);

  return E;
}

static StringRef getSourceTextForExpr(const MatchFinder::MatchResult &Result,
                                      const Expr *E) {
  return Lexer::getSourceText(
      CharSourceRange::getTokenRange(E->getBeginLoc(), E->getEndLoc()),
      *Result.SourceManager, Result.Context->getLangOpts());
}

static bool isCallExprNamed(const Expr *E, StringRef Name) {
  const auto *CE = dyn_cast<CallExpr>(E->IgnoreImplicit());
  if (!CE)
    return false;
  const auto *ND = dyn_cast<NamedDecl>(CE->getCalleeDecl());
  if (!ND)
    return false;
  return ND->getQualifiedNameAsString() == Name;
}

static void
initializeBindArgumentForCallExpr(const MatchFinder::MatchResult &Result,
                                  BindArgument &B, const CallExpr *CE,
                                  unsigned &CaptureIndex) {
  // std::ref(x) means to capture x by reference.
  if (isCallExprNamed(CE, "boost::ref") || isCallExprNamed(CE, "std::ref")) {
    B.Kind = BK_Other;
    if (tryCaptureAsLocalVariable(Result, B, CE->getArg(0)) ||
        tryCaptureAsMemberVariable(Result, B, CE->getArg(0))) {
      B.CE = CE_Var;
    } else {
      // The argument to std::ref is an expression that produces a reference.
      // Create a capture reference to hold it.
      B.CE = CE_InitExpression;
      B.UsageIdentifier = "capture" + llvm::utostr(CaptureIndex++);
    }
    // Strip off the reference wrapper.
    B.SourceTokens = getSourceTextForExpr(Result, CE->getArg(0));
    B.CM = CM_ByRef;
  } else {
    B.Kind = BK_CallExpr;
    B.CM = CM_ByValue;
    B.CE = CE_InitExpression;
    B.UsageIdentifier = "capture" + llvm::utostr(CaptureIndex++);
  }
  B.CaptureIdentifier = B.UsageIdentifier;
}

static bool anyDescendantIsLocal(const Stmt *Statement) {
  if (const auto *DeclRef = dyn_cast<DeclRefExpr>(Statement)) {
    const ValueDecl *Decl = DeclRef->getDecl();
    if (const auto *Var = dyn_cast_or_null<VarDecl>(Decl)) {
      if (Var->isLocalVarDeclOrParm())
        return true;
    }
  } else if (isa<CXXThisExpr>(Statement))
    return true;

  return any_of(Statement->children(), anyDescendantIsLocal);
}

static bool tryCaptureAsLocalVariable(const MatchFinder::MatchResult &Result,
                                      BindArgument &B, const Expr *E) {
  if (const auto *BTE = dyn_cast<CXXBindTemporaryExpr>(E)) {
    if (const auto *CE = dyn_cast<CXXConstructExpr>(BTE->getSubExpr()))
      return tryCaptureAsLocalVariable(Result, B, CE->getArg(0));
    return false;
  }

  const auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreImplicit());
  if (!DRE)
    return false;

  const auto *VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD || !VD->isLocalVarDeclOrParm())
    return false;

  B.CM = CM_ByValue;
  B.UsageIdentifier = std::string(getSourceTextForExpr(Result, E));
  B.CaptureIdentifier = B.UsageIdentifier;
  return true;
}

static bool tryCaptureAsMemberVariable(const MatchFinder::MatchResult &Result,
                                       BindArgument &B, const Expr *E) {
  if (const auto *BTE = dyn_cast<CXXBindTemporaryExpr>(E)) {
    if (const auto *CE = dyn_cast<CXXConstructExpr>(BTE->getSubExpr()))
      return tryCaptureAsMemberVariable(Result, B, CE->getArg(0));
    return false;
  }

  E = E->IgnoreImplicit();
  if (isa<CXXThisExpr>(E)) {
    // E is a direct use of "this".
    B.CM = CM_ByValue;
    B.UsageIdentifier = std::string(getSourceTextForExpr(Result, E));
    B.CaptureIdentifier = "this";
    return true;
  }

  const auto *ME = dyn_cast<MemberExpr>(E);
  if (!ME)
    return false;

  if (!ME->isLValue() || !isa<FieldDecl>(ME->getMemberDecl()))
    return false;

  if (isa<CXXThisExpr>(ME->getBase())) {
    // E refers to a data member without an explicit "this".
    B.CM = CM_ByValue;
    B.UsageIdentifier = std::string(getSourceTextForExpr(Result, E));
    B.CaptureIdentifier = "this";
    return true;
  }

  return false;
}

static SmallVector<BindArgument, 4>
buildBindArguments(const MatchFinder::MatchResult &Result,
                   const CallableInfo &Callable) {
  SmallVector<BindArgument, 4> BindArguments;
  static llvm::Regex MatchPlaceholder("^_([0-9]+)$");

  const auto *BindCall = Result.Nodes.getNodeAs<CallExpr>("bind");

  // Start at index 1 as first argument to bind is the function name.
  unsigned CaptureIndex = 0;
  for (size_t I = 1, ArgCount = BindCall->getNumArgs(); I < ArgCount; ++I) {

    const Expr *E = BindCall->getArg(I);
    BindArgument &B = BindArguments.emplace_back();

    size_t ArgIndex = I - 1;
    if (Callable.Type == CT_MemberFunction)
      --ArgIndex;

    bool IsObjectPtr = (I == 1 && Callable.Type == CT_MemberFunction);
    B.E = E;
    B.SourceTokens = getSourceTextForExpr(Result, E);

    if (!Callable.Decl || ArgIndex < Callable.Decl->getNumParams() ||
        IsObjectPtr)
      B.IsUsed = true;

    SmallVector<StringRef, 2> Matches;
    const auto *DRE = dyn_cast<DeclRefExpr>(E);
    if (MatchPlaceholder.match(B.SourceTokens, &Matches) ||
        // Check for match with qualifiers removed.
        (DRE && MatchPlaceholder.match(DRE->getDecl()->getName(), &Matches))) {
      B.Kind = BK_Placeholder;
      B.PlaceHolderIndex = std::stoi(std::string(Matches[1]));
      B.UsageIdentifier = "PH" + llvm::utostr(B.PlaceHolderIndex);
      B.CaptureIdentifier = B.UsageIdentifier;
      continue;
    }

    if (const auto *CE =
            dyn_cast<CallExpr>(ignoreTemporariesAndConstructors(E))) {
      initializeBindArgumentForCallExpr(Result, B, CE, CaptureIndex);
      continue;
    }

    if (tryCaptureAsLocalVariable(Result, B, B.E) ||
        tryCaptureAsMemberVariable(Result, B, B.E))
      continue;

    // If it's not something we recognize, capture it by init expression to be
    // safe.
    B.Kind = BK_Other;
    if (IsObjectPtr) {
      B.CE = CE_InitExpression;
      B.CM = CM_ByValue;
      B.UsageIdentifier = "ObjectPtr";
      B.CaptureIdentifier = B.UsageIdentifier;
    } else if (anyDescendantIsLocal(B.E)) {
      B.CE = CE_InitExpression;
      B.CM = CM_ByValue;
      B.CaptureIdentifier = "capture" + llvm::utostr(CaptureIndex++);
      B.UsageIdentifier = B.CaptureIdentifier;
    }
  }
  return BindArguments;
}

static int findPositionOfPlaceholderUse(ArrayRef<BindArgument> Args,
                                        size_t PlaceholderIndex) {
  for (size_t I = 0; I < Args.size(); ++I)
    if (Args[I].PlaceHolderIndex == PlaceholderIndex)
      return I;

  return -1;
}

static void addPlaceholderArgs(const LambdaProperties &LP,
                               llvm::raw_ostream &Stream,
                               bool PermissiveParameterList) {

  ArrayRef<BindArgument> Args = LP.BindArguments;

  const auto *MaxPlaceholderIt = llvm::max_element(
      Args, [](const BindArgument &B1, const BindArgument &B2) {
        return B1.PlaceHolderIndex < B2.PlaceHolderIndex;
      });

  // Placeholders (if present) have index 1 or greater.
  if (!PermissiveParameterList && (MaxPlaceholderIt == Args.end() ||
                                   MaxPlaceholderIt->PlaceHolderIndex == 0))
    return;

  size_t PlaceholderCount = MaxPlaceholderIt->PlaceHolderIndex;
  Stream << "(";
  StringRef Delimiter = "";
  for (size_t I = 1; I <= PlaceholderCount; ++I) {
    Stream << Delimiter << "auto &&";

    int ArgIndex = findPositionOfPlaceholderUse(Args, I);

    if (ArgIndex != -1 && Args[ArgIndex].IsUsed)
      Stream << " " << Args[ArgIndex].UsageIdentifier;
    Delimiter = ", ";
  }
  if (PermissiveParameterList)
    Stream << Delimiter << "auto && ...";
  Stream << ")";
}

static void addFunctionCallArgs(ArrayRef<BindArgument> Args,
                                llvm::raw_ostream &Stream) {
  StringRef Delimiter = "";

  for (const BindArgument &B : Args) {
    Stream << Delimiter;

    if (B.Kind == BK_Placeholder) {
      Stream << "std::forward<decltype(" << B.UsageIdentifier << ")>";
      Stream << "(" << B.UsageIdentifier << ")";
    } else if (B.CM != CM_None)
      Stream << B.UsageIdentifier;
    else
      Stream << B.SourceTokens;

    Delimiter = ", ";
  }
}

static bool isPlaceHolderIndexRepeated(const ArrayRef<BindArgument> Args) {
  llvm::SmallSet<size_t, 4> PlaceHolderIndices;
  for (const BindArgument &B : Args) {
    if (B.PlaceHolderIndex) {
      if (!PlaceHolderIndices.insert(B.PlaceHolderIndex).second)
        return true;
    }
  }
  return false;
}

static std::vector<const FunctionDecl *>
findCandidateCallOperators(const CXXRecordDecl *RecordDecl, size_t NumArgs) {
  std::vector<const FunctionDecl *> Candidates;

  for (const clang::CXXMethodDecl *Method : RecordDecl->methods()) {
    OverloadedOperatorKind OOK = Method->getOverloadedOperator();

    if (OOK != OverloadedOperatorKind::OO_Call)
      continue;

    if (Method->getNumParams() > NumArgs)
      continue;

    Candidates.push_back(Method);
  }

  // Find templated operator(), if any.
  for (const clang::Decl *D : RecordDecl->decls()) {
    const auto *FTD = dyn_cast<FunctionTemplateDecl>(D);
    if (!FTD)
      continue;
    const FunctionDecl *FD = FTD->getTemplatedDecl();

    OverloadedOperatorKind OOK = FD->getOverloadedOperator();
    if (OOK != OverloadedOperatorKind::OO_Call)
      continue;

    if (FD->getNumParams() > NumArgs)
      continue;

    Candidates.push_back(FD);
  }

  return Candidates;
}

static bool isFixitSupported(const CallableInfo &Callee,
                             ArrayRef<BindArgument> Args) {
  // Do not attempt to create fixits for nested std::bind or std::ref.
  // Supporting nested std::bind will be more difficult due to placeholder
  // sharing between outer and inner std::bind invocations, and std::ref
  // requires us to capture some parameters by reference instead of by value.
  if (any_of(Args, [](const BindArgument &B) {
        return isCallExprNamed(B.E, "boost::bind") ||
               isCallExprNamed(B.E, "std::bind");
      })) {
    return false;
  }

  // Do not attempt to create fixits when placeholders are reused.
  // Unused placeholders are supported by requiring C++14 generic lambdas.
  // FIXME: Support this case by deducing the common type.
  if (isPlaceHolderIndexRepeated(Args))
    return false;

  // If we can't determine the Decl being used, don't offer a fixit.
  if (!Callee.Decl)
    return false;

  if (Callee.Type == CT_Other || Callee.Materialization == CMK_Other)
    return false;

  return true;
}

static const FunctionDecl *getCallOperator(const CXXRecordDecl *Callable,
                                           size_t NumArgs) {
  std::vector<const FunctionDecl *> Candidates =
      findCandidateCallOperators(Callable, NumArgs);
  if (Candidates.size() != 1)
    return nullptr;

  return Candidates.front();
}

static const FunctionDecl *
getCallMethodDecl(const MatchFinder::MatchResult &Result, CallableType Type,
                  CallableMaterializationKind Materialization) {

  const Expr *Callee = Result.Nodes.getNodeAs<Expr>("ref");
  const Expr *CallExpression = ignoreTemporariesAndPointers(Callee);

  if (Type == CT_Object) {
    const auto *BindCall = Result.Nodes.getNodeAs<CallExpr>("bind");
    size_t NumArgs = BindCall->getNumArgs() - 1;
    return getCallOperator(Callee->getType()->getAsCXXRecordDecl(), NumArgs);
  }

  if (Materialization == CMK_Function) {
    if (const auto *DRE = dyn_cast<DeclRefExpr>(CallExpression))
      return dyn_cast<FunctionDecl>(DRE->getDecl());
  }

  // Maybe this is an indirect call through a function pointer or something
  // where we can't determine the exact decl.
  return nullptr;
}

static CallableType getCallableType(const MatchFinder::MatchResult &Result) {
  const auto *CallableExpr = Result.Nodes.getNodeAs<Expr>("ref");

  QualType QT = CallableExpr->getType();
  if (QT->isMemberFunctionPointerType())
    return CT_MemberFunction;

  if (QT->isFunctionPointerType() || QT->isFunctionReferenceType() ||
      QT->isFunctionType())
    return CT_Function;

  if (QT->isRecordType()) {
    const CXXRecordDecl *Decl = QT->getAsCXXRecordDecl();
    if (!Decl)
      return CT_Other;

    return CT_Object;
  }

  return CT_Other;
}

static CallableMaterializationKind
getCallableMaterialization(const MatchFinder::MatchResult &Result) {
  const auto *CallableExpr = Result.Nodes.getNodeAs<Expr>("ref");

  const auto *NoTemporaries = ignoreTemporariesAndPointers(CallableExpr);

  const auto *CE = dyn_cast<CXXConstructExpr>(NoTemporaries);
  const auto *FC = dyn_cast<CXXFunctionalCastExpr>(NoTemporaries);
  if ((isa<CallExpr>(NoTemporaries)) || (CE && (CE->getNumArgs() > 0)) ||
      (FC && (FC->getCastKind() == CK_ConstructorConversion)))
    // CE is something that looks like a call, with arguments - either
    // a function call or a constructor invocation.
    return CMK_CallExpression;

  if (isa<CXXFunctionalCastExpr>(NoTemporaries) || CE)
    return CMK_Function;

  if (const auto *DRE = dyn_cast<DeclRefExpr>(NoTemporaries)) {
    if (isa<FunctionDecl>(DRE->getDecl()))
      return CMK_Function;
    if (isa<VarDecl>(DRE->getDecl()))
      return CMK_VariableRef;
  }

  return CMK_Other;
}

static LambdaProperties
getLambdaProperties(const MatchFinder::MatchResult &Result) {
  const auto *CalleeExpr = Result.Nodes.getNodeAs<Expr>("ref");

  LambdaProperties LP;

  const auto *Bind = Result.Nodes.getNodeAs<CallExpr>("bind");
  const auto *Decl = cast<FunctionDecl>(Bind->getCalleeDecl());
  const auto *NS = cast<NamespaceDecl>(Decl->getEnclosingNamespaceContext());
  while (NS->isInlineNamespace())
    NS = cast<NamespaceDecl>(NS->getDeclContext());
  LP.BindNamespace = NS->getName();

  LP.Callable.Type = getCallableType(Result);
  LP.Callable.Materialization = getCallableMaterialization(Result);
  LP.Callable.Decl =
      getCallMethodDecl(Result, LP.Callable.Type, LP.Callable.Materialization);
  if (LP.Callable.Decl)
    if (const Type *ReturnType =
            LP.Callable.Decl->getReturnType().getCanonicalType().getTypePtr())
      LP.Callable.DoesReturn = !ReturnType->isVoidType();
  LP.Callable.SourceTokens = getSourceTextForExpr(Result, CalleeExpr);
  if (LP.Callable.Materialization == CMK_VariableRef) {
    LP.Callable.CE = CE_Var;
    LP.Callable.CM = CM_ByValue;
    LP.Callable.UsageIdentifier =
        std::string(getSourceTextForExpr(Result, CalleeExpr));
    LP.Callable.CaptureIdentifier = std::string(
        getSourceTextForExpr(Result, ignoreTemporariesAndPointers(CalleeExpr)));
  } else if (LP.Callable.Materialization == CMK_CallExpression) {
    LP.Callable.CE = CE_InitExpression;
    LP.Callable.CM = CM_ByValue;
    LP.Callable.UsageIdentifier = "Func";
    LP.Callable.CaptureIdentifier = "Func";
    LP.Callable.CaptureInitializer = getSourceTextForExpr(Result, CalleeExpr);
  }

  LP.BindArguments = buildBindArguments(Result, LP.Callable);

  LP.IsFixitSupported = isFixitSupported(LP.Callable, LP.BindArguments);

  return LP;
}

static bool emitCapture(llvm::StringSet<> &CaptureSet, StringRef Delimiter,
                        CaptureMode CM, CaptureExpr CE, StringRef Identifier,
                        StringRef InitExpression, raw_ostream &Stream) {
  if (CM == CM_None)
    return false;

  // This capture has already been emitted.
  if (CaptureSet.contains(Identifier))
    return false;

  Stream << Delimiter;

  if (CM == CM_ByRef)
    Stream << "&";
  Stream << Identifier;
  if (CE == CE_InitExpression)
    Stream << " = " << InitExpression;

  CaptureSet.insert(Identifier);
  return true;
}

static void emitCaptureList(const LambdaProperties &LP,
                            const MatchFinder::MatchResult &Result,
                            raw_ostream &Stream) {
  llvm::StringSet<> CaptureSet;
  bool AnyCapturesEmitted = false;

  AnyCapturesEmitted = emitCapture(
      CaptureSet, "", LP.Callable.CM, LP.Callable.CE,
      LP.Callable.CaptureIdentifier, LP.Callable.CaptureInitializer, Stream);

  for (const BindArgument &B : LP.BindArguments) {
    if (B.CM == CM_None || !B.IsUsed)
      continue;

    StringRef Delimiter = AnyCapturesEmitted ? ", " : "";

    if (emitCapture(CaptureSet, Delimiter, B.CM, B.CE, B.CaptureIdentifier,
                    B.SourceTokens, Stream))
      AnyCapturesEmitted = true;
  }
}

static ArrayRef<BindArgument>
getForwardedArgumentList(const LambdaProperties &P) {
  ArrayRef<BindArgument> Args = ArrayRef(P.BindArguments);
  if (P.Callable.Type != CT_MemberFunction)
    return Args;

  return Args.drop_front();
}
AvoidBindCheck::AvoidBindCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      PermissiveParameterList(Options.get("PermissiveParameterList", false)) {}

void AvoidBindCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "PermissiveParameterList", PermissiveParameterList);
}

void AvoidBindCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          callee(namedDecl(hasAnyName("::boost::bind", "::std::bind"))),
          hasArgument(
              0, anyOf(expr(hasType(memberPointerType())).bind("ref"),
                       expr(hasParent(materializeTemporaryExpr().bind("ref"))),
                       expr().bind("ref"))))
          .bind("bind"),
      this);
}

void AvoidBindCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<CallExpr>("bind");

  LambdaProperties LP = getLambdaProperties(Result);
  auto Diag =
      diag(MatchedDecl->getBeginLoc(),
           formatv("prefer a lambda to {0}::bind", LP.BindNamespace).str());
  if (!LP.IsFixitSupported)
    return;

  const auto *Ref = Result.Nodes.getNodeAs<Expr>("ref");

  std::string Buffer;
  llvm::raw_string_ostream Stream(Buffer);

  Stream << "[";
  emitCaptureList(LP, Result, Stream);
  Stream << "]";

  ArrayRef<BindArgument> FunctionCallArgs = ArrayRef(LP.BindArguments);

  addPlaceholderArgs(LP, Stream, PermissiveParameterList);

  Stream << " { ";

  if (LP.Callable.DoesReturn) {
    Stream << "return ";
  }

  if (LP.Callable.Type == CT_Function) {
    StringRef SourceTokens = LP.Callable.SourceTokens;
    SourceTokens.consume_front("&");
    Stream << SourceTokens;
  } else if (LP.Callable.Type == CT_MemberFunction) {
    const auto *MethodDecl = dyn_cast<CXXMethodDecl>(LP.Callable.Decl);
    const BindArgument &ObjPtr = FunctionCallArgs.front();

    if (MethodDecl->getOverloadedOperator() == OO_Call) {
      Stream << "(*" << ObjPtr.UsageIdentifier << ')';
    } else {
      if (!isa<CXXThisExpr>(ignoreTemporariesAndPointers(ObjPtr.E))) {
        Stream << ObjPtr.UsageIdentifier;
        Stream << "->";
      }
      Stream << MethodDecl->getNameAsString();
    }
  } else {
    switch (LP.Callable.CE) {
    case CE_Var:
      if (LP.Callable.UsageIdentifier != LP.Callable.CaptureIdentifier) {
        Stream << "(" << LP.Callable.UsageIdentifier << ")";
        break;
      }
      [[fallthrough]];
    case CE_InitExpression:
      Stream << LP.Callable.UsageIdentifier;
      break;
    default:
      Stream << getSourceTextForExpr(Result, Ref);
    }
  }

  Stream << "(";

  addFunctionCallArgs(getForwardedArgumentList(LP), Stream);
  Stream << "); }";

  Diag << FixItHint::CreateReplacement(MatchedDecl->getSourceRange(),
                                       Stream.str());
}

} // namespace clang::tidy::modernize
