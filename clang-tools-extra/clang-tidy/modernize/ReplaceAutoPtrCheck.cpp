//===--- ReplaceAutoPtrCheck.cpp - clang-tidy------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ReplaceAutoPtrCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

static const char AutoPtrTokenId[] = "AutoPrTokenId";
static const char AutoPtrOwnershipTransferId[] = "AutoPtrOwnershipTransferId";

/// \brief Matches expressions that are lvalues.
///
/// In the following example, a[0] matches expr(isLValue()):
/// \code
///   std::string a[2];
///   std::string b;
///   b = a[0];
///   b = "this string won't match";
/// \endcode
AST_MATCHER(Expr, isLValue) { return Node.getValueKind() == VK_LValue; }

/// Matches declarations whose declaration context is the C++ standard library
/// namespace std.
///
/// Note that inline namespaces are silently ignored during the lookup since
/// both libstdc++ and libc++ are known to use them for versioning purposes.
///
/// Given:
/// \code
///   namespace ns {
///     struct my_type {};
///     using namespace std;
///   }
///
///   using std::vector;
///   using ns:my_type;
///   using ns::list;
/// \code
///
/// usingDecl(hasAnyUsingShadowDecl(hasTargetDecl(isFromStdNamespace())))
/// matches "using std::vector" and "using ns::list".
AST_MATCHER(Decl, isFromStdNamespace) {
  const DeclContext *D = Node.getDeclContext();

  while (D->isInlineNamespace())
    D = D->getParent();

  if (!D->isNamespace() || !D->getParent()->isTranslationUnit())
    return false;

  const IdentifierInfo *Info = cast<NamespaceDecl>(D)->getIdentifier();

  return (Info && Info->isStr("std"));
}

/// \brief Matcher that finds auto_ptr declarations.
static DeclarationMatcher AutoPtrDecl =
    recordDecl(hasName("auto_ptr"), isFromStdNamespace());

/// \brief Matches types declared as auto_ptr.
static TypeMatcher AutoPtrType = qualType(hasDeclaration(AutoPtrDecl));

/// \brief Matcher that finds expressions that are candidates to be wrapped with
/// 'std::move'.
///
/// Binds the id \c AutoPtrOwnershipTransferId to the expression.
static StatementMatcher MovableArgumentMatcher =
    expr(allOf(isLValue(), hasType(AutoPtrType)))
        .bind(AutoPtrOwnershipTransferId);

/// \brief Creates a matcher that finds the locations of types referring to the
/// \c std::auto_ptr() type.
///
/// \code
///   std::auto_ptr<int> a;
///        ^~~~~~~~~~~~~
///
///   typedef std::auto_ptr<int> int_ptr_t;
///                ^~~~~~~~~~~~~
///
///   std::auto_ptr<int> fn(std::auto_ptr<int>);
///        ^~~~~~~~~~~~~         ^~~~~~~~~~~~~
///
///   <etc...>
/// \endcode
TypeLocMatcher makeAutoPtrTypeLocMatcher() {
  // Skip elaboratedType() as the named type will match soon thereafter.
  return typeLoc(loc(qualType(AutoPtrType, unless(elaboratedType()))))
      .bind(AutoPtrTokenId);
}

/// \brief Creates a matcher that finds the using declarations referring to
/// \c std::auto_ptr.
///
/// \code
///   using std::auto_ptr;
///   ^~~~~~~~~~~~~~~~~~~
/// \endcode
DeclarationMatcher makeAutoPtrUsingDeclMatcher() {
  return usingDecl(hasAnyUsingShadowDecl(hasTargetDecl(
                       allOf(hasName("auto_ptr"), isFromStdNamespace()))))
      .bind(AutoPtrTokenId);
}

/// \brief Creates a matcher that finds the \c std::auto_ptr copy-ctor and
/// assign-operator expressions.
///
/// \c AutoPtrOwnershipTransferId is assigned to the argument of the expression,
/// this is the part that has to be wrapped by \c std::move().
///
/// \code
///   std::auto_ptr<int> i, j;
///   i = j;
///   ~~~~^
/// \endcode
StatementMatcher makeTransferOwnershipExprMatcher() {
  return anyOf(
      cxxOperatorCallExpr(allOf(hasOverloadedOperatorName("="),
                                callee(cxxMethodDecl(ofClass(AutoPtrDecl))),
                                hasArgument(1, MovableArgumentMatcher))),
      cxxConstructExpr(allOf(hasType(AutoPtrType), argumentCountIs(1),
                             hasArgument(0, MovableArgumentMatcher))));
}

/// \brief Locates the \c auto_ptr token when it is referred by a \c TypeLoc.
///
/// \code
///   std::auto_ptr<int> i;
///        ^~~~~~~~~~~~~
/// \endcode
///
/// The caret represents the location returned and the tildes cover the
/// parameter \p AutoPtrTypeLoc.
///
/// \return An invalid \c SourceLocation if not found, otherwise the location
/// of the beginning of the \c auto_ptr token.
static SourceLocation locateFromTypeLoc(const TypeLoc *AutoPtrTypeLoc,
                                        const SourceManager &SM) {
  auto TL = AutoPtrTypeLoc->getAs<TemplateSpecializationTypeLoc>();
  if (TL.isNull())
    return SourceLocation();

  return TL.getTemplateNameLoc();
}

/// \brief Locates the \c auto_ptr token in using declarations.
///
/// \code
///   using std::auto_ptr;
///              ^
/// \endcode
///
/// The caret represents the location returned.
///
/// \return An invalid \c SourceLocation if not found, otherwise the location
/// of the beginning of the \c auto_ptr token.
static SourceLocation locateFromUsingDecl(const UsingDecl *UsingAutoPtrDecl,
                                          const SourceManager &SM) {
  return UsingAutoPtrDecl->getNameInfo().getBeginLoc();
}

/// \brief Verifies that the token at \p TokenStart is 'auto_ptr'.
static bool checkTokenIsAutoPtr(SourceLocation TokenStart,
                                const SourceManager &SM,
                                const LangOptions &LO) {
  SmallVector<char, 8> Buffer;
  bool Invalid = false;
  StringRef Res = Lexer::getSpelling(TokenStart, Buffer, SM, LO, &Invalid);

  return (!Invalid && Res == "auto_ptr");
}

ReplaceAutoPtrCheck::ReplaceAutoPtrCheck(StringRef Name,
                                         ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeStyle(utils::IncludeSorter::parseIncludeStyle(
          Options.get("IncludeStyle", "llvm"))) {}

void ReplaceAutoPtrCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle",
                utils::IncludeSorter::toString(IncludeStyle));
}

void ReplaceAutoPtrCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (getLangOpts().CPlusPlus) {
    Finder->addMatcher(makeAutoPtrTypeLocMatcher(), this);
    Finder->addMatcher(makeAutoPtrUsingDeclMatcher(), this);
    Finder->addMatcher(makeTransferOwnershipExprMatcher(), this);
  }
}

void ReplaceAutoPtrCheck::registerPPCallbacks(CompilerInstance &Compiler) {
  // Only register the preprocessor callbacks for C++; the functionality
  // currently does not provide any benefit to other languages, despite being
  // benign.
  if (getLangOpts().CPlusPlus) {
    Inserter.reset(new utils::IncludeInserter(
        Compiler.getSourceManager(), Compiler.getLangOpts(), IncludeStyle));
    Compiler.getPreprocessor().addPPCallbacks(Inserter->CreatePPCallbacks());
  }
}

void ReplaceAutoPtrCheck::check(const MatchFinder::MatchResult &Result) {
  SourceManager &SM = *Result.SourceManager;
  if (const auto *E =
          Result.Nodes.getNodeAs<Expr>(AutoPtrOwnershipTransferId)) {
    CharSourceRange Range = Lexer::makeFileCharRange(
        CharSourceRange::getTokenRange(E->getSourceRange()), SM, LangOptions());

    if (Range.isInvalid())
      return;

    auto Diag = diag(Range.getBegin(), "use std::move to transfer ownership")
                << FixItHint::CreateInsertion(Range.getBegin(), "std::move(")
                << FixItHint::CreateInsertion(Range.getEnd(), ")");

    auto Insertion =
        Inserter->CreateIncludeInsertion(SM.getMainFileID(), "utility",
                                         /*IsAngled=*/true);
    if (Insertion.hasValue())
      Diag << Insertion.getValue();

    return;
  }

  SourceLocation IdentifierLoc;
  if (const auto *TL = Result.Nodes.getNodeAs<TypeLoc>(AutoPtrTokenId)) {
    IdentifierLoc = locateFromTypeLoc(TL, SM);
  } else if (const auto *D =
                 Result.Nodes.getNodeAs<UsingDecl>(AutoPtrTokenId)) {
    IdentifierLoc = locateFromUsingDecl(D, SM);
  } else {
    llvm_unreachable("Bad Callback. No node provided.");
  }

  if (IdentifierLoc.isMacroID())
    IdentifierLoc = SM.getSpellingLoc(IdentifierLoc);

  // Ensure that only the 'auto_ptr' token is replaced and not the template
  // aliases.
  if (!checkTokenIsAutoPtr(IdentifierLoc, SM, LangOptions()))
    return;

  SourceLocation EndLoc =
      IdentifierLoc.getLocWithOffset(strlen("auto_ptr") - 1);
  diag(IdentifierLoc, "auto_ptr is deprecated, use unique_ptr instead")
      << FixItHint::CreateReplacement(SourceRange(IdentifierLoc, EndLoc),
                                      "unique_ptr");
}

} // namespace modernize
} // namespace tidy
} // namespace clang
