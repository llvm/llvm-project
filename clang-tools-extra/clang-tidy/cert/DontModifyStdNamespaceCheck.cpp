//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DontModifyStdNamespaceCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace {

AST_POLYMORPHIC_MATCHER_P(
    hasAnyTemplateArgumentIncludingPack,
    AST_POLYMORPHIC_SUPPORTED_TYPES(ClassTemplateSpecializationDecl,
                                    TemplateSpecializationType, FunctionDecl),
    clang::ast_matchers::internal::Matcher<TemplateArgument>, InnerMatcher) {
  ArrayRef<TemplateArgument> Args =
      clang::ast_matchers::internal::getTemplateSpecializationArgs(Node);
  for (const auto &Arg : Args) {
    if (Arg.getKind() != TemplateArgument::Pack)
      continue;
    ArrayRef<TemplateArgument> PackArgs = Arg.getPackAsArray();
    if (matchesFirstInRange(InnerMatcher, PackArgs.begin(), PackArgs.end(),
                            Finder, Builder) != PackArgs.end())
      return true;
  }
  return matchesFirstInRange(InnerMatcher, Args.begin(), Args.end(), Finder,
                             Builder) != Args.end();
}

} // namespace

namespace clang::tidy::cert {

void DontModifyStdNamespaceCheck::registerMatchers(MatchFinder *Finder) {
  auto HasStdParent =
      hasDeclContext(namespaceDecl(hasAnyName("std", "posix"),
                                   unless(hasParent(namespaceDecl())))
                         .bind("nmspc"));
  auto UserDefinedType = qualType(
      hasUnqualifiedDesugaredType(tagType(unless(hasDeclaration(tagDecl(
          hasAncestor(namespaceDecl(hasAnyName("std", "posix"),
                                    unless(hasParent(namespaceDecl()))))))))));
  auto HasNoProgramDefinedTemplateArgument = unless(
      hasAnyTemplateArgumentIncludingPack(refersToType(UserDefinedType)));
  auto InsideStdClassOrClassTemplateSpecialization = hasDeclContext(
      anyOf(cxxRecordDecl(HasStdParent),
            classTemplateSpecializationDecl(
                HasStdParent, HasNoProgramDefinedTemplateArgument)));

  // Try to follow exactly CERT rule DCL58-CPP (this text is taken from C++
  // standard into the CERT rule):
  // "
  // 1 The behavior of a C++ program is undefined if it adds declarations or
  // definitions to namespace std or to a namespace within namespace std unless
  // otherwise specified. A program may add a template specialization for any
  // standard library template to namespace std only if the declaration depends
  // on a user-defined type and the specialization meets the standard library
  // requirements for the original template and is not explicitly prohibited. 2
  // The behavior of a C++ program is undefined if it declares — an explicit
  // specialization of any member function of a standard library class template,
  // or — an explicit specialization of any member function template of a
  // standard library class or class template, or — an explicit or partial
  // specialization of any member class template of a standard library class or
  // class template.
  // "
  // The "standard library requirements" and explicit prohibition are not
  // checked.

  auto BadNonTemplateSpecializationDecl =
      decl(unless(anyOf(functionDecl(isExplicitTemplateSpecialization()),
                        varDecl(isExplicitTemplateSpecialization()),
                        cxxRecordDecl(isExplicitTemplateSpecialization()))),
           HasStdParent);
  auto BadClassTemplateSpec = classTemplateSpecializationDecl(
      HasNoProgramDefinedTemplateArgument, HasStdParent);
  auto BadInnerClassTemplateSpec = classTemplateSpecializationDecl(
      InsideStdClassOrClassTemplateSpecialization);
  auto BadFunctionTemplateSpec =
      functionDecl(unless(cxxMethodDecl()), isExplicitTemplateSpecialization(),
                   HasNoProgramDefinedTemplateArgument, HasStdParent);
  auto BadMemberFunctionSpec =
      cxxMethodDecl(isExplicitTemplateSpecialization(),
                    InsideStdClassOrClassTemplateSpecialization);

  Finder->addMatcher(decl(anyOf(BadNonTemplateSpecializationDecl,
                                BadClassTemplateSpec, BadInnerClassTemplateSpec,
                                BadFunctionTemplateSpec, BadMemberFunctionSpec),
                          unless(isExpansionInSystemHeader()))
                         .bind("decl"),
                     this);
}
} // namespace clang::tidy::cert

static const NamespaceDecl *getTopLevelLexicalNamespaceDecl(const Decl *D) {
  const NamespaceDecl *LastNS = nullptr;
  while (D) {
    if (const auto *NS = dyn_cast<NamespaceDecl>(D))
      LastNS = NS;
    D = dyn_cast_or_null<Decl>(D->getLexicalDeclContext());
  }
  return LastNS;
}

void clang::tidy::cert::DontModifyStdNamespaceCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *D = Result.Nodes.getNodeAs<Decl>("decl");
  const auto *NS = Result.Nodes.getNodeAs<NamespaceDecl>("nmspc");
  if (!D || !NS)
    return;

  diag(D->getLocation(),
       "modification of %0 namespace can result in undefined behavior")
      << NS;
  // 'NS' is not always the namespace declaration that lexically contains 'D',
  // try to find such a namespace.
  if (const NamespaceDecl *LexNS = getTopLevelLexicalNamespaceDecl(D)) {
    assert(NS->getCanonicalDecl() == LexNS->getCanonicalDecl() &&
           "Mismatch in found namespace");
    diag(LexNS->getLocation(), "%0 namespace opened here", DiagnosticIDs::Note)
        << LexNS;
  }
}
