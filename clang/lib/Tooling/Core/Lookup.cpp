//===--- Lookup.cpp - Framework for clang refactoring tools ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines helper methods for clang tools performing name lookup.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Core/Lookup.h"
#include "clang/AST/Decl.h"
using namespace clang;
using namespace clang::tooling;

// Returns true if the context in which the type is used and the context in
// which the type is declared are the same semantical namespace but different
// lexical namespaces.
static bool
usingFromDifferentCanonicalNamespace(const DeclContext *FromContext,
                                     const DeclContext *UseContext) {
  while (true) {
    // Look past non-namespaces and anonymous namespaces on FromContext.
    // We can skip anonymous namespace because:
    // 1. `FromContext` and `UseContext` must be in the same anonymous
    // namespaces since referencing across anonymous namespaces is not possible.
    // 2. If `FromContext` and `UseContext` are in the same anonymous namespace,
    // the function will still return `false` as expected.
    while (FromContext &&
           (!isa<NamespaceDecl>(FromContext) ||
            cast<NamespaceDecl>(FromContext)->isAnonymousNamespace()))
      FromContext = FromContext->getParent();

    // Look past non-namespaces and anonymous namespaces on UseContext.
    while (UseContext &&
           (!isa<NamespaceDecl>(UseContext) ||
            cast<NamespaceDecl>(UseContext)->isAnonymousNamespace()))
      UseContext = UseContext->getParent();

    // We hit the root, no namespace collision.
    if (!FromContext || !UseContext)
      return false;

    // Literally the same namespace, not a collision.
    if (FromContext == UseContext)
      return false;

    // Now check the names. If they match we have a different namespace with the
    // same name.
    if (cast<NamespaceDecl>(FromContext)->getDeclName() ==
        cast<NamespaceDecl>(UseContext)->getDeclName())
      return true;

    FromContext = FromContext->getParent();
    UseContext = UseContext->getParent();
  }
}

static StringRef getBestNamespaceSubstr(const DeclContext *DeclA,
                                        StringRef NewName,
                                        bool HadLeadingColonColon) {
  while (true) {
    while (DeclA && !isa<NamespaceDecl>(DeclA))
      DeclA = DeclA->getParent();

    // Fully qualified it is! Leave :: in place if it's there already.
    if (!DeclA)
      return HadLeadingColonColon ? NewName : NewName.substr(2);

    // Otherwise strip off redundant namespace qualifications from the new name.
    // We use the fully qualified name of the namespace and remove that part
    // from NewName if it has an identical prefix.
    std::string NS =
        "::" + cast<NamespaceDecl>(DeclA)->getQualifiedNameAsString() + "::";
    if (NewName.startswith(NS))
      return NewName.substr(NS.size());

    // No match yet. Strip of a namespace from the end of the chain and try
    // again. This allows to get optimal qualifications even if the old and new
    // decl only share common namespaces at a higher level.
    DeclA = DeclA->getParent();
  }
}

/// Check if the name specifier begins with a written "::".
static bool isFullyQualified(const NestedNameSpecifier *NNS) {
  while (NNS) {
    if (NNS->getKind() == NestedNameSpecifier::Global)
      return true;
    NNS = NNS->getPrefix();
  }
  return false;
}

std::string tooling::replaceNestedName(const NestedNameSpecifier *Use,
                                       const DeclContext *UseContext,
                                       const NamedDecl *FromDecl,
                                       StringRef ReplacementString) {
  assert(ReplacementString.startswith("::") &&
         "Expected fully-qualified name!");

  // We can do a raw name replacement when we are not inside the namespace for
  // the original function and it is not in the global namespace.  The
  // assumption is that outside the original namespace we must have a using
  // statement that makes this work out and that other parts of this refactor
  // will automatically fix using statements to point to the new function
  const bool class_name_only = !Use;
  const bool in_global_namespace =
      isa<TranslationUnitDecl>(FromDecl->getDeclContext());
  if (class_name_only && !in_global_namespace &&
      !usingFromDifferentCanonicalNamespace(FromDecl->getDeclContext(),
                                            UseContext)) {
    auto Pos = ReplacementString.rfind("::");
    return Pos != StringRef::npos ? ReplacementString.substr(Pos + 2)
                                  : ReplacementString;
  }
  // We did not match this because of a using statement, so we will need to
  // figure out how good a namespace match we have with our destination type.
  // We work backwards (from most specific possible namespace to least
  // specific).
  return getBestNamespaceSubstr(UseContext, ReplacementString,
                                isFullyQualified(Use));
}
