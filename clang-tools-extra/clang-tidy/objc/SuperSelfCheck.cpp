//===--- SuperSelfCheck.cpp - clang-tidy ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SuperSelfCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace objc {

namespace {

/// \brief Matches Objective-C methods in the initializer family.
///
/// Example matches -init and -initWithInt:.
///   (matcher = objcMethodDecl(isInitializer()))
/// \code
///   @interface Foo
///   - (instancetype)init;
///   - (instancetype)initWithInt:(int)i;
///   + (instancetype)init;
///   - (void)bar;
///   @end
/// \endcode
AST_MATCHER(ObjCMethodDecl, isInitializer) {
  return Node.getMethodFamily() == OMF_init;
}

/// \brief Matches Objective-C implementations of classes that directly or
/// indirectly have a superclass matching \c InterfaceDecl.
///
/// Note that a class is not considered to be a subclass of itself.
///
/// Example matches implementation declarations for Y and Z.
///   (matcher = objcInterfaceDecl(isSubclassOf(hasName("X"))))
/// \code
///   @interface X
///   @end
///   @interface Y : X
///   @end
///   @implementation Y  // directly derived
///   @end
///   @interface Z : Y
///   @end
///   @implementation Z  // indirectly derived
///   @end
/// \endcode
AST_MATCHER_P(ObjCImplementationDecl, isSubclassOf,
              ast_matchers::internal::Matcher<ObjCInterfaceDecl>,
              InterfaceDecl) {
  // Check if any of the superclasses of the class match.
  for (const ObjCInterfaceDecl *SuperClass =
           Node.getClassInterface()->getSuperClass();
       SuperClass != nullptr; SuperClass = SuperClass->getSuperClass()) {
    if (InterfaceDecl.matches(*SuperClass, Finder, Builder))
      return true;
  }

  // No matches found.
  return false;
}

/// \brief Matches Objective-C message expressions where the receiver is the
/// super instance.
///
/// Example matches the invocations of -banana and -orange.
///   (matcher = objcMessageExpr(isMessagingSuperInstance()))
/// \code
///   - (void)banana {
///     [self apple]
///     [super banana];
///     [super orange];
///   }
/// \endcode
AST_MATCHER(ObjCMessageExpr, isMessagingSuperInstance) {
  return Node.getReceiverKind() == ObjCMessageExpr::SuperInstance;
}

} // namespace

void SuperSelfCheck::registerMatchers(MatchFinder *Finder) {
  // This check should only be applied to Objective-C sources.
  if (!getLangOpts().ObjC)
    return;

  Finder->addMatcher(
      objcMessageExpr(
          hasSelector("self"), isMessagingSuperInstance(),
          hasAncestor(objcMethodDecl(isInitializer(),
                                     hasDeclContext(objcImplementationDecl(
                                         isSubclassOf(hasName("NSObject")))))))
          .bind("message"),
      this);
}

void SuperSelfCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Message = Result.Nodes.getNodeAs<ObjCMessageExpr>("message");

  auto Diag = diag(Message->getExprLoc(), "suspicious invocation of %0 in "
                                          "initializer; did you mean to "
                                          "invoke a superclass initializer?")
              << Message->getMethodDecl();

  SourceLocation ReceiverLoc = Message->getReceiverRange().getBegin();
  if (ReceiverLoc.isMacroID() || ReceiverLoc.isInvalid())
    return;

  SourceLocation SelectorLoc = Message->getSelectorStartLoc();
  if (SelectorLoc.isMacroID() || SelectorLoc.isInvalid())
    return;

  Diag << FixItHint::CreateReplacement(Message->getSourceRange(),
                                       StringRef("[super init]"));
}

} // namespace objc
} // namespace tidy
} // namespace clang
