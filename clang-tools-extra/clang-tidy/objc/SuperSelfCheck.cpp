//===--- SuperSelfCheck.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuperSelfCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::objc {

namespace {

/// Matches Objective-C methods in the initializer family.
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

/// Matches Objective-C implementations with interfaces that match
/// \c Base.
///
/// Example matches implementation declarations for X.
///   (matcher = objcImplementationDecl(hasInterface(hasName("X"))))
/// \code
///   @interface X
///   @end
///   @implementation X
///   @end
///   @interface Y
//    @end
///   @implementation Y
///   @end
/// \endcode
AST_MATCHER_P(ObjCImplementationDecl, hasInterface,
              ast_matchers::internal::Matcher<ObjCInterfaceDecl>, Base) {
  const ObjCInterfaceDecl *InterfaceDecl = Node.getClassInterface();
  return Base.matches(*InterfaceDecl, Finder, Builder);
}

/// Matches Objective-C message expressions where the receiver is the
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
  Finder->addMatcher(
      objcMessageExpr(hasSelector("self"), isMessagingSuperInstance(),
                      hasAncestor(objcMethodDecl(
                          isInitializer(),
                          hasDeclContext(objcImplementationDecl(hasInterface(
                              isDerivedFrom(hasName("NSObject"))))))))
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

} // namespace clang::tidy::objc
