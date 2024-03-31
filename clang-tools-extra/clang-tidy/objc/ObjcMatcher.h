//===- AbseilMatcher.h - clang-tidy ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_OBJCMATCHER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_OBJCMATCHER_H

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang::ast_matchers {

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

} // namespace clang::ast_matchers

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_OBJCMATCHER_H
