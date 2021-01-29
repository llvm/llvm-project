//===--- ForbiddenSubclassingCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ForbiddenSubclassingCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace objc {

namespace {

constexpr char DefaultForbiddenSuperClassNames[] =
    "ABNewPersonViewController;"
    "ABPeoplePickerNavigationController;"
    "ABPersonViewController;"
    "ABUnknownPersonViewController;"
    "NSHashTable;"
    "NSMapTable;"
    "NSPointerArray;"
    "NSPointerFunctions;"
    "NSTimer;"
    "UIActionSheet;"
    "UIAlertView;"
    "UIImagePickerController;"
    "UITextInputMode;"
    "UIWebView";

} // namespace

ForbiddenSubclassingCheck::ForbiddenSubclassingCheck(
    StringRef Name,
    ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ForbiddenSuperClassNames(
          utils::options::parseStringList(
              Options.get("ClassNames", DefaultForbiddenSuperClassNames))) {
}

void ForbiddenSubclassingCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      objcInterfaceDecl(
          isDerivedFrom(
              objcInterfaceDecl(
                  hasAnyName(
                      std::vector<StringRef>(
                          ForbiddenSuperClassNames.begin(),
                          ForbiddenSuperClassNames.end())))
              .bind("superclass")))
      .bind("subclass"),
      this);
}

void ForbiddenSubclassingCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *SubClass = Result.Nodes.getNodeAs<ObjCInterfaceDecl>(
      "subclass");
  assert(SubClass != nullptr);
  const auto *SuperClass = Result.Nodes.getNodeAs<ObjCInterfaceDecl>(
      "superclass");
  assert(SuperClass != nullptr);
  diag(SubClass->getLocation(),
       "Objective-C interface %0 subclasses %1, which is not "
       "intended to be subclassed")
      << SubClass
      << SuperClass;
}

void ForbiddenSubclassingCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(
      Opts,
      "ForbiddenSuperClassNames",
      utils::options::serializeStringList(ForbiddenSuperClassNames));
}

} // namespace objc
} // namespace tidy
} // namespace clang
