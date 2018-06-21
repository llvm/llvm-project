//===--- ForbiddenSubclassingCheck.cpp - clang-tidy -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ForbiddenSubclassingCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "../utils/OptionsUtils.h"

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

/// \brief Matches Objective-C classes that directly or indirectly
/// have a superclass matching \c Base.
///
/// Note that a class is not considered to be a subclass of itself.
///
/// Example matches Y, Z
/// (matcher = objcInterfaceDecl(hasName("X")))
/// \code
///   @interface X
///   @end
///   @interface Y : X  // directly derived
///   @end
///   @interface Z : Y  // indirectly derived
///   @end
/// \endcode
AST_MATCHER_P(ObjCInterfaceDecl, isSubclassOf,
              ast_matchers::internal::Matcher<ObjCInterfaceDecl>, Base) {
  for (const auto *SuperClass = Node.getSuperClass();
       SuperClass != nullptr;
       SuperClass = SuperClass->getSuperClass()) {
    if (Base.matches(*SuperClass, Finder, Builder)) {
      return true;
    }
  }
  return false;
}

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
  // this check should only be applied to ObjC sources.
  if (!getLangOpts().ObjC1 && !getLangOpts().ObjC2) {
    return;
  }
  Finder->addMatcher(
      objcInterfaceDecl(
          isSubclassOf(
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
