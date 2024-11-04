//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"

#include "uglify_attributes.hpp"

#include <algorithm>
#include <array>
#include <span>
#include <string_view>

namespace {
bool isUgly(std::string_view str) {
  if (str.size() < 2)
    return false;
  if (str[0] == '_' && str[1] >= 'A' && str[1] <= 'Z')
    return true;
  return str.find("__") != std::string_view::npos;
}

// Starting with Clang 17 ToT C++23 support is provided by CPlusPlus23 instead
// of C++23 support is provided by CPlusPlus2b. To allow a smooth transition for
// libc++ use "reflection" to select the proper member. Since the change
// happens in the development cycle it's not possible to use #ifdefs.
template <class T>
bool CPlusPlus23(const T& lang_opts)
  requires requires { T::CPlusPlus2b; }
{
  return lang_opts.CPlusPlus2b;
}

template <class T>
bool CPlusPlus23(const T& lang_opts)
  requires requires { T::CPlusPlus23; }
{
  return lang_opts.CPlusPlus23;
}

std::vector<const char*> get_standard_attributes(const clang::LangOptions& lang_opts) {
  std::vector<const char*> attributes;

  if (lang_opts.CPlusPlus11) {
    attributes.emplace_back("noreturn");
    attributes.emplace_back("carries_dependency");
  }

  if (lang_opts.CPlusPlus14)
    attributes.emplace_back("deprecated");

  if (lang_opts.CPlusPlus17) {
    attributes.emplace_back("fallthrough");
    attributes.emplace_back("nodiscard");
    attributes.emplace_back("maybe_unused");
  }

  if (lang_opts.CPlusPlus20) {
    attributes.emplace_back("likely");
    attributes.emplace_back("unlikely");
    attributes.emplace_back("no_unique_address");
  }

  if (CPlusPlus23(lang_opts)) {
    attributes.emplace_back("assume");
  }

  return attributes;
}

AST_MATCHER(clang::Attr, isPretty) {
  if (Node.isKeywordAttribute() || !Node.getAttrName())
    return false;
  if (Node.isCXX11Attribute() && !Node.hasScope()) {
    if (isUgly(Node.getAttrName()->getName()))
      return false;
    return !llvm::is_contained(
        get_standard_attributes(Finder->getASTContext().getLangOpts()), Node.getAttrName()->getName());
  }
  if (Node.hasScope())
    if (!isUgly(Node.getScopeName()->getName()))
      return true;
  return !isUgly(Node.getAttrName()->getName());

  return false;
}

std::optional<std::string> getUglyfiedCXX11Attr(const clang::Attr& attr) {
  // TODO: Don't emit FixItHints for attributes with `using` in them or emit correct fixes.

  std::string attr_string;
  if (attr.isClangScope())
    attr_string += "_Clang::";
  else if (attr.isGNUScope())
    attr_string += "__gnu__::";

  if (!attr.getAttrName()->getName().starts_with("__")) {
    attr_string += "__";
    attr_string += attr.getAttrName()->getName();
    attr_string += "__";
  } else {
    attr_string += attr.getAttrName()->getName();
  }
  return std::move(attr_string);
}

std::optional<std::string> getUglyfiedGNUAttr(const clang::Attr& attr) {
  return "__" + attr.getAttrName()->getName().str() + "__";
}

std::optional<std::string> getUglified(const clang::Attr& attr) {
  if (attr.isCXX11Attribute()) {
    return getUglyfiedCXX11Attr(attr);
  } else if (attr.isGNUAttribute()) {
    return getUglyfiedGNUAttr(attr);
  }

  return std::nullopt;
}
} // namespace

namespace libcpp {
uglify_attributes::uglify_attributes(llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context) {}

void uglify_attributes::registerMatchers(clang::ast_matchers::MatchFinder* finder) {
  using namespace clang::ast_matchers;
  finder->addMatcher(attr(isPretty()).bind("normal_attribute"), this);
}

void uglify_attributes::check(const clang::ast_matchers::MatchFinder::MatchResult& result) {
  if (const auto* attr = result.Nodes.getNodeAs<clang::Attr>("normal_attribute"); attr != nullptr) {
    auto diagnostic = diag(attr->getLoc(), "Non-standard attributes should use the _Ugly spelling");
    auto uglified   = getUglified(*attr);
    if (uglified.has_value()) {
      diagnostic << clang::FixItHint::CreateReplacement(attr->getRange(), *uglified);
    }
  }
}
} // namespace libcpp
