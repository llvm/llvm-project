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
#include <string_view>

namespace {
bool isUgly(std::string_view str) {
  if (str.size() < 2)
    return false;
  if (str[0] == '_' && str[1] >= 'A' && str[1] <= 'Z')
    return true;
  return str.find("__") != std::string_view::npos;
}

AST_MATCHER(clang::Attr, isPretty) {
  if (Node.isKeywordAttribute())
    return false;
  if (Node.isCXX11Attribute() && !Node.hasScope()) // TODO: reject standard attributes that are version extensions
    return false;
  if (Node.hasScope())
    if (!isUgly(Node.getScopeName()->getName()))
      return true;

  if (Node.getAttrName())
    return !isUgly(Node.getAttrName()->getName());

  return false;
}

std::optional<std::string> getUglyfiedCXX11Attr(const clang::Attr& attr) {
  // Don't try to fix attributes with `using` in them.
  if (std::ranges::search(std::string_view(attr.getSpelling()), std::string_view("::")).empty())
    return std::nullopt;

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
  if (const auto* call = result.Nodes.getNodeAs<clang::Attr>("normal_attribute"); call != nullptr) {
    auto diagnostic = diag(call->getLoc(), "Non-standard attributes should use the _Ugly spelling");
    auto uglified   = getUglified(*call);
    if (uglified.has_value()) {
      diagnostic << clang::FixItHint::CreateReplacement(call->getRange(), *uglified);
    }
  }
}
} // namespace libcpp
