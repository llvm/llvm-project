//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"

#include "clang/Basic/Module.h"

#include "llvm/ADT/ArrayRef.h"

#include "header_exportable_declarations.hpp"

#include <iostream>
#include <iterator>
#include <ranges>
#include <algorithm>

template <>
struct clang::tidy::OptionEnumMapping<libcpp::header_exportable_declarations::FileType> {
  static llvm::ArrayRef<std::pair<libcpp::header_exportable_declarations::FileType, llvm::StringRef>> getEnumMapping() {
    static constexpr std::pair<libcpp::header_exportable_declarations::FileType, llvm::StringRef> Mapping[] = {
        {libcpp::header_exportable_declarations::FileType::Header, "Header"},
        {libcpp::header_exportable_declarations::FileType::ModulePartition, "ModulePartition"},
        {libcpp::header_exportable_declarations::FileType::Module, "Module"}};
    return ArrayRef(Mapping);
  }
};

namespace libcpp {
header_exportable_declarations::header_exportable_declarations(
    llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context),
      filename_(Options.get("Filename", "")),
      file_type_(Options.get("FileType", header_exportable_declarations::FileType::Unknown)),
      extra_header_(Options.get("ExtraHeader", "")) {
  switch (file_type_) {
  case header_exportable_declarations::FileType::Header:
    if (filename_.empty())
      llvm::errs() << "No filename is provided.\n";
    if (extra_header_.empty())
      extra_header_ = "$^"; // Use a never matching regex to silence an error message.
    break;
  case header_exportable_declarations::FileType::ModulePartition:
    if (filename_.empty())
      llvm::errs() << "No filename is provided.\n";
    [[fallthrough]];
  case header_exportable_declarations::FileType::Module:
    if (!extra_header_.empty())
      llvm::errs() << "Extra headers are not allowed for modules.\n";
    if (Options.get("SkipDeclarations"))
      llvm::errs() << "Modules may not skip declarations.\n";
    if (Options.get("ExtraDeclarations"))
      llvm::errs() << "Modules may not have extra declarations.\n";
    break;
  case header_exportable_declarations::FileType::Unknown:
    llvm::errs() << "No file type is provided.\n";
    break;
  }

  std::optional<llvm::StringRef> list = Options.get("SkipDeclarations");
  // TODO(LLVM-17) Remove clang 15 work-around.
#if defined(__clang_major__) && __clang_major__ < 16
  if (list) {
    std::string_view s = *list;
    auto b             = s.begin();
    auto e             = std::find(b, s.end(), ' ');
    while (b != e) {
      decls_.emplace(b, e);
      if (e == s.end())
        break;
      b = e + 1;
      e = std::find(b, s.end(), ' ');
    }
  }
#else  // defined(__clang_major__) && __clang_major__ < 16
  if (list)
    for (auto decl : std::views::split(*list, ' ')) {
      std::string s;
      std::ranges::copy(decl, std::back_inserter(s)); // use range based constructor
      decls_.emplace(std::move(s));
    }
#endif // defined(__clang_major__) && __clang_major__ < 16

  list = Options.get("ExtraDeclarations");
  // TODO(LLVM-17) Remove clang 15 work-around.
#if defined(__clang_major__) && __clang_major__ < 16
  if (list) {
    std::string_view s = *list;
    auto b             = s.begin();
    auto e             = std::find(b, s.end(), ' ');
    while (b != e) {
      std::cout << "using " << std::string_view{b, e} << ";\n";
      if (e == s.end())
        break;
      b = e + 1;
      e = std::find(b, s.end(), ' ');
    }
  }
#else  // defined(__clang_major__) && __clang_major__ < 16
  if (list)
    for (auto decl : std::views::split(*list, ' '))
      std::cout << "using " << std::string_view{decl.data(), decl.size()} << ";\n";
#endif // defined(__clang_major__) && __clang_major__ < 16
}

void header_exportable_declarations::registerMatchers(clang::ast_matchers::MatchFinder* finder) {
  // there are no public names in the Standard starting with an underscore, so
  // no need to check the strict rules.
  using namespace clang::ast_matchers;

  switch (file_type_) {
  case FileType::Header:
    finder->addMatcher(
        namedDecl(
            // Looks at the common locations where headers store their data
            // * header
            // * __header/*.h
            // * __fwd/header.h
            anyOf(isExpansionInFileMatching(("v1/__" + filename_ + "/").str()),
                  isExpansionInFileMatching(extra_header_),
                  isExpansionInFileMatching(("v1/__fwd/" + filename_ + "\\.h$").str()),
                  isExpansionInFileMatching(("v1/" + filename_ + "$").str())),
            unless(hasAncestor(friendDecl())))
            .bind("header_exportable_declarations"),
        this);
    break;
  case FileType::ModulePartition:
    finder->addMatcher(namedDecl(isExpansionInFileMatching(filename_)).bind("header_exportable_declarations"), this);
    break;
  case FileType::Module:
    finder->addMatcher(namedDecl().bind("header_exportable_declarations"), this);
    break;
  case header_exportable_declarations::FileType::Unknown:
    llvm::errs() << "This should be unreachable.\n";
    break;
  }
}

/// Returns the qualified name of a declaration.
///
/// There is a small issue with qualified names. Typically the name returned is
/// in the namespace \c std instead of the namespace \c std::__1. Except when a
/// name is declared both in the namespace \c std and in the namespace
/// \c std::__1. In that case the returned value will adjust the name to use
/// the namespace \c std.
///
/// The reason this happens is due to some parts of libc++ using
/// \code namespace std \endcode instead of
/// \code _LIBCPP_BEGIN_NAMESPACE_STD \endcode
/// Some examples
/// * cstddef has bitwise operators for the type \c byte
/// * exception has equality operators for the type \c exception_ptr
/// * initializer_list has the functions \c begin and \c end
static std::string get_qualified_name(const clang::NamedDecl& decl) {
  std::string result = decl.getQualifiedNameAsString();

  if (result.starts_with("std::__1::"))
    result.erase(5, 5);

  return result;
}

static bool is_viable_declaration(const clang::NamedDecl* decl) {
  // Declarations nested in records are automatically exported with the record itself.
  if (!decl->getDeclContext()->isNamespace())
    return false;

  // Declarations that are a subobject of a friend Declaration are automatically exported with the record itself.
  if (decl->getFriendObjectKind() != clang::Decl::FOK_None)
    return false;

  // *** Function declarations ***

  if (clang::CXXMethodDecl::classof(decl))
    return false;

  if (clang::CXXDeductionGuideDecl::classof(decl))
    return false;

  if (clang::FunctionDecl::classof(decl))
    return true;

  if (clang::CXXConstructorDecl::classof(decl))
    return false;

  // implicit constructors disallowed
  if (const auto* r = llvm::dyn_cast_or_null<clang::RecordDecl>(decl))
    return !r->isLambda() && !r->isImplicit();

  // *** Unconditionally accepted declarations ***
  return llvm::isa<clang::EnumDecl, clang::VarDecl, clang::ConceptDecl, clang::TypedefNameDecl, clang::UsingDecl>(decl);
}

/// Returns the name is a reserved name.
///
/// Detected reserved names are names starting with __ or _[A-Z].
/// These names can be in the namespace std or any namespace inside std. For
/// example std::ranges contains reserved names to implement the Niebloids.
///
/// This test misses 2 candidates which are not used in libc++
/// * any identifier with two underscores not at the start
/// * a name with a leading underscore in the global namespace
bool is_reserved_name(const std::string& name) {
  std::size_t pos = name.find("::_");
  if (pos == std::string::npos)
    return false;

  if (pos + 3 > name.size())
    return false;

  return name[pos + 3] == '_' || std::isupper(name[pos + 3]);
}

void header_exportable_declarations::check(const clang::ast_matchers::MatchFinder::MatchResult& result) {
  if (const auto* decl = result.Nodes.getNodeAs<clang::NamedDecl>("header_exportable_declarations"); decl != nullptr) {
    if (!is_viable_declaration(decl))
      return;

    std::string name = get_qualified_name(*decl);
    if (is_reserved_name(name))
      return;

    // For modules only take the declarations exported.
    if (file_type_ == FileType::ModulePartition || file_type_ == FileType::Module)
      if (decl->getModuleOwnershipKind() != clang::Decl::ModuleOwnershipKind::VisibleWhenImported)
        return;

    if (decls_.contains(name)) {
      // For modules avoid exporting the same named declaration twice. For
      // header files this is common and valid.
      if (file_type_ == FileType::ModulePartition)
        // After the warning the script continues.
        // The test will fail since modules have duplicated entries and headers not.
        llvm::errs() << "Duplicated export of '" << name << "'.\n";
      else
        return;
    }

    std::cout << "using " << std::string{name} << ";\n";
    decls_.insert(name);
  }
}

} // namespace libcpp
