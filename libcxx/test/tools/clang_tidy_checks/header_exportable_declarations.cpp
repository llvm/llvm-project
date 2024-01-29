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
        {libcpp::header_exportable_declarations::FileType::Module, "Module"},
        {libcpp::header_exportable_declarations::FileType::CHeader, "CHeader"},
        {libcpp::header_exportable_declarations::FileType::CompatModulePartition, "CompatModulePartition"},
        {libcpp::header_exportable_declarations::FileType::CompatModule, "CompatModule"}};
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
  case header_exportable_declarations::FileType::CHeader:
  case header_exportable_declarations::FileType::Header:
    if (filename_.empty())
      llvm::errs() << "No filename is provided.\n";
    if (extra_header_.empty())
      extra_header_ = "$^"; // Use a never matching regex to silence an error message.
    break;
  case header_exportable_declarations::FileType::ModulePartition:
  case header_exportable_declarations::FileType::CompatModulePartition:
    if (filename_.empty())
      llvm::errs() << "No filename is provided.\n";
    [[fallthrough]];
  case header_exportable_declarations::FileType::Module:
  case header_exportable_declarations::FileType::CompatModule:
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
      skip_decls_.emplace(b, e);
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
      skip_decls_.emplace(std::move(s));
    }
#endif // defined(__clang_major__) && __clang_major__ < 16
  decls_ = skip_decls_;

  list = Options.get("ExtraDeclarations");
  // TODO(LLVM-17) Remove clang 15 work-around.
#if defined(__clang_major__) && __clang_major__ < 16
  if (list) {
    std::string_view s = *list;
    auto b             = s.begin();
    auto e             = std::find(b, s.end(), ' ');
    while (b != e) {
      std::cout << "using ::" << std::string_view{b, e} << ";\n";
      if (e == s.end())
        break;
      b = e + 1;
      e = std::find(b, s.end(), ' ');
    }
  }
#else  // defined(__clang_major__) && __clang_major__ < 16
  if (list)
    for (auto decl : std::views::split(*list, ' '))
      std::cout << "using ::" << std::string_view{decl.data(), decl.size()} << ";\n";
#endif // defined(__clang_major__) && __clang_major__ < 16
}

header_exportable_declarations::~header_exportable_declarations() {
  for (const auto& name : global_decls_)
    if (!skip_decls_.contains("std::" + name) && decls_.contains("std::" + name))
      std::cout << "using ::" << name << ";\n";
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
  case FileType::CHeader:
    // For C headers of the std.compat two matchers are used
    // - The cheader matcher; in libc++ these are never split in multiple
    //   headers so limiting the declarations to that header works.
    // - The header.h; where the declarations of this header are provided
    //   is not specified and depends on the libc used. Therefore it is not
    //   possible to restrict the location in a portable way.
    finder->addMatcher(namedDecl().bind("cheader_exportable_declarations"), this);

    [[fallthrough]];
  case FileType::ModulePartition:
  case FileType::CompatModulePartition:
    finder->addMatcher(namedDecl(isExpansionInFileMatching(filename_)).bind("header_exportable_declarations"), this);
    break;
  case FileType::Module:
  case FileType::CompatModule:
    finder->addMatcher(namedDecl().bind("header_exportable_declarations"), this);
    break;
  case header_exportable_declarations::FileType::Unknown:
    llvm::errs() << "This should be unreachable.\n";
    break;
  }
}

/// Returns the qualified name of a public declaration.
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
///
/// When the named declaration uses a reserved name the result is an
/// empty string.
static std::string get_qualified_name(const clang::NamedDecl& decl) {
  std::string result = decl.getNameAsString();
  // Reject reserved names (ignoring _ in global namespace).
  if (result.size() >= 2 && result[0] == '_')
    if (result[1] == '_' || std::isupper(result[1]))
      if (result != "_Exit")
        return "";

  for (auto* context = llvm::dyn_cast_or_null<clang::NamespaceDecl>(decl.getDeclContext()); //
       context;
       context = llvm::dyn_cast_or_null<clang::NamespaceDecl>(context->getDeclContext())) {
    std::string ns = std::string(context->getName());

    if (ns.starts_with("__")) {
      // When the reserved name is an inline namespace the namespace is
      // not added to the qualified name instead of removed. Libc++ uses
      // several inline namespace with reserved names. For example,
      // __1 for every declaration, __cpo in range-based algorithms.
      //
      // Note other inline namespaces are expanded. This resolves
      // ambiguity when two named declarations have the same name but in
      // different inline namespaces. These typically are the literal
      // conversion operators like operator""s which can be a
      // std::string or std::chrono::seconds.
      if (!context->isInline())
        return "";
    } else
      result = ns + "::" + result;
  }
  return result;
}

static bool is_viable_declaration(const clang::NamedDecl* decl) {
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

/// Some declarations in the global namespace are exported from the std module.
static bool is_global_name_exported_by_std_module(std::string_view name) {
  static const std::set<std::string_view> valid{
      "operator delete", "operator delete[]", "operator new", "operator new[]"};
  return valid.contains(name);
}

static bool is_valid_declaration_context(
    const clang::NamedDecl& decl, std::string_view name, header_exportable_declarations::FileType file_type) {
  const clang::DeclContext& context = *decl.getDeclContext();
  if (context.isNamespace())
    return true;

  if (context.isFunctionOrMethod() || context.isRecord())
    return false;

  if (is_global_name_exported_by_std_module(name))
    return true;

  return file_type != header_exportable_declarations::FileType::Header;
}

static bool is_module(header_exportable_declarations::FileType file_type) {
  switch (file_type) {
  case header_exportable_declarations::FileType::Module:
  case header_exportable_declarations::FileType::ModulePartition:
  case header_exportable_declarations::FileType::CompatModule:
  case header_exportable_declarations::FileType::CompatModulePartition:
    return true;

  case header_exportable_declarations::FileType::Header:
  case header_exportable_declarations::FileType::CHeader:
    return false;

  case header_exportable_declarations::FileType::Unknown:
    llvm::errs() << "This should be unreachable.\n";
    return false;
  }
}

void header_exportable_declarations::check(const clang::ast_matchers::MatchFinder::MatchResult& result) {
  if (const auto* decl = result.Nodes.getNodeAs<clang::NamedDecl>("header_exportable_declarations"); decl != nullptr) {
    if (!is_viable_declaration(decl))
      return;

    std::string name = get_qualified_name(*decl);
    if (name.empty())
      return;

    // For modules only take the declarations exported.
    if (is_module(file_type_))
      if (decl->getModuleOwnershipKind() != clang::Decl::ModuleOwnershipKind::VisibleWhenImported)
        return;

    if (!is_valid_declaration_context(*decl, name, file_type_))
      return;

    if (decls_.contains(name)) {
      // For modules avoid exporting the same named declaration twice. For
      // header files this is common and valid.
      if (file_type_ == FileType::ModulePartition || file_type_ == FileType::CompatModulePartition)
        // After the warning the script continues.
        // The test will fail since modules have duplicated entries and headers not.
        llvm::errs() << "Duplicated export of '" << name << "'.\n";
      else
        return;
    }

    // For named declarations in std this is valid
    //   using std::foo;
    // for named declarations it is invalid to use
    //   using bar;
    // Since fully qualifying named declarations in the std namespace is valid
    // using fully qualified names unconditionally.
    std::cout << "using ::" << std::string{name} << ";\n";
    decls_.insert(name);
  } else if (const auto* decl = result.Nodes.getNodeAs<clang::NamedDecl>("cheader_exportable_declarations");
             decl != nullptr) {
    if (decl->getDeclContext()->isNamespace())
      return;

    if (!is_viable_declaration(decl))
      return;

    std::string name = get_qualified_name(*decl);
    if (name.empty())
      return;

    if (global_decls_.contains(name))
      return;

    global_decls_.insert(name);
  }
}

} // namespace libcpp
