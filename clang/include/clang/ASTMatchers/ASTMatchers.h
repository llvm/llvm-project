//===- ASTMatchers.h - Structural query framework ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements matchers to be used together with the MatchFinder to
//  match AST nodes.
//
//  Matchers are created by generator functions, which can be combined in
//  a functional in-language DSL to express queries over the C++ AST.
//
//  For example, to match a class with a certain name, one would call:
//    cxxRecordDecl(hasName("MyClass"))
//  which returns a matcher that can be used to find all AST nodes that declare
//  a class named 'MyClass'.
//
//  For more complicated match expressions we're often interested in accessing
//  multiple parts of the matched AST nodes once a match is found. In that case,
//  call `.bind("name")` on match expressions that match the nodes you want to
//  access.
//
//  For example, when we're interested in child classes of a certain class, we
//  would write:
//    cxxRecordDecl(hasName("MyClass"), has(recordDecl().bind("child")))
//  When the match is found via the MatchFinder, a user provided callback will
//  be called with a BoundNodes instance that contains a mapping from the
//  strings that we provided for the `.bind()` calls to the nodes that were
//  matched.
//  In the given example, each time our matcher finds a match we get a callback
//  where "child" is bound to the RecordDecl node of the matching child
//  class declaration.
//
//  See ASTMatchersInternal.h for a more in-depth explanation of the
//  implementation details of the matcher framework.
//
//  See ASTMatchFinder.h for how to use the generated matchers to run over
//  an AST.
//
//  The doxygen comments on matchers are used to:
//   - create the doxygen documentation
//   - get information in the editor via signature help and goto definition
//   - generate the AST matcher reference html file
//   - test the documentation using a special syntax
//
//  Test Annotations:
//
//  The automatic testing uses doxygen commands (aliases) to extract the
//  relevant information about an example of using a matcher from the
//  documentation.
//
//      \header{a.h}
//      \endheader     <- zero or more header
//
//      \code
//        int a = 42;
//      \endcode
//      \compile_args{-std=c++,c23-or-later} <- optional, the std flag supports
//      std ranges and
//                                              whole languages
//
//      \matcher{expr()}              <- one or more matchers in succession
//      \matcher{integerLiteral()}    <- one or more matchers in succession
//                                    both matcher will have to match the
//                                    following matches
//      \match{42}                    <- one or more matches in succession
//
//      \matcher{varDecl()} <- new matcher resets the context, the above
//                             \match will not count for this new
//                             matcher(-group)
//      \match{int a  = 42} <- only applies to the previous matcher (not to the
//                             previous case)
//
//
//  The above block can be repeated inside a doxygen command for multiple code
//  examples for a single matcher. The test generation script will only look for
//  these annotations and ignore anything else like `\c` or the sentences where
//  these annotations are embedded into: `The matcher \matcher{expr()} matches
//  the number \match{42}.`.
//
//  Language Grammar:
//
//    [] denotes an optional, and <> denotes user-input
//
//    compile_args j:= \compile_args{[<compile_arg>;]<compile_arg>}
//    matcher_tag_key ::= type
//    match_tag_key ::= type || std || count || sub
//    matcher_tags ::= [matcher_tag_key=<value>;]matcher_tag_key=<value>
//    match_tags ::= [match_tag_key=<value>;]match_tag_key=<value>
//    matcher ::= \matcher{[matcher_tags$]<matcher>}
//    matchers ::= [matcher] matcher
//    match ::= \match{[match_tags$]<match>}
//    matches ::= [match] match
//    case ::= matchers matches
//    cases ::= [case] case
//    header-block ::= \header{<name>} <code> \endheader
//    code-block ::= \code <code> \endcode
//    testcase ::= code-block [compile_args] cases
//
//  Language Standard Versions:
//
//  The 'std' tag and '\compile_args' support specifying a specific language
//  version, a whole language and all of its versions, and thresholds (implies
//  ranges). Multiple arguments are passed with a ',' separator. For a language
//  and version to execute a tested matcher, it has to match the specified
//  '\compile_args' for the code, and the 'std' tag for the matcher. Predicates
//  for the 'std' compiler flag are used with disjunction between languages
//  (e.g. 'c || c++') and conjunction for all predicates specific to each
//  language (e.g. 'c++11-or-later && c++23-or-earlier').
//
//  Examples:
//   - `c`                                    all available versions of C
//   - `c++11`                                only C++11
//   - `c++11-or-later`                       C++11 or later
//   - `c++11-or-earlier`                     C++11 or earlier
//   - `c++11-or-later,c++23-or-earlier,c`    all of C and C++ between 11 and
//                                            23 (inclusive)
//   - `c++11-23,c`                             same as above
//
//  Tags
//
//  `type`:
//  **Match types** are used to select where the string that is used to check if
//  a node matches comes from. Available: `code`, `name`, `typestr`,
//  `typeofstr`. The default is `code`.
//
//   - `code`: Forwards to `tooling::fixit::getText(...)` and should be the
//   preferred way to show what matches.
//   - `name`: Casts the match to a `NamedDecl` and returns the result of
//   `getNameAsString`. Useful when the matched AST node is not easy to spell
//   out (`code` type), e.g., namespaces or classes with many members.
//   - `typestr`: Returns the result of `QualType::getAsString` for the type
//   derived from `Type` (otherwise, if it is derived from `Decl`, recurses with
//   `Node->getTypeForDecl()`)
//
//  **Matcher types** are used to mark matchers as sub-matcher with 'sub' or as
//  deactivated using 'none'. Testing sub-matcher is not implemented.
//
//  `count`:
//  Specifying a 'count=n' on a match will result in a test that requires that
//  the specified match will be matched n times. Default is 1.
//
//  `std`:
//  A match allows specifying if it matches only in specific language versions.
//  This may be needed when the AST differs between language versions.
//
//  `sub`:
//  The `sub` tag on a `\match` will indicate that the match is for a node of a
//  bound sub-matcher. E.g., `\matcher{expr(expr().bind("inner"))}` has a
//  sub-matcher that binds to `inner`, which is the value for the `sub` tag of
//  the expected match for the sub-matcher `\match{sub=inner$...}`. Currently,
//  sub-matchers are not tested in any way.
//
//
//  What if ...?
//
//  ... I want to add a matcher?
//
//  Add a doxygen comment to the matcher with a code example, corresponding
//  matchers and matches, that shows what the matcher is supposed to do. Specify
//  the compile arguments/supported languages if required, and run `ninja
//  check-clang-unit` to test the documentation.
//
//  ... the example I wrote is wrong?
//
//  The test-failure output of the generated test file will provide information
//  about
//   - where the generated test file is located
//   - which line in `ASTMatcher.h` the example is from
//   - which matches were: found, not-(yet)-found, expected
//   - in case of an unexpected match: what the node looks like using the
//   different `type`s
//   - the language version and if the test ran with a windows `-target` flag
//   (also in failure summary)
//
//  ... I don't adhere to the required order of the syntax?
//
//  The script will diagnose any found issues, such as `matcher is missing an
//  example` with a `file:line:` prefix, which should provide enough information
//  about the issue.
//
//  ... the script diagnoses a false-positive issue with a doxygen comment?
//
//  It hopefully shouldn't, but if you, e.g., added some non-matcher code and
//  documented it with doxygen, then the script will consider that as a matcher
//  documentation. As a result, the script will print that it detected a
//  mismatch between the actual and the expected number of failures. If the
//  diagnostic truly is a false-positive, change the
//  `expected_failure_statistics` at the top of the
//  `generate_ast_matcher_doc_tests.py` file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ASTMATCHERS_ASTMATCHERS_H
#define LLVM_CLANG_ASTMATCHERS_ASTMATCHERS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/LambdaCapture.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/ExceptionSpecificationType.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TypeTraits.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace ast_matchers {

/// Maps string IDs to AST nodes matched by parts of a matcher.
///
/// The bound nodes are generated by calling \c bind("id") on the node matchers
/// of the nodes we want to access later.
///
/// The instances of BoundNodes are created by \c MatchFinder when the user's
/// callbacks are executed every time a match is found.
class BoundNodes {
public:
  /// Returns the AST node bound to \c ID.
  ///
  /// Returns NULL if there was no node bound to \c ID or if there is a node but
  /// it cannot be converted to the specified type.
  template <typename T>
  const T *getNodeAs(StringRef ID) const {
    return MyBoundNodes.getNodeAs<T>(ID);
  }

  /// Type of mapping from binding identifiers to bound nodes. This type
  /// is an associative container with a key type of \c std::string and a value
  /// type of \c clang::DynTypedNode
  using IDToNodeMap = internal::BoundNodesMap::IDToNodeMap;

  /// Retrieve mapping from binding identifiers to bound nodes.
  const IDToNodeMap &getMap() const {
    return MyBoundNodes.getMap();
  }

private:
  friend class internal::BoundNodesTreeBuilder;

  /// Create BoundNodes from a pre-filled map of bindings.
  BoundNodes(internal::BoundNodesMap &MyBoundNodes)
      : MyBoundNodes(MyBoundNodes) {}

  internal::BoundNodesMap MyBoundNodes;
};

/// Types of matchers for the top-level classes in the AST class
/// hierarchy.
/// @{
using DeclarationMatcher = internal::Matcher<Decl>;
using StatementMatcher = internal::Matcher<Stmt>;
using TypeMatcher = internal::Matcher<QualType>;
using TypeLocMatcher = internal::Matcher<TypeLoc>;
using NestedNameSpecifierMatcher = internal::Matcher<NestedNameSpecifier>;
using NestedNameSpecifierLocMatcher = internal::Matcher<NestedNameSpecifierLoc>;
using CXXBaseSpecifierMatcher = internal::Matcher<CXXBaseSpecifier>;
using CXXCtorInitializerMatcher = internal::Matcher<CXXCtorInitializer>;
using TemplateArgumentMatcher = internal::Matcher<TemplateArgument>;
using TemplateArgumentLocMatcher = internal::Matcher<TemplateArgumentLoc>;
using LambdaCaptureMatcher = internal::Matcher<LambdaCapture>;
using AttrMatcher = internal::Matcher<Attr>;
/// @}

/// Matches any node.
///
/// Useful when another matcher requires a child matcher, but there's no
/// additional constraint. This will often be used with an explicit conversion
/// to an \c internal::Matcher<> type such as \c TypeMatcher.
///
/// Given
/// \code
///   int* p;
///   void f();
/// \endcode
/// The matcher \matcher{decl(anything())}
/// matches \match{int* p} and \match{void f()}.
/// Usable as: Any Matcher
inline internal::TrueMatcher anything() { return internal::TrueMatcher(); }

/// Matches the top declaration context.
///
/// Given
/// \code
///   int X;
///   namespace NS { int Y; }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{namedDecl(hasDeclContext(translationUnitDecl()))}
/// matches \match{int X} and \match{namespace NS { int Y; }},
/// but does not match \nomatch{int Y} because its decl-context is the
/// namespace \c NS .
extern const internal::VariadicDynCastAllOfMatcher<Decl, TranslationUnitDecl>
    translationUnitDecl;

/// Matches typedef declarations.
///
/// Given
/// \code
///   typedef int X;
///   using Y = int;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{typedefDecl()}
/// matches \match{typedef int X},
/// but does not match \nomatch{using Y = int}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, TypedefDecl>
    typedefDecl;

/// Matches typedef name declarations.
///
/// Given
/// \code
///   typedef int X;
///   using Y = int;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{typedefNameDecl()}
/// matches \match{typedef int X} and \match{using Y = int}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, TypedefNameDecl>
    typedefNameDecl;

/// Matches type alias declarations.
///
/// Given
/// \code
///   typedef int X;
///   using Y = int;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{typeAliasDecl()}
/// matches \match{using Y = int},
/// but does not match \nomatch{typedef int X}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, TypeAliasDecl>
    typeAliasDecl;

/// Matches type alias template declarations.
///
/// Given
/// \code
///   template <typename T> struct X {};
///   template <typename T> using Y = X<T>;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{typeAliasTemplateDecl()}
/// matches \match{template <typename T> using Y = X<T>}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, TypeAliasTemplateDecl>
    typeAliasTemplateDecl;

/// Matches AST nodes that were expanded within the main-file.
///
/// Given the header \c Y.h
/// \header{Y.h}
///   #pragma once
///   typedef int my_header_int;
/// \endheader
/// and the source file
/// \code
///   #include "Y.h"
///   typedef int my_main_file_int;
///   my_main_file_int a = 0;
///   my_header_int b = 1;
/// \endcode
///
/// The matcher \matcher{typedefDecl(isExpansionInMainFile())}
/// matches \match{typedef int my_main_file_int},
/// but does not match \nomatch{typedef int my_header_int}.
///
/// Usable as: Matcher<Decl>, Matcher<Stmt>, Matcher<TypeLoc>
AST_POLYMORPHIC_MATCHER(isExpansionInMainFile,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(Decl, Stmt, TypeLoc)) {
  auto &SourceManager = Finder->getASTContext().getSourceManager();
  return SourceManager.isInMainFile(
      SourceManager.getExpansionLoc(Node.getBeginLoc()));
}

/// Matches AST nodes that were expanded within system-header-files.
///
/// Given the header \c SystemHeader.h
/// \header{system_include/SystemHeader.h}
///   #pragma once
///   int header();
/// \endheader
/// and the source code
/// \code
///   #include <SystemHeader.h>
///   static int main_file();
/// \endcode
/// \compile_args{-isystemsystem_include/}
///
/// The matcher \matcher{type=none$functionDecl(isExpansionInSystemHeader())}
/// matches \match{int header()},
/// but does not match \nomatch{static int main_file()}.
///
/// Usable as: Matcher<Decl>, Matcher<Stmt>, Matcher<TypeLoc>
AST_POLYMORPHIC_MATCHER(isExpansionInSystemHeader,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(Decl, Stmt, TypeLoc)) {
  auto &SourceManager = Finder->getASTContext().getSourceManager();
  auto ExpansionLoc = SourceManager.getExpansionLoc(Node.getBeginLoc());
  if (ExpansionLoc.isInvalid()) {
    return false;
  }
  return SourceManager.isInSystemHeader(ExpansionLoc);
}

/// Matches AST nodes that were expanded within files whose name is
/// partially matching a given regex.
///
/// Given the headers \c Y.h
/// \header{Y.h}
///   #pragma once
///   typedef int my_y_int;
/// \endheader
/// and \c X.h
/// \header{X.h}
///   #pragma once
///   typedef int my_x_int;
/// \endheader
/// and the source code
/// \code
///   #include "X.h"
///   #include "Y.h"
///   typedef int my_main_file_int;
///   my_main_file_int a = 0;
///   my_x_int b = 1;
///   my_y_int c = 2;
/// \endcode
///
/// The matcher
/// \matcher{type=none$typedefDecl(isExpansionInFileMatching("Y.h"))}
/// matches \match{typedef int my_y_int},
/// but does not match \nomatch{typedef int my_main_file_int} or
/// \nomatch{typedef int my_x_int}.
///
/// Usable as: Matcher<Decl>, Matcher<Stmt>, Matcher<TypeLoc>
AST_POLYMORPHIC_MATCHER_REGEX(isExpansionInFileMatching,
                              AST_POLYMORPHIC_SUPPORTED_TYPES(Decl, Stmt,
                                                              TypeLoc),
                              RegExp) {
  auto &SourceManager = Finder->getASTContext().getSourceManager();
  auto ExpansionLoc = SourceManager.getExpansionLoc(Node.getBeginLoc());
  if (ExpansionLoc.isInvalid()) {
    return false;
  }
  auto FileEntry =
      SourceManager.getFileEntryRefForID(SourceManager.getFileID(ExpansionLoc));
  if (!FileEntry) {
    return false;
  }

  auto Filename = FileEntry->getName();
  return RegExp->match(Filename);
}

/// Matches statements that are (transitively) expanded from the named macro.
/// Does not match if only part of the statement is expanded from that macro or
/// if different parts of the statement are expanded from different
/// appearances of the macro.
///
/// Given
/// \code
///   #define A 0
///   #define B A
///   int c = B;
/// \endcode
///
/// The matcher \matcher{integerLiteral(isExpandedFromMacro("A"))}
/// matches the literal expanded at the initializer \match{B} of the variable
/// \c c .
AST_POLYMORPHIC_MATCHER_P(isExpandedFromMacro,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(Decl, Stmt, TypeLoc),
                          std::string, MacroName) {
  // Verifies that the statement' beginning and ending are both expanded from
  // the same instance of the given macro.
  auto& Context = Finder->getASTContext();
  std::optional<SourceLocation> B =
      internal::getExpansionLocOfMacro(MacroName, Node.getBeginLoc(), Context);
  if (!B) return false;
  std::optional<SourceLocation> E =
      internal::getExpansionLocOfMacro(MacroName, Node.getEndLoc(), Context);
  if (!E) return false;
  return *B == *E;
}

/// Matches declarations.
///
/// Given
/// \code
///   void X();
///   class C {
///     friend void X();
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{decl()}
/// matches \match{void X()} once, \match{type=name;count=2$C}
/// twice, once for the definition and once for the implicit class declaration,
/// and \match{count=2$friend void X()} twice, once for the declaration of the
/// friend, and once for the redeclaration of the function itself.
extern const internal::VariadicAllOfMatcher<Decl> decl;

/// Matches decomposition-declarations.
///
/// Given
/// \code
///   struct pair { int x; int y; };
///   pair make(int, int);
///   int number = 42;
///   auto [foo, bar] = make(42, 42);
/// \endcode
/// \compile_args{-std=c++17-or-later}
/// The matcher \matcher{decompositionDecl()}
/// matches \match{auto [foo, bar] = make(42, 42)},
/// but does not match \nomatch{type=name$number}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, DecompositionDecl>
    decompositionDecl;

/// Matches binding declarations
///
/// Given
/// \code
///   struct pair { int x; int y; };
///   pair make(int, int);
///   void f() {
///     auto [foo, bar] = make(42, 42);
///   }
/// \endcode
/// \compile_args{-std=c++17-or-later}
/// The matcher \matcher{bindingDecl()}
/// matches \match{type=name$foo} and \match{type=name$bar}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, BindingDecl>
    bindingDecl;

/// Matches a declaration of a linkage specification.
///
/// Given
/// \code
///   extern "C" {}
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{linkageSpecDecl()}
/// matches \match{extern "C" {}}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, LinkageSpecDecl>
    linkageSpecDecl;

/// Matches a declaration of anything that could have a name.
///
/// Example matches \c X, \c S, the anonymous union type, \c i, and \c U;
/// Given
/// \code
///   typedef int X;
///   struct S { union { int i; } U; };
/// \endcode
/// The matcher \matcher{namedDecl()}
/// matches \match{typedef int X},
/// \match{std=c$struct S { union { int i; } U; }}, \match{int i},
/// the unnamed union\match{type=name$} and the variable
/// \match{union { int i; } U},
/// with \match{type=name;count=2;std=c++$S} matching twice in C++.
/// Once for the implicit class declaration and once for the declaration itself.
extern const internal::VariadicDynCastAllOfMatcher<Decl, NamedDecl> namedDecl;

/// Matches a declaration of label.
///
/// Given
/// \code
///   void bar();
///   void foo() {
///     goto FOO;
///     FOO: bar();
///   }
/// \endcode
/// The matcher \matcher{type=none$labelDecl()}
/// matches \match{FOO: bar()}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, LabelDecl> labelDecl;

/// Matches a declaration of a namespace.
///
/// Given
/// \code
///   namespace {}
///   namespace test {}
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{namespaceDecl()}
/// matches \match{namespace {}} and \match{namespace test {}}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, NamespaceDecl>
    namespaceDecl;

/// Matches a declaration of a namespace alias.
///
/// Given
/// \code
///   namespace test {}
///   namespace alias = ::test;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{namespaceAliasDecl()}
/// matches \match{namespace alias = ::test},
/// but does not match \nomatch{namespace test {}}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, NamespaceAliasDecl>
    namespaceAliasDecl;

/// Matches class, struct, and union declarations.
///
/// Given
/// \code
///   class X;
///   template<class T> class Z {};
///   struct S {};
///   union U {};
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{recordDecl()}
/// matches \match{class X} once, and the rest of the declared records twice,
/// once for their written definition and once for their implicit declaration:
/// \match{type=name;count=2$Z}, \match{type=name;count=2$S} and
/// \match{type=name;count=2$U}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, RecordDecl> recordDecl;

/// Matches C++ class declarations.
///
/// Given
/// \code
///   class X;
///   template<class T> class Z {};
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{cxxRecordDecl()}
/// matches \match{class X} once, and \match{type=name;count=2$Z} twice,
/// once for the written definition and once for the implicit declaration.
extern const internal::VariadicDynCastAllOfMatcher<Decl, CXXRecordDecl>
    cxxRecordDecl;

/// Matches C++ class template declarations.
///
/// Given
/// \code
///   template<class T> class Z {};
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{classTemplateDecl()}
/// matches \match{template<class T> class Z {}}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, ClassTemplateDecl>
    classTemplateDecl;

/// Matches C++ class template specializations.
///
/// Given
/// \code
///   template<typename T> class A {};
///   template<> class A<double> {};
///   A<int> a;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{classTemplateSpecializationDecl()}
/// matches \match{type=typestr$class A<int>}
/// and \match{type=typestr$class A<double>}.
extern const internal::VariadicDynCastAllOfMatcher<
    Decl, ClassTemplateSpecializationDecl>
    classTemplateSpecializationDecl;

/// Matches C++ class template partial specializations.
///
/// Given
/// \code
///   template<class T1, class T2, int I>
///   class A {};
///
///   template<class T, int I> class A<T, T*, I> {};
///
///   template<>
///   class A<int, int, 1> {};
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{classTemplatePartialSpecializationDecl()}
/// matches \match{template<class T, int I> class A<T, T*, I> {}},
/// but does not match \nomatch{A<int, int, 1>}.
extern const internal::VariadicDynCastAllOfMatcher<
    Decl, ClassTemplatePartialSpecializationDecl>
    classTemplatePartialSpecializationDecl;

/// Matches declarator declarations (field, variable, function
/// and non-type template parameter declarations).
///
/// Given
/// \code
///   class X { int y; };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{declaratorDecl()}
/// matches \match{int y}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, DeclaratorDecl>
    declaratorDecl;

/// Matches parameter variable declarations.
///
/// Given
/// \code
///   void f(int x);
/// \endcode
/// The matcher \matcher{parmVarDecl()}
/// matches \match{int x}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, ParmVarDecl>
    parmVarDecl;

/// Matches C++ access specifier declarations.
///
/// Given
/// \code
///   class C {
///   public:
///     int a;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{accessSpecDecl()}
/// matches \match{public:}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, AccessSpecDecl>
    accessSpecDecl;

/// Matches class bases.
///
/// Given
/// \code
///   class B {};
///   class C : public B {};
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxRecordDecl(hasDirectBase(cxxBaseSpecifier()))}
/// matches \match{class C : public B {}}.
extern const internal::VariadicAllOfMatcher<CXXBaseSpecifier> cxxBaseSpecifier;

/// Matches constructor initializers.
///
/// Given
/// \code
///   class C {
///     C() : i(42) {}
///     int i;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxCtorInitializer()}
/// matches \match{i(42)}.
extern const internal::VariadicAllOfMatcher<CXXCtorInitializer>
    cxxCtorInitializer;

/// Matches template arguments.
///
/// Given
/// \code
///   template <typename T> struct C {};
///   C<int> c;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{templateSpecializationType(hasAnyTemplateArgument(templateArgument()))}
/// matches \match{type=typestr$C<int>}.
extern const internal::VariadicAllOfMatcher<TemplateArgument> templateArgument;

/// Matches template arguments (with location info).
///
/// Given
/// \code
///   template <typename T> struct C {};
///   C<int> c;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{templateArgumentLoc()}
/// matches \match{int} in C<int>.
extern const internal::VariadicAllOfMatcher<TemplateArgumentLoc>
    templateArgumentLoc;

/// Matches template name.
///
/// Given
/// \code
///   template<template <typename> class S> class X {};
///   template<typename T> class Y {};
///   X<Y> xi;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{classTemplateSpecializationDecl(hasAnyTemplateArgument(
///               refersToTemplate(templateName())))}
/// matches the specialization \match{type=typestr$class X<Y>}
extern const internal::VariadicAllOfMatcher<TemplateName> templateName;

/// Matches non-type template parameter declarations.
///
/// Given
/// \code
///   template <typename T, int N> struct C {};
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{nonTypeTemplateParmDecl()}
/// matches \match{int N},
/// but does not match \nomatch{typename T}.
extern const internal::VariadicDynCastAllOfMatcher<Decl,
                                                   NonTypeTemplateParmDecl>
    nonTypeTemplateParmDecl;

/// Matches template type parameter declarations.
///
/// Given
/// \code
///   template <typename T, int N> struct C {};
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{templateTypeParmDecl()}
/// matches \match{typename T},
/// but does not \nomatch{int N}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, TemplateTypeParmDecl>
    templateTypeParmDecl;

/// Matches template template parameter declarations.
///
/// Given
/// \code
///   template <template <typename> class Z, int N> struct C {};
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{templateTemplateParmDecl()}
/// matches \match{template <typename> class Z},
/// but does not match \nomatch{int N}.
extern const internal::VariadicDynCastAllOfMatcher<Decl,
                                                   TemplateTemplateParmDecl>
    templateTemplateParmDecl;

/// Matches public C++ declarations and C++ base specifers that specify public
/// inheritance.
///
/// Given
/// \code
///   class C {
///   public:    int a;
///   protected: int b;
///   private:   int c;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{fieldDecl(isPublic())}
/// matches \match{int a}.
///
/// Given
/// \code
///   class Base {};
///   class Derived1 : public Base {};
///   struct Derived2 : Base {};
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(hasAnyBase(cxxBaseSpecifier(isPublic()).bind("base")))}
/// matches \match{class Derived1 : public Base {}} and
/// \match{struct Derived2 : Base {}},
/// with \matcher{type=sub$cxxBaseSpecifier(isPublic())} matching
/// \match{sub=base$public Base} and \match{sub=base$Base}.
AST_POLYMORPHIC_MATCHER(isPublic,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(Decl,
                                                        CXXBaseSpecifier)) {
  return getAccessSpecifier(Node) == AS_public;
}

/// Matches protected C++ declarations and C++ base specifers that specify
/// protected inheritance.
///
/// Given
/// \code
///   class C {
///   public:    int a;
///   protected: int b;
///   private:   int c;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{fieldDecl(isProtected())}
/// matches \match{int b}.
///
/// \code
///   class Base {};
///   class Derived : protected Base {};
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(hasAnyBase(cxxBaseSpecifier(isProtected()).bind("base")))}
/// matches \match{class Derived : protected Base {}}, with
/// \matcher{type=sub$cxxBaseSpecifier(isProtected())} matching
/// \match{sub=base$Base}.
AST_POLYMORPHIC_MATCHER(isProtected,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(Decl,
                                                        CXXBaseSpecifier)) {
  return getAccessSpecifier(Node) == AS_protected;
}

/// Matches private C++ declarations and C++ base specifers that specify private
/// inheritance.
///
/// Given
/// \code
///   class C {
///   public:    int a;
///   protected: int b;
///   private:   int c;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{fieldDecl(isPrivate())}
/// matches \match{int c}.
///
/// \code
///   struct Base {};
///   struct Derived1 : private Base {}; // \match{Base}
///   class Derived2 : Base {}; // \match{Base}
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(hasAnyBase(cxxBaseSpecifier(isPrivate()).bind("base")))}
/// matches \match{struct Derived1 : private Base {}} and
/// \match{class Derived2 : Base {}}, with
/// \matcher{type=sub$cxxBaseSpecifier(isPrivate())} matching
/// \match{sub=base;count=2$Base} each time.
AST_POLYMORPHIC_MATCHER(isPrivate,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(Decl,
                                                        CXXBaseSpecifier)) {
  return getAccessSpecifier(Node) == AS_private;
}

/// Matches non-static data members that are bit-fields.
///
/// Given
/// \code
///   class C {
///     int a : 2;
///     int b;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{fieldDecl(isBitField())}
/// matches \match{int a : 2},
/// but does not match \nomatch{int b}.
AST_MATCHER(FieldDecl, isBitField) { return Node.isBitField(); }

/// Matches non-static data members that are bit-fields of the specified
/// bit width.
///
/// Given
/// \code
///   class C {
///     int a : 2;
///     int b : 4;
///     int c : 2;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{fieldDecl(hasBitWidth(2))}
/// matches \match{int a : 2} and \match{int c : 2},
/// but not \nomatch{int b : 4}.
AST_MATCHER_P(FieldDecl, hasBitWidth, unsigned, Width) {
  return Node.isBitField() &&
         Node.getBitWidthValue(Finder->getASTContext()) == Width;
}

/// Matches non-static data members that have an in-class initializer.
///
/// Given
/// \code
///   class C {
///     int a = 2;
///     int b = 3;
///     int c;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{fieldDecl(hasInClassInitializer(integerLiteral(equals(2))))}
/// matches \match{int a = 2},
/// but does not match \nomatch{int b = 3}.
/// The matcher \matcher{fieldDecl(hasInClassInitializer(anything()))}
/// matches \match{int a = 2} and \match{int b = 3},
/// but does not match \nomatch{int c}.
AST_MATCHER_P(FieldDecl, hasInClassInitializer, internal::Matcher<Expr>,
              InnerMatcher) {
  const Expr *Initializer = Node.getInClassInitializer();
  return (Initializer != nullptr &&
          InnerMatcher.matches(*Initializer, Finder, Builder));
}

/// Determines whether the function is "main", which is the entry point
/// into an executable program.
///
/// Given
/// \code
///   void f();
///   int main() {}
/// \endcode
///
/// The matcher \matcher{functionDecl(isMain())} matches \match{int main() {}}.
AST_MATCHER(FunctionDecl, isMain) { return Node.isMain(); }

/// Matches the specialized template of a specialization declaration.
///
/// Given
/// \code
///   template<typename T> class A {}; // #1
///   template<> class A<int> {}; // #2
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{classTemplateSpecializationDecl(hasSpecializedTemplate(classTemplateDecl().bind("ctd")))}
/// matches \match{template<> class A<int> {}},
/// with \matcher{type=sub$classTemplateDecl()} matching the class template
/// declaration \match{sub=ctd$template <typename T> class A {}}.
AST_MATCHER_P(ClassTemplateSpecializationDecl, hasSpecializedTemplate,
              internal::Matcher<ClassTemplateDecl>, InnerMatcher) {
  const ClassTemplateDecl* Decl = Node.getSpecializedTemplate();
  return (Decl != nullptr &&
          InnerMatcher.matches(*Decl, Finder, Builder));
}

/// Matches an entity that has been implicitly added by the compiler (e.g.
/// implicit default/copy constructors).
///
/// Given
/// \code
///   struct S {};
///   void f(S obj) {
///     S copy = obj;
///     [&](){ return copy; };
///   }
/// \endcode
/// \compile_args{-std=c++11}
///
/// The matcher \matcher{cxxConstructorDecl(isImplicit(), isCopyConstructor())}
/// matches the implicit copy constructor of \match{S}.
/// The matcher \matcher{lambdaExpr(forEachLambdaCapture(
///     lambdaCapture(isImplicit())))} matches \match{[&](){ return copy; }},
/// because it implicitly captures \c copy .
AST_POLYMORPHIC_MATCHER(isImplicit,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(Decl, Attr,
                                                        LambdaCapture)) {
  return Node.isImplicit();
}

/// Matches templateSpecializationTypes, class template specializations,
/// variable template specializations, and function template specializations
/// that have at least one TemplateArgument matching the given InnerMatcher.
///
/// Given
/// \code
///   template<typename T> class A {};
///   template<> class A<double> {};
///   A<int> a;
///
///   template<typename T> void f() {};
///   void func() { f<int>(); };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
///
/// The matcher \matcher{classTemplateSpecializationDecl(
///                         hasAnyTemplateArgument(
///                           refersToType(asString("int"))))}
/// matches \match{type=typestr$class A<int>}.
///
/// The matcher
/// \matcher{functionDecl(hasAnyTemplateArgument(
///               refersToType(asString("int"))))}
/// matches the instantiation of
/// \match{void f() {}}.
AST_POLYMORPHIC_MATCHER_P(
    hasAnyTemplateArgument,
    AST_POLYMORPHIC_SUPPORTED_TYPES(ClassTemplateSpecializationDecl,
                                    VarTemplateSpecializationDecl, FunctionDecl,
                                    TemplateSpecializationType),
    internal::Matcher<TemplateArgument>, InnerMatcher) {
  ArrayRef<TemplateArgument> List =
      internal::getTemplateSpecializationArgs(Node);
  return matchesFirstInRange(InnerMatcher, List.begin(), List.end(), Finder,
                             Builder) != List.end();
}

/// Causes all nested matchers to be matched with the specified traversal kind.
///
/// Given
/// \code
///   void foo()
///   {
///       int i = 3.0;
///   }
/// \endcode
/// The matcher
/// \matcher{traverse(TK_IgnoreUnlessSpelledInSource,
///     varDecl(hasInitializer(floatLiteral().bind("init")))
///   )}
///   matches \match{int i = 3.0} with "init" bound to \match{sub=init$3.0}.
template <typename T>
internal::Matcher<T> traverse(TraversalKind TK,
                              const internal::Matcher<T> &InnerMatcher) {
  return internal::DynTypedMatcher::constructRestrictedWrapper(
             new internal::TraversalMatcher<T>(TK, InnerMatcher),
             InnerMatcher.getID().first)
      .template unconditionalConvertTo<T>();
}

template <typename T>
internal::BindableMatcher<T>
traverse(TraversalKind TK, const internal::BindableMatcher<T> &InnerMatcher) {
  return internal::BindableMatcher<T>(
      internal::DynTypedMatcher::constructRestrictedWrapper(
          new internal::TraversalMatcher<T>(TK, InnerMatcher),
          InnerMatcher.getID().first)
          .template unconditionalConvertTo<T>());
}

template <typename... T>
internal::TraversalWrapper<internal::VariadicOperatorMatcher<T...>>
traverse(TraversalKind TK,
         const internal::VariadicOperatorMatcher<T...> &InnerMatcher) {
  return internal::TraversalWrapper<internal::VariadicOperatorMatcher<T...>>(
      TK, InnerMatcher);
}

template <template <typename ToArg, typename FromArg> class ArgumentAdapterT,
          typename T, typename ToTypes>
internal::TraversalWrapper<
    internal::ArgumentAdaptingMatcherFuncAdaptor<ArgumentAdapterT, T, ToTypes>>
traverse(TraversalKind TK, const internal::ArgumentAdaptingMatcherFuncAdaptor<
                               ArgumentAdapterT, T, ToTypes> &InnerMatcher) {
  return internal::TraversalWrapper<
      internal::ArgumentAdaptingMatcherFuncAdaptor<ArgumentAdapterT, T,
                                                   ToTypes>>(TK, InnerMatcher);
}

template <template <typename T, typename... P> class MatcherT, typename... P,
          typename ReturnTypesF>
internal::TraversalWrapper<
    internal::PolymorphicMatcher<MatcherT, ReturnTypesF, P...>>
traverse(TraversalKind TK,
         const internal::PolymorphicMatcher<MatcherT, ReturnTypesF, P...>
             &InnerMatcher) {
  return internal::TraversalWrapper<
      internal::PolymorphicMatcher<MatcherT, ReturnTypesF, P...>>(TK,
                                                                  InnerMatcher);
}

template <typename... T>
internal::Matcher<typename internal::GetClade<T...>::Type>
traverse(TraversalKind TK, const internal::MapAnyOfHelper<T...> &InnerMatcher) {
  return traverse(TK, InnerMatcher.with());
}

/// Matches expressions that match InnerMatcher after any implicit AST
/// nodes are stripped off.
///
/// Parentheses and explicit casts are not discarded.
///
/// Given
/// \code
///   void f(int param) {
///     int a = 0;
///     int b = param;
///     const int c = 0;
///     const int d = param;
///     int e = (0U);
///     int f = (int)0.0;
///     const int g = ((int)(((0))));
///   }
/// \endcode
///
/// The matcher
/// \matcher{varDecl(hasInitializer(ignoringImplicit(integerLiteral())))}
/// matches \match{int a = 0} and \match{const int c = 0},
/// but not \nomatch{int e = (0U)} and \nomatch{((int)(((0)))}.
/// The matcher
/// \matcher{varDecl(hasInitializer(integerLiteral()))}
/// matches \match{int a = 0} and \match{const int c = 0},
/// but not \nomatch{int e = (0U)} and \nomatch{((int)(((0)))}.
///
/// The matcher
/// \matcher{varDecl(hasInitializer(ignoringImplicit(declRefExpr())))}
/// matches \match{int b = param} and \match{const int d = param}.
/// The matcher
/// \matcher{varDecl(hasInitializer(declRefExpr()))}
/// matches neither \nomatch{int b = param} nor \nomatch{const int d = param},
/// because an l-to-r-value cast happens.
AST_MATCHER_P(Expr, ignoringImplicit, internal::Matcher<Expr>, InnerMatcher) {
  return InnerMatcher.matches(*Node.IgnoreImplicit(), Finder, Builder);
}

/// Matches expressions that match InnerMatcher after any implicit casts
/// are stripped off.
///
/// Parentheses and explicit casts are not discarded.
/// Given
/// \code
///   int arr[5];
///   const int a = 0;
///   char b = 0;
///   const int c = a;
///   int *d = arr;
///   long e = (long) 0l;
/// \endcode
/// The matcher
/// \matcher{varDecl(hasInitializer(ignoringImpCasts(integerLiteral())))}
/// matches \match{const int a = 0} and \match{char b = 0},
/// but does not match \nomatch{long e = (long) 0l} because of the c-style
/// case.
///
/// The matcher
/// \matcher{varDecl(hasInitializer(ignoringImpCasts(declRefExpr())))}
/// matches \match{const int c = a} and \match{int *d = arr}.
///
/// The matcher
/// \matcher{varDecl(hasInitializer(integerLiteral()))}
/// matches \match{const int a = 0},
/// but does not match \nomatch{char b = 0} because of the implicit cast to
/// \c char or \nomatch{long e = (long) 0l} because of the c-style cast.
///
/// The matcher \matcher{varDecl(hasInitializer(declRefExpr()))}
/// does not match \nomatch{const int c = a} because \c a is cast from an
/// l- to an r-value or \nomatch{int *d = arr} because the array-to-pointer
/// decay.
AST_MATCHER_P(Expr, ignoringImpCasts,
              internal::Matcher<Expr>, InnerMatcher) {
  return InnerMatcher.matches(*Node.IgnoreImpCasts(), Finder, Builder);
}

/// Matches expressions that match InnerMatcher after parentheses and
/// casts are stripped off.
///
/// Implicit and non-C Style casts are also discarded.
/// Given
/// \code
///   int a = 0;
///   char b = (0);
///   void* c = reinterpret_cast<char*>(0);
///   char d = char(0);
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{varDecl(hasInitializer(ignoringParenCasts(integerLiteral())))}
/// matches \match{int a = 0}, \match{char b = (0)},
/// \match{void* c = reinterpret_cast<char*>(0)} and \match{type=name$d}.
///
/// The matcher
/// \matcher{varDecl(hasInitializer(integerLiteral()))}
/// matches \match{int a = 0}.
AST_MATCHER_P(Expr, ignoringParenCasts, internal::Matcher<Expr>, InnerMatcher) {
  return InnerMatcher.matches(*Node.IgnoreParenCasts(), Finder, Builder);
}

/// Matches expressions that match InnerMatcher after implicit casts and
/// parentheses are stripped off.
///
/// Explicit casts are not discarded.
/// Given
/// \code
///   int arr[5];
///   int a = 0;
///   char b = (0);
///   const int c = a;
///   int *d = (arr);
///   long e = ((long) 0l);
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher
/// \matcher{varDecl(hasInitializer(ignoringParenImpCasts(integerLiteral())))}
/// matches \match{int a = 0} and \match{char b = (0)},
/// but does not match \nomatch{long e = ((long) 0l)} because of the c-style
/// cast.
///
/// The matcher
/// \matcher{varDecl(hasInitializer(ignoringParenImpCasts(declRefExpr())))}
/// matches \match{const int c = a} and \match{int *d = (arr)}.
///
/// The matcher \matcher{varDecl(hasInitializer(integerLiteral()))} matches
/// \match{int a = 0}, but does not match \nomatch{char b = (0)} or
/// \nomatch{long e = ((long) 0l)} because of the casts.
///
/// The matcher \matcher{varDecl(hasInitializer(declRefExpr()))}
/// does not match \nomatch{const int c = a} because of the l- to r-value cast,
/// or \nomatch{int *d = (arr)} because of the array-to-pointer decay.
AST_MATCHER_P(Expr, ignoringParenImpCasts,
              internal::Matcher<Expr>, InnerMatcher) {
  return InnerMatcher.matches(*Node.IgnoreParenImpCasts(), Finder, Builder);
}

/// Matches types that match InnerMatcher after any parens are stripped.
///
/// Given
/// \code
///   void (*fp)(void);
/// \endcode
/// The matcher
/// \matcher{varDecl(hasType(pointerType(pointee(ignoringParens(functionType())))))}
/// matches \match{void (*fp)(void)}.
AST_MATCHER_P_OVERLOAD(QualType, ignoringParens, internal::Matcher<QualType>,
                       InnerMatcher, 0) {
  return InnerMatcher.matches(Node.IgnoreParens(), Finder, Builder);
}

/// Overload \c ignoringParens for \c Expr.
///
/// Given
/// \code
///   const char* str = ("my-string");
/// \endcode
/// The matcher
/// \matcher{implicitCastExpr(hasSourceExpression(ignoringParens(stringLiteral())))}
/// would match the implicit cast resulting from the assignment
/// \match{("my-string")}.
AST_MATCHER_P_OVERLOAD(Expr, ignoringParens, internal::Matcher<Expr>,
                       InnerMatcher, 1) {
  const Expr *E = Node.IgnoreParens();
  return InnerMatcher.matches(*E, Finder, Builder);
}

/// Matches expressions that are instantiation-dependent even if it is
/// neither type- nor value-dependent.
///
/// In the following example, the expression sizeof(sizeof(T() + T()))
/// is instantiation-dependent (since it involves a template parameter T),
/// but is neither type- nor value-dependent, since the type of the inner
/// sizeof is known (std::size_t) and therefore the size of the outer
/// sizeof is known.
/// \code
///   template<typename T>
///   void f(T x, T y) { sizeof(T() + T()); }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{expr(isInstantiationDependent())}
/// matches \match{sizeof(T() + T())},
/// \match{(T() + T())},
/// \match{T() + T()} and two time \match{count=2$T()}.
AST_MATCHER(Expr, isInstantiationDependent) {
  return Node.isInstantiationDependent();
}

/// Matches expressions that are type-dependent because the template type
/// is not yet instantiated.
///
/// For example, the expressions "x" and "x + y" are type-dependent in
/// the following code, but "y" is not type-dependent:
/// \code
///   template<typename T>
///   void add(T x, int y) {
///     x + y;
///   }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{expr(isTypeDependent())}
/// matches \match{x + y} and \match{x}.
AST_MATCHER(Expr, isTypeDependent) { return Node.isTypeDependent(); }

/// Matches expression that are value-dependent because they contain a
/// non-type template parameter.
///
/// For example, the array bound of "Chars" in the following example is
/// value-dependent.
/// \code
///   template<int Size> int f() { return Size; }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{expr(isValueDependent())}
/// matches the return value \match{Size}.
AST_MATCHER(Expr, isValueDependent) { return Node.isValueDependent(); }

/// Matches templateSpecializationType, class template specializations,
/// variable template specializations, and function template specializations
/// where the n'th TemplateArgument matches the given InnerMatcher.
///
/// Given
/// \code
///   template<typename T, typename U> class A {};
///   A<double, int> b;
///   A<int, double> c;
///
///   template<typename T> void f() {}
///   void func() { f<int>(); };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{classTemplateSpecializationDecl(hasTemplateArgument(
///     1, refersToType(asString("int"))))}
/// matches the specialization \match{type=typestr$class A<double, int>}.
///
/// The matcher \matcher{functionDecl(hasTemplateArgument(0,
///                         refersToType(asString("int"))))}
/// matches the specialization of \match{void f() {}}.
AST_POLYMORPHIC_MATCHER_P2(
    hasTemplateArgument,
    AST_POLYMORPHIC_SUPPORTED_TYPES(ClassTemplateSpecializationDecl,
                                    VarTemplateSpecializationDecl, FunctionDecl,
                                    TemplateSpecializationType),
    unsigned, N, internal::Matcher<TemplateArgument>, InnerMatcher) {
  ArrayRef<TemplateArgument> List =
      internal::getTemplateSpecializationArgs(Node);
  if (List.size() <= N)
    return false;
  return InnerMatcher.matches(List[N], Finder, Builder);
}

/// Matches if the number of template arguments equals \p N.
///
/// Given
/// \code
///   template<typename T> struct C {};
///   C<int> c;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{classTemplateSpecializationDecl(templateArgumentCountIs(1))}
/// matches \match{type=typestr$struct C<int>}.
AST_POLYMORPHIC_MATCHER_P(
    templateArgumentCountIs,
    AST_POLYMORPHIC_SUPPORTED_TYPES(ClassTemplateSpecializationDecl,
                                    TemplateSpecializationType),
    unsigned, N) {
  return internal::getTemplateSpecializationArgs(Node).size() == N;
}

/// Matches a TemplateArgument that refers to a certain type.
///
/// Given
/// \code
///   struct X {};
///   template<typename T> struct A {};
///   A<X> a;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{classTemplateSpecializationDecl(hasAnyTemplateArgument(refersToType(
///             recordType(hasDeclaration(recordDecl(hasName("X")))))))}
/// matches the specialization \match{type=typestr$struct A<struct X>}.
AST_MATCHER_P(TemplateArgument, refersToType,
              internal::Matcher<QualType>, InnerMatcher) {
  if (Node.getKind() != TemplateArgument::Type)
    return false;
  return InnerMatcher.matches(Node.getAsType(), Finder, Builder);
}

/// Matches a TemplateArgument that refers to a certain template.
///
/// Given
/// \code
///   template<template <typename> class S> class X {};
///   template<typename T> class Y {};
///   X<Y> xi;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{classTemplateSpecializationDecl(hasAnyTemplateArgument(
///               refersToTemplate(templateName())))}
/// matches the specialization \match{type=typestr$class X<Y>}
AST_MATCHER_P(TemplateArgument, refersToTemplate,
              internal::Matcher<TemplateName>, InnerMatcher) {
  if (Node.getKind() != TemplateArgument::Template)
    return false;
  return InnerMatcher.matches(Node.getAsTemplate(), Finder, Builder);
}

/// Matches a canonical TemplateArgument that refers to a certain
/// declaration.
///
/// Given
/// \code
///   struct B { int next; };
///   template<int(B::*next_ptr)> struct A {};
///   A<&B::next> a;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{classTemplateSpecializationDecl(hasAnyTemplateArgument(
///     refersToDeclaration(fieldDecl(hasName("next")).bind("next"))))}
/// matches the specialization \match{type=typestr$struct A<&B::next>}
/// with \matcher{type=sub$fieldDecl(hasName("next"))} matching
/// \match{sub=next$int next}.
AST_MATCHER_P(TemplateArgument, refersToDeclaration,
              internal::Matcher<Decl>, InnerMatcher) {
  if (Node.getKind() == TemplateArgument::Declaration)
    return InnerMatcher.matches(*Node.getAsDecl(), Finder, Builder);
  return false;
}

/// Matches a sugar TemplateArgument that refers to a certain expression.
///
/// Given
/// \code
///   struct B { int next; };
///   template<int(B::*next_ptr)> struct A {};
///   A<&B::next> a;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{templateSpecializationType(hasAnyTemplateArgument(
///   isExpr(hasDescendant(declRefExpr(to(fieldDecl(hasName("next")).bind("next")))))))}
/// matches the specialization \match{type=typestr$A<&struct B::next>}
/// with \matcher{type=sub$fieldDecl(hasName("next"))} matching
/// \match{sub=next$int next}.
AST_MATCHER_P(TemplateArgument, isExpr, internal::Matcher<Expr>, InnerMatcher) {
  if (Node.getKind() == TemplateArgument::Expression)
    return InnerMatcher.matches(*Node.getAsExpr(), Finder, Builder);
  return false;
}

/// Matches a TemplateArgument that is an integral value.
///
/// Given
/// \code
///   template<int T> struct C {};
///   C<42> c;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{classTemplateSpecializationDecl(
///   hasAnyTemplateArgument(isIntegral()))}
/// matches the implicitly declared specialization
/// \match{type=typestr$struct C<42>} from the instantiation for the type of the
/// variable \c c .
AST_MATCHER(TemplateArgument, isIntegral) {
  return Node.getKind() == TemplateArgument::Integral;
}

/// Matches a TemplateArgument that refers to an integral type.
///
/// Given
/// \code
///   template<int T> struct C {};
///   C<42> c;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{classTemplateSpecializationDecl(
///   hasAnyTemplateArgument(refersToIntegralType(asString("int"))))}
/// matches the implicitly declared specialization
/// \match{type=typestr$struct C<42>} from the instantiation for the type of the
/// variable \c c .
AST_MATCHER_P(TemplateArgument, refersToIntegralType,
              internal::Matcher<QualType>, InnerMatcher) {
  if (Node.getKind() != TemplateArgument::Integral)
    return false;
  return InnerMatcher.matches(Node.getIntegralType(), Finder, Builder);
}

/// Matches a TemplateArgument of integral type with a given value.
///
/// Note that 'Value' is a string as the template argument's value is
/// an arbitrary precision integer. 'Value' must be euqal to the canonical
/// representation of that integral value in base 10.
///
/// Given
/// \code
///   template<int T> struct C {};
///   C<42> c;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{classTemplateSpecializationDecl(
///   hasAnyTemplateArgument(equalsIntegralValue("42")))}
/// matches the implicitly declared specialization
/// \match{type=typestr$struct C<42>} from the instantiation for the type of the
/// variable \c c .
AST_MATCHER_P(TemplateArgument, equalsIntegralValue,
              std::string, Value) {
  if (Node.getKind() != TemplateArgument::Integral)
    return false;
  return toString(Node.getAsIntegral(), 10) == Value;
}

/// Matches an Objective-C autorelease pool statement.
///
/// Given
/// \code
///   @autoreleasepool {
///     int x = 0;
///   }
/// \endcode
/// \compile_args{-ObjC}
/// The matcher \matcher{autoreleasePoolStmt(stmt())} matches the declaration of
/// \match{int x = 0} inside the autorelease pool.
extern const internal::VariadicDynCastAllOfMatcher<Stmt,
       ObjCAutoreleasePoolStmt> autoreleasePoolStmt;

/// Matches any value declaration.
///
/// Given
/// \code
///   enum X { A, B, C };
///   void F();
///   int V = 0;
/// \endcode
/// The matcher \matcher{valueDecl()}
/// matches \match{A}, \match{B}, \match{C}, \match{void F()}
/// and \match{int V = 0}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, ValueDecl> valueDecl;

/// Matches C++ constructor declarations.
///
/// Given
/// \code
///   class Foo {
///    public:
///     Foo();
///     Foo(int);
///     int DoSomething();
///   };
///
///   struct Bar {};
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxConstructorDecl()}
/// matches \match{Foo()} and \match{Foo(int)}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, CXXConstructorDecl>
    cxxConstructorDecl;

/// Matches explicit C++ destructor declarations.
///
/// Given
/// \code
///   class Foo {
///    public:
///     virtual ~Foo();
///   };
///
///   struct Bar {};
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxDestructorDecl()}
/// matches \match{virtual ~Foo()}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, CXXDestructorDecl>
    cxxDestructorDecl;

/// Matches enum declarations.
///
/// Given
/// \code
///   enum X { A, B, C };
/// \endcode
///
/// The matcher \matcher{enumDecl()}
/// matches the enum \match{enum X { A, B, C }}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, EnumDecl> enumDecl;

/// Matches enum constants.
///
/// Given
/// \code
///   enum X {
///     A, B, C
///   };
/// \endcode
/// The matcher \matcher{enumConstantDecl()}
/// matches \match{A}, \match{B} and \match{C}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, EnumConstantDecl>
    enumConstantDecl;

/// Matches tag declarations.
///
/// Given
/// \code
///   class X;
///   template<class T> class Z {};
///   struct S {};
///   union U {};
///   enum E { A, B, C };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
///
/// The matcher \matcher{tagDecl()}
/// matches \match{class X}, \match{class Z {}}, the implicit class
/// declaration \match{class Z}, \match{struct S {}},
/// the implicit class declaration \match{struct S}, \match{union U {}},
/// the implicit class declaration \match{union U}
/// and \match{enum E { A, B, C }}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, TagDecl> tagDecl;

/// Matches method declarations.
///
/// Given
/// \code
///   class X { void y(); };
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxMethodDecl()}
/// matches \match{void y()}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, CXXMethodDecl>
    cxxMethodDecl;

/// Matches conversion operator declarations.
///
/// Given
/// \code
///   class X { operator int() const; };
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxConversionDecl()}
/// matches \match{operator int() const}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, CXXConversionDecl>
    cxxConversionDecl;

/// Matches user-defined and implicitly generated deduction guide.
///
/// Given
/// \code
///   template<typename T>
///   class X { X(int); };
///   X(int) -> X<int>;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++17-or-later}
///
/// The matcher \matcher{cxxDeductionGuideDecl()}
/// matches the written deduction guide
/// \match{type=typestr$auto (int) -> X<int>},
/// the implicit copy deduction guide \match{type=typestr$auto (int) -> X<T>}
/// and the implicitly declared deduction guide
/// \match{type=typestr$auto (X<T>) -> X<T>}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, CXXDeductionGuideDecl>
    cxxDeductionGuideDecl;

/// Matches concept declarations.
///
/// Given
/// \code
///   template<typename T> concept my_concept = true;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++20-or-later}
///
/// The matcher \matcher{conceptDecl()}
/// matches \match{template<typename T>
/// concept my_concept = true}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, ConceptDecl>
    conceptDecl;

/// Matches variable declarations.
///
/// Note: this does not match declarations of member variables, which are
/// "field" declarations in Clang parlance.
///
/// Example matches a
/// \code
///   int a;
///   struct Foo {
///     int x;
///   };
///   void bar(int val);
/// \endcode
///
/// The matcher \matcher{varDecl()}
/// matches \match{int a} and \match{int val}, but not \nomatch{int x}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, VarDecl> varDecl;

/// Matches field declarations.
///
/// Given
/// \code
///   int a;
///   struct Foo {
///     int x;
///   };
///   void bar(int val);
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{fieldDecl()}
/// matches \match{int x}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, FieldDecl> fieldDecl;

/// Matches indirect field declarations.
///
/// Given
/// \code
///   struct X { struct { int a; }; };
/// \endcode
/// The matcher \matcher{indirectFieldDecl()}
/// matches \match{a}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, IndirectFieldDecl>
    indirectFieldDecl;

/// Matches function declarations.
///
/// Given
/// \code
///   void f();
/// \endcode
///
/// The matcher \matcher{functionDecl()}
/// matches \match{void f()}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, FunctionDecl>
    functionDecl;

/// Matches C++ function template declarations.
///
/// Example matches f
/// \code
///   template<class T> void f(T t) {}
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
///
/// The matcher \matcher{functionTemplateDecl()}
/// matches \match{template<class T> void f(T t) {}}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, FunctionTemplateDecl>
    functionTemplateDecl;

/// Matches friend declarations.
///
/// Given
/// \code
///   class X { friend void foo(); };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{friendDecl()}
/// matches \match{friend void foo()}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, FriendDecl> friendDecl;

/// Matches statements.
///
/// Given
/// \code
///   void foo(int a) { { ++a; } }
/// \endcode
/// The matcher \matcher{stmt()}
/// matches the function body itself \match{{ { ++a; } }}, the compound
/// statement \match{{ ++a; }}, the expression \match{++a} and \match{a}.
extern const internal::VariadicAllOfMatcher<Stmt> stmt;

/// Matches declaration statements.
///
/// Given
/// \code
///   void foo() {
///     int a;
///   }
/// \endcode
/// The matcher \matcher{declStmt()}
/// matches \match{int a;}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, DeclStmt> declStmt;

/// Matches member expressions.
///
/// Given
/// \code
///   class Y {
///     void x() { this->x(); x(); Y y; y.x(); a; this->b; Y::b; }
///     int a; static int b;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{memberExpr()}
/// matches \match{this->x}, \match{x}, \match{y.x}, \match{a}, \match{this->b}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, MemberExpr> memberExpr;

/// Matches unresolved member expressions.
///
/// Given
/// \code
///   struct X {
///     template <class T> void f();
///     void g();
///   };
///   template <class T> void h() { X x; x.f<T>(); x.g(); }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{unresolvedMemberExpr()}
/// matches \match{x.f<T>}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, UnresolvedMemberExpr>
    unresolvedMemberExpr;

/// Matches member expressions where the actual member referenced could not be
/// resolved because the base expression or the member name was dependent.
///
/// Given
/// \code
///   template <class T> void f() { T t; t.g(); }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{cxxDependentScopeMemberExpr()}
///   matches \match{t.g}
extern const internal::VariadicDynCastAllOfMatcher<Stmt,
                                                   CXXDependentScopeMemberExpr>
    cxxDependentScopeMemberExpr;

/// Matches call expressions.
///
/// Example matches x.y() and y()
/// \code
///   struct X { void foo(); };
///   void bar();
///   void foobar() {
///     X x;
///     x.foo();
///     bar();
///   }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{callExpr()}
/// matches \match{x.foo()} and \match{bar()};
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CallExpr> callExpr;

/// Matches call expressions which were resolved using ADL.
///
/// Given
/// \code
///   namespace NS {
///     struct X {};
///     void y(X);
///   }
///
///   void y(...);
///
///   void test() {
///     NS::X x;
///     y(x); // Matches
///     NS::y(x); // Doesn't match
///     y(42); // Doesn't match
///     using NS::y;
///     y(x); // Found by both unqualified lookup and ADL, doesn't match
///    }
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{callExpr(usesADL())}
/// matches \match{y(x)}, but not \nomatch{y(42)} or \nomatch{NS::y(x)}.
AST_MATCHER(CallExpr, usesADL) { return Node.usesADL(); }

/// Matches lambda expressions.
///
/// Given
/// \code
///   void f() {
///     []() { return 5; };
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{lambdaExpr()} matches \match{[]() { return 5; }}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, LambdaExpr> lambdaExpr;

/// Matches member call expressions.
///
/// Given
/// \code
///   struct X {
///     void y();
///     void m() { y(); }
///   };
///   void f();
///   void g() {
///     X x;
///     x.y();
///     f();
///   }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxMemberCallExpr()} matches \match{x.y()} and
/// \match{y()}, but not \nomatch{f()}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXMemberCallExpr>
    cxxMemberCallExpr;

/// Matches ObjectiveC Message invocation expressions.
///
/// The innermost message send invokes the "alloc" class method on the
/// NSString class, while the outermost message send invokes the
/// "initWithString" instance method on the object returned from
/// NSString's "alloc". This matcher should match both message sends.
/// \code
///   [[NSString alloc] initWithString:@"Hello"]
/// \endcode
/// \compile_args{-ObjC}
///
/// The matcher \matcher{objcMessageExpr()} matches
/// \match{[[NSString alloc] initWithString:@"Hello"]}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ObjCMessageExpr>
    objcMessageExpr;

/// Matches ObjectiveC String literal expressions.
///
/// Example matches @"abcd"
/// \code
///   NSString *s = @"abcd";
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ObjCStringLiteral>
    objcStringLiteral;

/// Matches Objective-C interface declarations.
///
/// Example matches Foo
/// \code
///   @interface Foo
///   @end
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Decl, ObjCInterfaceDecl>
    objcInterfaceDecl;

/// Matches Objective-C implementation declarations.
///
/// Example matches Foo
/// \code
///   @implementation Foo
///   @end
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Decl, ObjCImplementationDecl>
    objcImplementationDecl;

/// Matches Objective-C protocol declarations.
///
/// Example matches FooDelegate
/// \code
///   @protocol FooDelegate
///   @end
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Decl, ObjCProtocolDecl>
    objcProtocolDecl;

/// Matches Objective-C category declarations.
///
/// Example matches Foo (Additions)
/// \code
///   @interface Foo (Additions)
///   @end
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Decl, ObjCCategoryDecl>
    objcCategoryDecl;

/// Matches Objective-C category definitions.
///
/// Example matches Foo (Additions)
/// \code
///   @implementation Foo (Additions)
///   @end
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Decl, ObjCCategoryImplDecl>
    objcCategoryImplDecl;

/// Matches Objective-C method declarations.
///
/// Example matches both declaration and definition of -[Foo method]
/// \code
///   @interface Foo
///   - (void)method;
///   @end
///
///   @implementation Foo
///   - (void)method {}
///   @end
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Decl, ObjCMethodDecl>
    objcMethodDecl;

/// Matches block declarations.
///
/// Example matches the declaration of the nameless block printing an input
/// integer.
///
/// \code
///   myFunc(^(int p) {
///     printf("%d", p);
///   })
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Decl, BlockDecl>
    blockDecl;

/// Matches Objective-C instance variable declarations.
///
/// Example matches _enabled
/// \code
///   @implementation Foo {
///     BOOL _enabled;
///   }
///   @end
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Decl, ObjCIvarDecl>
    objcIvarDecl;

/// Matches Objective-C property declarations.
///
/// Example matches enabled
/// \code
///   @interface Foo
///   @property BOOL enabled;
///   @end
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Decl, ObjCPropertyDecl>
    objcPropertyDecl;

/// Matches Objective-C \@throw statements.
///
/// Example matches \@throw
/// \code
///   @throw obj;
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ObjCAtThrowStmt>
    objcThrowStmt;

/// Matches Objective-C @try statements.
///
/// Example matches @try
/// \code
///   @try {}
///   @catch (...) {}
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ObjCAtTryStmt>
    objcTryStmt;

/// Matches Objective-C @catch statements.
///
/// Example matches @catch
/// \code
///   @try {}
///   @catch (...) {}
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ObjCAtCatchStmt>
    objcCatchStmt;

/// Matches Objective-C @finally statements.
///
/// Example matches @finally
/// \code
///   @try {}
///   @finally {}
/// \endcode
/// \compile_args{-ObjC}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ObjCAtFinallyStmt>
    objcFinallyStmt;

/// Matches expressions that introduce cleanups to be run at the end
/// of the sub-expression's evaluation.
///
/// Example matches std::string()
/// \code
///   struct A { ~A(); };
///   void f(A);
///   void g(A&);
///   void h() {
///     A a = A{};
///     f(A{});
///     f(a);
///     g(a);
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{exprWithCleanups()} matches \match{A{}},
/// \match{f(A{})} and \match{f(a)},
/// but does not match passing \nomatch{g(a)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ExprWithCleanups>
    exprWithCleanups;

/// Matches init list expressions.
///
/// Given
/// \code
///   int a[] = { 1, 2 };
///   struct B { int x, y; };
///   struct B b = { 5, 6 };
/// \endcode
/// The matcher \matcher{initListExpr()}
/// matches \match{{ 1, 2 }} and \match{{ 5, 6 }}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, InitListExpr>
    initListExpr;

/// Matches the syntactic form of init list expressions
/// (if expression have it).
///
/// Given
/// \code
///   int a[] = { 1, 2 };
///   struct B { int x, y; };
///   struct B b = { 5, 6 };
/// \endcode
/// \compile_args{-std=c}
///
/// The matcher
/// \matcher{initListExpr(hasSyntacticForm(expr().bind("syntactic")))}
/// matches \match{{ 1, 2 }} and \match{{ 5, 6 }}.
AST_MATCHER_P(InitListExpr, hasSyntacticForm, internal::Matcher<Expr>,
              InnerMatcher) {
  const Expr *SyntForm = Node.getSyntacticForm();
  return (SyntForm != nullptr &&
          InnerMatcher.matches(*SyntForm, Finder, Builder));
}

/// Matches C++ initializer list expressions.
///
/// Given
/// \code
///   namespace std {
///     template <typename T>
///     class initializer_list {
///       const T* begin;
///       const T* end;
///     };
///   }
///   template <typename T> class vector {
///     public: vector(std::initializer_list<T>) {}
///   };
///
///   vector<int> a({ 1, 2, 3 });
///   vector<int> b = { 4, 5 };
///   int c[] = { 6, 7 };
///   struct pair { int x; int y; };
///   pair d = { 8, 9 };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++11-or-later,-nostdinc++}
/// The matcher \matcher{cxxStdInitializerListExpr()}
/// matches \match{{ 1, 2, 3 }} and \match{{ 4, 5 }}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt,
                                                   CXXStdInitializerListExpr>
    cxxStdInitializerListExpr;

/// Matches implicit initializers of init list expressions.
///
/// Given
/// \code
///   struct point { double x; double y; };
///   struct point pt = { .x = 42.0 };
/// \endcode
/// The matcher
/// \matcher{initListExpr(has(implicitValueInitExpr().bind("implicit")))}
/// matches \match{{ .x = 42.0 }}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ImplicitValueInitExpr>
    implicitValueInitExpr;

/// Matches paren list expressions.
/// ParenListExprs don't have a predefined type and are used for late parsing.
/// In the final AST, they can be met in template declarations.
///
/// Given
/// \code
///   template<typename T> class X {
///     void f() {
///       X x(*this);
///       int a = 0, b = 1; int i = (a, b);
///     }
///   };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{parenListExpr()}
/// matches \match{(*this)},
/// but does not match \nomatch{(a, b)}
/// because (a, b) has a predefined type and is a ParenExpr, not a
/// ParenListExpr.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ParenListExpr>
    parenListExpr;

/// Matches substitutions of non-type template parameters.
///
/// Given
/// \code
///   template <int N>
///   struct A { static const int n = N; };
///   struct B : public A<42> {};
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{substNonTypeTemplateParmExpr()}
/// matches \match{N} in the right-hand side of "static const int n = N;"
extern const internal::VariadicDynCastAllOfMatcher<Stmt,
                                                   SubstNonTypeTemplateParmExpr>
    substNonTypeTemplateParmExpr;

/// Matches using declarations.
///
/// Given
/// \code
///   namespace X { int x; }
///   using X::x;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{usingDecl()}
///   matches \match{using X::x}
extern const internal::VariadicDynCastAllOfMatcher<Decl, UsingDecl> usingDecl;

/// Matches using-enum declarations.
///
/// Given
/// \code
///   namespace X { enum x { val1, val2 }; }
///   using enum X::x;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{usingEnumDecl()}
///   matches \match{using enum X::x}
extern const internal::VariadicDynCastAllOfMatcher<Decl, UsingEnumDecl>
    usingEnumDecl;

/// Matches using namespace declarations.
///
/// Given
/// \code
///   namespace X { int x; }
///   using namespace X;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{usingDirectiveDecl()}
///   matches \match{using namespace X}
extern const internal::VariadicDynCastAllOfMatcher<Decl, UsingDirectiveDecl>
    usingDirectiveDecl;

/// Matches reference to a name that can be looked up during parsing
/// but could not be resolved to a specific declaration.
///
/// Given
/// \code
///   template<typename T>
///   T foo() { T a; return a; }
///   template<typename T>
///   void bar() {
///     foo<T>();
///   }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{unresolvedLookupExpr()}
/// matches \match{foo<T>}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, UnresolvedLookupExpr>
    unresolvedLookupExpr;

/// Matches unresolved using value declarations.
///
/// Given
/// \code
///   template<typename X>
///   class C : private X {
///     using X::x;
///   };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{unresolvedUsingValueDecl()}
///   matches \match{using X::x}
extern const internal::VariadicDynCastAllOfMatcher<Decl,
                                                   UnresolvedUsingValueDecl>
    unresolvedUsingValueDecl;

/// Matches unresolved using value declarations that involve the
/// typename.
///
/// Given
/// \code
///   template <typename T>
///   struct Base { typedef T Foo; };
///
///   template<typename T>
///   struct S : private Base<T> {
///     using typename Base<T>::Foo;
///   };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{unresolvedUsingTypenameDecl()}
///   matches \match{using typename Base<T>::Foo}
extern const internal::VariadicDynCastAllOfMatcher<Decl,
                                                   UnresolvedUsingTypenameDecl>
    unresolvedUsingTypenameDecl;

/// Matches a constant expression wrapper.
///
/// Given
/// \code
///   void f(int a) {
///     switch (a) {
///       case 37: break;
///     }
///   }
/// \endcode
///
/// The matcher \matcher{constantExpr()} matches \match{37}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ConstantExpr>
    constantExpr;

/// Matches parentheses used in expressions.
///
/// Given
/// \code
///   int foo() { return 1; }
///   int bar() {
///     int a = (foo() + 1);
///   }
/// \endcode
///
/// The matcher \matcher{parenExpr()} matches \match{(foo() + 1)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ParenExpr> parenExpr;

/// Matches constructor call expressions (including implicit ones).
///
/// Given
/// \code
///   struct string {
///     string(const char*);
///     string(const char*s, int n);
///   };
///   void f(const string &a, const string &b);
///   void foo(char *ptr, int n) {
///     f(string(ptr, n), ptr);
///   }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxConstructExpr()} matches \match{string(ptr, n)}
/// and \match{ptr} within arguments of \c f .
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXConstructExpr>
    cxxConstructExpr;

/// Matches unresolved constructor call expressions.
///
/// Given
/// \code
///   template <typename T>
///   void f(const T& t) { return T(t); }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
///
/// The matcher \matcher{cxxUnresolvedConstructExpr()} matches
/// \match{T(t)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt,
                                                   CXXUnresolvedConstructExpr>
    cxxUnresolvedConstructExpr;

/// Matches implicit and explicit this expressions.
///
/// Given
/// \code
///   struct foo {
///     int i;
///     int f() { return i; }
///     int g() { return this->i; }
///   };
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxThisExpr()}
/// matches \match{this} of \c this->i and the implicit \c this expression
/// of \match{i}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXThisExpr>
    cxxThisExpr;

/// Matches nodes where temporaries are created.
///
/// Given
/// \code
///   struct S {
///     S() { }  // User defined constructor makes S non-POD.
///     ~S() { } // User defined destructor makes it non-trivial.
///   };
///   void test() {
///     const S &s_ref = S(); // Requires a CXXBindTemporaryExpr.
///   }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxBindTemporaryExpr()}
/// matches the constructor call \match{S()}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXBindTemporaryExpr>
    cxxBindTemporaryExpr;

/// Matches nodes where temporaries are materialized.
///
/// Example: Given
/// \code
///   struct T {void func();};
///   T f();
///   void g(T);
///   void foo() {
///     T u(f());
///     g(f());
///     f().func();
///     f(); // does not match
///   }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{materializeTemporaryExpr()} matches
/// \match{std=c++14-or-earlier;count=3$f()} three times before C++17 and it
/// matches \match{std=c++17-or-later$f()} one time with C++17 and later for
/// \c f().func() , but it does not match the \nomatch{f()} in the last line in
/// any version.
extern const internal::VariadicDynCastAllOfMatcher<Stmt,
                                                   MaterializeTemporaryExpr>
    materializeTemporaryExpr;

/// Matches new expressions.
///
/// Given
/// \code
///   void* operator new(decltype(sizeof(void*)));
///   struct X {};
///   void foo() {
///     auto* x = new X;
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{cxxNewExpr()}
/// matches \match{new X}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXNewExpr> cxxNewExpr;

/// Matches delete expressions.
///
/// Given
/// \code
///   void* operator new(decltype(sizeof(void*)));
///   void operator delete(void*);
///   struct X {};
///   void foo() {
///     auto* x = new X;
///     delete x;
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{cxxDeleteExpr()}
/// matches \match{delete x}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXDeleteExpr>
    cxxDeleteExpr;

/// Matches noexcept expressions.
///
/// Given
/// \code
///   bool a() noexcept;
///   bool b() noexcept(true);
///   bool c() noexcept(false);
///   bool d() noexcept(noexcept(a()));
///   bool e = noexcept(b()) || noexcept(c());
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{cxxNoexceptExpr()}
/// matches \match{noexcept(a())}, \match{noexcept(b())} and
/// \match{noexcept(c())}, but does not match the noexcept specifier in the
/// declarations a, b, c or d.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXNoexceptExpr>
    cxxNoexceptExpr;

/// Matches a loop initializing the elements of an array in a number of
/// contexts:
///  * in the implicit copy/move constructor for a class with an array member
///  * when a lambda-expression captures an array by value
///  * when a decomposition declaration decomposes an array
///
/// Given
/// \code
///   void testLambdaCapture() {
///     int a[10];
///     [a]() {};
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{arrayInitLoopExpr()} matches the implicit loop that
/// initializes each element of the implicit array field inside the lambda
/// object, that represents the array \match{a} captured by value.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ArrayInitLoopExpr>
    arrayInitLoopExpr;

/// The arrayInitIndexExpr consists of two subexpressions: a common expression
/// (the source array) that is evaluated once up-front, and a per-element
/// initializer that runs once for each array element. Within the per-element
/// initializer, the current index may be obtained via an ArrayInitIndexExpr.
///
/// Given
/// \code
///   void testStructuredBinding() {
///     int a[2] = {1, 2};
///     auto [x, y] = a;
///   }
/// \endcode
/// \compile_args{-std=c++17-or-later}
///
/// The matcher \matcher{type=none$arrayInitIndexExpr()} matches the array index
/// that implicitly iterates over the array `a` to copy each element to the
/// anonymous array that backs the structured binding.
/// \match{}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ArrayInitIndexExpr>
    arrayInitIndexExpr;

/// Matches array subscript expressions.
///
/// Given
/// \code
///   void foo() {
///     int a[2] = {0, 1};
///     int i = a[1];
///   }
/// \endcode
/// The matcher \matcher{arraySubscriptExpr()}
/// matches \match{a[1]}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ArraySubscriptExpr>
    arraySubscriptExpr;

/// Matches the value of a default argument at the call site.
///
/// Given
/// \code
///   void f(int x, int y = 0);
///   void g() {
///     f(42);
///   }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{callExpr(has(cxxDefaultArgExpr()))}
/// matches the \c CXXDefaultArgExpr placeholder inserted for the default value
/// of the second parameter in the call expression \match{f(42)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXDefaultArgExpr>
    cxxDefaultArgExpr;

/// Matches overloaded operator calls.
///
/// Note that if an operator isn't overloaded, it won't match. Instead, use
/// binaryOperator matcher.
/// Currently it does not match operators such as new delete.
/// FIXME: figure out why these do not match?
///
/// Given
/// \code
///   struct ostream;
///   ostream &operator<< (ostream &out, int i) { };
///   void f(ostream& o, int b, int c) {
///     o << b << c;
///   }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxOperatorCallExpr()} matches \match{o << b << c}
/// and \match{o << b}.
/// See also the binaryOperation() matcher for more-general matching of binary
/// uses of this AST node.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXOperatorCallExpr>
    cxxOperatorCallExpr;

/// Matches C++17 fold expressions.
///
/// Given
/// \code
///   template <typename... Args>
///   auto sum(Args... args) {
///       return (0 + ... + args);
///   }
/// \endcode
/// \compile_args{-std=c++17-or-later}
///
/// The matcher \matcher{cxxFoldExpr()} matches \match{(0 + ... + args)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXFoldExpr>
    cxxFoldExpr;

/// Matches rewritten binary operators
///
/// Example matches use of "<":
/// \code
///   struct HasSpaceshipMem {
///     int a;
///     constexpr bool operator==(const HasSpaceshipMem&) const = default;
///   };
///   void compare() {
///     HasSpaceshipMem hs1, hs2;
///     if (hs1 != hs2)
///         return;
///   }
/// \endcode
/// \compile_args{-std=c++20-or-later}
///
/// The matcher \matcher{cxxRewrittenBinaryOperator()} matches
/// \match{hs1 != hs2}.
///
/// See also the binaryOperation() matcher for more-general matching
/// of this AST node.
extern const internal::VariadicDynCastAllOfMatcher<Stmt,
                                                   CXXRewrittenBinaryOperator>
    cxxRewrittenBinaryOperator;

/// Matches expressions.
///
/// Given
/// \code
///   int f(int x, int y) { return x + y; }
/// \endcode
///
/// The matcher \matcher{expr()} matches \match{x + y} once,
/// \match{count=2$x} twice and \match{count=2$y} twice, matching the
/// \c DeclRefExpr , and the \c ImplicitCastExpr that does an l- to r-value
/// cast.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, Expr> expr;

/// Matches expressions that refer to declarations.
///
/// Given
/// \code
///   void f(bool x) {
///     if (x) {}
///   }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
///
/// The matcher \matcher{declRefExpr()} matches \match{x}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, DeclRefExpr>
    declRefExpr;

/// Matches a reference to an ObjCIvar.
///
/// Given
/// \code
/// @implementation A {
///   NSString *a;
/// }
/// - (void) init {
///   a = @"hello";
/// }
/// \endcode
/// \compile_args{-ObjC}
///
/// The matcher \matcher{objcIvarRefExpr()} matches \match{a}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ObjCIvarRefExpr>
    objcIvarRefExpr;

/// Matches a reference to a block.
///
/// Given
/// \code
///   void f() { ^{}(); }
/// \endcode
/// \compile_args{-ObjC}
///
/// The matcher \matcher{blockExpr()} matches \match{^{}}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, BlockExpr> blockExpr;

/// Matches if statements.
///
/// Given
/// \code
///   void foo(int x) {
///     if (x) {}
///   }
/// \endcode
///
/// The matcher \matcher{ifStmt()} matches \match{if (x) {}}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, IfStmt> ifStmt;

/// Matches for statements.
///
/// Given
/// \code
///   void foo() {
///     for (;;) {}
///     int i[] =  {1, 2, 3}; for (auto a : i);
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{forStmt()} matches \match{for (;;) {}},
/// but not \nomatch{for (auto a : i);}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ForStmt> forStmt;

/// Matches the increment statement of a for loop.
///
/// Given
/// \code
/// void foo(int N) {
///     for (int x = 0; x < N; ++x) { }
/// }
/// \endcode
/// The matcher
/// \matcher{forStmt(hasIncrement(unaryOperator(hasOperatorName("++"))))}
/// matches \match{for (int x = 0; x < N; ++x) { }}
AST_MATCHER_P(ForStmt, hasIncrement, internal::Matcher<Stmt>,
              InnerMatcher) {
  const Stmt *const Increment = Node.getInc();
  return (Increment != nullptr &&
          InnerMatcher.matches(*Increment, Finder, Builder));
}

/// Matches the initialization statement of a for loop.
///
/// Given
/// \code
/// void foo(int N) {
///     for (int x = 0; x < N; ++x) { }
/// }
/// \endcode
/// The matcher \matcher{forStmt(hasLoopInit(declStmt()))}
/// matches \match{for (int x = 0; x < N; ++x) { }}
AST_MATCHER_P(ForStmt, hasLoopInit, internal::Matcher<Stmt>,
              InnerMatcher) {
  const Stmt *const Init = Node.getInit();
  return (Init != nullptr && InnerMatcher.matches(*Init, Finder, Builder));
}

/// Matches range-based for statements.
///
/// Given
/// \code
///   void foo() {
///     int i[] =  {1, 2, 3}; for (auto a : i);
///     for(int j = 0; j < 5; ++j);
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{cxxForRangeStmt()}
/// matches \match{for (auto a : i);}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXForRangeStmt>
    cxxForRangeStmt;

/// Matches the initialization statement of a for loop.
///
/// Given
/// \code
///   void foo() {
///     int a[42] = {};
///     for (int x : a) { }
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{cxxForRangeStmt(hasLoopVariable(anything()))}
/// matches \match{for (int x : a) { }}
AST_MATCHER_P(CXXForRangeStmt, hasLoopVariable, internal::Matcher<VarDecl>,
              InnerMatcher) {
  const VarDecl *const Var = Node.getLoopVariable();
  return (Var != nullptr && InnerMatcher.matches(*Var, Finder, Builder));
}

/// Matches the range initialization statement of a for loop.
///
/// Given
/// \code
///   void foo() {
///     int a[42] = {};
///     for (int x : a) { }
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{cxxForRangeStmt(hasRangeInit(anything()))}
/// matches \match{for (int x : a) { }}
AST_MATCHER_P(CXXForRangeStmt, hasRangeInit, internal::Matcher<Expr>,
              InnerMatcher) {
  const Expr *const Init = Node.getRangeInit();
  return (Init != nullptr && InnerMatcher.matches(*Init, Finder, Builder));
}

/// Matches while statements.
///
/// Given
/// \code
/// void foo() {
///   while (true) {}
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{whileStmt()}
/// matches \match{while (true) {}}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, WhileStmt> whileStmt;

/// Matches do statements.
///
/// Given
/// \code
/// void foo() {
///   do {} while (true);
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{doStmt()}
/// matches \match{do {} while (true)}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, DoStmt> doStmt;

/// Matches break statements.
///
/// Given
/// \code
/// void foo() {
///   while (true) { break; }
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{breakStmt()}
/// matches \match{break}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, BreakStmt> breakStmt;

/// Matches continue statements.
///
/// Given
/// \code
/// void foo() {
///   while (true) { continue; }
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{continueStmt()}
/// matches \match{continue}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ContinueStmt>
    continueStmt;

/// Matches co_return statements.
///
/// Given
/// \code
///   namespace std {
///   template <typename T = void>
///   struct coroutine_handle {
///       static constexpr coroutine_handle from_address(void* addr) {
///         return {};
///       }
///   };
///
///   struct always_suspend {
///       bool await_ready() const noexcept;
///       bool await_resume() const noexcept;
///       template <typename T>
///       bool await_suspend(coroutine_handle<T>) const noexcept;
///   };
///
///   template <typename T>
///   struct coroutine_traits {
///       using promise_type = T::promise_type;
///   };
///   }  // namespace std
///
///   struct generator {
///       struct promise_type {
///           void return_value(int v);
///           std::always_suspend yield_value(int&&);
///           std::always_suspend initial_suspend() const noexcept;
///           std::always_suspend final_suspend() const noexcept;
///           void unhandled_exception();
///           generator get_return_object();
///       };
///   };
///
///   generator f() {
///       co_return 10;
///   }
///
/// \endcode
/// \compile_args{-std=c++20-or-later}
/// The matcher \matcher{coreturnStmt(has(integerLiteral()))}
/// matches \match{co_return 10}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CoreturnStmt>
    coreturnStmt;

/// Matches return statements.
///
/// Given
/// \code
/// int foo() {
///   return 1;
/// }
/// \endcode
/// The matcher \matcher{returnStmt()}
/// matches \match{return 1}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ReturnStmt> returnStmt;

/// Matches goto statements.
///
/// Given
/// \code
/// void bar();
/// void foo() {
///   goto FOO;
///   FOO: bar();
/// }
/// \endcode
/// The matcher \matcher{gotoStmt()}
/// matches \match{goto FOO}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, GotoStmt> gotoStmt;

/// Matches label statements.
///
/// Given
/// \code
/// void bar();
/// void foo() {
///   goto FOO;
///   FOO: bar();
/// }
/// \endcode
/// The matcher \matcher{labelStmt()}
/// matches \match{FOO: bar()}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, LabelStmt> labelStmt;

/// Matches address of label statements (GNU extension).
///
/// Given
/// \code
/// void bar();
/// void foo() {
///   FOO: bar();
///   void *ptr = &&FOO;
///   goto *ptr;
/// }
/// \endcode
/// The matcher \matcher{addrLabelExpr()}
/// matches \match{&&FOO}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, AddrLabelExpr>
    addrLabelExpr;

/// Matches switch statements.
///
/// Given
/// \code
/// void foo(int a) {
///   switch(a) { case 42: break; default: break; }
/// }
/// \endcode
/// The matcher \matcher{switchStmt()}
/// matches \match{switch(a) { case 42: break; default: break; }}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, SwitchStmt> switchStmt;

/// Matches case and default statements inside switch statements.
///
/// Given
/// \code
/// void foo(int a) {
///   switch(a) { case 42: break; default: break; }
/// }
/// \endcode
/// The matcher \matcher{switchCase()}
/// matches \match{case 42: break} and \match{default: break}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, SwitchCase> switchCase;

/// Matches case statements inside switch statements.
///
/// Given
/// \code
/// void foo(int a) {
///   switch(a) { case 42: break; default: break; }
/// }
/// \endcode
/// The matcher \matcher{caseStmt()}
/// matches \match{case 42: break}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CaseStmt> caseStmt;

/// Matches default statements inside switch statements.
///
/// Given
/// \code
/// void foo(int a) {
///   switch(a) { case 42: break; default: break; }
/// }
/// \endcode
/// The matcher \matcher{defaultStmt()}
/// matches \match{default: break}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, DefaultStmt>
    defaultStmt;

/// Matches compound statements.
///
/// Given
/// \code
/// void foo() { for (;;) {{}} }
/// \endcode
///
/// The matcher \matcher{compoundStmt()} matches
/// \match{{ for (;;) {{}} }}, \match{{{}}} and \match{{}}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CompoundStmt>
    compoundStmt;

/// Matches catch statements.
///
/// \code
/// void foo() {
///   try {} catch(int i) {}
/// }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxCatchStmt()}
/// matches \match{catch(int i) {}}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXCatchStmt>
    cxxCatchStmt;

/// Matches try statements.
///
/// \code
/// void foo() {
///   try {} catch(int i) {}
/// }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxTryStmt()}
/// matches \match{try {} catch(int i) {}}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXTryStmt> cxxTryStmt;

/// Matches throw expressions.
///
/// \code
/// void foo() {
///   try { throw 5; } catch(int i) {}
/// }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxThrowExpr()}
/// matches \match{throw 5}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXThrowExpr>
    cxxThrowExpr;

/// Matches null statements.
///
/// \code
/// void foo() {
///   foo();;
/// }
/// \endcode
/// The matcher \matcher{nullStmt()}
/// matches the second \match{;}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, NullStmt> nullStmt;

/// Matches asm statements.
///
/// \code
/// void foo() {
///  int i = 100;
///   __asm("mov %al, 2");
/// }
/// \endcode
/// The matcher \matcher{asmStmt()}
/// matches \match{__asm("mov %al, 2")}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, AsmStmt> asmStmt;

/// Matches bool literals.
///
/// Example matches true
/// \code
///   bool Flag = true;
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
///
/// The matcher \matcher{cxxBoolLiteral()} matches \match{true}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXBoolLiteralExpr>
    cxxBoolLiteral;

/// Matches string literals (also matches wide string literals).
///
/// Given
/// \code
///   char *s = "abcd";
///   wchar_t *ws = L"abcd";
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{stringLiteral()} matches \match{"abcd"} and
/// \match{L"abcd"}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, StringLiteral>
    stringLiteral;

/// Matches character literals (also matches wchar_t).
///
/// Not matching Hex-encoded chars (e.g. 0x1234, which is a IntegerLiteral),
/// though.
///
/// Given
/// \code
///   char ch = 'a';
///   wchar_t chw = L'a';
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{characterLiteral()} matches \match{'a'} and
/// \match{L'a'}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CharacterLiteral>
    characterLiteral;

/// Matches integer literals of all sizes / encodings, e.g.
/// 1, 1L, 0x1 and 1U.
///
/// Does not match character-encoded integers such as L'a'.
///
/// Given
/// \code
///   int a = 1;
///   int b = 1L;
///   int c = 0x1;
///   int d = 1U;
///   int e = 1.0;
/// \endcode
///
/// The matcher \matcher{integerLiteral()} matches
/// \match{1}, \match{1L}, \match{0x1} and \match{1U}, but does not match
/// \nomatch{1.0}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, IntegerLiteral>
    integerLiteral;

/// Matches float literals of all sizes / encodings, e.g.
/// 1.0, 1.0f, 1.0L and 1e10.
///
/// Given
/// \code
///   int a = 1.0;
///   int b = 1.0F;
///   int c = 1.0L;
///   int d = 1e10;
///   int e = 1;
/// \endcode
///
/// The matcher \matcher{floatLiteral()} matches
/// \match{1.0}, \match{1.0F}, \match{1.0L} and \match{1e10}, but does not match
/// \nomatch{1}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, FloatingLiteral>
    floatLiteral;

/// Matches imaginary literals, which are based on integer and floating
/// point literals e.g.: 1i, 1.0i
///
/// Given
/// \code
///   auto a = 1i;
///   auto b = 1.0i;
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{imaginaryLiteral()} matches \match{1i} and
/// \match{1.0i}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ImaginaryLiteral>
    imaginaryLiteral;

/// Matches fixed point literals
///
/// Given
/// \code
///   void f() {
///     0.0k;
///   }
/// \endcode
/// \compile_args{-ffixed-point}
///
/// The matcher \matcher{type=none$fixedPointLiteral()} matches \match{0.0k}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, FixedPointLiteral>
    fixedPointLiteral;

/// Matches user defined literal operator call.
///
/// Example match: "foo"_suffix
/// Given
/// \code
///   float operator ""_foo(long double);
///   float a = 1234.5_foo;
/// \endcode
/// \compile_args{-std=c++11}
///
/// The matcher \matcher{userDefinedLiteral()} matches \match{1234.5_foo}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, UserDefinedLiteral>
    userDefinedLiteral;

/// Matches compound (i.e. non-scalar) literals
///
/// Example match: {1}, (1, 2)
/// \code
///   struct vector { int x; int y; };
///   struct vector myvec = (struct vector){ 1, 2 };
/// \endcode
///
/// The matcher \matcher{compoundLiteralExpr()}
/// matches \match{(struct vector){ 1, 2 }}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CompoundLiteralExpr>
    compoundLiteralExpr;

/// Matches co_await expressions.
///
/// Given
/// \code
///   namespace std {
///   template <typename T = void>
///   struct coroutine_handle {
///       static constexpr coroutine_handle from_address(void* addr) {
///         return {};
///       }
///   };
///
///   struct always_suspend {
///       bool await_ready() const noexcept;
///       bool await_resume() const noexcept;
///       template <typename T>
///       bool await_suspend(coroutine_handle<T>) const noexcept;
///   };
///
///   template <typename T>
///   struct coroutine_traits {
///       using promise_type = T::promise_type;
///   };
///   }  // namespace std
///
///   struct generator {
///       struct promise_type {
///           std::always_suspend yield_value(int&&);
///           std::always_suspend initial_suspend() const noexcept;
///           std::always_suspend final_suspend() const noexcept;
///           void return_void();
///           void unhandled_exception();
///           generator get_return_object();
///       };
///   };
///
///   std::always_suspend h();
///
///   generator g() { co_await h(); }
/// \endcode
/// \compile_args{-std=c++20-or-later}
/// The matcher
/// \matcher{coawaitExpr(has(callExpr(callee(functionDecl(hasName("h"))))))}
/// matches \match{co_await h()}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CoawaitExpr>
    coawaitExpr;

/// Matches co_await expressions where the type of the promise is dependent
///
/// Given
/// \code
///   namespace std {
///   template <typename T = void>
///   struct coroutine_handle {
///       static constexpr coroutine_handle from_address(void* addr) {
///         return {};
///       }
///   };
///
///   struct always_suspend {
///       bool await_ready() const noexcept;
///       bool await_resume() const noexcept;
///       template <typename T>
///       bool await_suspend(coroutine_handle<T>) const noexcept;
///   };
///
///   template <typename T>
///   struct coroutine_traits {
///       using promise_type = T::promise_type;
///   };
///   }  // namespace std
///
///   template <typename T>
///   struct generator {
///       struct promise_type {
///           std::always_suspend yield_value(int&&);
///           std::always_suspend initial_suspend() const noexcept;
///           std::always_suspend final_suspend() const noexcept;
///           void return_void();
///           void unhandled_exception();
///           generator get_return_object();
///       };
///   };
///
///   template <typename T>
///   std::always_suspend h();
///
///   template <>
///   std::always_suspend h<void>();
///
///   template<typename T>
///   generator<T> g() { co_await h<T>(); }
/// \endcode
/// \compile_args{-std=c++20-or-later}
/// The matcher \matcher{dependentCoawaitExpr()}
/// matches \match{co_await h<T>()}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, DependentCoawaitExpr>
    dependentCoawaitExpr;

/// Matches co_yield expressions.
///
/// Given
/// \code
///   namespace std {
///   template <typename T = void>
///   struct coroutine_handle {
///       static constexpr coroutine_handle from_address(void* addr) {
///         return {};
///       }
///   };
///
///   struct always_suspend {
///       bool await_ready() const noexcept;
///       bool await_resume() const noexcept;
///       template <typename T>
///       bool await_suspend(coroutine_handle<T>) const noexcept;
///   };
///
///   template <typename T>
///   struct coroutine_traits {
///       using promise_type = T::promise_type;
///   };
///   }  // namespace std
///
///   struct generator {
///       struct promise_type {
///           std::always_suspend yield_value(int&&);
///           std::always_suspend initial_suspend() const noexcept;
///           std::always_suspend final_suspend() const noexcept;
///           void return_void();
///           void unhandled_exception();
///           generator get_return_object();
///       };
///   };
///
///   generator f() {
///       while (true) {
///           co_yield 10;
///       }
///   }
/// \endcode
/// \compile_args{-std=c++20-or-later}
/// The matcher \matcher{coyieldExpr()}
/// matches \match{co_yield 10}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CoyieldExpr>
    coyieldExpr;

/// Matches coroutine body statements.
///
/// Given
/// \code
///   namespace std {
///   template <typename T = void>
///   struct coroutine_handle {
///       static constexpr coroutine_handle from_address(void* addr) {
///         return {};
///       }
///   };
///
///   struct suspend_always {
///       bool await_ready() const noexcept;
///       bool await_resume() const noexcept;
///       template <typename T>
///       bool await_suspend(coroutine_handle<T>) const noexcept;
///   };
///
///   template <typename...>
///   struct coroutine_traits {
///       struct promise_type {
///           std::suspend_always initial_suspend() const noexcept;
///           std::suspend_always final_suspend() const noexcept;
///           void return_void();
///           void unhandled_exception();
///           coroutine_traits get_return_object();
///       };
///   };
///   }  // namespace std
///
///   void f() { while (true) { co_return; } }
///
/// \endcode
/// \compile_args{-std=c++20-or-later}
///
/// The matcher \matcher{coroutineBodyStmt()} matches
/// \match{{ while (true) { co_return; } }}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CoroutineBodyStmt>
    coroutineBodyStmt;

/// Matches nullptr literal.
///
/// Given
/// \code
///   int a = 0;
///   int* b = 0;
///   int *c = nullptr;
/// \endcode
/// \compile_args{-std=c++11,c23-or-later}
///
/// The matcher \matcher{cxxNullPtrLiteralExpr()} matches \match{nullptr}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXNullPtrLiteralExpr>
    cxxNullPtrLiteralExpr;

/// Matches GNU __builtin_choose_expr.
///
/// Given
/// \code
///   void f() { (void)__builtin_choose_expr(1, 2, 3); }
/// \endcode
///
/// The matcher \matcher{chooseExpr()} matches
/// \match{__builtin_choose_expr(1, 2, 3)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ChooseExpr> chooseExpr;

/// Matches builtin function __builtin_convertvector.
///
/// Given
/// \code
///   typedef double vector4double __attribute__((__vector_size__(32)));
///   typedef float  vector4float  __attribute__((__vector_size__(16)));
///   vector4float vf;
///   void f() { (void)__builtin_convertvector(vf, vector4double); }
/// \endcode
///
/// The matcher \matcher{convertVectorExpr()} matches
/// \match{__builtin_convertvector(vf, vector4double)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ConvertVectorExpr>
    convertVectorExpr;

/// Matches GNU __null expression.
///
/// Given
/// \code
///   auto val = __null;
/// \endcode
/// \compile_args{-std=c++11}
///
/// The matcher \matcher{gnuNullExpr()} matches \match{__null}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, GNUNullExpr>
    gnuNullExpr;

/// Matches C11 _Generic expression.
///
/// Given
/// \code
///   double fdouble(double);
///   float ffloat(float);
///   #define GENERIC_MACRO(X) _Generic((X), double: fdouble, float: ffloat)(X)
///
///   void f() {
///       GENERIC_MACRO(0.0);
///       GENERIC_MACRO(0.0F);
///   }
/// \endcode
/// \compile_args{-std=c}
///
/// The matcher \matcher{type=none$genericSelectionExpr()} matches
/// the generic selection expression that is expanded in
/// \match{GENERIC_MACRO(0.0)} and \match{GENERIC_MACRO(0.0F)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, GenericSelectionExpr>
    genericSelectionExpr;

/// Matches atomic builtins.
///
/// Given
/// \code
///   void foo() { int *ptr; __atomic_load_n(ptr, 1); }
/// \endcode
///
/// The matcher \matcher{atomicExpr()} matches \match{__atomic_load_n(ptr, 1)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, AtomicExpr> atomicExpr;

/// Matches statement expression (GNU extension).
///
/// Given
/// \code
///   void f() {
///     int C = ({ int X = 4; X; });
///   }
/// \endcode
///
/// The matcher \matcher{stmtExpr()} matches \match{({ int X = 4; X; })}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, StmtExpr> stmtExpr;

/// Matches binary operator expressions.
///
/// Given
/// \code
///   void foo(bool a, bool b) {
///     !(a || b);
///   }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
///
/// The matcher \matcher{binaryOperator()} matches \match{a || b}.
///
/// See also the binaryOperation() matcher for more-general matching.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, BinaryOperator>
    binaryOperator;

/// Matches unary operator expressions.
///
/// Example matches !a
/// \code
///   void foo(bool a, bool b) {
///     !a || b;
///   }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
///
/// The matcher \matcher{unaryOperator()} matches \match{!a}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, UnaryOperator>
    unaryOperator;

/// Matches conditional operator expressions.
///
/// Given
/// \code
///   int f(int a, int b, int c) {
///     return (a ? b : c) + 42;
///   }
/// \endcode
///
/// The matcher \matcher{conditionalOperator()} matches \match{a ? b : c}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ConditionalOperator>
    conditionalOperator;

/// Matches binary conditional operator expressions (GNU extension).
///
/// Given
/// \code
///   int f(int a, int b) {
///     return (a ?: b) + 42;
///   }
/// \endcode
///
/// The matcher \matcher{binaryConditionalOperator()} matches \match{a ?: b}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt,
                                                   BinaryConditionalOperator>
    binaryConditionalOperator;

/// Matches opaque value expressions. They are used as helpers
/// to reference another expressions and can be met
/// in BinaryConditionalOperators, for example.
///
/// Given
/// \code
///   int f(int a, int b) {
///     return (a ?: b) + 42;
///   }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{opaqueValueExpr()} matches \match{count=2$a} twice,
/// once for the check and once for the expression of the true path.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, OpaqueValueExpr>
    opaqueValueExpr;

/// Matches a C++ static_assert declaration.
///
/// Given
/// \code
///   struct S {
///     int x;
///   };
///   static_assert(sizeof(S) == sizeof(int));
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{staticAssertDecl()}
/// matches \match{static_assert(sizeof(S) == sizeof(int))}.
extern const internal::VariadicDynCastAllOfMatcher<Decl, StaticAssertDecl>
    staticAssertDecl;

/// Matches a reinterpret_cast expression.
///
/// Either the source expression or the destination type can be matched
/// using has(), but hasDestinationType() is more specific and can be
/// more readable.
///
/// Example matches reinterpret_cast<char*>(&p) in
/// \code
///   void* p = reinterpret_cast<char*>(&p);
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxReinterpretCastExpr()}
/// matches \match{reinterpret_cast<char*>(&p)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXReinterpretCastExpr>
    cxxReinterpretCastExpr;

/// Matches a C++ static_cast expression.
///
/// \see hasDestinationType
/// \see reinterpretCast
///
/// Given
/// \code
///   long eight(static_cast<long>(8));
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxStaticCastExpr()}
/// matches \match{static_cast<long>(8)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXStaticCastExpr>
    cxxStaticCastExpr;

/// Matches a dynamic_cast expression.
///
/// Given
/// \code
///   struct B { virtual ~B() {} }; struct D : B {};
///   B b;
///   D* p = dynamic_cast<D*>(&b);
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxDynamicCastExpr()}
/// matches \match{dynamic_cast<D*>(&b)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXDynamicCastExpr>
    cxxDynamicCastExpr;

/// Matches a const_cast expression.
///
/// Given
/// \code
///   int n = 42;
///   const int &r(n);
///   int* p = const_cast<int*>(&r);
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxConstCastExpr()}
/// matches \match{const_cast<int*>(&r)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXConstCastExpr>
    cxxConstCastExpr;

/// Matches a C-style cast expression.
///
/// Given
/// \code
///   int i = (int) 2.2f;
/// \endcode
///
/// The matcher \matcher{cStyleCastExpr()}
/// matches \match{(int) 2.2f}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CStyleCastExpr>
    cStyleCastExpr;

/// Matches explicit cast expressions.
///
/// Matches any cast expression written in user code, whether it be a
/// C-style cast, a functional-style cast, or a keyword cast.
///
/// Does not match implicit conversions.
///
/// Note: the name "explicitCast" is chosen to match Clang's terminology, as
/// Clang uses the term "cast" to apply to implicit conversions as well as to
/// actual cast expressions.
///
/// \see hasDestinationType.
///
/// \code
///   struct S {};
///   const S* s;
///   S* s2 = const_cast<S*>(s);
///
///   const int val = 0;
///   char val0 = val;
///   char val1 = (char)val;
///   char val2 = static_cast<char>(val);
///   int* val3 = reinterpret_cast<int*>(val);
///   char val4 = char(val);
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{explicitCastExpr()}
/// matches \match{(char)val}, \match{static_cast<char>(val)},
/// \match{reinterpret_cast<int*>(val)}, \match{const_cast<S*>(s)}
/// and \match{char(val)}, but not the initialization of \c val0 with
/// \nomatch{val}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ExplicitCastExpr>
    explicitCastExpr;

/// Matches the implicit cast nodes of Clang's AST.
///
/// This matches many different places, including function call return value
/// eliding, as well as any type conversions.
///
/// \code
/// void f(int);
/// void g(int val1, int val2) {
///   unsigned int a = val1;
///   f(val2);
/// }
/// \endcode
///
/// The matcher \matcher{implicitCastExpr()}
/// matches \match{count=2$val1} for the implicit cast from an l- to an r-value
/// and for the cast to \c{unsigned int}, \match{f} for the function pointer
/// decay, and \match{val2} for the cast from an l- to an r-value.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, ImplicitCastExpr>
    implicitCastExpr;

/// Matches any cast nodes of Clang's AST.
///
/// Given
/// \code
///   struct S {};
///   const S* s;
///   S* s2 = const_cast<S*>(s);
///
///   const int val = 0;
///   char val0 = 1;
///   char val1 = (char)2;
///   char val2 = static_cast<char>(3);
///   int* val3 = reinterpret_cast<int*>(4);
///   char val4 = char(5);
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{castExpr()}
/// matches
/// \match{const_cast<S*>(s)} and the implicit l- to r-value cast for \match{s},
/// the implicit cast to \c char for the initializer \match{1},
/// the c-style cast \match{(char)2} and it's implicit cast to \c char
/// (part of the c-style cast) \match{2},
/// \match{static_cast<char>(3)} and it's implicit cast to \c char
/// (part of the \c static_cast) \match{3},
/// \match{reinterpret_cast<int*>(4)},
/// \match{char(5)} and it's implicit cast to \c char
/// (part of the functional cast) \match{5}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CastExpr> castExpr;

/// Matches functional cast expressions
///
/// Given
/// \code
///   struct Foo {
///     Foo(int x);
///   };
///
///   void foo(int bar) {
///     Foo f = bar;
///     Foo g = (Foo) bar;
///     Foo h = Foo(bar);
///   }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxFunctionalCastExpr()}
/// matches \match{Foo(bar)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXFunctionalCastExpr>
    cxxFunctionalCastExpr;

/// Matches functional cast expressions having N != 1 arguments
///
/// Given
/// \code
///   struct Foo {
///     Foo(int x, int y);
///   };
///
///   void foo(int bar) {
///     Foo h = Foo(bar, bar);
///   }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxTemporaryObjectExpr()}
/// matches \match{Foo(bar, bar)}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CXXTemporaryObjectExpr>
    cxxTemporaryObjectExpr;

/// Matches predefined identifier expressions [C99 6.4.2.2].
///
/// Example: Matches __func__
/// \code
///   void f() {
///     const char* func_name = __func__;
///   }
/// \endcode
///
/// The matcher \matcher{predefinedExpr()}
/// matches \match{__func__}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, PredefinedExpr>
    predefinedExpr;

/// Matches C99 designated initializer expressions [C99 6.7.8].
///
/// Example: Given
/// \code
///   struct point2 { double x; double y; };
///   struct point2 ptarray[10] = { [0].x = 1.0 };
///   struct point2 pt = { .x = 2.0 };
/// \endcode
///
/// The matcher \matcher{designatedInitExpr()}
/// matches \match{[0].x = 1.0} and \match{.x = 2.0}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, DesignatedInitExpr>
    designatedInitExpr;

/// Matches designated initializer expressions that contain
/// a specific number of designators.
///
/// Example: Given
/// \code
///   struct point2 { double x; double y; };
///   struct point2 ptarray[10] = { [0].x = 1.0 };
///   struct point2 pt = { .x = 2.0 };
/// \endcode
///
/// The matcher \matcher{designatedInitExpr(designatorCountIs(2))}
/// matches \match{[0].x = 1.0}, but not \nomatch{.x = 2.0}.
AST_MATCHER_P(DesignatedInitExpr, designatorCountIs, unsigned, N) {
  return Node.size() == N;
}

/// Matches \c QualTypes in the clang AST.
///
/// Given
/// \code
///   int a = 0;
///   const int b = 1;
/// \endcode
///
/// The matcher \matcher{varDecl(hasType(qualType(isConstQualified())))}
/// matches \match{const int b = 1}, but not \nomatch{int a = 0}.
extern const internal::VariadicAllOfMatcher<QualType> qualType;

/// Matches \c Types in the clang AST.
///
/// Given
/// \code
///   const int b = 1;
/// \endcode
///
/// The matcher \matcher{varDecl(hasType(type().bind("type")))}
/// matches \match{const int b = 1}, with \matcher{type=sub$type()}
/// matching \match{sub=type$int}.
extern const internal::VariadicAllOfMatcher<Type> type;

/// Matches \c TypeLocs in the clang AST.
///
/// That is, information about a type and where it was written.
///
/// \code
///   void foo(int val);
/// \endcode
///
/// The matcher \matcher{declaratorDecl(hasTypeLoc(typeLoc().bind("type")))}
/// matches \match{void foo(int val)} and \match{int val}, with
/// \matcher{type=sub$typeLoc()} matching \match{sub=type$void} and
/// \match{sub=type$int} respectively.
extern const internal::VariadicAllOfMatcher<TypeLoc> typeLoc;

/// Matches if any of the given matchers matches.
///
/// Unlike \c anyOf, \c eachOf will generate a match result for each
/// matching submatcher.
///
/// Given
/// \code
///   void f(int a, int b);
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{functionDecl(hasAnyParameter(
/// eachOf(parmVarDecl(hasName("a")).bind("v"),
///        parmVarDecl(hasName("b")).bind("v"))))}
/// matches \match{void f(int a, int b)},
/// with \matcher{type=sub$parmVarDecl(hasName("a"))} matching \match{sub=v$a}
/// for one match,
/// and with \matcher{type=sub$parmVarDecl(hasName("b"))} matching
/// \match{sub=v$b} for the other match.
///
/// Usable as: Any Matcher
extern const internal::VariadicOperatorMatcherFunc<
    2, std::numeric_limits<unsigned>::max()>
    eachOf;

/// Matches if any of the given matchers matches.
///
/// Usable as: Any Matcher
///
/// \code
///   char v0 = 'a';
///   int v1 = 1;
///   float v2 = 2.0;
/// \endcode
///
/// The matcher \matcher{varDecl(anyOf(hasName("v0"), hasType(isInteger())))}
/// matches \match{char v0 = 'a'} and \match{int v1 = 1}.
extern const internal::VariadicOperatorMatcherFunc<
    2, std::numeric_limits<unsigned>::max()>
    anyOf;

/// Matches if all given matchers match.
///
/// Usable as: Any Matcher
///
/// \code
///   int v0 = 0;
///   int v1 = 1;
/// \endcode
///
/// The matcher \matcher{varDecl(allOf(hasName("v0"), hasType(isInteger())))}
/// matches \match{int v0 = 0}.
extern const internal::VariadicOperatorMatcherFunc<
    2, std::numeric_limits<unsigned>::max()>
    allOf;

/// Matches any node regardless of the submatcher.
///
/// However, \c optionally will retain any bindings generated by the submatcher.
/// Useful when additional information which may or may not present about a main
/// matching node is desired.
///
/// Given
/// \code
///   int a = 0;
///   int b;
/// \endcode
///
/// The matcher \matcher{varDecl(optionally(hasInitializer(expr())))}
/// matches \match{int a = 0} and \match{int b}.
///
/// Usable as: Any Matcher
extern const internal::VariadicOperatorMatcherFunc<1, 1> optionally;

/// Matches sizeof (C99), alignof (C++11) and vec_step (OpenCL)
///
/// Given
/// \code
///   int x = 42;
///   int y = sizeof(x) + alignof(x);
/// \endcode
/// \compile_args{-std=c++11-or-later,c23-or-later}
/// The matcher \matcher{unaryExprOrTypeTraitExpr()}
/// matches \match{sizeof(x)} and \match{alignof(x)}
extern const internal::VariadicDynCastAllOfMatcher<Stmt,
                                                   UnaryExprOrTypeTraitExpr>
    unaryExprOrTypeTraitExpr;

/// Matches any of the \p NodeMatchers with InnerMatchers nested within
///
/// Given
/// \code
///   void f() {
///     if (true);
///     for (; true; );
///   }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
///
/// The matcher \matcher{stmt(mapAnyOf(ifStmt, forStmt).with(
///     hasCondition(cxxBoolLiteral(equals(true)))
///     ))},
/// which is equivalent to
/// \matcher{stmt(anyOf(
///     ifStmt(hasCondition(cxxBoolLiteral(equals(true)))).bind("trueCond"),
///     forStmt(hasCondition(cxxBoolLiteral(equals(true)))).bind("trueCond")
///     ))},
/// matches \match{if (true);} and \match{for (; true; );}.
///
/// The with() chain-call accepts zero or more matchers which are combined
/// as-if with allOf() in each of the node matchers.
///
/// Usable as: Any Matcher
template <typename T, typename... U>
auto mapAnyOf(internal::VariadicDynCastAllOfMatcher<T, U> const &...) {
  return internal::MapAnyOfHelper<U...>();
}

/// Matches nodes which can be used with binary operators.
///
/// A comparison of two expressions might be represented in the clang AST as a
/// \c binaryOperator, a \c cxxOperatorCallExpr or a
/// \c cxxRewrittenBinaryOperator, depending on
///
/// * whether the types of var1 and var2 are fundamental (binaryOperator) or at
///   least one is a class type (\c cxxOperatorCallExpr)
/// * whether the code appears in a template declaration, if at least one of the
///   vars is a dependent-type (\c binaryOperator)
/// * whether the code relies on a rewritten binary operator, such as a
/// spaceship operator or an inverted equality operator
/// (\c cxxRewrittenBinaryOperator)
///
/// This matcher elides details in places where the matchers for the nodes are
/// compatible.
///
/// Given
/// \code
///   struct S{
///       bool operator!=(const S&) const;
///   };
///
///   void foo()
///   {
///      1 != 2;
///      S() != S();
///   }
///
///   template<typename T>
///   void templ()
///   {
///      3 != 4;
///      T() != S();
///   }
///   struct HasOpEq
///   {
///       friend bool
///       operator==(const HasOpEq &, const HasOpEq&) noexcept = default;
///   };
///
///   void inverse()
///   {
///       HasOpEq e1;
///       HasOpEq e2;
///       if (e1 != e2)
///           return;
///   }
///
///   struct HasSpaceship
///   {
///       friend bool
///       operator<=>(const HasSpaceship &,
///                   const HasSpaceship&) noexcept = default;
///   };
///
///   void use_spaceship()
///   {
///       HasSpaceship s1;
///       HasSpaceship s2;
///       if (s1 != s2)
///           return;
///   }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++20-or-later}
///
/// The matcher \matcher{binaryOperation(
///     hasOperatorName("!="),
///     hasLHS(expr().bind("lhs")),
///     hasRHS(expr().bind("rhs"))
///   )}
/// matches \match{1 != 2}, \match{S() != S()}, \match{3 != 4},
/// \match{T() != S()}, \match{e1 != e2} and \match{s1 != s2}.
extern const internal::MapAnyOfMatcher<BinaryOperator, CXXOperatorCallExpr,
                                       CXXRewrittenBinaryOperator>
    binaryOperation;

/// Matches function calls and constructor calls
///
/// Because \c CallExpr and \c CXXConstructExpr do not share a common
/// base class with API accessing arguments etc, AST Matchers for code
/// which should match both are typically duplicated. This matcher
/// removes the need for duplication.
///
/// Given
/// \code
/// struct ConstructorTakesInt
/// {
///   ConstructorTakesInt(int i) {}
/// };
///
/// void callTakesInt(int i)
/// {
/// }
///
/// void doCall()
/// {
///   callTakesInt(42);
/// }
///
/// void doConstruct()
/// {
///   ConstructorTakesInt cti(42);
/// }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher
/// \matcher{expr(invocation(hasArgument(0, integerLiteral(equals(42)))))}
/// matches the expressions \match{callTakesInt(42)}
/// and \match{cti(42)}.
extern const internal::MapAnyOfMatcher<CallExpr, CXXConstructExpr> invocation;

/// Matches unary expressions that have a specific type of argument.
///
/// Given
/// \code
///   int a, c; float b; int s = sizeof(a) + sizeof(b) + alignof(c);
/// \endcode
/// \compile_args{-std=c++11-or-later,c23-or-later}
/// The matcher
/// \matcher{unaryExprOrTypeTraitExpr(hasArgumentOfType(asString("int")))}
/// matches \match{sizeof(a)} and \match{alignof(c)}
AST_MATCHER_P(UnaryExprOrTypeTraitExpr, hasArgumentOfType,
              internal::Matcher<QualType>, InnerMatcher) {
  const QualType ArgumentType = Node.getTypeOfArgument();
  return InnerMatcher.matches(ArgumentType, Finder, Builder);
}

/// Matches unary expressions of a certain kind.
///
/// Given
/// \code
///   int x;
///   int s = sizeof(x) + alignof(x);
/// \endcode
/// \compile_args{-std=c++11-or-later,c23-or-later}
/// The matcher \matcher{unaryExprOrTypeTraitExpr(ofKind(UETT_SizeOf))}
/// matches \match{sizeof(x)}
///
/// If the matcher is use from clang-query, UnaryExprOrTypeTrait parameter
/// should be passed as a quoted string. e.g., ofKind("UETT_SizeOf").
AST_MATCHER_P(UnaryExprOrTypeTraitExpr, ofKind, UnaryExprOrTypeTrait, Kind) {
  return Node.getKind() == Kind;
}

/// Same as unaryExprOrTypeTraitExpr, but only matching
/// alignof.
///
/// Given
/// \code
///   int align = alignof(int);
/// \endcode
/// \compile_args{-std=c++11-or-later,c23-or-later}
///
/// The matcher \matcher{alignOfExpr(expr())}
/// matches \match{alignof(int)}.
inline internal::BindableMatcher<Stmt>
alignOfExpr(const internal::Matcher<UnaryExprOrTypeTraitExpr> &InnerMatcher) {
  return stmt(unaryExprOrTypeTraitExpr(
      allOf(anyOf(ofKind(UETT_AlignOf), ofKind(UETT_PreferredAlignOf)),
            InnerMatcher)));
}

/// Same as unaryExprOrTypeTraitExpr, but only matching
/// sizeof.
///
/// Given
/// \code
///   struct S { double x; double y; };
///   int size = sizeof(struct S);
/// \endcode
///
/// The matcher \matcher{sizeOfExpr(expr())}
/// matches \match{sizeof(struct S)}.
inline internal::BindableMatcher<Stmt> sizeOfExpr(
    const internal::Matcher<UnaryExprOrTypeTraitExpr> &InnerMatcher) {
  return stmt(unaryExprOrTypeTraitExpr(
      allOf(ofKind(UETT_SizeOf), InnerMatcher)));
}

/// Matches NamedDecl nodes that have the specified name.
///
/// Supports specifying enclosing namespaces or classes by prefixing the name
/// with '<enclosing>::'.
/// Does not match typedefs of an underlying type with the given name.
///
/// Given
/// \code
///   class X;
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{namedDecl(hasName("X"))}
/// matches \match{class X}.
///
/// Given
/// \code
///   namespace a { namespace b { class X; } }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matchers \matcher{namedDecl(hasName("::a::b::X"))},
/// \matcher{namedDecl(hasName("a::b::X"))},
/// \matcher{namedDecl(hasName("b::X"))} and
/// \matcher{namedDecl(hasName("X"))}
/// match \match{class X}.
inline internal::Matcher<NamedDecl> hasName(StringRef Name) {
  return internal::Matcher<NamedDecl>(
      new internal::HasNameMatcher({std::string(Name)}));
}

/// Matches NamedDecl nodes that have any of the specified names.
///
/// This matcher is only provided as a performance optimization of hasName.
///
/// Given
/// \code
///   void f(int a, int b);
/// \endcode
///
/// The matcher \matcher{namedDecl(hasAnyName("a", "b"))},
/// which is equivalent to the matcher
/// \matcher{namedDecl(hasAnyName("a", "b"))},
/// matches \match{int a} and \match{int b}, but not
/// \nomatch{void f(int a, int b)}.
extern const internal::VariadicFunction<internal::Matcher<NamedDecl>, StringRef,
                                        internal::hasAnyNameFunc>
    hasAnyName;

/// Matches NamedDecl nodes whose fully qualified names contain
/// a substring matched by the given RegExp.
///
/// Supports specifying enclosing namespaces or classes by
/// prefixing the name with '<enclosing>::'.  Does not match typedefs
/// of an underlying type with the given name.
///
/// Given
/// \code
///   namespace foo { namespace bar { class X; } }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{namedDecl(matchesName("^::foo:.*X"))}
/// matches \match{class X}.
AST_MATCHER_REGEX(NamedDecl, matchesName, RegExp) {
  std::string FullNameString = "::" + Node.getQualifiedNameAsString();
  return RegExp->match(FullNameString);
}

/// Matches overloaded operator names.
///
/// Matches overloaded operator names specified in strings without the
/// "operator" prefix: e.g. "<<".
///
/// Given
/// \code
///   struct A { int operator*(); };
///   const A &operator<<(const A &a, const A &b);
///   void f(A a) {
///     a << a;   // <-- This matches
///   }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxOperatorCallExpr(hasOverloadedOperatorName("<<"))}
/// matches \match{a << a}.
/// The matcher
/// \matcher{cxxRecordDecl(hasMethod(hasOverloadedOperatorName("*")))}
/// matches \match{struct A { int operator*(); }}.
///
/// Usable as: Matcher<CXXOperatorCallExpr>, Matcher<FunctionDecl>
inline internal::PolymorphicMatcher<
    internal::HasOverloadedOperatorNameMatcher,
    AST_POLYMORPHIC_SUPPORTED_TYPES(CXXOperatorCallExpr, FunctionDecl),
    std::vector<std::string>>
hasOverloadedOperatorName(StringRef Name) {
  return internal::PolymorphicMatcher<
      internal::HasOverloadedOperatorNameMatcher,
      AST_POLYMORPHIC_SUPPORTED_TYPES(CXXOperatorCallExpr, FunctionDecl),
      std::vector<std::string>>({std::string(Name)});
}

/// Matches overloaded operator names.
///
/// Matches overloaded operator names specified in strings without the
/// "operator" prefix: e.g. "<<".
///
///   hasAnyOverloadedOperatorName("+", "-")
///
/// Given
/// \code
///   struct Point { double x; double y; };
///   Point operator+(const Point&, const Point&);
///   Point operator-(const Point&, const Point&);
///
///   Point sub(Point a, Point b) {
///     return b - a;
///   }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{functionDecl(hasAnyOverloadedOperatorName("+", "-"))},
/// which is equivalent to
/// \matcher{functionDecl(anyOf(hasAnyOverloadedOperatorName("+"),
/// hasOverloadedOperatorName("-")))},
/// matches \match{Point operator+(const Point&, const Point&)} and
/// \match{Point operator-(const Point&, const Point&)}.
/// The matcher
/// \matcher{cxxOperatorCallExpr(hasAnyOverloadedOperatorName("+", "-"))},
/// which is equivalent to
/// \matcher{cxxOperatorCallExpr(anyOf(hasOverloadedOperatorName("+"),
/// hasOverloadedOperatorName("-")))},
/// matches \match{b - a}.
///
/// Is equivalent to
///   anyOf(hasOverloadedOperatorName("+"), hasOverloadedOperatorName("-"))
extern const internal::VariadicFunction<
    internal::PolymorphicMatcher<internal::HasOverloadedOperatorNameMatcher,
                                 AST_POLYMORPHIC_SUPPORTED_TYPES(
                                     CXXOperatorCallExpr, FunctionDecl),
                                 std::vector<std::string>>,
    StringRef, internal::hasAnyOverloadedOperatorNameFunc>
    hasAnyOverloadedOperatorName;

/// Matches template-dependent, but known, member names.
///
/// In template declarations, dependent members are not resolved and so can
/// not be matched to particular named declarations.
///
/// This matcher allows to match on the known name of members.
///
/// Given
/// \code
///   template <typename T>
///   struct S {
///       void mem();
///   };
///   template <typename T>
///   void x() {
///       S<T> s;
///       s.mem();
///   }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{cxxDependentScopeMemberExpr(hasMemberName("mem"))}
/// matches \match{s.mem}.
AST_MATCHER_P(CXXDependentScopeMemberExpr, hasMemberName, std::string, N) {
  return Node.getMember().getAsString() == N;
}

/// Matches template-dependent, but known, member names against an already-bound
/// node
///
/// In template declarations, dependent members are not resolved and so can
/// not be matched to particular named declarations.
///
/// This matcher allows to match on the name of already-bound VarDecl, FieldDecl
/// and CXXMethodDecl nodes.
///
/// Given
/// \code
///   template <typename T>
///   struct S {
///       void mem();
///   };
///   template <typename T>
///   void x() {
///       S<T> s;
///       s.mem();
///   }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{cxxDependentScopeMemberExpr(
///   hasObjectExpression(declRefExpr(hasType(
///     elaboratedType(namesType(templateSpecializationType(
///       hasDeclaration(classTemplateDecl(has(cxxRecordDecl(has(
///           cxxMethodDecl(hasName("mem")).bind("templMem")
///           )))))
///     )))
///   ))),
///   memberHasSameNameAsBoundNode("templMem")
/// )}
/// matches \match{s.mem}, with the inner matcher
/// \matcher{type=sub$cxxMethodDecl(hasName("mem"))} matching
/// \match{sub=templMem$void mem()} of the \c S template.
AST_MATCHER_P(CXXDependentScopeMemberExpr, memberHasSameNameAsBoundNode,
              std::string, BindingID) {
  auto MemberName = Node.getMember().getAsString();

  return Builder->removeBindings(
      [this, MemberName](const BoundNodesMap &Nodes) {
        const auto &BN = Nodes.getNode(this->BindingID);
        if (const auto *ND = BN.get<NamedDecl>()) {
          if (!isa<FieldDecl, CXXMethodDecl, VarDecl>(ND))
            return true;
          return ND->getName() != MemberName;
        }
        return true;
      });
}

/// Matches C++ classes that are directly or indirectly derived from a class
/// matching \c Base, or Objective-C classes that directly or indirectly
/// subclass a class matching \c Base.
///
/// Note that a class is not considered to be derived from itself.
///
/// Example matches Y, Z, C (Base == hasName("X"))
/// \code
///   class X {};
///   class Y : public X {};  // directly derived
///   class Z : public Y {};  // indirectly derived
///   typedef X A;
///   typedef A B;
///   class C : public B {};  // derived from a typedef of X
///
///   class Foo {};
///   typedef Foo Alias;
///   class Bar : public Alias {};
///   // derived from a type that Alias is a typedef of Foo
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxRecordDecl(isDerivedFrom(hasName("X")))}
/// matches \match{class Y : public X {}}, \match{class Z : public Y {}}
/// and \match{class C : public B {}}.
///
/// The matcher \matcher{cxxRecordDecl(isDerivedFrom(hasName("Foo")))}
/// matches \match{class Bar : public Alias {}}.
///
/// In the following example, Bar matches isDerivedFrom(hasName("NSObject"))
/// \code
///   @interface NSObject @end
///   @interface Bar : NSObject @end
/// \endcode
/// \compile_args{-ObjC}
///
/// Usable as: Matcher<CXXRecordDecl>, Matcher<ObjCInterfaceDecl>
AST_POLYMORPHIC_MATCHER_P(
    isDerivedFrom,
    AST_POLYMORPHIC_SUPPORTED_TYPES(CXXRecordDecl, ObjCInterfaceDecl),
    internal::Matcher<NamedDecl>, Base) {
  // Check if the node is a C++ struct/union/class.
  if (const auto *RD = dyn_cast<CXXRecordDecl>(&Node))
    return Finder->classIsDerivedFrom(RD, Base, Builder, /*Directly=*/false);

  // The node must be an Objective-C class.
  const auto *InterfaceDecl = cast<ObjCInterfaceDecl>(&Node);
  return Finder->objcClassIsDerivedFrom(InterfaceDecl, Base, Builder,
                                        /*Directly=*/false);
}

/// Overloaded method as shortcut for \c isDerivedFrom(hasName(...)).
///
/// Matches C++ classes that are directly or indirectly derived from a class
/// matching \c Base, or Objective-C classes that directly or indirectly
/// subclass a class matching \c Base.
///
/// Note that a class is not considered to be derived from itself.
///
/// Example matches Y, Z, C (Base == hasName("X"))
/// \code
///   class X {};
///   class Y : public X {};  // directly derived
///   class Z : public Y {};  // indirectly derived
///   typedef X A;
///   typedef A B;
///   class C : public B {};  // derived from a typedef of X
///
///   class Foo {};
///   typedef Foo Alias;
///   class Bar : public Alias {};  // derived from Alias, which is a
///                                 // typedef of Foo
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxRecordDecl(isDerivedFrom("X"))}
/// matches \match{class Y : public X {}}, \match{class Z : public Y {}}
/// and \match{class C : public B {}}.
///
/// The matcher \matcher{cxxRecordDecl(isDerivedFrom("Foo"))}
/// matches \match{class Bar : public Alias {}}.
///
/// In the following example, Bar matches isDerivedFrom(hasName("NSObject"))
/// \code
///   @interface NSObject @end
///   @interface Bar : NSObject @end
/// \endcode
/// \compile_args{-ObjC}
///
/// Usable as: Matcher<CXXRecordDecl>, Matcher<ObjCInterfaceDecl>
AST_POLYMORPHIC_MATCHER_P_OVERLOAD(
    isDerivedFrom,
    AST_POLYMORPHIC_SUPPORTED_TYPES(CXXRecordDecl, ObjCInterfaceDecl),
    std::string, BaseName, 1) {
  if (BaseName.empty())
    return false;

  const auto M = isDerivedFrom(hasName(BaseName));

  if (const auto *RD = dyn_cast<CXXRecordDecl>(&Node))
    return Matcher<CXXRecordDecl>(M).matches(*RD, Finder, Builder);

  const auto *InterfaceDecl = cast<ObjCInterfaceDecl>(&Node);
  return Matcher<ObjCInterfaceDecl>(M).matches(*InterfaceDecl, Finder, Builder);
}

/// Matches C++ classes that have a direct or indirect base matching \p
/// BaseSpecMatcher.
///
/// Given
/// \code
///   class Foo {};
///   class Bar : Foo {};
///   class Baz : Bar {};
///   class SpecialBase {};
///   class Proxy : SpecialBase {};  // matches Proxy
///   class IndirectlyDerived : Proxy {};  //matches IndirectlyDerived
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher
/// \matcher{cxxRecordDecl(hasAnyBase(hasType(cxxRecordDecl(hasName("SpecialBase")))))}
/// matches \match{class Proxy : SpecialBase {}} and
/// \match{class IndirectlyDerived : Proxy {}}.
// FIXME: Refactor this and isDerivedFrom to reuse implementation.
AST_MATCHER_P(CXXRecordDecl, hasAnyBase, internal::Matcher<CXXBaseSpecifier>,
              BaseSpecMatcher) {
  return internal::matchesAnyBase(Node, BaseSpecMatcher, Finder, Builder);
}

/// Matches C++ classes that have a direct base matching \p BaseSpecMatcher.
///
/// Given
/// \code
///   class Foo {};
///   class Bar : Foo {};
///   class Baz : Bar {};
///   class SpecialBase {};
///   class Proxy : SpecialBase {};  // matches Proxy
///   class IndirectlyDerived : Proxy {};  // doesn't match
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(hasDirectBase(hasType(cxxRecordDecl(hasName("SpecialBase")))))}
/// matches \match{class Proxy : SpecialBase {}}.
AST_MATCHER_P(CXXRecordDecl, hasDirectBase, internal::Matcher<CXXBaseSpecifier>,
              BaseSpecMatcher) {
  return Node.hasDefinition() &&
         llvm::any_of(Node.bases(), [&](const CXXBaseSpecifier &Base) {
           return BaseSpecMatcher.matches(Base, Finder, Builder);
         });
}

/// Similar to \c isDerivedFrom(), but also matches classes that directly
/// match \c Base.
///
/// Given
/// \code
///   class X {};
///   class Y : public X {};  // directly derived
///   class Z : public Y {};  // indirectly derived
///   typedef X A;
///   typedef A B;
///   class C : public B {};  // derived from a typedef of X
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(isSameOrDerivedFrom(cxxRecordDecl(hasName("X"))),
/// isDefinition())}
/// matches \match{class X {}}, \match{class Y : public X {}},
/// \match{class Z : public Y {}} and \match{class C : public B {}}.
AST_POLYMORPHIC_MATCHER_P_OVERLOAD(
    isSameOrDerivedFrom,
    AST_POLYMORPHIC_SUPPORTED_TYPES(CXXRecordDecl, ObjCInterfaceDecl),
    internal::Matcher<NamedDecl>, Base, 0) {
  const auto M = anyOf(Base, isDerivedFrom(Base));

  if (const auto *RD = dyn_cast<CXXRecordDecl>(&Node))
    return Matcher<CXXRecordDecl>(M).matches(*RD, Finder, Builder);

  const auto *InterfaceDecl = cast<ObjCInterfaceDecl>(&Node);
  return Matcher<ObjCInterfaceDecl>(M).matches(*InterfaceDecl, Finder, Builder);
}

/// Similar to \c isDerivedFrom(), but also matches classes that directly
/// match \c Base.
/// Overloaded method as shortcut for
/// \c isSameOrDerivedFrom(hasName(...)).
///
/// Given
/// \code
///   class X {};
///   class Y : public X {};  // directly derived
///   class Z : public Y {};  // indirectly derived
///   typedef X A;
///   typedef A B;
///   class C : public B {};  // derived from a typedef of X
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(isSameOrDerivedFrom("X"), isDefinition())}
/// matches \match{class X {}}, \match{class Y : public X {}},
/// \match{class Z : public Y {}} and \match{class C : public B {}}.
AST_POLYMORPHIC_MATCHER_P_OVERLOAD(
    isSameOrDerivedFrom,
    AST_POLYMORPHIC_SUPPORTED_TYPES(CXXRecordDecl, ObjCInterfaceDecl),
    std::string, BaseName, 1) {
  if (BaseName.empty())
    return false;

  const auto M = isSameOrDerivedFrom(hasName(BaseName));

  if (const auto *RD = dyn_cast<CXXRecordDecl>(&Node))
    return Matcher<CXXRecordDecl>(M).matches(*RD, Finder, Builder);

  const auto *InterfaceDecl = cast<ObjCInterfaceDecl>(&Node);
  return Matcher<ObjCInterfaceDecl>(M).matches(*InterfaceDecl, Finder, Builder);
}

/// Matches C++ or Objective-C classes that are directly derived from a class
/// matching \c Base.
///
/// Note that a class is not considered to be derived from itself.
///
/// Given
/// \code
///   class X {};
///   class Y : public X {};  // directly derived
///   class Z : public Y {};  // indirectly derived
///   typedef X A;
///   typedef A B;
///   class C : public B {};  // derived from a typedef of X
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(isDirectlyDerivedFrom(namedDecl(hasName("X"))))}
/// matches \match{class Y : public X {}} and \match{class C : public B {}}
/// (Base == hasName("X").
///
/// In the following example, Bar matches isDerivedFrom(hasName("X")):
/// \code
///   class Foo {};
///   typedef Foo X;
///   class Bar : public Foo {};  // derived from a type that X is a typedef of
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxRecordDecl(isDerivedFrom(hasName("X")))}
/// matches \match{class Bar : public Foo {}}.
AST_POLYMORPHIC_MATCHER_P_OVERLOAD(
    isDirectlyDerivedFrom,
    AST_POLYMORPHIC_SUPPORTED_TYPES(CXXRecordDecl, ObjCInterfaceDecl),
    internal::Matcher<NamedDecl>, Base, 0) {
  // Check if the node is a C++ struct/union/class.
  if (const auto *RD = dyn_cast<CXXRecordDecl>(&Node))
    return Finder->classIsDerivedFrom(RD, Base, Builder, /*Directly=*/true);

  // The node must be an Objective-C class.
  const auto *InterfaceDecl = cast<ObjCInterfaceDecl>(&Node);
  return Finder->objcClassIsDerivedFrom(InterfaceDecl, Base, Builder,
                                        /*Directly=*/true);
}

/// Overloaded method as shortcut for \c isDirectlyDerivedFrom(hasName(...)).
///
/// Given
/// \code
///   struct Base {};
///   struct DirectlyDerived : public Base {};
///   struct IndirectlyDerived : public DirectlyDerived {};
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxRecordDecl(isDirectlyDerivedFrom("Base"))}
/// matches \match{struct DirectlyDerived : public Base {}}, but not
/// \nomatch{struct IndirectlyDerived : public DirectlyDerived {}}.
AST_POLYMORPHIC_MATCHER_P_OVERLOAD(
    isDirectlyDerivedFrom,
    AST_POLYMORPHIC_SUPPORTED_TYPES(CXXRecordDecl, ObjCInterfaceDecl),
    std::string, BaseName, 1) {
  if (BaseName.empty())
    return false;
  const auto M = isDirectlyDerivedFrom(hasName(BaseName));

  if (const auto *RD = dyn_cast<CXXRecordDecl>(&Node))
    return Matcher<CXXRecordDecl>(M).matches(*RD, Finder, Builder);

  const auto *InterfaceDecl = cast<ObjCInterfaceDecl>(&Node);
  return Matcher<ObjCInterfaceDecl>(M).matches(*InterfaceDecl, Finder, Builder);
}
/// Matches the first method of a class or struct that satisfies \c
/// InnerMatcher.
///
/// Given
/// \code
///   class A { void func(); };
///   class B { void member(); };
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxRecordDecl(hasMethod(hasName("func")))}
/// matches the declaration of \match{class A { void func(); }}
/// but does not match \nomatch{class B { void member(); }}
AST_MATCHER_P(CXXRecordDecl, hasMethod, internal::Matcher<CXXMethodDecl>,
              InnerMatcher) {
  BoundNodesTreeBuilder Result(*Builder);
  auto MatchIt = matchesFirstInPointerRange(InnerMatcher, Node.method_begin(),
                                            Node.method_end(), Finder, &Result);
  if (MatchIt == Node.method_end())
    return false;

  if (Finder->isTraversalIgnoringImplicitNodes() && (*MatchIt)->isImplicit())
    return false;
  *Builder = std::move(Result);
  return true;
}

/// Matches the generated class of lambda expressions.
///
/// Given
/// \code
///   auto x = []{};
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{varDecl(hasType(cxxRecordDecl(isLambda())))}
/// matches \match{auto x = []{}}.
AST_MATCHER(CXXRecordDecl, isLambda) {
  return Node.isLambda();
}

/// Matches AST nodes that have child AST nodes that match the
/// provided matcher.
///
/// Given
/// \code
///   class X {};  // Matches X, because X::X is a class of name X inside X.
///   class Y { class X {}; };
///   class Z { class Y { class X {}; }; };  // Does not match Z.
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxRecordDecl(has(cxxRecordDecl(hasName("X"))))}
/// matches \match{count=3$class X {}} three times for the definitions of \c X
/// that contain the implicit class declarations of \c X ,
/// and \match{count=2$class Y { class X {}; }} two times for the two different
/// definitions of \c Y that contain \c X .
///
/// ChildT must be an AST base type.
///
/// Usable as: Any Matcher
/// Note that has is direct matcher, so it also matches things like implicit
/// casts and paren casts. If you are matching with expr then you should
/// probably consider using ignoringParenImpCasts:
///
/// Given
/// \code
///   int x =0;
///   double y = static_cast<double>(x);
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxStaticCastExpr(has(ignoringParenImpCasts(declRefExpr())))}.
/// matches \match{static_cast<double>(x)}
extern const internal::ArgumentAdaptingMatcherFunc<internal::HasMatcher> has;

/// Matches AST nodes that have descendant AST nodes that match the
/// provided matcher.
///
/// Given
/// \code
///   class X {};  // Matches X, because X::X is a class of name X inside X.
///   class Y { class X {}; };
///   class Z { class Y { class X {}; }; };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(hasDescendant(cxxRecordDecl(hasName("X"))))}
/// matches \match{count=3$class X {}} three times for the definitions of \c X
/// that contain the implicit class declarations of \c X ,
/// \match{count=2$class Y { class X {}; }} two times for the declaration of
/// \c X they contain, and \match{class Z { class Y { class X {}; }; }}.
///
/// DescendantT must be an AST base type.
///
/// Usable as: Any Matcher
extern const internal::ArgumentAdaptingMatcherFunc<
    internal::HasDescendantMatcher>
    hasDescendant;

/// Matches AST nodes that have child AST nodes that match the
/// provided matcher.
///
/// Given
/// \code
///   class X {};
///   class Y { class X {}; };  // Matches Y, because Y::X is a class of name X
///                             // inside Y.
///   class Z { class Y { class X {}; }; };  // Does not match Z.
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxRecordDecl(forEach(cxxRecordDecl(hasName("X"))))}
/// matches \match{type=typestr$class X},
/// \match{type=typestr$class Y},
/// \match{type=typestr$class Y::X},
/// \match{type=typestr$class Z::Y::X} and \match{type=typestr$class Z::Y}
///
/// ChildT must be an AST base type.
///
/// As opposed to 'has', 'forEach' will cause a match for each result that
///   matches instead of only on the first one.
///
/// Usable as: Any Matcher
extern const internal::ArgumentAdaptingMatcherFunc<internal::ForEachMatcher>
    forEach;

/// Matches AST nodes that have descendant AST nodes that match the
/// provided matcher.
///
/// Given
/// \code
///   class X {};
///   class A { class X {}; };  // Matches A, because A::X is a class of name
///                             // X inside A.
///   class B { class C { class X {}; }; };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(forEachDescendant(cxxRecordDecl(hasName("X"))))}
/// matches \match{count=3$class X {}} three times, once for each of the
/// declared classes \c X and their implicit class declaration,
/// \match{class A { class X {}; }},
/// \match{class B { class C { class X {}; }; }} and
/// \match{class C { class X {}; }}.
///
/// DescendantT must be an AST base type.
///
/// As opposed to 'hasDescendant', 'forEachDescendant' will cause a match for
/// each result that matches instead of only on the first one.
///
/// Note: Recursively combined ForEachDescendant can cause many matches:
/// \code
///   struct A {
///     struct B {
///       struct C {};
///       struct D {};
///     };
///   };
///
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxRecordDecl(forEachDescendant(cxxRecordDecl(
///     forEachDescendant(cxxRecordDecl().bind("inner"))
///   ).bind("middle")))}
/// will match 9 times:
///
/// It matches the definition of \match{type=name$A} with the definition of
/// \match{sub=middle;type=name$B} in the middle and the implicit class
/// declaration of \match{sub=inner;type=name$B} as the innermost \c
/// cxxRecordDecl.
///
/// It matches the definition of \match{type=name$A} with the definition of
/// \match{sub=middle;type=name$C} in the middle and the definition of
/// \match{sub=inner;type=name$B} as the innermost \c cxxRecordDecl.
///
/// It matches the definition of \match{type=name$A} with the definition of
/// \match{sub=middle;type=name$C} in the middle and the implicit class
/// declaration of \match{sub=inner;type=name$B} as the innermost \c
/// cxxRecordDecl.
///
/// It matches the definition of \match{type=name$A} with the definition of
/// \match{sub=middle;type=name$B} in the middle and the definition of
/// \match{sub=inner;type=name$D} as the innermost \c cxxRecordDecl.
///
/// It matches the definition of \match{type=name$A} with the definition of
/// \match{sub=middle;type=name$B} in the middle and the implicit class
/// declaration of \match{sub=inner;type=name$D} as the innermost \c
/// cxxRecordDecl.
///
/// It matches the definition of \match{type=name$A} with the definition of
/// \match{sub=middle;type=name$C} in the middle and the implicit class
/// declaration of \match{sub=inner;type=name$C} as the innermost \c
/// cxxRecordDecl.
///
/// It matches the definition of \match{type=name$A} with the definition of
/// \match{sub=middle;type=name$D} in the middle and the implicit class
/// declaration of \match{sub=inner;type=name$D} as the innermost \c
/// cxxRecordDecl.
///
/// It matches the definition of \match{type=name$B} with the definition of
/// \match{sub=middle;type=name$C} in the middle and the implicit class
/// declaration of \match{sub=inner;type=name$C} as the innermost \c
/// cxxRecordDecl.
///
/// It matches the definition of \match{type=name$B} with the definition of
/// \match{sub=middle;type=name$D} in the middle and the implicit class
/// declaration of \match{sub=inner;type=name$D} as the innermost \c
/// cxxRecordDecl.
///
/// Usable as: Any Matcher
extern const internal::ArgumentAdaptingMatcherFunc<
    internal::ForEachDescendantMatcher>
    forEachDescendant;

/// Matches if the node or any descendant matches.
///
/// Generates results for each match.
///
/// For example, in:
/// \code
///   class A { class B {}; class C {}; };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(hasName("::A"),
///                 findAll(cxxRecordDecl(isDefinition()).bind("m")))}
/// matches \match{count=3$class A { class B {}; class C {}; }} three times,
/// with \matcher{type=sub$cxxRecordDecl(isDefinition()).bind("m")}
/// matching \match{type=name;sub=m$A},
/// \match{type=name;sub=m$B} and \match{type=name;sub=m$C}.
///
/// Usable as: Any Matcher
template <typename T>
internal::Matcher<T> findAll(const internal::Matcher<T> &Matcher) {
  return eachOf(Matcher, forEachDescendant(Matcher));
}

/// Matches AST nodes that have a parent that matches the provided
/// matcher.
///
/// Given
/// \code
/// void f() { for (;;) { int x = 42; if (true) { int x = 43; } } }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{compoundStmt(hasParent(ifStmt()))}
/// matches \match{{ int x = 43; }}
///
/// Usable as: Any Matcher
extern const internal::ArgumentAdaptingMatcherFunc<
    internal::HasParentMatcher,
    internal::TypeList<Decl, NestedNameSpecifierLoc, Stmt, TypeLoc, Attr>,
    internal::TypeList<Decl, NestedNameSpecifierLoc, Stmt, TypeLoc, Attr>>
    hasParent;

/// Matches AST nodes that have an ancestor that matches the provided
/// matcher.
///
/// Given
/// \code
/// void f() { if (true) { int x = 42; } }
/// void g() { for (;;) { int x = 43; } }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{expr(integerLiteral(hasAncestor(ifStmt())))}
/// matches \match{42}
/// but does not match \nomatch{43}
///
/// Usable as: Any Matcher
extern const internal::ArgumentAdaptingMatcherFunc<
    internal::HasAncestorMatcher,
    internal::TypeList<Decl, NestedNameSpecifierLoc, Stmt, TypeLoc, Attr>,
    internal::TypeList<Decl, NestedNameSpecifierLoc, Stmt, TypeLoc, Attr>>
    hasAncestor;

/// Matches if the provided matcher does not match.
///
/// Given
/// \code
///   int x;
///   int y = 0;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{varDecl(unless(hasInitializer(expr())))}
/// matches \match{int x}, but not \nomatch{int y = 0}.
///
/// Usable as: Any Matcher
extern const internal::VariadicOperatorMatcherFunc<1, 1> unless;

/// Matches a node if the declaration associated with that node
///   matches the given matcher.
///
/// The associated declaration is:
/// - for type nodes, the declaration of the underlying type
/// - for CallExpr, the declaration of the callee
/// - for MemberExpr, the declaration of the referenced member
/// - for CXXConstructExpr, the declaration of the constructor
/// - for CXXNewExpr, the declaration of the operator new
/// - for ObjCIvarExpr, the declaration of the ivar
///
/// Given
/// \code
///   class X {};
///   typedef X Y;
///   Y y;
/// \endcode
/// \compile_args{-std=c++}
/// For type nodes, hasDeclaration will generally match the declaration of the
/// sugared type, i.e., the matcher
/// \matcher{varDecl(hasType(qualType(hasDeclaration(decl().bind("d")))))},
/// matches \match{Y y}, with
/// the matcher \matcher{type=sub$decl()} matching
/// \match{sub=d$typedef X Y;}.
/// A common use case is to match the underlying, desugared type.
/// This can be achieved by using the hasUnqualifiedDesugaredType matcher:
/// \matcher{varDecl(hasType(hasUnqualifiedDesugaredType(
///       recordType(hasDeclaration(decl().bind("d"))))))}
/// matches \match{Y y}.
/// In this matcher, the matcher \matcher{type=sub$decl()} will match the
/// CXXRecordDecl
/// \match{sub=d$class X {};}.
///
/// Usable as: Matcher<AddrLabelExpr>, Matcher<CallExpr>,
///   Matcher<CXXConstructExpr>, Matcher<CXXNewExpr>, Matcher<DeclRefExpr>,
///   Matcher<EnumType>, Matcher<InjectedClassNameType>, Matcher<LabelStmt>,
///   Matcher<MemberExpr>, Matcher<QualType>, Matcher<RecordType>,
///   Matcher<TagType>, Matcher<TemplateSpecializationType>,
///   Matcher<TemplateTypeParmType>, Matcher<TypedefType>,
///   Matcher<UnresolvedUsingType>
inline internal::PolymorphicMatcher<
    internal::HasDeclarationMatcher,
    void(internal::HasDeclarationSupportedTypes), internal::Matcher<Decl>>
hasDeclaration(const internal::Matcher<Decl> &InnerMatcher) {
  return internal::PolymorphicMatcher<
      internal::HasDeclarationMatcher,
      void(internal::HasDeclarationSupportedTypes), internal::Matcher<Decl>>(
      InnerMatcher);
}

/// Matches a \c NamedDecl whose underlying declaration matches the given
/// matcher.
///
/// Given
/// \code
///   namespace N { template<class T> void f(T t); }
///   template <class T> void g() { using N::f; f(T()); }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++11-or-later}
/// The matcher \matcher{unresolvedLookupExpr(hasAnyDeclaration(
///     namedDecl(hasUnderlyingDecl(hasName("::N::f")))))}
///   matches \match{f} in \c g().
AST_MATCHER_P(NamedDecl, hasUnderlyingDecl, internal::Matcher<NamedDecl>,
              InnerMatcher) {
  const NamedDecl *UnderlyingDecl = Node.getUnderlyingDecl();

  return UnderlyingDecl != nullptr &&
         InnerMatcher.matches(*UnderlyingDecl, Finder, Builder);
}

/// Matches on the implicit object argument of a member call expression, after
/// stripping off any parentheses or implicit casts.
///
/// Given
/// \code
///   class Y { public: void m(); };
///   Y g();
///   class X : public Y {};
///   void z(Y y, X x) { y.m(); (g()).m(); x.m(); }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxMemberCallExpr(on(hasType(cxxRecordDecl(hasName("Y")))))}
///   matches \match{y.m()} and \match{(g()).m()}.
/// The matcher
/// \matcher{cxxMemberCallExpr(on(hasType(cxxRecordDecl(hasName("X")))))}
///   matches \match{x.m()}.
/// The matcher \matcher{cxxMemberCallExpr(on(callExpr()))}
///   matches \match{(g()).m()}.
///
/// FIXME: Overload to allow directly matching types?
AST_MATCHER_P(CXXMemberCallExpr, on, internal::Matcher<Expr>,
              InnerMatcher) {
  const Expr *ExprNode = Node.getImplicitObjectArgument()
                            ->IgnoreParenImpCasts();
  return (ExprNode != nullptr &&
          InnerMatcher.matches(*ExprNode, Finder, Builder));
}

/// Matches on the receiver of an ObjectiveC Message expression.
///
/// \code
///   NSString *webViewJavaScript = ...
///   UIWebView *webView = ...
///   [webView stringByEvaluatingJavaScriptFromString:webViewJavascript];
/// \endcode
/// \compile_args{-ObjC}
/// The matcher \matcher{objCMessageExpr(hasReceiverType(asString("UIWebView
/// *")))} matches
/// \match{[webViewstringByEvaluatingJavaScriptFromString:webViewJavascript];}
AST_MATCHER_P(ObjCMessageExpr, hasReceiverType, internal::Matcher<QualType>,
              InnerMatcher) {
  const QualType TypeDecl = Node.getReceiverType();
  return InnerMatcher.matches(TypeDecl, Finder, Builder);
}

/// Returns true when the Objective-C method declaration is a class method.
///
/// Given
/// \code
/// @interface I + (void)foo; @end
/// @interface I - (void)bar; @end
/// \endcode
/// \compile_args{-ObjC}
/// The matcher \matcher{objcMethodDecl(isClassMethod())}
/// matches \match{@interface I + (void)foo; @end}
/// but does not match \nomatch{interface I + (void)foo; @end}
AST_MATCHER(ObjCMethodDecl, isClassMethod) {
  return Node.isClassMethod();
}

/// Returns true when the Objective-C method declaration is an instance method.
///
/// Given
/// \code
/// @interface I - (void)bar; @end
/// @interface I + (void)foo; @end
/// \endcode
/// \compile_args{-ObjC}
/// The matcher \matcher{objcMethodDecl(isInstanceMethod())}
/// matches \match{@interface I - (void)bar; @end}
/// but does not match \nomatch{@interface I - (void)foo; @end}
/// \compile_args{-ObjC}
AST_MATCHER(ObjCMethodDecl, isInstanceMethod) {
  return Node.isInstanceMethod();
}

/// Returns true when the Objective-C message is sent to a class.
///
/// Given
/// \code
///   [NSString stringWithFormat:@"format"];
///   NSString *x = @"hello";
///   [x containsString:@"h"];
/// \endcode
/// \compile_args{-ObjC}
/// The matcher \matcher{objcMessageExpr(isClassMessage())}
/// matches \match{[NSString stringWithFormat:@"format"];}
/// but does not match \nomatch{[[x containsString:@"h"]}
AST_MATCHER(ObjCMessageExpr, isClassMessage) {
  return Node.isClassMessage();
}

/// Returns true when the Objective-C message is sent to an instance.
///
/// Given
/// \code
///   NSString *x = @"hello";
///   [x containsString:@"h"];
///   [NSString stringWithFormat:@"format"];
/// \endcode
/// \compile_args{-ObjC}
/// The matcher \matcher{objcMessageExpr(isInstanceMessage())}
/// matches \match{[x containsString:@"h"];}
/// but does not match \nomatch{[NSString stringWithFormat:@"format"];}
AST_MATCHER(ObjCMessageExpr, isInstanceMessage) {
  return Node.isInstanceMessage();
}

/// Matches if the Objective-C message is sent to an instance,
/// and the inner matcher matches on that instance.
///
/// Given
/// \code
///   NSString *x = @"hello";
///   [x containsString:@"h"];
/// \endcode
/// \compile_args{-ObjC}
/// The matcher
/// \matcher{objcMessageExpr(hasReceiver(declRefExpr(to(varDecl(hasName("x"))))))}
/// matches \match{[x containsString:@"h"];}
AST_MATCHER_P(ObjCMessageExpr, hasReceiver, internal::Matcher<Expr>,
              InnerMatcher) {
  const Expr *ReceiverNode = Node.getInstanceReceiver();
  return (ReceiverNode != nullptr &&
          InnerMatcher.matches(*ReceiverNode->IgnoreParenImpCasts(), Finder,
                               Builder));
}

/// Matches when BaseName == Selector.getAsString()
///
/// \code
///     [self.bodyView loadHTMLString:html baseURL:NULL];
/// \endcode
/// \compile_args{-ObjC}
/// The matcher
/// \matcher{objCMessageExpr(hasSelector("loadHTMLString:baseURL:"));} matches
/// the outer message expr in the code below, but NOT the message invocation
/// for self.bodyView.
AST_MATCHER_P(ObjCMessageExpr, hasSelector, std::string, BaseName) {
  Selector Sel = Node.getSelector();
  return BaseName == Sel.getAsString();
}

/// Matches when at least one of the supplied string equals to the
/// Selector.getAsString()
///
/// \code
///     [myObj methodA:argA];
///     [myObj methodB:argB];
/// \endcode
/// \compile_args{-ObjC}
///  The matcher \matcher{objCMessageExpr(hasSelector("methodA:", "methodB:"));}
///  matches \match{[myObj methodA:argA];} and \match{[myObj methodB:argB];}
extern const internal::VariadicFunction<internal::Matcher<ObjCMessageExpr>,
                                        StringRef,
                                        internal::hasAnySelectorFunc>
                                        hasAnySelector;

/// Matches ObjC selectors whose name contains
/// a substring matched by the given RegExp.
///
/// Given
/// \code
///     [self.bodyView loadHTMLString:html baseURL:NULL];
/// \endcode
/// \compile_args{-ObjC}
///
/// The matcher
/// \matcher{objCMessageExpr(matchesSelector("loadHTMLString\:baseURL?"))}
/// matches the outer message expr in the code below, but NOT the message
/// invocation for self.bodyView.
AST_MATCHER_REGEX(ObjCMessageExpr, matchesSelector, RegExp) {
  std::string SelectorString = Node.getSelector().getAsString();
  return RegExp->match(SelectorString);
}

/// Matches when the selector is the empty selector
///
/// Matches only when the selector of the objCMessageExpr is NULL. This may
/// represent an error condition in the tree!
AST_MATCHER(ObjCMessageExpr, hasNullSelector) {
  return Node.getSelector().isNull();
}

/// Matches when the selector is a Unary Selector
///
/// Given
/// \code
///     [self.bodyView loadHTMLString:html baseURL:NULL];
/// \endcode
/// \compile_args{-ObjC}
///
///  The matcher \matcher{objCMessageExpr(matchesSelector(hasUnarySelector());}
///  matches \match{self.bodyView}, but does not match the outer message
///  invocation of "loadHTMLString:baseURL:".
AST_MATCHER(ObjCMessageExpr, hasUnarySelector) {
  return Node.getSelector().isUnarySelector();
}

/// Matches when the selector is a keyword selector
///
/// Given
/// \code
///   UIWebView *webView = ...;
///   CGRect bodyFrame = webView.frame;
///   bodyFrame.size.height = self.bodyContentHeight;
///   webView.frame = bodyFrame;
///   //     ^---- matches here
/// \endcode
/// \compile_args{-ObjC}
///
/// The matcher \matcher{objCMessageExpr(hasKeywordSelector())} matches the
/// generated setFrame message expression in
AST_MATCHER(ObjCMessageExpr, hasKeywordSelector) {
  return Node.getSelector().isKeywordSelector();
}

/// Matches when the selector has the specified number of arguments
///
/// \code
///     [self.bodyView loadHTMLString:html baseURL:NULL];
/// \endcode
/// \compile_args{-ObjC}
/// The matcher \matcher{objCMessageExpr(numSelectorArgs(0))}
/// matches \match{self.bodyView}.
/// The matcher \matcher{objCMessageExpr(numSelectorArgs(2))}
/// matches the invocation of \match{loadHTMLString:baseURL:}
/// but does not match \nomatch{self.bodyView}
AST_MATCHER_P(ObjCMessageExpr, numSelectorArgs, unsigned, N) {
  return Node.getSelector().getNumArgs() == N;
}

/// Matches if the call or fold expression's callee expression matches.
///
/// Given
/// \code
///   class Y { void x() { this->x(); x(); Y y; y.x(); } };
///   void f() { f(); }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{callExpr(callee(expr().bind("callee")))}
/// matches \match{this->x()}, \match{x()}, \match{y.x()}, \match{f()}
/// with \matcher{type=sub$expr()} inside of \c callee
/// matching \match{sub=callee$this->x}, \match{sub=callee$x},
/// \match{sub=callee$y.x}, \match{sub=callee$f} respectively
///
/// Given
/// \code
///   struct Dummy {};
///   // makes sure there is a callee, otherwise there would be no callee,
///   // just a builtin operator
///   Dummy operator+(Dummy, Dummy);
///   // not defining a '*' operator
///
///   template <typename... Args>
///   auto sum(Args... args) {
///       return (0 + ... + args);
///   }
///
///   template <typename... Args>
///   auto multiply(Args... args) {
///       return (args * ... * 1);
///   }
/// \endcode
/// \compile_args{-std=c++17-or-later}
/// The matcher \matcher{cxxFoldExpr(callee(expr().bind("op")))}
/// matches \match{(0 + ... + args)}
/// with \matcher{type=sub$callee(...)} matching \match{sub=op$*},
/// but does not match \nomatch{(args * ... * 1)}.
/// A \c CXXFoldExpr only has an \c UnresolvedLookupExpr as a callee.
/// When there are no define operators that could be used instead of builtin
/// ones, then there will be no \c callee .
///
/// Note: Callee cannot take the more general internal::Matcher<Expr>
/// because this introduces ambiguous overloads with calls to Callee taking a
/// internal::Matcher<Decl>, as the matcher hierarchy is purely
/// implemented in terms of implicit casts.
AST_POLYMORPHIC_MATCHER_P_OVERLOAD(callee,
                                   AST_POLYMORPHIC_SUPPORTED_TYPES(CallExpr,
                                                                   CXXFoldExpr),
                                   internal::Matcher<Stmt>, InnerMatcher, 0) {
  const auto *ExprNode = Node.getCallee();
  return (ExprNode != nullptr &&
          InnerMatcher.matches(*ExprNode, Finder, Builder));
}

/// Matches 1) if the call expression's callee's declaration matches the
/// given matcher; or 2) if the Obj-C message expression's callee's method
/// declaration matches the given matcher.
///
/// Example 1
/// \code
///   class Y { public: void x(); };
///   void z() { Y y; y.x(); }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{callExpr(callee(cxxMethodDecl(hasName("x"))))}
/// matches \match{y.x()}
///
/// Example 2
/// \code
///   @interface I: NSObject
///   +(void)foo;
///   @end
///   ...
///   [I foo]
/// \endcode
/// \compile_args{-ObjC}
/// The matcher
/// \matcher{objcMessageExpr(callee(objcMethodDecl(hasName("foo"))))}
/// matches \match{[I foo]}
AST_POLYMORPHIC_MATCHER_P_OVERLOAD(
    callee, AST_POLYMORPHIC_SUPPORTED_TYPES(ObjCMessageExpr, CallExpr),
    internal::Matcher<Decl>, InnerMatcher, 1) {
  if (isa<CallExpr>(&Node))
    return callExpr(hasDeclaration(InnerMatcher))
        .matches(Node, Finder, Builder);
  else {
    // The dynamic cast below is guaranteed to succeed as there are only 2
    // supported return types.
    const auto *MsgNode = cast<ObjCMessageExpr>(&Node);
    const Decl *DeclNode = MsgNode->getMethodDecl();
    return (DeclNode != nullptr &&
            InnerMatcher.matches(*DeclNode, Finder, Builder));
  }
}

/// Matches if the expression's or declaration's type matches a type
/// matcher.
///
/// Exmaple
/// \code
///  class X {};
///  void y(X &x) { x; X z; }
///  typedef int U;
///  class Y { friend class X; };
///  class Z : public virtual X {};
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{expr(hasType(cxxRecordDecl(hasName("X"))))}
/// matches \match{x} and \match{z}.
/// The matcher \matcher{varDecl(hasType(cxxRecordDecl(hasName("X"))))}
/// matches \match{X z}
/// The matcher \matcher{typedefDecl(hasType(asString("int")))}
/// matches \match{typedef int U}
/// The matcher \matcher{friendDecl(hasType(asString("class X")))}
/// matches \match{friend class X}
/// The matcher \matcher{cxxRecordDecl(hasAnyBase(cxxBaseSpecifier(hasType(
/// asString("X"))).bind("b")))} matches \match{class Z : public virtual X {}},
/// with \matcher{type=sub$cxxBaseSpecifier(...)}
/// matching \match{sub=b$public virtual X}.
AST_POLYMORPHIC_MATCHER_P_OVERLOAD(
    hasType,
    AST_POLYMORPHIC_SUPPORTED_TYPES(Expr, FriendDecl, TypedefNameDecl,
                                    ValueDecl, CXXBaseSpecifier),
    internal::Matcher<QualType>, InnerMatcher, 0) {
  QualType QT = internal::getUnderlyingType(Node);
  if (!QT.isNull())
    return InnerMatcher.matches(QT, Finder, Builder);
  return false;
}

/// Overloaded to match the declaration of the expression's or value
/// declaration's type.
///
/// In case of a value declaration (for example a variable declaration),
/// this resolves one layer of indirection. For example, in the value
/// declaration "X x;", cxxRecordDecl(hasName("X")) matches the declaration of
/// X, while varDecl(hasType(cxxRecordDecl(hasName("X")))) matches the
/// declaration of x.
///
/// \code
///  class X {};
///  void y(X &x) { x; X z; }
///  class Y { friend class X; };
///  class Z : public virtual X {};
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{expr(hasType(cxxRecordDecl(hasName("X"))))}
/// matches \match{x} and \match{z}.
/// The matcher \matcher{varDecl(hasType(cxxRecordDecl(hasName("X"))))}
/// matches \match{X z}.
/// The matcher \matcher{friendDecl(hasType(asString("class X")))}
/// matches \match{friend class X}.
/// The matcher \matcher{cxxRecordDecl(hasAnyBase(cxxBaseSpecifier(hasType(
/// asString("X"))).bind("b")))} matches
/// \match{class Z : public virtual X {}},
/// with \matcher{type=sub$cxxBaseSpecifier(...)}
/// matching \match{sub=b$public virtual X}.
///
/// Given
/// \code
/// class Base {};
/// class Derived : Base {};
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(hasAnyBase(hasType(cxxRecordDecl(hasName("Base")))))}
/// matches \match{class Derived : Base {}}.
///
/// Usable as: Matcher<Expr>, Matcher<FriendDecl>, Matcher<ValueDecl>,
/// Matcher<CXXBaseSpecifier>
AST_POLYMORPHIC_MATCHER_P_OVERLOAD(
    hasType,
    AST_POLYMORPHIC_SUPPORTED_TYPES(Expr, FriendDecl, ValueDecl,
                                    CXXBaseSpecifier),
    internal::Matcher<Decl>, InnerMatcher, 1) {
  QualType QT = internal::getUnderlyingType(Node);
  if (!QT.isNull())
    return qualType(hasDeclaration(InnerMatcher)).matches(QT, Finder, Builder);
  return false;
}

/// Matches if the type location of a node matches the inner matcher.
///
/// Given
/// \code
///   int x;
/// \endcode
/// The matcher \matcher{declaratorDecl(hasTypeLoc(loc(asString("int"))))}
/// matches \match{int x}.
///
/// Given
/// \code
/// struct point { point(double, double); };
/// point p = point(1.0, -1.0);
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxTemporaryObjectExpr(hasTypeLoc(loc(asString("point"))))}
/// matches \match{point(1.0, -1.0)}.
///
/// Given
/// \code
/// struct Foo { Foo(int, int); };
/// Foo x = Foo(1, 2);
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxTemporaryObjectExpr(hasTypeLoc(
///                           loc(asString("Foo"))))}
/// matches \match{Foo(1, 2)}.
///
/// Usable as: Matcher<BlockDecl>, Matcher<CXXBaseSpecifier>,
///   Matcher<CXXCtorInitializer>, Matcher<CXXFunctionalCastExpr>,
///   Matcher<CXXNewExpr>, Matcher<CXXTemporaryObjectExpr>,
///   Matcher<CXXUnresolvedConstructExpr>,
///   Matcher<CompoundLiteralExpr>,
///   Matcher<DeclaratorDecl>, Matcher<ExplicitCastExpr>,
///   Matcher<ObjCPropertyDecl>, Matcher<TemplateArgumentLoc>,
///   Matcher<TypedefNameDecl>
AST_POLYMORPHIC_MATCHER_P(
    hasTypeLoc,
    AST_POLYMORPHIC_SUPPORTED_TYPES(
        BlockDecl, CXXBaseSpecifier, CXXCtorInitializer, CXXFunctionalCastExpr,
        CXXNewExpr, CXXTemporaryObjectExpr, CXXUnresolvedConstructExpr,
        CompoundLiteralExpr, DeclaratorDecl, ExplicitCastExpr, ObjCPropertyDecl,
        TemplateArgumentLoc, TypedefNameDecl),
    internal::Matcher<TypeLoc>, Inner) {
  TypeSourceInfo *source = internal::GetTypeSourceInfo(Node);
  if (source == nullptr) {
    // This happens for example for implicit destructors.
    return false;
  }
  return Inner.matches(source->getTypeLoc(), Finder, Builder);
}

/// Matches if the matched type is represented by the given string.
///
/// Given
/// \code
///   class Y { public: void x(); };
///   void z() { Y* y; y->x(); }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMemberCallExpr(on(hasType(asString("Y *"))))}
/// matches \match{y->x()}
AST_MATCHER_P(QualType, asString, std::string, Name) {
  return Name == Node.getAsString();
}

/// Matches if the matched type is a pointer type and the pointee type
///   matches the specified matcher.
///
/// Given
/// \code
///   class Y { public: void x(); };
///   void z() { Y *y; y->x(); }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMemberCallExpr(on(hasType(pointsTo(
///      qualType()))))}
/// matches \match{y->x()}
AST_MATCHER_P(
    QualType, pointsTo, internal::Matcher<QualType>,
    InnerMatcher) {
  return (!Node.isNull() && Node->isAnyPointerType() &&
          InnerMatcher.matches(Node->getPointeeType(), Finder, Builder));
}

/// Matches if the matched type is a pointer type and the pointee type
///   matches the specified matcher.
/// Overloaded to match the pointee type's declaration.
///
/// Given
/// \code
///   class Y { public: void x(); };
///   void z() { Y *y; y->x(); }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMemberCallExpr(on(hasType(pointsTo(
///      cxxRecordDecl(hasName("Y"))))))}
/// matches \match{y->x()}
AST_MATCHER_P_OVERLOAD(QualType, pointsTo, internal::Matcher<Decl>,
                       InnerMatcher, 1) {
  return pointsTo(qualType(hasDeclaration(InnerMatcher)))
      .matches(Node, Finder, Builder);
}

/// Matches if the matched type matches the unqualified desugared
/// type of the matched node.
///
/// For example, in:
/// \code
///   class A {};
///   using B = A;
///   B b;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher
/// \matcher{varDecl(hasType(hasUnqualifiedDesugaredType(recordType())))}
/// matches \match{B b}.
AST_MATCHER_P(Type, hasUnqualifiedDesugaredType, internal::Matcher<Type>,
              InnerMatcher) {
  return InnerMatcher.matches(*Node.getUnqualifiedDesugaredType(), Finder,
                              Builder);
}

/// Matches if the matched type is a reference type and the referenced
/// type matches the specified matcher.
///
/// Given
/// \code
///   class X {
///     void a(X b) {
///       X &x = b;
///       const X &y = b;
///     }
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{varDecl(hasType(references(qualType())))} matches
/// \match{X &x = b} and \match{const X &y = b}.
AST_MATCHER_P(QualType, references, internal::Matcher<QualType>, InnerMatcher) {
  return (!Node.isNull() && Node->isReferenceType() &&
          InnerMatcher.matches(Node->getPointeeType(), Finder, Builder));
}

/// Matches QualTypes whose canonical type matches InnerMatcher.
///
/// Given
/// \code
///   typedef int &int_ref;
///   int a;
///   int_ref b = a;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{varDecl(hasType(qualType(referenceType())))}
/// does not match \nomatch{int_ref b = a},
/// but the matcher
/// \matcher{varDecl(hasType(qualType(hasCanonicalType(referenceType()))))}
/// does match \match{int_ref b = a}.
AST_MATCHER_P(QualType, hasCanonicalType, internal::Matcher<QualType>,
              InnerMatcher) {
  if (Node.isNull())
    return false;
  return InnerMatcher.matches(Node.getCanonicalType(), Finder, Builder);
}

/// Matches if the matched type is a reference type and the referenced
/// type matches the specified matcher.
/// Overloaded to match the referenced type's declaration.
///
/// Given
/// \code
///   class X {
///     void a(X b) {
///       X &x = b;
///       const X &y = b;
///     }
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{varDecl(hasType(references(cxxRecordDecl(hasName("X")))))} matches
/// \match{X &x = b} and \match{const X &y = b}.
AST_MATCHER_P_OVERLOAD(QualType, references, internal::Matcher<Decl>,
                       InnerMatcher, 1) {
  return references(qualType(hasDeclaration(InnerMatcher)))
      .matches(Node, Finder, Builder);
}

/// Matches on the implicit object argument of a member call expression. Unlike
/// `on`, matches the argument directly without stripping away anything.
///
/// Given
/// \code
///   class Y { public: void m(); };
///   Y g();
///   class X : public Y { public: void g(); };
///   void z(Y y, X x) { y.m(); x.m(); x.g(); (g()).m(); }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMemberCallExpr(onImplicitObjectArgument(hasType(
///     cxxRecordDecl(hasName("Y")))))}
/// matches \match{y.m()}, \match{x.m()} and \match{(g()).m()}
/// but does not match \nomatch{x.g()}.
/// The matcher \matcher{cxxMemberCallExpr(on(callExpr()))}
/// matches \match{(g()).m()}, because the parens are ignored.
/// FIXME: should they be ignored? (ignored bc of `on`)
///
/// FIXME: Overload to allow directly matching types?
AST_MATCHER_P(CXXMemberCallExpr, onImplicitObjectArgument,
              internal::Matcher<Expr>, InnerMatcher) {
  const Expr *ExprNode = Node.getImplicitObjectArgument();
  return (ExprNode != nullptr &&
          InnerMatcher.matches(*ExprNode, Finder, Builder));
}

/// Matches if the type of the expression's implicit object argument either
///   matches the InnerMatcher, or is a pointer to a type that matches the
/// InnerMatcher.
///
/// Given
/// \code
///   class Y { public: void m() const; };
///   class X : public Y { public: void g(); };
///   void z() { const Y y; y.m(); const Y *p; p->m(); X x; x.m(); x.g(); }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxMemberCallExpr(thisPointerType(isConstQualified()))}
/// matches \match{y.m()}, \match{x.m()} and \match{p->m()},
/// but not \nomatch{x.g()}.
AST_MATCHER_P_OVERLOAD(CXXMemberCallExpr, thisPointerType,
                       internal::Matcher<QualType>, InnerMatcher, 0) {
  return onImplicitObjectArgument(
      anyOf(hasType(InnerMatcher), hasType(pointsTo(InnerMatcher))))
      .matches(Node, Finder, Builder);
}

/// Overloaded to match the type's declaration.
///
/// Given
/// \code
///   class Y { public: void m(); };
///   class X : public Y { public: void g(); };
///   void z() { Y y; y.m(); Y *p; p->m(); X x; x.m(); x.g(); }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMemberCallExpr(thisPointerType(
///     cxxRecordDecl(hasName("Y"))))}
///   matches \match{y.m()}, \match{p->m()} and \match{x.m()}.
/// The matcher \matcher{cxxMemberCallExpr(thisPointerType(
///     cxxRecordDecl(hasName("X"))))}
///   matches \match{x.g()}.
AST_MATCHER_P_OVERLOAD(CXXMemberCallExpr, thisPointerType,
                       internal::Matcher<Decl>, InnerMatcher, 1) {
  return onImplicitObjectArgument(
      anyOf(hasType(InnerMatcher), hasType(pointsTo(InnerMatcher))))
      .matches(Node, Finder, Builder);
}

/// Matches a DeclRefExpr that refers to a declaration that matches the
/// specified matcher.
///
/// Given
/// \code
///   void foo() {
///     bool x;
///     if (x) {}
///   }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{declRefExpr(to(varDecl(hasName("x"))))}
/// matches \match{x} inside the condition of the if-stmt.
AST_MATCHER_P(DeclRefExpr, to, internal::Matcher<Decl>, InnerMatcher) {
  const Decl *DeclNode = Node.getDecl();
  return (DeclNode != nullptr &&
          InnerMatcher.matches(*DeclNode, Finder, Builder));
}

/// Matches if a node refers to a declaration through a specific
/// using shadow declaration.
///
/// Given
/// \code
///   namespace a { int f(); }
///   using a::f;
///   int x = f();
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{declRefExpr(throughUsingDecl(anything()))}
/// matches \match{f}
///
/// \code
///   namespace a { class X{}; }
///   using a::X;
///   X x;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{typeLoc(loc(usingType(throughUsingDecl(anything()))))}
/// matches \match{X}
///
/// Usable as: Matcher<DeclRefExpr>, Matcher<UsingType>
AST_POLYMORPHIC_MATCHER_P(throughUsingDecl,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(DeclRefExpr,
                                                          UsingType),
                          internal::Matcher<UsingShadowDecl>, Inner) {
  const NamedDecl *FoundDecl = Node.getFoundDecl();
  if (const UsingShadowDecl *UsingDecl = dyn_cast<UsingShadowDecl>(FoundDecl))
    return Inner.matches(*UsingDecl, Finder, Builder);
  return false;
}

/// Matches an \c OverloadExpr if any of the declarations in the set of
/// overloads matches the given matcher.
///
/// Given
/// \code
///   template <typename T> void foo(T);
///   template <typename T> void bar(T);
///   template <typename T> void baz(T t) {
///     foo(t);
///     bar(t);
///   }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{unresolvedLookupExpr(hasAnyDeclaration(
///     functionTemplateDecl(hasName("foo"))))}
/// matches \match{foo} in \c foo(t);
/// but does not match \nomatch{bar} in \c bar(t);
AST_MATCHER_P(OverloadExpr, hasAnyDeclaration, internal::Matcher<Decl>,
              InnerMatcher) {
  return matchesFirstInPointerRange(InnerMatcher, Node.decls_begin(),
                                    Node.decls_end(), Finder,
                                    Builder) != Node.decls_end();
}

/// Matches the Decl of a DeclStmt which has a single declaration.
///
/// Given
/// \code
///   void foo() {
///     int a, b;
///     int c;
///   }
/// \endcode
/// The matcher \matcher{declStmt(hasSingleDecl(anything()))}
/// matches \match{int c;}
/// but does not match \nomatch{int a, b;}
AST_MATCHER_P(DeclStmt, hasSingleDecl, internal::Matcher<Decl>, InnerMatcher) {
  if (Node.isSingleDecl()) {
    const Decl *FoundDecl = Node.getSingleDecl();
    return InnerMatcher.matches(*FoundDecl, Finder, Builder);
  }
  return false;
}

/// Matches a variable declaration that has an initializer expression
/// that matches the given matcher.
///
/// Given
/// \code
///   int y() { return 0; }
///   void foo() {
///     int x = y();
///   }
/// \endcode
/// The matcher \matcher{varDecl(hasInitializer(callExpr()))}
/// matches \match{int x = y()}
AST_MATCHER_P(
    VarDecl, hasInitializer, internal::Matcher<Expr>,
    InnerMatcher) {
  const Expr *Initializer = Node.getAnyInitializer();
  return (Initializer != nullptr &&
          InnerMatcher.matches(*Initializer, Finder, Builder));
}

/// Matches a variable serving as the implicit variable for a lambda init-
/// capture.
///
/// Given
/// \code
/// auto f = [x = 3]() { return x; };
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{varDecl(isInitCapture())}
/// matches \match{x = 3}.
AST_MATCHER(VarDecl, isInitCapture) { return Node.isInitCapture(); }

/// Matches each lambda capture in a lambda expression.
///
/// Given
/// \code
///   int main() {
///     int x;
///     int y;
///     float z;
///     auto f = [=]() { return x + y + z; };
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{lambdaExpr(forEachLambdaCapture(
///     lambdaCapture(capturesVar(
///     varDecl(hasType(isInteger())).bind("captured")))))}
/// matches \match{count=2$[=]() { return x + y + z; }} two times,
/// with \matcher{type=sub$varDecl(hasType(isInteger()))} matching
/// \match{sub=captured$int x} and \match{sub=captured$int y}.
AST_MATCHER_P(LambdaExpr, forEachLambdaCapture,
              internal::Matcher<LambdaCapture>, InnerMatcher) {
  BoundNodesTreeBuilder Result;
  bool Matched = false;
  for (const auto &Capture : Node.captures()) {
    if (Finder->isTraversalIgnoringImplicitNodes() && Capture.isImplicit())
      continue;
    BoundNodesTreeBuilder CaptureBuilder(*Builder);
    if (InnerMatcher.matches(Capture, Finder, &CaptureBuilder)) {
      Matched = true;
      Result.addMatch(CaptureBuilder);
    }
  }
  *Builder = std::move(Result);
  return Matched;
}

/// \brief Matches a static variable with local scope.
///
/// Given
/// \code
/// void f() {
///   int x;
///   static int y;
/// }
/// static int z;
/// \endcode
/// The matcher \matcher{varDecl(isStaticLocal())}
/// matches \match{static int y}.
AST_MATCHER(VarDecl, isStaticLocal) {
  return Node.isStaticLocal();
}

/// Matches a variable declaration that has function scope and is a
/// non-static local variable.
///
/// Given
/// \code
/// void f() {
///   int x;
///   static int y;
/// }
/// int z;
/// \endcode
/// The matcher \matcher{varDecl(hasLocalStorage())}
/// matches \match{int x}.
AST_MATCHER(VarDecl, hasLocalStorage) {
  return Node.hasLocalStorage();
}

/// Matches a variable declaration that does not have local storage.
///
/// Given
/// \code
/// void f() {
///   int x;
///   static int y;
/// }
/// int z;
/// \endcode
/// The matcher \matcher{varDecl(hasGlobalStorage())}
/// matches \match{static int y} and \match{int z}.
AST_MATCHER(VarDecl, hasGlobalStorage) {
  return Node.hasGlobalStorage();
}

/// Matches a variable declaration that has automatic storage duration.
///
/// Given
/// \code
/// void f() {
///   int x;
///   static int y;
///   thread_local int z;
/// }
/// int a;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{varDecl(hasAutomaticStorageDuration())}
/// matches \match{int x}
/// but does not match \nomatch{static int y}, \nomatch{thread_local int z} or
/// \nomatch{int a}
AST_MATCHER(VarDecl, hasAutomaticStorageDuration) {
  return Node.getStorageDuration() == SD_Automatic;
}

/// Matches a variable declaration that has static storage duration.
/// It includes the variable declared at namespace scope and those declared
/// with "static" and "extern" storage class specifiers.
///
/// \code
/// void f() {
///   int x;
///   static int y;
///   thread_local int z;
/// }
/// int a;
/// static int b;
/// extern int c;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{varDecl(hasStaticStorageDuration())}
/// matches \match{static int y}, \match{int a}, \match{static int b} and
/// \match{extern int c}
AST_MATCHER(VarDecl, hasStaticStorageDuration) {
  return Node.getStorageDuration() == SD_Static;
}

/// Matches a variable declaration that has thread storage duration.
///
/// Given
/// \code
/// void f() {
///   int x;
///   static int y;
///   thread_local int z;
/// }
/// int a;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{varDecl(hasThreadStorageDuration())}
/// matches \match{thread_local int z}
/// but does not match \nomatch{int x} or \nomatch{type=name$a}.
AST_MATCHER(VarDecl, hasThreadStorageDuration) {
  return Node.getStorageDuration() == SD_Thread;
}

/// Matches a variable declaration that is an exception variable from
/// a C++ catch block, or an Objective-C \@catch statement.
///
/// Given
/// \code
/// void f(int y) {
///   try {
///   } catch (int x) {
///   }
/// }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{varDecl(isExceptionVariable())}
/// matches \match{int x}.
AST_MATCHER(VarDecl, isExceptionVariable) {
  return Node.isExceptionVariable();
}

/// Checks that a call expression or a constructor call expression has
/// a specific number of arguments (including absent default arguments).
///
/// Given
/// \code
///   void f(int x, int y);
///   void foo() {
///     f(0, 0);
///   }
/// \endcode
/// The matcher \matcher{callExpr(argumentCountIs(2))}
/// matches \match{f(0, 0)}.
AST_POLYMORPHIC_MATCHER_P(argumentCountIs,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(
                              CallExpr, CXXConstructExpr,
                              CXXUnresolvedConstructExpr, ObjCMessageExpr),
                          unsigned, N) {
  unsigned NumArgs = Node.getNumArgs();
  if (!Finder->isTraversalIgnoringImplicitNodes())
    return NumArgs == N;
  while (NumArgs) {
    if (!isa<CXXDefaultArgExpr>(Node.getArg(NumArgs - 1)))
      break;
    --NumArgs;
  }
  return NumArgs == N;
}

/// Checks that a call expression or a constructor call expression has at least
/// the specified number of arguments (including absent default arguments).
///
/// Given
/// \code
///   void f(int x, int y);
///   void g(int x, int y, int z);
///   void foo() {
///     f(0, 0);
///     g(0, 0, 0);
///   }
/// \endcode
/// The matcher \matcher{callExpr(argumentCountAtLeast(2))}
/// matches \match{f(0, 0)} and \match{g(0, 0, 0)}
AST_POLYMORPHIC_MATCHER_P(argumentCountAtLeast,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(
                              CallExpr, CXXConstructExpr,
                              CXXUnresolvedConstructExpr, ObjCMessageExpr),
                          unsigned, N) {
  unsigned NumArgs = Node.getNumArgs();
  if (!Finder->isTraversalIgnoringImplicitNodes())
    return NumArgs >= N;
  while (NumArgs) {
    if (!isa<CXXDefaultArgExpr>(Node.getArg(NumArgs - 1)))
      break;
    --NumArgs;
  }
  return NumArgs >= N;
}

/// Matches the n'th argument of a call expression or a constructor
/// call expression.
///
/// Given
/// \code
///   void x(int) { int y; x(y); }
/// \endcode
/// The matcher \matcher{callExpr(hasArgument(0, declRefExpr().bind("arg")))}
/// matches \match{x(y)},
/// with \matcher{type=sub$declRefExpr()} matching \match{sub="arg"$y}.
AST_POLYMORPHIC_MATCHER_P2(hasArgument,
                           AST_POLYMORPHIC_SUPPORTED_TYPES(
                               CallExpr, CXXConstructExpr,
                               CXXUnresolvedConstructExpr, ObjCMessageExpr),
                           unsigned, N, internal::Matcher<Expr>, InnerMatcher) {
  if (N >= Node.getNumArgs())
    return false;
  const Expr *Arg = Node.getArg(N);
  if (Finder->isTraversalIgnoringImplicitNodes() && isa<CXXDefaultArgExpr>(Arg))
    return false;
  return InnerMatcher.matches(*Arg->IgnoreParenImpCasts(), Finder, Builder);
}

/// Matches the operand that does not contain the parameter pack.
///
/// Given
/// \code
///   template <typename... Args>
///   auto sum(Args... args) {
///       return (0 + ... + args);
///   }
///
///   template <typename... Args>
///   auto multiply(Args... args) {
///       return (args * ... * 1);
///   }
/// \endcode
/// \compile_args{-std=c++17-or-later}
///
/// The matcher \matcher{cxxFoldExpr(hasFoldInit(expr().bind("init")))}
/// matches \match{(0 + ... + args)} and \match{(args * ... * 1)}
/// with \matcher{type=sub$hasFoldInit(expr().bind("init"))} matching
/// \match{sub=init$0} and \match{sub=init$1}.
AST_MATCHER_P(CXXFoldExpr, hasFoldInit, internal::Matcher<Expr>, InnerMacher) {
  const auto *const Init = Node.getInit();
  return Init && InnerMacher.matches(*Init, Finder, Builder);
}

/// Matches the operand that contains the parameter pack.
///
/// Given
/// \code
///   template <typename... Args>
///   auto sum(Args... args) {
///       return (0 + ... + args);
///   }
///
///   template <typename... Args>
///   auto multiply(Args... args) {
///       return (args * ... * 1);
///   }
/// \endcode
/// \compile_args{-std=c++17-or-later}
///
/// The matcher \matcher{cxxFoldExpr(hasPattern(expr().bind("pattern")))}
/// matches \match{(0 + ... + args)} and \match{(args * ... * 1)},
/// with \matcher{type=sub$hasPattern(expr().bind("pattern"))} matching
/// \match{count=2;sub=pattern$args} two times.
AST_MATCHER_P(CXXFoldExpr, hasPattern, internal::Matcher<Expr>, InnerMacher) {
  const Expr *const Pattern = Node.getPattern();
  return Pattern && InnerMacher.matches(*Pattern, Finder, Builder);
}

/// Matches right-folding fold expressions.
///
/// Given
/// \code
///   template <typename... Args>
///   auto sum(Args... args) {
///       return (0 + ... + args);
///   }
///
///   template <typename... Args>
///   auto multiply(Args... args) {
///       return (args * ... * 1);
///   }
/// \endcode
/// \compile_args{-std=c++17-or-later}
///
/// The matcher \matcher{cxxFoldExpr(isRightFold())}
/// matches \match{(args * ... * 1)}.
AST_MATCHER(CXXFoldExpr, isRightFold) { return Node.isRightFold(); }

/// Matches left-folding fold expressions.
///
/// Given
/// \code
///   template <typename... Args>
///   auto sum(Args... args) {
///       return (0 + ... + args);
///   }
///
///   template <typename... Args>
///   auto multiply(Args... args) {
///       return (args * ... * 1);
///   }
/// \endcode
/// \compile_args{-std=c++17-or-later}
///
/// The matcher \matcher{cxxFoldExpr(isLeftFold())}
/// matches \match{(0 + ... + args)}.
AST_MATCHER(CXXFoldExpr, isLeftFold) { return Node.isLeftFold(); }

/// Matches unary fold expressions, i.e. fold expressions without an
/// initializer.
///
/// Given
/// \code
///   template <typename... Args>
///   auto sum(Args... args) {
///       return (0 + ... + args);
///   }
///
///   template <typename... Args>
///   auto multiply(Args... args) {
///       return (args * ...);
///   }
/// \endcode
/// \compile_args{-std=c++17-or-later}
///
/// The matcher \matcher{cxxFoldExpr(isUnaryFold())}
/// matches \match{(args * ...)}, but not \nomatch{(0 + ... + args)}.
AST_MATCHER(CXXFoldExpr, isUnaryFold) { return Node.getInit() == nullptr; }

/// Matches binary fold expressions, i.e. fold expressions with an initializer.
///
/// Given
/// \code
///   template <typename... Args>
///   auto sum(Args... args) {
///       return (0 + ... + args);
///   }
///
///   template <typename... Args>
///   auto multiply(Args... args) {
///       return (args * ...);
///   }
/// \endcode
/// \compile_args{-std=c++17-or-later}
///
/// The matcher \matcher{cxxFoldExpr(isBinaryFold())}
/// matches \match{(0 + ... + args)}.
AST_MATCHER(CXXFoldExpr, isBinaryFold) { return Node.getInit() != nullptr; }

/// Matches the n'th item of an initializer list expression.
///
/// Given
/// \code
///   int y = 42;
///   int x{y};
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{initListExpr(hasInit(0, expr()))}
/// matches \match{{y}}.
AST_MATCHER_P2(InitListExpr, hasInit, unsigned, N, internal::Matcher<Expr>,
               InnerMatcher) {
  return N < Node.getNumInits() &&
         InnerMatcher.matches(*Node.getInit(N), Finder, Builder);
}

/// Matches declaration statements that contain a specific number of
/// declarations.
///
/// Given
/// \code
///   void foo() {
///     int a, b;
///     int c;
///     int d = 2, e;
///   }
/// \endcode
/// The matcher \matcher{declStmt(declCountIs(2))}
/// matches \match{int a, b;} and \match{int d = 2, e;},
/// but does not match \nomatch{int c;}
AST_MATCHER_P(DeclStmt, declCountIs, unsigned, N) {
  return std::distance(Node.decl_begin(), Node.decl_end()) == (ptrdiff_t)N;
}

/// Matches the n'th declaration of a declaration statement.
///
/// Note that this does not work for global declarations because the AST
/// breaks up multiple-declaration DeclStmt's into multiple single-declaration
/// DeclStmt's.
///
/// Given non-global declarations
/// \code
///   void foo() {
///     int a, b = 0;
///     int c;
///     int d = 2, e;
///   }
/// \endcode
/// The matcher \matcher{declStmt(containsDeclaration(
///       0, varDecl(hasInitializer(anything()))))}
/// matches \match{int d = 2, e;}.
/// The matcher \matcher{declStmt(containsDeclaration(1, varDecl()))}
/// matches \match{int a, b = 0;} and \match{int d = 2, e;}
/// but does not match \nomatch{int c;}.
AST_MATCHER_P2(DeclStmt, containsDeclaration, unsigned, N,
               internal::Matcher<Decl>, InnerMatcher) {
  const unsigned NumDecls = std::distance(Node.decl_begin(), Node.decl_end());
  if (N >= NumDecls)
    return false;
  DeclStmt::const_decl_iterator Iterator = Node.decl_begin();
  std::advance(Iterator, N);
  return InnerMatcher.matches(**Iterator, Finder, Builder);
}

/// Matches a C++ catch statement that has a catch-all handler.
///
/// Given
/// \code
///   void foo() {
///     try {}
///     catch (int) {}
///     catch (...) {}
///   }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxCatchStmt(isCatchAll())}
/// matches \match{catch (...) {}}
/// but does not match \nomatch{catch(int)}
AST_MATCHER(CXXCatchStmt, isCatchAll) {
  return Node.getExceptionDecl() == nullptr;
}

/// Matches a constructor initializer.
///
/// Given
/// \code
///   struct Foo {
///     Foo() : foo_(1) { }
///     int foo_;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxConstructorDecl(
///   hasAnyConstructorInitializer(cxxCtorInitializer().bind("ctor_init"))
/// )}
/// matches \match{Foo() : foo_(1) { }},
/// with \matcher{type=sub$cxxCtorInitializer()}
/// matching \match{sub=ctor_init$foo_(1)}.
AST_MATCHER_P(CXXConstructorDecl, hasAnyConstructorInitializer,
              internal::Matcher<CXXCtorInitializer>, InnerMatcher) {
  auto MatchIt = matchesFirstInPointerRange(InnerMatcher, Node.init_begin(),
                                            Node.init_end(), Finder, Builder);
  if (MatchIt == Node.init_end())
    return false;
  return (*MatchIt)->isWritten() || !Finder->isTraversalIgnoringImplicitNodes();
}

/// Matches the field declaration of a constructor initializer.
///
/// Given
/// \code
///   struct Foo {
///     Foo() : foo_(1) { }
///     int foo_;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxConstructorDecl(hasAnyConstructorInitializer(
///     forField(fieldDecl(hasName("foo_")).bind("field"))))}
/// matches \match{Foo() : foo_(1) { }},
/// with \matcher{type=sub$fieldDecl(hasName("foo_"))}
/// matching \match{sub=field$foo_(1)}.
AST_MATCHER_P(CXXCtorInitializer, forField,
              internal::Matcher<FieldDecl>, InnerMatcher) {
  const FieldDecl *NodeAsDecl = Node.getAnyMember();
  return (NodeAsDecl != nullptr &&
      InnerMatcher.matches(*NodeAsDecl, Finder, Builder));
}

/// Matches the initializer expression of a constructor initializer.
///
/// Given
/// \code
///   struct Foo {
///     Foo() : foo_(1) { }
///     int foo_;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxConstructorDecl(hasAnyConstructorInitializer(
///     withInitializer(integerLiteral(equals(1)).bind("literal"))))}
/// matches \match{Foo() : foo_(1) { }},
/// with \matcher{type=sub$integerLiteral(equals(1))} matching
/// \match{sub=literal$1}.
AST_MATCHER_P(CXXCtorInitializer, withInitializer,
              internal::Matcher<Expr>, InnerMatcher) {
  const Expr* NodeAsExpr = Node.getInit();
  return (NodeAsExpr != nullptr &&
      InnerMatcher.matches(*NodeAsExpr, Finder, Builder));
}

/// Matches a constructor initializer if it is explicitly written in
/// code (as opposed to implicitly added by the compiler).
///
/// Given
/// \code
///   struct Bar { explicit Bar(const char*); };
///   struct Foo {
///     Foo() { }
///     Foo(int) : foo_("A") { }
///     Bar foo_{""};
///   };
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher
/// \matcher{cxxConstructorDecl(hasAnyConstructorInitializer(isWritten()))} will
/// match \match{Foo(int) : foo_("A") { }}, but not \nomatch{Foo() { }}
AST_MATCHER(CXXCtorInitializer, isWritten) {
  return Node.isWritten();
}

/// Matches a constructor initializer if it is initializing a base, as
/// opposed to a member.
///
/// Given
/// \code
///   struct B {};
///   struct D : B {
///     int I;
///     D(int i) : I(i) {}
///   };
///   struct E : B {
///     E() : B() {}
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxConstructorDecl(hasAnyConstructorInitializer(isBaseInitializer()))}
/// matches \match{E() : B() {}} and \match{D(int i) : I(i) {}}.
/// The constructor of \c D is matched, because it implicitly has a constructor
/// initializer for \c B .
AST_MATCHER(CXXCtorInitializer, isBaseInitializer) {
  return Node.isBaseInitializer();
}

/// Matches a constructor initializer if it is initializing a member, as
/// opposed to a base.
///
/// Given
/// \code
///   struct B {};
///   struct D : B {
///     int I;
///     D(int i) : I(i) {}
///   };
///   struct E : B {
///     E() : B() {}
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxConstructorDecl(hasAnyConstructorInitializer(isMemberInitializer()))}
///   will match \match{D(int i) : I(i) {}}, but not match \nomatch{E() : B()
///   {}}.
AST_MATCHER(CXXCtorInitializer, isMemberInitializer) {
  return Node.isMemberInitializer();
}

/// Matches any argument of a call expression or a constructor call
/// expression, or an ObjC-message-send expression.
///
/// Given
/// \code
///   void x(int, int, int) { int y = 42; x(1, y, 42); }
/// \endcode
/// The matcher
/// \matcher{callExpr(hasAnyArgument(ignoringImplicit(declRefExpr())))} matches
/// \match{x(1, y, 42)} with hasAnyArgument(...)
///   matching y
///
/// For ObjectiveC, given
/// \code
///   @interface I - (void) f:(int) y; @end
///   void foo(I *i) { [i f:12]; }
/// \endcode
/// \compile_args{-ObjC}
/// The matcher
/// \matcher{objcMessageExpr(hasAnyArgument(integerLiteral(equals(12))))}
/// matches \match{[i f:12]}
AST_POLYMORPHIC_MATCHER_P(hasAnyArgument,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(
                              CallExpr, CXXConstructExpr,
                              CXXUnresolvedConstructExpr, ObjCMessageExpr),
                          internal::Matcher<Expr>, InnerMatcher) {
  for (const Expr *Arg : Node.arguments()) {
    if (Finder->isTraversalIgnoringImplicitNodes() &&
        isa<CXXDefaultArgExpr>(Arg))
      break;
    BoundNodesTreeBuilder Result(*Builder);
    if (InnerMatcher.matches(*Arg, Finder, &Result)) {
      *Builder = std::move(Result);
      return true;
    }
  }
  return false;
}

/// Matches lambda captures.
///
/// Given
/// \code
///   int main() {
///     int x;
///     auto f = [x](){};
///     auto g = [x = 1](){};
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher
/// \matcher{lambdaExpr(hasAnyCapture(lambdaCapture().bind("capture")))},
/// matches \match{[x](){}} and \match{[x = 1](){}},
/// with \matcher{type=sub$lambdaCapture()} matching
/// \match{sub=capture$x} and \match{sub=capture$x = 1}.
extern const internal::VariadicAllOfMatcher<LambdaCapture> lambdaCapture;

/// Matches any capture in a lambda expression.
///
/// Given
/// \code
///   void foo() {
///     int t = 5;
///     auto f = [=](){ return t; };
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{lambdaExpr(hasAnyCapture(lambdaCapture()))} and
/// \matcher{lambdaExpr(hasAnyCapture(lambdaCapture(capturesVar(hasName("t")))))}
///   both match \match{[=](){ return t; }}.
AST_MATCHER_P(LambdaExpr, hasAnyCapture, internal::Matcher<LambdaCapture>,
              InnerMatcher) {
  for (const LambdaCapture &Capture : Node.captures()) {
    clang::ast_matchers::internal::BoundNodesTreeBuilder Result(*Builder);
    if (InnerMatcher.matches(Capture, Finder, &Result)) {
      *Builder = std::move(Result);
      return true;
    }
  }
  return false;
}

/// Matches a `LambdaCapture` that refers to the specified `VarDecl`. The
/// `VarDecl` can be a separate variable that is captured by value or
/// reference, or a synthesized variable if the capture has an initializer.
///
/// Given
/// \code
///   void foo() {
///     int x;
///     auto f = [x](){};
///     auto g = [x = 1](){};
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher
/// \matcher{lambdaExpr(hasAnyCapture(
///     lambdaCapture(capturesVar(hasName("x"))).bind("capture")))}
/// matches \match{[x](){}} and \match{[x = 1](){}}, with
/// \matcher{type=sub$lambdaCapture(capturesVar(hasName("x"))).bind("capture")}
/// matching \match{sub=capture$x} and \match{sub=capture$x = 1}.
AST_MATCHER_P(LambdaCapture, capturesVar, internal::Matcher<ValueDecl>,
              InnerMatcher) {
  if (!Node.capturesVariable())
    return false;
  auto *capturedVar = Node.getCapturedVar();
  return capturedVar && InnerMatcher.matches(*capturedVar, Finder, Builder);
}

/// Matches a `LambdaCapture` that refers to 'this'.
///
/// Given
/// \code
/// class C {
///   int cc;
///   int f() {
///     auto l = [this]() { return cc; };
///     return l();
///   }
/// };
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher
/// \matcher{lambdaExpr(hasAnyCapture(lambdaCapture(capturesThis())))}
/// matches \match{[this]() { return cc; }}.
AST_MATCHER(LambdaCapture, capturesThis) { return Node.capturesThis(); }

/// Matches a constructor call expression which uses list initialization.
///
/// Given
/// \code
///   namespace std {
///     template <typename T>
///     class initializer_list {
///       const T* begin;
///       const T* end;
///     };
///   }
///   template <typename T> class vector {
///     public: vector(std::initializer_list<T>) {}
///   };
///
///   vector<int> a({ 1, 2, 3 });
///   vector<int> b = { 4, 5 };
///   int c[] = { 6, 7 };
///   struct pair { int x; int y; };
///   pair d = { 8, 9 };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++11-or-later}
/// The matcher \matcher{cxxConstructExpr(isListInitialization())}
/// matches \match{{ 4, 5 }}.
AST_MATCHER(CXXConstructExpr, isListInitialization) {
  return Node.isListInitialization();
}

/// Matches a constructor call expression which requires
/// zero initialization.
///
/// Given
/// \code
/// void foo() {
///   struct Foo {
///     double x;
///   };
///   auto Val = Foo();
/// }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxConstructExpr(requiresZeroInitialization())}
/// matches \match{Foo()} because the \c x member has to be zero initialized.
AST_MATCHER(CXXConstructExpr, requiresZeroInitialization) {
  return Node.requiresZeroInitialization();
}

/// Matches the n'th parameter of a function or an ObjC method
/// declaration or a block.
///
/// Given
/// \code
///   class X { void f(int x) {} };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxMethodDecl(hasParameter(0, hasType(asString("int"))))}
/// matches \match{void f(int x) {}}
/// with hasParameter(...)
/// matching int x.
///
/// For ObjectiveC, given
/// \code
///   @interface I - (void) f:(int) y; @end
/// \endcode
/// \compile_args{-ObjC}
///
/// The matcher \matcher{objcMethodDecl(hasParameter(0, hasName("y")))}
/// matches the declaration of method f with hasParameter
/// matching y.
AST_POLYMORPHIC_MATCHER_P2(hasParameter,
                           AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                           ObjCMethodDecl,
                                                           BlockDecl),
                           unsigned, N, internal::Matcher<ParmVarDecl>,
                           InnerMatcher) {
  return (N < Node.parameters().size()
          && InnerMatcher.matches(*Node.parameters()[N], Finder, Builder));
}

/// Matches if the given method declaration declares a member function with an
/// explicit object parameter.
///
/// Given
/// \code
/// struct A {
///  int operator-(this A, int);
///  void fun(this A &&self);
///  static int operator()(int);
///  int operator+(int);
/// };
/// \endcode
/// \compile_args{-std=c++23-or-later}
///
/// The matcher \matcher{cxxMethodDecl(isExplicitObjectMemberFunction())}
/// matches \match{int operator-(this A, int)} and
/// \match{void fun(this A &&self)},
/// but not \nomatch{static int operator()(int)} or
/// \nomatch{int operator+(int)}.
AST_MATCHER(CXXMethodDecl, isExplicitObjectMemberFunction) {
  return Node.isExplicitObjectMemberFunction();
}

/// Matches all arguments and their respective ParmVarDecl.
///
/// Given
/// \code
///   void f(int i);
///   int y;
///   void foo() {
///     f(y);
///   }
/// \endcode
/// The matcher \matcher{callExpr(
///   forEachArgumentWithParam(
///     declRefExpr(to(varDecl(hasName("y")))),
///     parmVarDecl(hasType(isInteger()))
/// ))}
///   matches \match{f(y)};
/// with declRefExpr(...)
///   matching int y
/// and parmVarDecl(...)
///   matching int i
AST_POLYMORPHIC_MATCHER_P2(forEachArgumentWithParam,
                           AST_POLYMORPHIC_SUPPORTED_TYPES(CallExpr,
                                                           CXXConstructExpr),
                           internal::Matcher<Expr>, ArgMatcher,
                           internal::Matcher<ParmVarDecl>, ParamMatcher) {
  BoundNodesTreeBuilder Result;
  // The first argument of an overloaded member operator is the implicit object
  // argument of the method which should not be matched against a parameter, so
  // we skip over it here.
  BoundNodesTreeBuilder Matches;
  unsigned ArgIndex =
      cxxOperatorCallExpr(
          callee(cxxMethodDecl(unless(isExplicitObjectMemberFunction()))))
              .matches(Node, Finder, &Matches)
          ? 1
          : 0;
  int ParamIndex = 0;
  bool Matched = false;
  for (; ArgIndex < Node.getNumArgs(); ++ArgIndex) {
    BoundNodesTreeBuilder ArgMatches(*Builder);
    if (ArgMatcher.matches(*(Node.getArg(ArgIndex)->IgnoreParenCasts()),
                           Finder, &ArgMatches)) {
      BoundNodesTreeBuilder ParamMatches(ArgMatches);
      if (expr(anyOf(cxxConstructExpr(hasDeclaration(cxxConstructorDecl(
                         hasParameter(ParamIndex, ParamMatcher)))),
                     callExpr(callee(functionDecl(
                         hasParameter(ParamIndex, ParamMatcher))))))
              .matches(Node, Finder, &ParamMatches)) {
        Result.addMatch(ParamMatches);
        Matched = true;
      }
    }
    ++ParamIndex;
  }
  *Builder = std::move(Result);
  return Matched;
}

/// Matches all arguments and their respective types for a \c CallExpr or
/// \c CXXConstructExpr. It is very similar to \c forEachArgumentWithParam but
/// it works on calls through function pointers as well.
///
/// The difference is, that function pointers do not provide access to a
/// \c ParmVarDecl, but only the \c QualType for each argument.
///
/// Given
/// \code
///   void f(int i);
///   void foo(int y) {
///     f(y);
///     void (*f_ptr)(int) = f;
///     f_ptr(y);
///   }
/// \endcode
/// The matcher \matcher{callExpr(
///   forEachArgumentWithParamType(
///     declRefExpr(to(varDecl(hasName("y")))),
///     qualType(isInteger()).bind("type")
/// ))}
///   matches \match{f(y)} and \match{f_ptr(y)}
/// with declRefExpr(...)
///   matching int y
/// and qualType(...)
///   matching int
AST_POLYMORPHIC_MATCHER_P2(forEachArgumentWithParamType,
                           AST_POLYMORPHIC_SUPPORTED_TYPES(CallExpr,
                                                           CXXConstructExpr),
                           internal::Matcher<Expr>, ArgMatcher,
                           internal::Matcher<QualType>, ParamMatcher) {
  BoundNodesTreeBuilder Result;
  // The first argument of an overloaded member operator is the implicit object
  // argument of the method which should not be matched against a parameter, so
  // we skip over it here.
  BoundNodesTreeBuilder Matches;
  unsigned ArgIndex =
      cxxOperatorCallExpr(
          callee(cxxMethodDecl(unless(isExplicitObjectMemberFunction()))))
              .matches(Node, Finder, &Matches)
          ? 1
          : 0;
  const FunctionProtoType *FProto = nullptr;

  if (const auto *Call = dyn_cast<CallExpr>(&Node)) {
    if (const auto *Value =
            dyn_cast_or_null<ValueDecl>(Call->getCalleeDecl())) {
      QualType QT = Value->getType().getCanonicalType();

      // This does not necessarily lead to a `FunctionProtoType`,
      // e.g. K&R functions do not have a function prototype.
      if (QT->isFunctionPointerType())
        FProto = QT->getPointeeType()->getAs<FunctionProtoType>();

      if (QT->isMemberFunctionPointerType()) {
        const auto *MP = QT->getAs<MemberPointerType>();
        assert(MP && "Must be member-pointer if its a memberfunctionpointer");
        FProto = MP->getPointeeType()->getAs<FunctionProtoType>();
        assert(FProto &&
               "The call must have happened through a member function "
               "pointer");
      }
    }
  }

  unsigned ParamIndex = 0;
  bool Matched = false;
  unsigned NumArgs = Node.getNumArgs();
  if (FProto && FProto->isVariadic())
    NumArgs = std::min(NumArgs, FProto->getNumParams());

  for (; ArgIndex < NumArgs; ++ArgIndex, ++ParamIndex) {
    BoundNodesTreeBuilder ArgMatches(*Builder);
    if (ArgMatcher.matches(*(Node.getArg(ArgIndex)->IgnoreParenCasts()), Finder,
                           &ArgMatches)) {
      BoundNodesTreeBuilder ParamMatches(ArgMatches);

      // This test is cheaper compared to the big matcher in the next if.
      // Therefore, please keep this order.
      if (FProto && FProto->getNumParams() > ParamIndex) {
        QualType ParamType = FProto->getParamType(ParamIndex);
        if (ParamMatcher.matches(ParamType, Finder, &ParamMatches)) {
          Result.addMatch(ParamMatches);
          Matched = true;
          continue;
        }
      }
      if (expr(anyOf(cxxConstructExpr(hasDeclaration(cxxConstructorDecl(
                         hasParameter(ParamIndex, hasType(ParamMatcher))))),
                     callExpr(callee(functionDecl(
                         hasParameter(ParamIndex, hasType(ParamMatcher)))))))
              .matches(Node, Finder, &ParamMatches)) {
        Result.addMatch(ParamMatches);
        Matched = true;
        continue;
      }
    }
  }
  *Builder = std::move(Result);
  return Matched;
}

/// Matches the ParmVarDecl nodes that are at the N'th position in the parameter
/// list. The parameter list could be that of either a block, function, or
/// objc-method.
///
///
/// Given
///
/// \code
/// void f(int a, int b, int c) {
/// }
/// \endcode
///
/// The matcher \matcher{parmVarDecl(isAtPosition(0))} matches
/// \match{int a}.
///
/// The matcher \matcher{parmVarDecl(isAtPosition(1))}
/// matches \match{int b}.
AST_MATCHER_P(ParmVarDecl, isAtPosition, unsigned, N) {
  const clang::DeclContext *Context = Node.getParentFunctionOrMethod();

  if (const auto *Decl = dyn_cast_or_null<FunctionDecl>(Context))
    return N < Decl->param_size() && Decl->getParamDecl(N) == &Node;
  if (const auto *Decl = dyn_cast_or_null<BlockDecl>(Context))
    return N < Decl->param_size() && Decl->getParamDecl(N) == &Node;
  if (const auto *Decl = dyn_cast_or_null<ObjCMethodDecl>(Context))
    return N < Decl->param_size() && Decl->getParamDecl(N) == &Node;

  return false;
}

/// Matches any parameter of a function or an ObjC method declaration or a
/// block.
///
/// Does not match the 'this' parameter of a method.
///
/// Given
/// \code
///   class X { void f(int x, int y, int z) {} };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxMethodDecl(hasAnyParameter(
///   parmVarDecl(hasName("y")).bind("parm")))}
/// matches \match{void f(int x, int y, int z) {}},
/// with \matcher{type=sub$parmVarDecl(hasName("y"))}
/// matching \match{sub=parm$int y}.
///
/// For ObjectiveC, given
/// \code
///   @interface I - (void) f:(int) y; @end
/// \endcode
/// \compile_args{-ObjC}
///
/// the matcher \matcher{objcMethodDecl(hasAnyParameter(hasName("y")))}
///   matches the declaration of method f with hasParameter
/// matching y.
///
/// For blocks, given
/// \code
///   b = ^(int y) { printf("%d", y) };
/// \endcode
/// \compile_args{-ObjC}
///
/// the matcher \matcher{blockDecl(hasAnyParameter(hasName("y")))}
///   matches the declaration of the block b with hasParameter
/// matching y.
AST_POLYMORPHIC_MATCHER_P(hasAnyParameter,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                          ObjCMethodDecl,
                                                          BlockDecl),
                          internal::Matcher<ParmVarDecl>,
                          InnerMatcher) {
  return matchesFirstInPointerRange(InnerMatcher, Node.param_begin(),
                                    Node.param_end(), Finder,
                                    Builder) != Node.param_end();
}

/// Matches \c FunctionDecls and \c FunctionProtoTypes that have a
/// specific parameter count.
///
/// Given
/// \code
///   void f(int i) {}
///   void g(int i, int j) {}
///   void h(int i, int j);
///   void j(int i);
///   void k(int x, int y, int z, ...);
/// \endcode
/// The matcher \matcher{functionDecl(parameterCountIs(2))}
/// matches \match{void g(int i, int j) {}} and \match{void h(int i, int j)}
/// The matcher \matcher{functionProtoType(parameterCountIs(1))}
/// matches the type \match{type=typestr;count=2$void (int)} of \c f and \c j.
/// The matcher \matcher{functionProtoType(parameterCountIs(3))} matches the
/// type \match{type=typestr$void (int, int, int, ...)} of \c k.
AST_POLYMORPHIC_MATCHER_P(parameterCountIs,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                          FunctionProtoType),
                          unsigned, N) {
  return Node.getNumParams() == N;
}

/// Matches templateSpecializationType, class template specialization,
/// variable template specialization, and function template specialization
/// nodes where the template argument matches the inner matcher. This matcher
/// may produce multiple matches.
///
/// Given
/// \code
///   template <typename T, unsigned N, unsigned M>
///   struct Matrix {};
///
///   constexpr unsigned R = 2;
///   Matrix<int, R * 2, R * 4> M;
///
///   template <typename T, typename U>
///   void f(T&& t, U&& u) {}
///
///   void foo() {
///     bool B = false;
///     f(R, B);
///   }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++11-or-later}
/// The matcher
/// \matcher{templateSpecializationType(forEachTemplateArgument(isExpr(expr().bind("t_arg"))))}
/// matches \match{type=typestr;count=2$Matrix<int, R * 2, R * 4>} twice, with
/// \matcher{type=sub$expr()} matching \match{sub=t_arg$R * 2} and
/// \match{sub=t_arg$R * 4}.
/// The matcher
/// \matcher{functionDecl(forEachTemplateArgument(refersToType(qualType().bind("type"))))}
/// matches the specialization of \match{count=2$void f(T&& t, U&& u) {}} twice
/// for each of the template arguments, with \matcher{type=sub$qualType()}
/// matching
/// \match{sub=type;type=typestr$unsigned} and
/// \match{sub=type;type=typestr$bool}.
AST_POLYMORPHIC_MATCHER_P(
    forEachTemplateArgument,
    AST_POLYMORPHIC_SUPPORTED_TYPES(ClassTemplateSpecializationDecl,
                                    VarTemplateSpecializationDecl, FunctionDecl,
                                    TemplateSpecializationType),
    internal::Matcher<TemplateArgument>, InnerMatcher) {
  ArrayRef<TemplateArgument> TemplateArgs =
      clang::ast_matchers::internal::getTemplateSpecializationArgs(Node);
  clang::ast_matchers::internal::BoundNodesTreeBuilder Result;
  bool Matched = false;
  for (const auto &Arg : TemplateArgs) {
    clang::ast_matchers::internal::BoundNodesTreeBuilder ArgBuilder(*Builder);
    if (InnerMatcher.matches(Arg, Finder, &ArgBuilder)) {
      Matched = true;
      Result.addMatch(ArgBuilder);
    }
  }
  *Builder = std::move(Result);
  return Matched;
}

/// Matches \c FunctionDecls that have a noreturn attribute.
///
/// Given
/// \code
///   void nope();
///   [[noreturn]] void a();
///   __attribute__((noreturn)) void b();
///   struct C { [[noreturn]] void c(); };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{functionDecl(isNoReturn())}
/// matches \match{void a()}, \match{__attribute__((noreturn)) void b()} and
/// \match{void c()} but does not match \nomatch{void nope()}.
AST_MATCHER(FunctionDecl, isNoReturn) { return Node.isNoReturn(); }

/// Matches the return type of a function declaration.
///
/// Given
/// \code
///   class X { int f() { return 1; } };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMethodDecl(returns(asString("int")))}
/// matches \match{int f() { return 1; }}.
AST_MATCHER_P(FunctionDecl, returns,
              internal::Matcher<QualType>, InnerMatcher) {
  return InnerMatcher.matches(Node.getReturnType(), Finder, Builder);
}

/// Matches extern "C" function or variable declarations.
///
/// Given
/// \code
///   extern "C" void f() {}
///   extern "C" { void g() {} }
///   void h() {}
///   extern "C" int x = 1;
///   extern "C" { int y = 2; }
///   int z = 3;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{functionDecl(isExternC())}
/// matches \match{void f() {}} and \match{void g() {}}.
/// The matcher \matcher{varDecl(isExternC())}
/// matches \match{int x = 1} and \match{int y = 2}, but does not
/// match \nomatch{int z = 3}.
AST_POLYMORPHIC_MATCHER(isExternC, AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                                   VarDecl)) {
  return Node.isExternC();
}

/// Matches variable/function declarations that have "static" storage
/// class specifier ("static" keyword) written in the source.
///
/// Given
/// \code
///   static void f() {}
///   static int i = 0;
///   extern int j;
///   int k;
/// \endcode
/// The matcher \matcher{functionDecl(isStaticStorageClass())}
/// matches \match{static void f() {}}.
/// The matcher \matcher{varDecl(isStaticStorageClass())}
/// matches \match{static int i = 0}.
AST_POLYMORPHIC_MATCHER(isStaticStorageClass,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                        VarDecl)) {
  return Node.getStorageClass() == SC_Static;
}

/// Matches deleted function declarations.
///
/// Given
/// \code
///   void Func();
///   void DeletedFunc() = delete;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{functionDecl(isDeleted())}
/// matches \match{void DeletedFunc()},
/// but does not match \nomatch{void Func()}.
AST_MATCHER(FunctionDecl, isDeleted) {
  return Node.isDeleted();
}

/// Matches defaulted function declarations.
///
/// Given
/// \code
///   class A { ~A(); };
///   class B { ~B() = default; };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{functionDecl(isDefaulted())}
///   matches \match{~B() = default},
/// but does not match \nomatch{~A()}.
AST_MATCHER(FunctionDecl, isDefaulted) {
  return Node.isDefaulted();
}

/// Matches weak function declarations.
///
/// Given
/// \code
///   static void f();
///   void g() __attribute__((weak));
/// \endcode
/// The matcher \matcher{functionDecl(isWeak())}
///   matches the weak declaration
/// \match{void g() __attribute__((weak))},
/// but does not match \nomatch{static void foo_v1()}.
AST_MATCHER(FunctionDecl, isWeak) { return Node.isWeak(); }

/// Matches functions that have a dynamic exception specification.
///
/// Given
/// \code
///   void f(int);
///   void g(int) noexcept;
///   void h(int) noexcept(true);
///   void i(int) noexcept(false);
///   void j(int) throw();
///   void k(int) throw(int);
///   void l(int) throw(...);
/// \endcode
/// \compile_args{-std=c++11-14}
/// The matcher \matcher{functionDecl(hasDynamicExceptionSpec())}
/// matches the declarations \match{void j(int) throw()},
/// \match{void k(int) throw(int)}
/// and \match{void l(int) throw(...)},
/// but does not match \nomatch{void f(int)}, \nomatch{void g(int) noexcept},
/// \nomatch{void h(int) noexcept(true)}
/// or \nomatch{void i(int) noexcept(true)}.
/// The matcher
/// \matcher{functionProtoType(hasDynamicExceptionSpec())} matches
/// the type \match{type=typestr$void (int) throw()} of \c j ,
/// the type \match{type=typestr$void (int) throw(int)} of \c k and
/// the type \match{type=typestr$void (int) throw(...)} of \c l .
/// It does not match
/// the type \nomatch{type=typestr$void (int) noexcept} of \c f ,
/// the type \nomatch{type=typestr$void (int) noexcept} of \c g ,
/// the type \nomatch{type=typestr$void (int) noexcept(int)} of \c h or
/// the type \nomatch{type=typestr$void (int) noexcept(...)} of \c i .
AST_POLYMORPHIC_MATCHER(hasDynamicExceptionSpec,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                        FunctionProtoType)) {
  if (const FunctionProtoType *FnTy = internal::getFunctionProtoType(Node))
    return FnTy->hasDynamicExceptionSpec();
  return false;
}

/// Matches functions that have a non-throwing exception specification.
///
/// Given
/// \code
///   void f(int);
///   void g(int) noexcept;
///   void h(int) noexcept(false);
///   void i(int) throw();
///   void j(int) throw(int);
/// \endcode
/// \compile_args{-std=c++11-14}
/// The matcher \matcher{functionDecl(isNoThrow())}
/// matches the declaration \match{void g(int) noexcept}
/// and \match{void i(int) throw()},
/// but does not match \nomatch{void f(int)},
/// \nomatch{void h(int) noexcept(false)}
/// or \nomatch{void j(int) throw(int)}.
/// The matcher
/// \matcher{functionProtoType(isNoThrow())}
/// matches the type \match{type=typestr$void (int) throw()} of \c i
/// and the type \match{type=typestr$void (int) noexcept} of \c g,
/// but does not match
/// the type \nomatch{type=typestr$void (int)} of \c f ,
/// the type \nomatch{type=typestr$void (int) noexcept(false)} of \c h or
/// the type \nomatch{type=typestr$void (int) throw(int)} of \c j .
AST_POLYMORPHIC_MATCHER(isNoThrow,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                        FunctionProtoType)) {
  const FunctionProtoType *FnTy = internal::getFunctionProtoType(Node);

  // If the function does not have a prototype, then it is assumed to be a
  // throwing function (as it would if the function did not have any exception
  // specification).
  if (!FnTy)
    return false;

  // Assume the best for any unresolved exception specification.
  if (isUnresolvedExceptionSpec(FnTy->getExceptionSpecType()))
    return true;

  return FnTy->isNothrow();
}

/// Matches consteval function declarations and if consteval/if ! consteval
/// statements.
///
/// Given
/// \code
///   consteval int a();
///   void b() { if consteval {} }
///   void c() { if ! consteval {} }
///   void d() { if ! consteval {} else {} }
/// \endcode
/// \compile_args{-std=c++23-or-later}
/// The matcher \matcher{functionDecl(isConsteval())}
/// matches \match{consteval int a()}.
/// The matcher \matcher{ifStmt(isConsteval())}
/// matches the if statements
/// \match{if consteval {}}, \match{if ! consteval {}} and
/// \match{if ! consteval {} else {}}.
AST_POLYMORPHIC_MATCHER(isConsteval,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl, IfStmt)) {
  return Node.isConsteval();
}

/// Matches constexpr variable and function declarations,
///        and if constexpr.
///
/// Given
/// \code
///   constexpr int foo = 42;
///   constexpr int bar();
///   void baz() { if constexpr(1 > 0) {} }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{varDecl(isConstexpr())}
/// matches \match{constexpr int foo = 42}.
/// The matcher \matcher{functionDecl(isConstexpr())}
/// matches \match{constexpr int bar()}.
/// The matcher \matcher{ifStmt(isConstexpr())}
/// matches \match{if constexpr(1 > 0) {}}.
AST_POLYMORPHIC_MATCHER(isConstexpr,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(VarDecl,
                                                        FunctionDecl,
                                                        IfStmt)) {
  return Node.isConstexpr();
}

/// Matches constinit variable declarations.
///
/// Given
/// \code
///   constinit int foo = 42;
///   constinit const char* bar = "bar";
///   int baz = 42;
///   [[clang::require_constant_initialization]] int xyz = 42;
/// \endcode
/// \compile_args{-std=c++20-or-later}
/// The matcher \matcher{varDecl(isConstinit())}
/// matches the declaration of \match{constinit int foo = 42}
/// and \match{constinit const char* bar = "bar"},
/// but does not match \nomatch{int baz = 42} or
/// \nomatch{[[clang::require_constant_initialization]] int xyz = 42}.
AST_MATCHER(VarDecl, isConstinit) {
  if (const auto *CIA = Node.getAttr<ConstInitAttr>())
    return CIA->isConstinit();
  return false;
}

/// Matches selection statements with initializer.
///
/// Given
/// \code
///  struct vec { int* begin(); int* end(); };
///  int foobar();
///  vec& get_range();
///  void foo() {
///    if (int i = foobar(); i > 0) {}
///    switch (int i = foobar(); i) {}
///    for (auto& a = get_range(); auto& x : a) {}
///  }
///  void bar() {
///    if (foobar() > 0) {}
///    switch (foobar()) {}
///    for (auto& x : get_range()) {}
///  }
/// \endcode
/// \compile_args{-std=c++20-or-later}
/// The matcher \matcher{ifStmt(hasInitStatement(anything()))}
///   matches the if statement \match{if (int i = foobar(); i > 0) {}}
///   in foo but not \nomatch{if (foobar() > 0) {}} in bar.
/// The matcher \matcher{switchStmt(hasInitStatement(anything()))}
///   matches the switch statement \match{switch (int i = foobar(); i) {}}
///   in foo but not \nomatch{switch (foobar()) {}} in bar.
/// The matcher \matcher{cxxForRangeStmt(hasInitStatement(anything()))}
///   matches the range for statement
///   \match{for (auto& a = get_range(); auto& x : a) {}} in foo
///   but not \nomatch{for (auto& x : get_range()) {}} in bar.
AST_POLYMORPHIC_MATCHER_P(hasInitStatement,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(IfStmt, SwitchStmt,
                                                          CXXForRangeStmt),
                          internal::Matcher<Stmt>, InnerMatcher) {
  const Stmt *Init = Node.getInit();
  return Init != nullptr && InnerMatcher.matches(*Init, Finder, Builder);
}

/// Matches the condition expression of an if statement, for loop,
/// switch statement or conditional operator.
///
/// Given
/// \code
/// void foo() {
///   if (true) {}
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{ifStmt(hasCondition(cxxBoolLiteral(equals(true))))}
/// \match{if (true) {}}
AST_POLYMORPHIC_MATCHER_P(
    hasCondition,
    AST_POLYMORPHIC_SUPPORTED_TYPES(IfStmt, ForStmt, WhileStmt, DoStmt,
                                    SwitchStmt, AbstractConditionalOperator),
    internal::Matcher<Expr>, InnerMatcher) {
  const Expr *const Condition = Node.getCond();
  return (Condition != nullptr &&
          InnerMatcher.matches(*Condition, Finder, Builder));
}

/// Matches the then-statement of an if statement.
///
/// Given
/// \code
/// void foo() {
///   if (false) true; else false;
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{ifStmt(hasThen(cxxBoolLiteral(equals(true))))}
/// \match{if (false) true; else false}
AST_MATCHER_P(IfStmt, hasThen, internal::Matcher<Stmt>, InnerMatcher) {
  const Stmt *const Then = Node.getThen();
  return (Then != nullptr && InnerMatcher.matches(*Then, Finder, Builder));
}

/// Matches the else-statement of an if statement.
///
/// Given
/// \code
/// void foo() {
///   if (false) false; else true;
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{ifStmt(hasElse(cxxBoolLiteral(equals(true))))}
/// \match{if (false) false; else true}
AST_MATCHER_P(IfStmt, hasElse, internal::Matcher<Stmt>, InnerMatcher) {
  const Stmt *const Else = Node.getElse();
  return (Else != nullptr && InnerMatcher.matches(*Else, Finder, Builder));
}

/// Matches if a node equals a previously bound node.
///
/// Matches a node if it equals the node previously bound to \p ID.
///
/// Given
/// \code
///   class X { int a; int b; };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxRecordDecl(
///     has(fieldDecl(hasName("a"), hasType(type().bind("t")))),
///     has(fieldDecl(hasName("b"), hasType(type(equalsBoundNode("t"))))))}
///   matches \match{class X { int a; int b; }}, as \c a and \c b have the same
///   type.
///
/// Note that when multiple matches are involved via \c forEach* matchers,
/// \c equalsBoundNodes acts as a filter.
/// For example:
/// compoundStmt(
///     forEachDescendant(varDecl().bind("d")),
///     forEachDescendant(declRefExpr(to(decl(equalsBoundNode("d"))))))
/// will trigger a match for each combination of variable declaration
/// and reference to that variable declaration within a compound statement.
AST_POLYMORPHIC_MATCHER_P(equalsBoundNode,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(Stmt, Decl, Type,
                                                          QualType),
                          std::string, ID) {
  // FIXME: Figure out whether it makes sense to allow this
  // on any other node types.
  // For *Loc it probably does not make sense, as those seem
  // unique. For NestedNameSepcifier it might make sense, as
  // those also have pointer identity, but I'm not sure whether
  // they're ever reused.
  internal::NotEqualsBoundNodePredicate Predicate;
  Predicate.ID = ID;
  Predicate.Node = DynTypedNode::create(Node);
  return Builder->removeBindings(Predicate);
}

/// Matches the condition variable statement in an if statement.
///
/// Given
/// \code
/// struct A {};
/// A* GetAPointer();
/// void foo() {
///   if (A* a = GetAPointer()) {}
/// }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{ifStmt(hasConditionVariableStatement(declStmt()))}
/// \match{if (A* a = GetAPointer()) {}}
AST_MATCHER_P(IfStmt, hasConditionVariableStatement,
              internal::Matcher<DeclStmt>, InnerMatcher) {
  const DeclStmt *const DeclarationStatement =
      Node.getConditionVariableDeclStmt();
  return DeclarationStatement != nullptr &&
         InnerMatcher.matches(*DeclarationStatement, Finder, Builder);
}

/// Matches the index expression of an array subscript expression.
///
/// Given
/// \code
///   int i[5];
///   void f() { i[1] = 42; }
/// \endcode
/// The matcher \matcher{arraySubscriptExpr(hasIndex(integerLiteral()))}
///   matches \match{i[1]} with the \c integerLiteral() matching \c 1
AST_MATCHER_P(ArraySubscriptExpr, hasIndex, internal::Matcher<Expr>,
              InnerMatcher) {
  if (const Expr *Expression = Node.getIdx())
    return InnerMatcher.matches(*Expression, Finder, Builder);
  return false;
}

/// Matches the base expression of an array subscript expression.
///
/// Given
/// \code
///   int i[5];
///   void f() { i[1] = 42; }
/// \endcode
/// The matcher \matcher{arraySubscriptExpr(hasBase(implicitCastExpr(
///     hasSourceExpression(declRefExpr()))))}
///   matches \match{i[1]} with the \c declRefExpr() matching \c i
AST_MATCHER_P(ArraySubscriptExpr, hasBase,
              internal::Matcher<Expr>, InnerMatcher) {
  if (const Expr* Expression = Node.getBase())
    return InnerMatcher.matches(*Expression, Finder, Builder);
  return false;
}

/// Matches a 'for', 'while', 'while' statement or a function or coroutine
/// definition that has a given body. Note that in case of functions or
/// coroutines this matcher only matches the definition itself and not the
/// other declarations of the same function or coroutine.
///
/// Given
/// \code
/// void foo() {
///   for (;;) {}
/// }
/// \endcode
/// The matcher \matcher{forStmt(hasBody(compoundStmt().bind("body")))}
/// matches \match{for (;;) {}}
/// with \matcher{type=sub$compoundStmt()}
///   matching \match{sub=body${}}
///
/// Given
/// \code
///   void f();
///   void f() {}
/// \endcode
/// The matcher \matcher{functionDecl(hasBody(compoundStmt().bind("compound")))}
/// \match{void f() {}}
/// with \matcher{type=sub$compoundStmt()}
/// matching \match{sub=compound${}},
/// but it does not match \nomatch{void f();}.
AST_POLYMORPHIC_MATCHER_P(
    hasBody,
    AST_POLYMORPHIC_SUPPORTED_TYPES(DoStmt, ForStmt, WhileStmt, CXXForRangeStmt,
                                    FunctionDecl, CoroutineBodyStmt),
    internal::Matcher<Stmt>, InnerMatcher) {
  if (Finder->isTraversalIgnoringImplicitNodes() && isDefaultedHelper(&Node))
    return false;
  const Stmt *const Statement = internal::GetBodyMatcher<NodeType>::get(Node);
  return (Statement != nullptr &&
          InnerMatcher.matches(*Statement, Finder, Builder));
}

/// Matches a function declaration that has a given body present in the AST.
/// Note that this matcher matches all the declarations of a function whose
/// body is present in the AST.
///
/// Given
/// \code
///   void f();
///   void f() {}
///   void g();
/// \endcode
/// The matcher \matcher{functionDecl(hasAnyBody(compoundStmt().bind("body")))}
/// matches \match{void f() {}} and the declaration \match{void f()},
/// with \matcher{type=sub$compoundStmt()} matching \match{sub=body${}}, but it
/// does not match \nomatch{void g()}.
AST_MATCHER_P(FunctionDecl, hasAnyBody, internal::Matcher<Stmt>, InnerMatcher) {
  const Stmt *const Statement = Node.getBody();
  return (Statement != nullptr &&
          InnerMatcher.matches(*Statement, Finder, Builder));
}

/// Matches compound statements where at least one substatement matches
/// a given matcher. Also matches StmtExprs that have CompoundStmt as children.
///
/// Given
/// \code
/// void foo() { { {}; 1+2; } }
/// \endcode
/// The matcher
/// \matcher{compoundStmt(hasAnySubstatement(compoundStmt().bind("compound")))}
/// \match{{ {}; 1+2; }} and \match{{ { {}; 1+2; } }}
/// with \matcher{type=sub$compoundStmt()}
/// matching \match{sub=compound${}} and \match{sub=compound${ {}; 1+2; }}.
AST_POLYMORPHIC_MATCHER_P(hasAnySubstatement,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(CompoundStmt,
                                                          StmtExpr),
                          internal::Matcher<Stmt>, InnerMatcher) {
  const CompoundStmt *CS = CompoundStmtMatcher<NodeType>::get(Node);
  return CS && matchesFirstInPointerRange(InnerMatcher, CS->body_begin(),
                                          CS->body_end(), Finder,
                                          Builder) != CS->body_end();
}

/// Checks that a compound statement contains a specific number of
/// child statements.
///
/// Example: Given
/// \code
/// void foo() {
///   { for (;;) {} }
/// }
/// \endcode
/// The matcher \matcher{compoundStmt(statementCountIs(0))}
/// \match{{}}
///   but does not match the outer compound statement.
AST_MATCHER_P(CompoundStmt, statementCountIs, unsigned, N) {
  return Node.size() == N;
}

/// Matches literals that are equal to the given value of type ValueT.
///
/// Given
/// \code
/// void f(char, bool, double, int);
/// void foo() {
///   f('\0', false, 3.14, 42);
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{characterLiteral(equals(0U))} matches \match{'\0'}.
/// The matchers \matcher{cxxBoolLiteral(equals(false))} and
/// \matcher{cxxBoolLiteral(equals(0))} match \match{false}.
/// The matcher \matcher{floatLiteral(equals(3.14))} matches \match{3.14}.
/// The matcher \matcher{integerLiteral(equals(42))} matches \match{42}.
///
/// Note that you cannot directly match a negative numeric literal because the
/// minus sign is not part of the literal: It is a unary operator whose operand
/// is the positive numeric literal. Instead, you must use a unaryOperator()
/// matcher to match the minus sign:
///
/// Given
/// \code
///   int val = -1;
/// \endcode
///
/// The matcher \matcher{unaryOperator(hasOperatorName("-"),
///               hasUnaryOperand(integerLiteral(equals(1))))}
/// matches \match{-1}.
///
/// Usable as: Matcher<CharacterLiteral>, Matcher<CXXBoolLiteralExpr>,
///            Matcher<FloatingLiteral>, Matcher<IntegerLiteral>
template <typename ValueT>
internal::PolymorphicMatcher<internal::ValueEqualsMatcher,
                             void(internal::AllNodeBaseTypes), ValueT>
equals(const ValueT &Value) {
  return internal::PolymorphicMatcher<internal::ValueEqualsMatcher,
                                      void(internal::AllNodeBaseTypes), ValueT>(
      Value);
}

AST_POLYMORPHIC_MATCHER_P_OVERLOAD(equals,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(CharacterLiteral,
                                                          CXXBoolLiteralExpr,
                                                          IntegerLiteral),
                          bool, Value, 0) {
  return internal::ValueEqualsMatcher<NodeType, ParamT>(Value)
    .matchesNode(Node);
}

AST_POLYMORPHIC_MATCHER_P_OVERLOAD(equals,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(CharacterLiteral,
                                                          CXXBoolLiteralExpr,
                                                          IntegerLiteral),
                          unsigned, Value, 1) {
  return internal::ValueEqualsMatcher<NodeType, ParamT>(Value)
    .matchesNode(Node);
}

AST_POLYMORPHIC_MATCHER_P_OVERLOAD(equals,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(CharacterLiteral,
                                                          CXXBoolLiteralExpr,
                                                          FloatingLiteral,
                                                          IntegerLiteral),
                          double, Value, 2) {
  return internal::ValueEqualsMatcher<NodeType, ParamT>(Value)
    .matchesNode(Node);
}

/// Matches the operator Name of operator expressions and fold expressions
/// (binary or unary).
///
/// Given
/// \code
//// void foo(bool a, bool b) {
///   !(a || b);
///  }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{binaryOperator(hasOperatorName("||"))}
/// matches \match{a || b}
///
/// Given
/// \code
///   template <typename... Args>
///   auto sum(Args... args) {
///       return (0 + ... + args);
///   }
/// \endcode
/// \compile_args{-std=c++17-or-later}
/// The matcher \matcher{cxxFoldExpr(hasOperatorName("+"))}
///  matches \match{(0 + ... + args)}.
AST_POLYMORPHIC_MATCHER_P(
    hasOperatorName,
    AST_POLYMORPHIC_SUPPORTED_TYPES(BinaryOperator, CXXOperatorCallExpr,
                                    CXXRewrittenBinaryOperator, CXXFoldExpr,
                                    UnaryOperator),
    std::string, Name) {
  if (std::optional<StringRef> OpName = internal::getOpName(Node))
    return *OpName == Name;
  return false;
}

/// Matches operator expressions (binary or unary) that have any of the
/// specified names.
///
/// It provides a compact way of writing if an operator has any of the specified
/// names:
/// The matcher
///    \c hasAnyOperatorName("+", "-")
/// Is equivalent to
///    \c{anyOf(hasOperatorName("+"), hasOperatorName("-"))}
///
/// Given
/// \code
//// void foo(bool a, bool b) {
///   !(a || b);
///  }
///
//// void bar(bool a, bool b) {
///   a && b;
///  }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{binaryOperator(hasAnyOperatorName("||", "&&"))}
/// matches \match{a || b} and \match{a && b}.
/// The matcher \matcher{unaryOperator(hasAnyOperatorName("-", "!"))}
/// matches \match{!(a || b)}.
extern const internal::VariadicFunction<
    internal::PolymorphicMatcher<internal::HasAnyOperatorNameMatcher,
                                 AST_POLYMORPHIC_SUPPORTED_TYPES(
                                     BinaryOperator, CXXOperatorCallExpr,
                                     CXXRewrittenBinaryOperator, UnaryOperator),
                                 std::vector<std::string>>,
    StringRef, internal::hasAnyOperatorNameFunc>
    hasAnyOperatorName;

/// Matches all kinds of assignment operators.
///
/// Given
/// \code
/// void foo(int a, int b) {
///   if (a == b)
///     a += b;
/// }
/// \endcode
/// The matcher \matcher{binaryOperator(isAssignmentOperator())}
/// matches \match{a += b}.
///
/// Given
/// \code
///   struct S { S& operator=(const S&); };
///   void x() { S s1, s2; s1 = s2; }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxOperatorCallExpr(isAssignmentOperator())}
/// matches \match{s1 = s2}.
AST_POLYMORPHIC_MATCHER(
    isAssignmentOperator,
    AST_POLYMORPHIC_SUPPORTED_TYPES(BinaryOperator, CXXOperatorCallExpr,
                                    CXXRewrittenBinaryOperator)) {
  return Node.isAssignmentOp();
}

/// Matches comparison operators.
///
/// Given
/// \code
/// void foo(int a, int b) {
///   if (a == b)
///     a += b;
/// }
/// \endcode
/// The matcher \matcher{binaryOperator(isComparisonOperator())}
/// matches \match{a == b}
///
/// Given
/// \code
///   struct S { bool operator<(const S& other); };
///   void x(S s1, S s2) { bool b1 = s1 < s2; }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxOperatorCallExpr(isComparisonOperator())}
/// matches \match{s1 < s2}
AST_POLYMORPHIC_MATCHER(
    isComparisonOperator,
    AST_POLYMORPHIC_SUPPORTED_TYPES(BinaryOperator, CXXOperatorCallExpr,
                                    CXXRewrittenBinaryOperator)) {
  return Node.isComparisonOp();
}

/// Matches the left hand side of binary operator expressions.
///
/// Given
/// \code
/// void foo(bool a, bool b) {
///   a || b;
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{binaryOperator(hasLHS(expr().bind("lhs")))}
/// matches \match{a || b},
/// with \matcher{type=sub$expr()}
/// matching \match{sub=lhs$a}.
AST_POLYMORPHIC_MATCHER_P(
    hasLHS,
    AST_POLYMORPHIC_SUPPORTED_TYPES(BinaryOperator, CXXOperatorCallExpr,
                                    CXXRewrittenBinaryOperator,
                                    ArraySubscriptExpr, CXXFoldExpr),
    internal::Matcher<Expr>, InnerMatcher) {
  const Expr *LeftHandSide = internal::getLHS(Node);
  return (LeftHandSide != nullptr &&
          InnerMatcher.matches(*LeftHandSide, Finder, Builder));
}

/// Matches the right hand side of binary operator expressions.
///
/// Given
/// \code
/// void foo(bool a, bool b) {
///   a || b;
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{binaryOperator(hasRHS(expr().bind("rhs")))}
/// matches \match{a || b},
/// with \matcher{type=sub$expr()}
/// matching \match{sub=rhs$b}.
AST_POLYMORPHIC_MATCHER_P(
    hasRHS,
    AST_POLYMORPHIC_SUPPORTED_TYPES(BinaryOperator, CXXOperatorCallExpr,
                                    CXXRewrittenBinaryOperator,
                                    ArraySubscriptExpr, CXXFoldExpr),
    internal::Matcher<Expr>, InnerMatcher) {
  const Expr *RightHandSide = internal::getRHS(Node);
  return (RightHandSide != nullptr &&
          InnerMatcher.matches(*RightHandSide, Finder, Builder));
}

/// Matches if either the left hand side or the right hand side of a
/// binary operator or fold expression matches.
///
/// Given
/// \code
///   struct S {};
///   bool operator ==(const S&, const S&);
///
///   void f(int a, const S&lhs, const S&rhs) {
///       a + 0;
///       lhs == rhs;
///       lhs != rhs;
///   }
///
///   template <typename ...Ts>
///   auto sum(Ts... args) {
///     return (0 + ... + args);
///   }
/// \endcode
/// \compile_args{-std=c++20-or-later}
///
/// The matcher \matcher{binaryOperator(hasEitherOperand(integerLiteral()))}
/// matches \match{a + 0}.
/// The matcher \matcher{cxxOperatorCallExpr(hasEitherOperand(declRefExpr(to(
/// parmVarDecl(hasName("lhs"))))))} matches \match{lhs == rhs} and
/// \match{lhs != rhs}.
/// The matcher \matcher{cxxFoldExpr(hasEitherOperand(integerLiteral()))}
/// matches \match{(0 + ... + args)}.
AST_POLYMORPHIC_MATCHER_P(
    hasEitherOperand,
    AST_POLYMORPHIC_SUPPORTED_TYPES(BinaryOperator, CXXOperatorCallExpr,
                                    CXXFoldExpr, CXXRewrittenBinaryOperator),
    internal::Matcher<Expr>, InnerMatcher) {
  return internal::VariadicDynCastAllOfMatcher<Stmt, NodeType>()(
             anyOf(hasLHS(InnerMatcher), hasRHS(InnerMatcher)))
      .matches(Node, Finder, Builder);
}

/// Matches if both matchers match with opposite sides of the binary operator
/// or fold expression.
///
/// Given
/// \code
/// void foo() {
///   1 + 2; // Match
///   2 + 1; // Match
///   1 + 1; // No match
///   2 + 2; // No match
/// }
/// \endcode
/// The matcher \matcher{binaryOperator(hasOperands(integerLiteral(equals(1)),
///                                             integerLiteral(equals(2))))}
/// matches \match{1 + 2} and \match{2 + 1},
/// but does not match \nomatch{1 + 1}
/// or \nomatch{2 + 2}.
AST_POLYMORPHIC_MATCHER_P2(
    hasOperands,
    AST_POLYMORPHIC_SUPPORTED_TYPES(BinaryOperator, CXXOperatorCallExpr,
                                    CXXFoldExpr, CXXRewrittenBinaryOperator),
    internal::Matcher<Expr>, Matcher1, internal::Matcher<Expr>, Matcher2) {
  return internal::VariadicDynCastAllOfMatcher<Stmt, NodeType>()(
             anyOf(allOf(hasLHS(Matcher1), hasRHS(Matcher2)),
                   allOf(hasRHS(Matcher1), hasLHS(Matcher2))))
      .matches(Node, Finder, Builder);
}

/// Matches if the operand of a unary operator matches.
///
/// \code
/// void foo() {
///   !true;
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher
/// \matcher{unaryOperator(hasUnaryOperand(cxxBoolLiteral(equals(true))))}
/// matches \match{!true}.
AST_POLYMORPHIC_MATCHER_P(hasUnaryOperand,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(UnaryOperator,
                                                          CXXOperatorCallExpr),
                          internal::Matcher<Expr>, InnerMatcher) {
  const Expr *const Operand = internal::getSubExpr(Node);
  return (Operand != nullptr &&
          InnerMatcher.matches(*Operand, Finder, Builder));
}

/// Matches if the cast's source expression
/// or opaque value's source expression matches the given matcher.
///
/// Given
/// \code
///  struct URL { URL(const char*); };
///  URL url = "a string";
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{castExpr(hasSourceExpression(cxxConstructExpr()))}
/// matches \match{"a string"}.
///
/// Given
/// \code
/// void foo(bool b) {
///   int a = b ?: 1;
/// }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher
/// \matcher{opaqueValueExpr(hasSourceExpression(
///               implicitCastExpr(has(
///                 implicitCastExpr(has(declRefExpr()))))))}
/// matches \match{count=2$b} twice, for the condition and the true expression.
AST_POLYMORPHIC_MATCHER_P(hasSourceExpression,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(CastExpr,
                                                          OpaqueValueExpr),
                          internal::Matcher<Expr>, InnerMatcher) {
  const Expr *const SubExpression =
      internal::GetSourceExpressionMatcher<NodeType>::get(Node);
  return (SubExpression != nullptr &&
          InnerMatcher.matches(*SubExpression, Finder, Builder));
}

/// Matches casts that has a given cast kind.
///
/// Given
/// \code
///   int *p = 0;
/// \endcode
/// The matcher \matcher{castExpr(hasCastKind(CK_NullToPointer))}
/// matches the implicit cast around \match{0}
///
/// If the matcher is use from clang-query, CastKind parameter
/// should be passed as a quoted string. e.g., hasCastKind("CK_NullToPointer").
AST_MATCHER_P(CastExpr, hasCastKind, CastKind, Kind) {
  return Node.getCastKind() == Kind;
}

/// Matches casts whose destination type matches a given matcher.
///
/// (Note: Clang's AST refers to other conversions as "casts" too, and calls
/// actual casts "explicit" casts.)
///
/// \code
///   unsigned int a = (unsigned int)0;
/// \endcode
///
/// The matcher \matcher{explicitCastExpr(hasDestinationType(
/// qualType(isUnsignedInteger())))} matches \match{(unsigned int)0}.
AST_MATCHER_P(ExplicitCastExpr, hasDestinationType, internal::Matcher<QualType>,
              InnerMatcher) {
  const QualType NodeType = Node.getTypeAsWritten();
  return InnerMatcher.matches(NodeType, Finder, Builder);
}

/// Matches implicit casts whose destination type matches a given
/// matcher.
///
/// Given
/// \code
///   unsigned int a = 0;
/// \endcode
///
/// The matcher
/// \matcher{implicitCastExpr(hasImplicitDestinationType(
/// qualType(isUnsignedInteger())))} matches \match{0}.
AST_MATCHER_P(ImplicitCastExpr, hasImplicitDestinationType,
              internal::Matcher<QualType>, InnerMatcher) {
  return InnerMatcher.matches(Node.getType(), Finder, Builder);
}

/// Matches TagDecl object that are spelled with "struct."
///
/// Example matches S, but not C, U or E.
/// \code
///   struct S;
///   class C;
///   union U;
///   enum E {};
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{tagDecl(isStruct())} matches \match{struct S},
/// but does not match \nomatch{class C}, \nomatch{union U} or
/// \nomatch{enum E {}}.
AST_MATCHER(TagDecl, isStruct) {
  return Node.isStruct();
}

/// Matches TagDecl object that are spelled with "union."
///
/// Given
/// \code
///   struct S;
///   class C;
///   union U;
///   enum E {};
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{tagDecl(isUnion())} matches \match{union U},
/// but does not match \nomatch{struct S}, \nomatch{class C} or
/// \nomatch{enum E {}}.
AST_MATCHER(TagDecl, isUnion) {
  return Node.isUnion();
}

/// Matches TagDecl object that are spelled with "class."
///
/// Given
/// \code
///   struct S;
///   class C;
///   union U;
///   enum E {};
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{tagDecl(isClass())} matches \match{class C},
/// but does not match \nomatch{struct S}, \nomatch{union U} or
/// \nomatch{enum E {}}.
AST_MATCHER(TagDecl, isClass) {
  return Node.isClass();
}

/// Matches TagDecl object that are spelled with "enum."
///
/// Given
/// \code
///   struct S;
///   class C;
///   union U;
///   enum E {};
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{tagDecl(isEnum())} matches \match{enum E {}},
/// but does not match \nomatch{struct S}, \nomatch{class C} or
/// \nomatch{union U}.
AST_MATCHER(TagDecl, isEnum) {
  return Node.isEnum();
}

/// Matches the true branch expression of a conditional operator.
///
/// Example 1 (conditional ternary operator): matches a
/// Given
/// \code
///   void foo(bool condition, int a, int b) {
///     condition ? a : b;
///   }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher
/// \matcher{conditionalOperator(hasTrueExpression(expr().bind("true")))}
/// matches \match{condition ? a : b},
/// with \matcher{type=sub$expr()} matching \match{sub=true$a}.
///
/// Example 2 (conditional binary operator): matches opaqueValueExpr(condition)
/// Given
/// \code
///   void foo(bool condition, int a, int b) {
///     condition ?: b;
///   }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher \matcher{binaryConditionalOperator(hasTrueExpression(expr()))}
/// matches \match{condition ?: b},
/// with \matcher{type=sub$expr()} matching \match{sub=true$conditoin}.
AST_MATCHER_P(AbstractConditionalOperator, hasTrueExpression,
              internal::Matcher<Expr>, InnerMatcher) {
  const Expr *Expression = Node.getTrueExpr();
  return (Expression != nullptr &&
          InnerMatcher.matches(*Expression, Finder, Builder));
}

/// Matches the false branch expression of a conditional operator
/// (binary or ternary).
///
/// Example matches b
/// \code
///   void foo(bool condition, int a, int b) {
///     condition ? a : b;
///     condition ?: b;
///   }
/// \endcode
/// \compile_args{-std=c++,c23-or-later}
/// The matcher
/// \matcher{conditionalOperator(hasFalseExpression(expr().bind("false")))}
/// matches \match{condition ? a : b},
/// with \matcher{type=sub$expr()} matching \match{sub=false$b}.
/// The matcher
/// \matcher{binaryConditionalOperator(hasFalseExpression(expr().bind("false")))}
/// matches \match{condition ?: b},
/// with \matcher{type=sub$expr()} matching \match{sub=false$b}.
AST_MATCHER_P(AbstractConditionalOperator, hasFalseExpression,
              internal::Matcher<Expr>, InnerMatcher) {
  const Expr *Expression = Node.getFalseExpr();
  return (Expression != nullptr &&
          InnerMatcher.matches(*Expression, Finder, Builder));
}

/// Matches if a declaration has a body attached.
///
/// Example matches A, va, fa
/// \code
///   class A {};
///   class B;  // Doesn't match, as it has no body.
///   int va;
///   extern int vb;  // Doesn't match, as it doesn't define the variable.
///   void fa() {}
///   void fb();  // Doesn't match, as it has no body.
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{tagDecl(isDefinition())}
/// matches \match{class A {}}.
/// The matcher \matcher{varDecl(isDefinition())}
/// matches \match{int va}.
/// The matcher \matcher{functionDecl(isDefinition())}
/// matches \match{void fa() {}}.
///
/// \code
///   @interface X
///   - (void)ma; // Doesn't match, interface is declaration.
///   @end
///   @implementation X
///   - (void)ma {}
///   @end
/// \endcode
/// \compile_args{-ObjC}
/// The matcher \matcher{objcMethodDecl(isDefinition())}
/// matches \match{- (void)ma {}}
///
/// Usable as: Matcher<TagDecl>, Matcher<VarDecl>, Matcher<FunctionDecl>,
///   Matcher<ObjCMethodDecl>
AST_POLYMORPHIC_MATCHER(isDefinition,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(TagDecl, VarDecl,
                                                        ObjCMethodDecl,
                                                        FunctionDecl)) {
  return Node.isThisDeclarationADefinition();
}

/// Matches if a function declaration is variadic.
///
/// Example matches f, but not g or h. The function i will not match, even when
/// compiled in C mode.
/// \code
///   void f(...);
///   void g(int);
///   template <typename... Ts> void h(Ts...);
///   void i();
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{functionDecl(isVariadic())}
/// matches \match{void f(...)},
/// but does not match \nomatch{void g(int)},
/// \nomatch{template <typename... Ts> void h(Ts...)},
/// or \nomatch{void i()}.
AST_MATCHER(FunctionDecl, isVariadic) {
  return Node.isVariadic();
}

/// Matches the class declaration that the given method declaration
/// belongs to.
///
/// FIXME: Generalize this for other kinds of declarations.
/// FIXME: What other kind of declarations would we need to generalize
/// this to?
///
/// Given
/// \code
///   class A {
///    public:
///     A();
///     void foo();
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMethodDecl(ofClass(hasName("A")))}
/// matches \match{A()} and \match{void foo()}.
AST_MATCHER_P(CXXMethodDecl, ofClass,
              internal::Matcher<CXXRecordDecl>, InnerMatcher) {

  ASTChildrenNotSpelledInSourceScope RAII(Finder, false);

  const CXXRecordDecl *Parent = Node.getParent();
  return (Parent != nullptr &&
          InnerMatcher.matches(*Parent, Finder, Builder));
}

/// Matches each method overridden by the given method. This matcher may
/// produce multiple matches.
///
/// Given
/// \code
///   class A { virtual void f(); };
///   class B : public A { void f(); };
///   class C : public B { void f(); };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMethodDecl(ofClass(hasName("C")),
///               forEachOverridden(cxxMethodDecl().bind("b")))}
/// matches \match{void f()} of \c C ,
/// with \matcher{type=sub$cxxMethodDecl()} matching
/// \match{sub=b$virtual void f()} of \c A ,
/// but the matcher does not match \nomatch{void f()} of \c B because
/// it is not overridden by C::f.
///
/// The check can produce multiple matches in case of multiple inheritance, e.g.
/// \code
///   class A1 { virtual void f(); };
///   class A2 { virtual void f(); };
///   class C : public A1, public A2 { void f(); };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMethodDecl(ofClass(hasName("C")),
///               forEachOverridden(cxxMethodDecl().bind("b")))}
/// matches \match{void f()} of \c C with the inner
/// \matcher{type=sub$cxxMethodDecl()} matching \match{sub=b$virtual void f()}
/// inside of \c A1 , and \match{void f()} of \c C with the inner
/// \matcher{type=sub$cxxMethodDecl()} matching \match{sub=b$virtual void f()}
/// inside of \c A2.
AST_MATCHER_P(CXXMethodDecl, forEachOverridden,
              internal::Matcher<CXXMethodDecl>, InnerMatcher) {
  BoundNodesTreeBuilder Result;
  bool Matched = false;
  for (const auto *Overridden : Node.overridden_methods()) {
    BoundNodesTreeBuilder OverriddenBuilder(*Builder);
    const bool OverriddenMatched =
        InnerMatcher.matches(*Overridden, Finder, &OverriddenBuilder);
    if (OverriddenMatched) {
      Matched = true;
      Result.addMatch(OverriddenBuilder);
    }
  }
  *Builder = std::move(Result);
  return Matched;
}

/// Matches declarations of virtual methods and C++ base specifers that specify
/// virtual inheritance.
///
/// Given
/// \code
///   class A {
///    public:
///     virtual void x(); // matches x
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMethodDecl(isVirtual())}
/// matches \match{virtual void x()}.
///
/// Given
/// \code
///   struct Base {};
///   struct DirectlyDerived : virtual Base {}; // matches Base
///   struct IndirectlyDerived : DirectlyDerived, Base {}; // matches Base
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{cxxRecordDecl(hasDirectBase(cxxBaseSpecifier(isVirtual())))}
/// matches \match{struct DirectlyDerived : virtual Base {}}.
///
/// Usable as: Matcher<CXXMethodDecl>, Matcher<CXXBaseSpecifier>
AST_POLYMORPHIC_MATCHER(isVirtual,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(CXXMethodDecl,
                                                        CXXBaseSpecifier)) {
  return Node.isVirtual();
}

/// Matches if the given method declaration has an explicit "virtual".
///
/// Given
/// \code
///   class A {
///    public:
///     virtual void x();
///   };
///   class B : public A {
///    public:
///     void x();
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMethodDecl(isVirtualAsWritten())}
/// matches \match{virtual void x()} of \c A,
/// but does not match \notmatch{void x()} of \c B .
AST_MATCHER(CXXMethodDecl, isVirtualAsWritten) {
  return Node.isVirtualAsWritten();
}

AST_MATCHER(CXXConstructorDecl, isInheritingConstructor) {
  return Node.isInheritingConstructor();
}

/// Matches if the given method or class declaration is final.
///
/// Given
/// \code
///   class A final {};
///
///   struct B {
///     virtual void f();
///   };
///
///   struct C : B {
///     void f() final;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxRecordDecl(isFinal())}
/// matches \match{class A final {}},
/// but does not match \nomatch{type=name$B} or \nomatch{type=name$C}.
/// The matcher \matcher{cxxMethodDecl(isFinal())}
/// matches \match{void f() final} in \c C ,
/// but it does not match \nomatch{virtual void f()} in \c B .
AST_POLYMORPHIC_MATCHER(isFinal,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(CXXRecordDecl,
                                                        CXXMethodDecl)) {
  return Node.template hasAttr<FinalAttr>();
}

/// Matches if the given method declaration is pure.
///
/// Given
/// \code
///   class A {
///    public:
///     virtual void x() = 0;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMethodDecl(isPure())}
/// matches \match{virtual void x() = 0}
AST_MATCHER(CXXMethodDecl, isPure) { return Node.isPureVirtual(); }

/// Matches if the given method declaration is const.
///
/// Given
/// \code
/// struct A {
///   void foo() const;
///   void bar();
/// };
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxMethodDecl(isConst())}
/// matches \match{void foo() const} but not \nomatch{void bar()}.
AST_MATCHER(CXXMethodDecl, isConst) {
  return Node.isConst();
}

/// Matches if the given method declaration declares a copy assignment
/// operator.
///
/// Given
/// \code
/// struct A {
///   A &operator=(const A &);
///   A &operator=(A &&);
/// };
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxMethodDecl(isCopyAssignmentOperator())}
/// matches \match{A &operator=(const A &)}
/// but does not match \nomatch{A &operator=(A &&)}
AST_MATCHER(CXXMethodDecl, isCopyAssignmentOperator) {
  return Node.isCopyAssignmentOperator();
}

/// Matches if the given method declaration declares a move assignment
/// operator.
///
/// Given
/// \code
/// struct A {
///   A &operator=(const A &);
///   A &operator=(A &&);
/// };
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxMethodDecl(isMoveAssignmentOperator())}
/// matches \match{A &operator=(A &&)}
/// but does not match \nomatch{A &operator=(const A &)}
AST_MATCHER(CXXMethodDecl, isMoveAssignmentOperator) {
  return Node.isMoveAssignmentOperator();
}

/// Matches if the given method declaration overrides another method.
///
/// Given
/// \code
///   class A {
///    public:
///     virtual void x();
///   };
///   class B : public A {
///    public:
///     void x() override;
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxMethodDecl(isOverride())}
///   matches \match{void x() override}
AST_MATCHER(CXXMethodDecl, isOverride) {
  return Node.size_overridden_methods() > 0 || Node.hasAttr<OverrideAttr>();
}

/// Matches method declarations that are user-provided.
///
/// Given
/// \code
///   struct S {
///     S(); // #1
///     S(const S &) = default; // #2
///     S(S &&) = delete; // #3
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxConstructorDecl(isUserProvided())}
/// will match \match{S()}, but not \notmatch{S(const S &) = default} or
/// \notmatch{S(S &&) = delete}
AST_MATCHER(CXXMethodDecl, isUserProvided) {
  return Node.isUserProvided();
}

/// Matches member expressions that are called with '->' as opposed
/// to '.'.
///
/// Member calls on the implicit this pointer match as called with '->'.
///
/// Given
/// \code
///   class Y {
///     void x() { this->x(); x(); Y y; y.x(); a; this->b; Y::b; }
///     template <class T> void f() { this->f<T>(); f<T>(); }
///     int a;
///     static int b;
///   };
///   template <class T>
///   class Z {
///     void x() {
///       this->m;
///       this->t;
///       this->t->m;
///     }
///     int m;
///     T* t;
///   };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{memberExpr(isArrow())}
/// matches \match{this->x}, \match{x}, \match{a},
/// \match{this->b}, \match{this->m} and two times \match{count=2$this->t},
/// once for the standalone member expression, and once for the member
/// expression that later accesses \c m .
/// Additionally, it does not match \nomatch{this->t->t}.
/// The matcher \matcher{cxxDependentScopeMemberExpr(isArrow())}
/// matches \match{this->t->m}, but not \nomatch{this->m} or \nomatch{this->t}.
/// The matcher \matcher{unresolvedMemberExpr(isArrow())}
/// matches \match{this->f<T>}, \match{f<T>}
AST_POLYMORPHIC_MATCHER(
    isArrow, AST_POLYMORPHIC_SUPPORTED_TYPES(MemberExpr, UnresolvedMemberExpr,
                                             CXXDependentScopeMemberExpr)) {
  return Node.isArrow();
}

/// Matches QualType nodes that are of integer type.
///
/// Given
/// \code
///   void a(int);
///   void b(long);
///   void c(double);
/// \endcode
/// The matcher \matcher{functionDecl(hasAnyParameter(hasType(isInteger())))}
/// \match{void a(int)}, \match{void b(long)}, but not \nomatch{void c(double)}.
AST_MATCHER(QualType, isInteger) {
    return Node->isIntegerType();
}

/// Matches QualType nodes that are of unsigned integer type.
///
/// Given
/// \code
///   void a(int);
///   void b(unsigned long);
///   void c(double);
/// \endcode
/// The matcher
/// \matcher{functionDecl(hasAnyParameter(hasType(isUnsignedInteger())))}
/// matches \match{void b(unsigned long)},
/// but it does not match \nomatch{void a(int)} and \nomatch{void c(double)}.
AST_MATCHER(QualType, isUnsignedInteger) {
    return Node->isUnsignedIntegerType();
}

/// Matches QualType nodes that are of signed integer type.
///
/// Given
/// \code
///   void a(int);
///   void b(unsigned long);
///   void c(double);
/// \endcode
/// The matcher
/// \matcher{functionDecl(hasAnyParameter(hasType(isSignedInteger())))} matches
/// \match{void a(int)}, but not \notmatch{void b(unsigned long)} or
/// \notmatch{void c(double)}.
AST_MATCHER(QualType, isSignedInteger) {
    return Node->isSignedIntegerType();
}

/// Matches QualType nodes that are of character type.
///
/// Given
/// \code
///   void a(char);
///   void b(wchar_t);
///   void c(double);
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher
/// \matcher{functionDecl(hasAnyParameter(hasType(isAnyCharacter())))}
/// \match{void a(char)}, \match{void b(wchar_t)}, but not
/// \notmatch{void c(double)}.
AST_MATCHER(QualType, isAnyCharacter) {
    return Node->isAnyCharacterType();
}

/// Matches QualType nodes that are of any pointer type; this includes
/// the Objective-C object pointer type, which is different despite being
/// syntactically similar.
///
/// Given
/// \code
///   int *i = nullptr;
///
///   @interface Foo
///   @end
///   Foo *f;
///
///   int j;
/// \endcode
/// \compile_args{-ObjC}
/// The matcher \matcher{varDecl(hasType(isAnyPointer()))}
/// \match{int *i} and \match{Foo *f}, but not \nomatch{int j}.
AST_MATCHER(QualType, isAnyPointer) {
  return Node->isAnyPointerType();
}

/// Matches QualType nodes that are const-qualified, i.e., that
/// include "top-level" const.
///
/// Given
/// \code
///   void a(int);
///   void b(int const);
///   void c(const int);
///   void d(const int*);
/// \endcode
/// The matcher
/// \matcher{functionDecl(hasAnyParameter(hasType(isConstQualified())))}
/// matches \match{void b(int const)} and \match{void c(const int)}.
/// It does not match \notmatch{void d(const int*)} as there
/// is no top-level \c const on the parameter type \c{const int *}.
AST_MATCHER(QualType, isConstQualified) {
  return Node.isConstQualified();
}

/// Matches QualType nodes that are volatile-qualified, i.e., that
/// include "top-level" volatile.
///
/// Given
/// \code
///   void a(int);
///   void b(int volatile);
///   void c(volatile int);
///   void d(volatile int*);
/// \endcode
/// The matcher
/// \matcher{functionDecl(hasAnyParameter(hasType(isVolatileQualified())))}
/// matches \match{void b(int volatile)} and \match{void c(volatile int)}.
/// It does not match \notmatch{void d(volatile int*)} as there
/// is no top-level volatile on the parameter type "volatile int *".
AST_MATCHER(QualType, isVolatileQualified) {
  return Node.isVolatileQualified();
}

/// Matches QualType nodes that have local CV-qualifiers attached to
/// the node, not hidden within a typedef.
///
/// Given
/// \code
///   typedef const int const_int;
///   const_int i = 0;
///   int *const j = nullptr;
///   int *volatile k;
///   int m;
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{varDecl(hasType(hasLocalQualifiers()))} matches
/// \match{int *const j = nullptr} and \match{int *volatile k},
///  bot not \notmatch{const_int i} because the const qualifier is not local.
AST_MATCHER(QualType, hasLocalQualifiers) {
  return Node.hasLocalQualifiers();
}

/// Matches a member expression where the member is matched by a
/// given matcher.
///
/// Given
/// \code
///   struct { int first = 0, second = 1; } first, second;
///   int i = second.first;
///   int j = first.second;
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{memberExpr(member(hasName("first")))}
/// matches \match{second.first}
/// but not \notmatch{first.second}.
AST_MATCHER_P(MemberExpr, member,
              internal::Matcher<ValueDecl>, InnerMatcher) {
  return InnerMatcher.matches(*Node.getMemberDecl(), Finder, Builder);
}

/// Matches a member expression where the object expression is matched by a
/// given matcher. Implicit object expressions are included; that is, it matches
/// use of implicit `this`.
///
/// Given
/// \code
///   struct X {
///     int m;
///     int f(X x) { x.m; return m; }
///   };
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher
/// \matcher{memberExpr(hasObjectExpression(hasType(cxxRecordDecl(hasName("X")))))}
/// matches \match{x.m}, but not \nomatch{m}; however,
/// The matcher \matcher{memberExpr(hasObjectExpression(hasType(pointsTo(
/// cxxRecordDecl(hasName("X"))))))}
/// matches \match{m} (aka. this->m), but not \nomatch{x.m}.
AST_POLYMORPHIC_MATCHER_P(
    hasObjectExpression,
    AST_POLYMORPHIC_SUPPORTED_TYPES(MemberExpr, UnresolvedMemberExpr,
                                    CXXDependentScopeMemberExpr),
    internal::Matcher<Expr>, InnerMatcher) {
  if (const auto *E = dyn_cast<UnresolvedMemberExpr>(&Node))
    if (E->isImplicitAccess())
      return false;
  if (const auto *E = dyn_cast<CXXDependentScopeMemberExpr>(&Node))
    if (E->isImplicitAccess())
      return false;
  return InnerMatcher.matches(*Node.getBase(), Finder, Builder);
}

/// Matches any using shadow declaration.
///
/// Given
/// \code
///   namespace X { void b(); }
///   using X::b;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{usingDecl(hasAnyUsingShadowDecl(hasName("b")))}
///   matches \match{using X::b}
AST_MATCHER_P(BaseUsingDecl, hasAnyUsingShadowDecl,
              internal::Matcher<UsingShadowDecl>, InnerMatcher) {
  return matchesFirstInPointerRange(InnerMatcher, Node.shadow_begin(),
                                    Node.shadow_end(), Finder,
                                    Builder) != Node.shadow_end();
}

/// Matches a using shadow declaration where the target declaration is
/// matched by the given matcher.
///
/// Given
/// \code
///   namespace X { int a; void b(); }
///   using X::a;
///   using X::b;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher
/// \matcher{usingDecl(hasAnyUsingShadowDecl(hasTargetDecl(functionDecl())))}
///   matches \match{using X::b}
///   but not \notmatch{using X::a}
AST_MATCHER_P(UsingShadowDecl, hasTargetDecl,
              internal::Matcher<NamedDecl>, InnerMatcher) {
  return InnerMatcher.matches(*Node.getTargetDecl(), Finder, Builder);
}

/// Matches template instantiations of function, class, or static
/// member variable template instantiations.
///
/// Given
/// \code
///   template <typename T> class X {};
///   class A {};
///   X<A> x;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{cxxRecordDecl(hasName("::X"),
/// isTemplateInstantiation())}
/// matches \match{type=typestr$class X<class A>}.
/// \code
///   template <typename T> class X {};
///   class A {};
///   template class X<A>;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{cxxRecordDecl(hasName("::X"),
/// isTemplateInstantiation())}
/// matches \match{template class X<A>}
/// \code
///   template <typename T> class X {};
///   class A {};
///   extern template class X<A>;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{cxxRecordDecl(hasName("::X"),
/// isTemplateInstantiation())}
/// matches \match{extern template class X<A>}
///
/// But given
/// \code
///   template <typename T>  class X {};
///   class A {};
///   template <> class X<A> {};
///   X<A> x;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{cxxRecordDecl(hasName("::X"),
/// isTemplateInstantiation())}
///   \nomatch{} does not match, as X<A> is an explicit template specialization.
///
/// Usable as: Matcher<FunctionDecl>, Matcher<VarDecl>, Matcher<CXXRecordDecl>
AST_POLYMORPHIC_MATCHER(isTemplateInstantiation,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl, VarDecl,
                                                        CXXRecordDecl)) {
  return (Node.getTemplateSpecializationKind() == TSK_ImplicitInstantiation ||
          Node.getTemplateSpecializationKind() ==
              TSK_ExplicitInstantiationDefinition ||
          Node.getTemplateSpecializationKind() ==
              TSK_ExplicitInstantiationDeclaration);
}

/// Matches declarations that are template instantiations or are inside
/// template instantiations.
///
/// Given
/// \code
///   template<typename T> void A(T t) { T i; }
///   void foo() {
///     A(0);
///     A(0U);
///   }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{functionDecl(isInstantiated())}
/// matches the two instantiations of \match{count=2$void A(T t) { T i; }} that
/// are generated for \c int , and for \c{unsigned int}.
AST_MATCHER_FUNCTION(internal::Matcher<Decl>, isInstantiated) {
  auto IsInstantiation = decl(anyOf(cxxRecordDecl(isTemplateInstantiation()),
                                    functionDecl(isTemplateInstantiation())));
  return decl(anyOf(IsInstantiation, hasAncestor(IsInstantiation)));
}

/// Matches statements inside of a template instantiation.
///
/// Given
/// \code
///   int j;
///   template<typename T> void A(T t) { T i; }
///   void foo() {
///     A(0);
///     A(0U);
///   }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{declStmt(isInTemplateInstantiation())}
/// matches \match{count=2$T i;} twice, once for \c int and once for
/// \c{unsigned int}.
/// The matcher \matcher{declStmt(unless(isInTemplateInstantiation()))} will
/// match \match{T i;} once inside the template definition, but not for any of
/// the instantiated bodies.
AST_MATCHER_FUNCTION(internal::Matcher<Stmt>, isInTemplateInstantiation) {
  return stmt(
      hasAncestor(decl(anyOf(cxxRecordDecl(isTemplateInstantiation()),
                             functionDecl(isTemplateInstantiation())))));
}

/// Matches explicit template specializations of function, class, or
/// static member variable template instantiations.
///
/// Given
/// \code
///   template<typename T> void A(T t) { }
///   template<> void A(int N) { }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{functionDecl(isExplicitTemplateSpecialization())}
///   matches the specialization \match{template<> void A(int N) { }}.
///
/// Usable as: Matcher<FunctionDecl>, Matcher<VarDecl>, Matcher<CXXRecordDecl>
AST_POLYMORPHIC_MATCHER(isExplicitTemplateSpecialization,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl, VarDecl,
                                                        CXXRecordDecl)) {
  return (Node.getTemplateSpecializationKind() == TSK_ExplicitSpecialization);
}

/// Matches \c TypeLocs for which the given inner
/// QualType-matcher matches.
///
/// \code
///   int a = 10;
/// \endcode
///
/// The matcher \matcher{typeLoc(loc(qualType(isInteger())))}
/// matches the \match{int} of \c a .
AST_MATCHER_FUNCTION_P_OVERLOAD(internal::BindableMatcher<TypeLoc>, loc,
                                internal::Matcher<QualType>, InnerMatcher, 0) {
  return internal::BindableMatcher<TypeLoc>(
      new internal::TypeLocTypeMatcher(InnerMatcher));
}

/// Matches `QualifiedTypeLoc`s in the clang AST.
///
/// Given
/// \code
///   const int x = 0;
/// \endcode
///
/// The matcher \matcher{qualifiedTypeLoc()}
/// matches the type of the variable declaration \c x . However, the
/// current implementation of \c QualifiedTypeLoc does not store the source
/// locations for the qualifiers of the type \match{int}.
extern const internal::VariadicDynCastAllOfMatcher<TypeLoc, QualifiedTypeLoc>
    qualifiedTypeLoc;

/// Matches `QualifiedTypeLoc`s that have an unqualified `TypeLoc` matching
/// `InnerMatcher`.
///
/// Given
/// \code
///   int* const x = nullptr;
///   const int y = 0;
/// \endcode
/// \compile_args{-std=c++11-or-later,c23-or-later}
///
/// The matcher \matcher{qualifiedTypeLoc(hasUnqualifiedLoc(pointerTypeLoc()))}
/// matches the type \match{int*} of the variable declaration \c{x}, but
/// not \c{y}.
AST_MATCHER_P(QualifiedTypeLoc, hasUnqualifiedLoc, internal::Matcher<TypeLoc>,
              InnerMatcher) {
  return InnerMatcher.matches(Node.getUnqualifiedLoc(), Finder, Builder);
}

/// Matches a function declared with the specified return `TypeLoc`.
///
/// Given
/// \code
///   int f() { return 5; }
///   void g() {}
/// \endcode
/// The matcher
/// \matcher{functionDecl(hasReturnTypeLoc(typeLoc(loc(asString("int")))))}
/// matches the declaration of \match{int f() { return 5; }} with
/// \matcher{type=sub$typeLoc(loc(asString("int")))} matching the spelling of
/// \match{sub=loc$int}, but the matcher does not match \notmatch{void g() {}}.
AST_MATCHER_P(FunctionDecl, hasReturnTypeLoc, internal::Matcher<TypeLoc>,
              ReturnMatcher) {
  auto Loc = Node.getFunctionTypeLoc();
  return Loc && ReturnMatcher.matches(Loc.getReturnLoc(), Finder, Builder);
}

/// Matches pointer `TypeLoc`s.
///
/// Given
/// \code
///   int* x;
/// \endcode
/// The matcher \matcher{pointerTypeLoc()}
///   matches \match{int*}.
extern const internal::VariadicDynCastAllOfMatcher<TypeLoc, PointerTypeLoc>
    pointerTypeLoc;

/// Matches pointer `TypeLoc`s that have a pointee `TypeLoc` matching
/// `PointeeMatcher`.
///
/// Given
/// \code
///   int* x;
/// \endcode
/// The matcher \matcher{pointerTypeLoc(hasPointeeLoc(loc(asString("int"))))}
///   matches \match{int*}.
AST_MATCHER_P(PointerTypeLoc, hasPointeeLoc, internal::Matcher<TypeLoc>,
              PointeeMatcher) {
  return PointeeMatcher.matches(Node.getPointeeLoc(), Finder, Builder);
}

/// Matches reference `TypeLoc`s.
///
/// Given
/// \code
///   int x = 3;
///   int& l = x;
///   int&& r = 3;
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{referenceTypeLoc()}
///   matches \match{int&} and \match{int&&}.
extern const internal::VariadicDynCastAllOfMatcher<TypeLoc, ReferenceTypeLoc>
    referenceTypeLoc;

/// Matches reference `TypeLoc`s that have a referent `TypeLoc` matching
/// `ReferentMatcher`.
///
/// Given
/// \code
///   int x = 3;
///   int& xx = x;
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{referenceTypeLoc(hasReferentLoc(loc(asString("int"))))}
///   matches \match{int&}.
AST_MATCHER_P(ReferenceTypeLoc, hasReferentLoc, internal::Matcher<TypeLoc>,
              ReferentMatcher) {
  return ReferentMatcher.matches(Node.getPointeeLoc(), Finder, Builder);
}

/// Matches template specialization `TypeLoc`s.
///
/// Given
/// \code
///   template <typename T> class C {};
///   C<char> var;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{varDecl(hasTypeLoc(elaboratedTypeLoc(hasNamedTypeLoc(
/// templateSpecializationTypeLoc(typeLoc())))))}
/// matches \match{C<char> var}.
extern const internal::VariadicDynCastAllOfMatcher<
    TypeLoc, TemplateSpecializationTypeLoc>
    templateSpecializationTypeLoc;

/// Matches template specialization `TypeLoc`s, class template specializations,
/// variable template specializations, and function template specializations
/// that have at least one `TemplateArgumentLoc` matching the given
/// `InnerMatcher`.
///
/// Given
/// \code
///   template<typename T> class A {};
///   A<int> a;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{varDecl(hasTypeLoc(elaboratedTypeLoc(hasNamedTypeLoc(
/// templateSpecializationTypeLoc(hasAnyTemplateArgumentLoc(
/// hasTypeLoc(loc(asString("int")))))))))} matches \match{A<int> a}.
AST_POLYMORPHIC_MATCHER_P(
    hasAnyTemplateArgumentLoc,
    AST_POLYMORPHIC_SUPPORTED_TYPES(ClassTemplateSpecializationDecl,
                                    VarTemplateSpecializationDecl, FunctionDecl,
                                    DeclRefExpr, TemplateSpecializationTypeLoc),
    internal::Matcher<TemplateArgumentLoc>, InnerMatcher) {
  auto Args = internal::getTemplateArgsWritten(Node);
  return matchesFirstInRange(InnerMatcher, Args.begin(), Args.end(), Finder,
                             Builder) != Args.end();
  return false;
}

/// Matches template specialization `TypeLoc`s, class template specializations,
/// variable template specializations, and function template specializations
/// where the n'th `TemplateArgumentLoc` matches the given `InnerMatcher`.
///
/// Given
/// \code
///   template<typename T, typename U> class A {};
///   A<double, int> b;
///   A<int, double> c;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{varDecl(hasTypeLoc(elaboratedTypeLoc(hasNamedTypeLoc(
/// templateSpecializationTypeLoc(hasTemplateArgumentLoc(0,
/// hasTypeLoc(loc(asString("double")))))))))}
/// matches \match{A<double, int> b}, but not \notmatch{A<int, double> c}.
AST_POLYMORPHIC_MATCHER_P2(
    hasTemplateArgumentLoc,
    AST_POLYMORPHIC_SUPPORTED_TYPES(ClassTemplateSpecializationDecl,
                                    VarTemplateSpecializationDecl, FunctionDecl,
                                    DeclRefExpr, TemplateSpecializationTypeLoc),
    unsigned, Index, internal::Matcher<TemplateArgumentLoc>, InnerMatcher) {
  auto Args = internal::getTemplateArgsWritten(Node);
  return Index < Args.size() &&
         InnerMatcher.matches(Args[Index], Finder, Builder);
}

/// Matches C or C++ elaborated `TypeLoc`s.
///
/// Given
/// \code
///   struct s {};
///   struct s ss;
/// \endcode
/// The matcher \matcher{elaboratedTypeLoc()}
/// matches the type \match{struct s} of \c ss.
extern const internal::VariadicDynCastAllOfMatcher<TypeLoc, ElaboratedTypeLoc>
    elaboratedTypeLoc;

/// Matches elaborated `TypeLoc`s that have a named `TypeLoc` matching
/// `InnerMatcher`.
///
/// Given
/// \code
///   template <typename T>
///   class C {};
///   class C<int> c;
///
///   class D {};
///   class D d;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{elaboratedTypeLoc(hasNamedTypeLoc(templateSpecializationTypeLoc()))}
///   matches \match{class C<int>}, but not \notmatch{ckass D}
AST_MATCHER_P(ElaboratedTypeLoc, hasNamedTypeLoc, internal::Matcher<TypeLoc>,
              InnerMatcher) {
  return InnerMatcher.matches(Node.getNamedTypeLoc(), Finder, Builder);
}

/// Matches type \c bool.
///
/// Given
/// \code
///  struct S { bool func(); };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{functionDecl(returns(booleanType()))}
/// \match{bool func()}.
AST_MATCHER(Type, booleanType) {
  return Node.isBooleanType();
}

/// Matches type \c void.
///
/// Given
/// \code
///  struct S { void func(); };
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{functionDecl(returns(voidType()))}
/// \match{void func()}.
AST_MATCHER(Type, voidType) {
  return Node.isVoidType();
}

template <typename NodeType>
using AstTypeMatcher = internal::VariadicDynCastAllOfMatcher<Type, NodeType>;

/// Matches builtin Types.
///
/// Given
/// \code
///   enum E { Ok };
///   enum E e;
///   int b;
///   float c;
/// \endcode
/// The matcher \matcher{varDecl(hasType(builtinType()))}
/// matches \match{int b} and \match{float c}.
extern const AstTypeMatcher<BuiltinType> builtinType;

/// Matches all kinds of arrays.
///
/// Given
/// \code
///   int a[] = { 2, 3 };
///   int b[4];
///   void f() { int c[a[0]]; }
/// \endcode
/// The matcher \matcher{arrayType()}
/// \match{type=typestr$int[4]}, \match{type=typestr$int[a[0]]} and
/// \match{type=typestr$int[]};
extern const AstTypeMatcher<ArrayType> arrayType;

/// Matches C99 complex types.
///
/// Given
/// \code
///   _Complex float f;
/// \endcode
/// The matcher \matcher{complexType()}
/// \match{type=typestr$_Complex float}
extern const AstTypeMatcher<ComplexType> complexType;

/// Matches any real floating-point type (float, double, long double).
///
/// Given
/// \code
///   int i;
///   float f;
/// \endcode
/// The matcher \matcher{type(realFloatingPointType())}
/// matches \match{type=typestr$float}
/// but does not match \nomatch{type=typestr$int}.
AST_MATCHER(Type, realFloatingPointType) {
  return Node.isRealFloatingType();
}

/// Matches arrays and C99 complex types that have a specific element
/// type.
///
/// Given
/// \code
///   struct A {};
///   A a[7];
///   int b[7];
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{arrayType(hasElementType(builtinType()))}
/// \match{type=typestr$int[7]}
///
/// Usable as: Matcher<ArrayType>, Matcher<ComplexType>
AST_TYPELOC_TRAVERSE_MATCHER_DECL(hasElementType, getElement,
                                  AST_POLYMORPHIC_SUPPORTED_TYPES(ArrayType,
                                                                  ComplexType));

/// Matches C arrays with a specified constant size.
///
/// Given
/// \code
///   void foo() {
///     int a[2];
///     int b[] = { 2, 3 };
///     int c[b[0]];
///   }
/// \endcode
/// The matcher \matcher{constantArrayType()}
/// \match{type=typestr$int[2]}
extern const AstTypeMatcher<ConstantArrayType> constantArrayType;

/// Matches nodes that have the specified size.
///
/// Given
/// \code
///   int a[42];
///   int b[2 * 21];
///   int c[41], d[43];
///   char *s = "abcd";
///   wchar_t *ws = L"abcd";
///   char *w = "a";
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{constantArrayType(hasSize(42))}
/// matches \match{type=typestr;count=2$int[42]} twice.
/// The matcher \matcher{stringLiteral(hasSize(4))}
/// matches \match{"abcd"} and \match{L"abcd"}.
AST_POLYMORPHIC_MATCHER_P(hasSize,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(ConstantArrayType,
                                                          StringLiteral),
                          unsigned, N) {
  return internal::HasSizeMatcher<NodeType>::hasSize(Node, N);
}

/// Matches C++ arrays whose size is a value-dependent expression.
///
/// Given
/// \code
///   template<typename T, int Size>
///   class array {
///     T data[Size];
///   };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{dependentSizedArrayType()}
/// \match{type=typestr$T[Size]}
extern const AstTypeMatcher<DependentSizedArrayType> dependentSizedArrayType;

/// Matches C++ extended vector type where either the type or size is
/// dependent.
///
/// Given
/// \code
///   template<typename T, int Size>
///   class vector {
///     typedef T __attribute__((ext_vector_type(Size))) type;
///   };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{dependentSizedExtVectorType()}
/// \match{type=typestr$T __attribute__((ext_vector_type(Size)))}
extern const AstTypeMatcher<DependentSizedExtVectorType>
    dependentSizedExtVectorType;

/// Matches C arrays with unspecified size.
///
/// Given
/// \code
///   int a[] = { 2, 3 };
///   int b[42];
///   void f(int c[]) { int d[a[0]]; };
/// \endcode
/// The matcher \matcher{incompleteArrayType()}
/// \match{type=typestr$int[]} and \match{type=typestr$int[]}
extern const AstTypeMatcher<IncompleteArrayType> incompleteArrayType;

/// Matches C arrays with a specified size that is not an
/// integer-constant-expression.
///
/// Given
/// \code
///   void f() {
///     int a[] = { 2, 3 };
///     int b[42];
///     int c[a[0]];
///   }
/// \endcode
/// The matcher \matcher{variableArrayType()}
/// \match{type=typestr$int[a[0]]}
extern const AstTypeMatcher<VariableArrayType> variableArrayType;

/// Matches \c VariableArrayType nodes that have a specific size
/// expression.
///
/// Given
/// \code
///   void f(int b) {
///     int a[b];
///   }
/// \endcode
/// The matcher
/// \matcher{variableArrayType(hasSizeExpr(ignoringImpCasts(declRefExpr(to(
///   varDecl(hasName("b")))))))}
/// matches \match{type=typestr$int[b]}
AST_MATCHER_P(VariableArrayType, hasSizeExpr,
              internal::Matcher<Expr>, InnerMatcher) {
  return InnerMatcher.matches(*Node.getSizeExpr(), Finder, Builder);
}

/// Matches atomic types.
///
/// Given
/// \code
///   _Atomic(int) i;
/// \endcode
/// The matcher \matcher{atomicType()}
/// \match{type=typestr$_Atomic(int)}
extern const AstTypeMatcher<AtomicType> atomicType;

/// Matches atomic types with a specific value type.
///
/// Given
/// \code
///   _Atomic(int) i;
///   _Atomic(float) f;
/// \endcode
/// The matcher \matcher{atomicType(hasValueType(isInteger()))}
/// \match{type=typestr$_Atomic(int)}.
///
/// Usable as: Matcher<AtomicType>
AST_TYPELOC_TRAVERSE_MATCHER_DECL(hasValueType, getValue,
                                  AST_POLYMORPHIC_SUPPORTED_TYPES(AtomicType));

/// Matches types nodes representing C++11 auto types.
///
/// Given
/// \code
///   void foo() {
///     auto n = 4;
///     int v[] = { 2, 3 };
///     for (auto i : v) { };
///   }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{autoType()}
/// matches the \match{type=typestr;count=5$auto} of \c n and \c i ,
/// as well as the auto types for the implicitly generated code of the range-for
/// loop (for the range, the begin iterator and the end iterator).
extern const AstTypeMatcher<AutoType> autoType;

/// Matches types nodes representing C++11 decltype(<expr>) types.
///
/// Given
/// \code
///   short i = 1;
///   int j = 42;
///   decltype(i + j) result = i + j;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{decltypeType()}
/// \match{type=typestr$decltype(i + j)}
extern const AstTypeMatcher<DecltypeType> decltypeType;

/// Matches \c AutoType nodes where the deduced type is a specific type.
///
/// Note: There is no \c TypeLoc for the deduced type and thus no
/// \c getDeducedLoc() matcher.
///
/// Given
/// \code
///   auto a = 1;
///   auto b = 2.0;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher
/// \matcher{varDecl(hasType(autoType(hasDeducedType(isInteger()))))}
/// matches \match{auto a = 1}, but does not match \nomatch{auto b = 2.0}.
///
/// Usable as: Matcher<AutoType>
AST_TYPE_TRAVERSE_MATCHER(hasDeducedType, getDeducedType,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(AutoType));

/// Matches \c DecltypeType or \c UsingType nodes to find the underlying type.
///
/// Given
/// \code
///   decltype(1) a = 1;
///   decltype(2.0) b = 2.0;
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{decltypeType(hasUnderlyingType(isInteger()))}
/// matches the type \match{type=typestr$decltype(1)} of the variable
/// declaration of \c a .
///
/// Usable as: Matcher<DecltypeType>, Matcher<UsingType>
AST_TYPE_TRAVERSE_MATCHER(hasUnderlyingType, getUnderlyingType,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(DecltypeType,
                                                          UsingType));

/// Matches \c FunctionType nodes.
///
/// Given
/// \code
///   int (*f)(int);
///   void g();
/// \endcode
/// The matcher \matcher{functionType()}
/// \match{type=typestr$int (int)} and the type of
/// \match{std=c++,c23-or-later;type=typestr$void (void)} in C++ and in C23 and
/// later. Before C23, the function type for \c f will be matched the same way,
/// but the function type for \c g will match
/// \match{std=c17-or-earlier;type=typestr$void ()}.
extern const AstTypeMatcher<FunctionType> functionType;

/// Matches \c FunctionProtoType nodes.
///
/// Given
/// \code
///   int (*f)(int);
///   void g();
/// \endcode
/// The matcher \matcher{functionProtoType()}
/// matches the type \match{type=typestr$int (int)} of 'f' and the type
/// \match{std=c++,c23-or-later;type=typestr$void (void)} of 'g' in C++ mode.
/// In C, the type \nomatch{std=c;type=typestr$void ()} of 'g' is not
/// matched because it does not contain a prototype.
extern const AstTypeMatcher<FunctionProtoType> functionProtoType;

/// Matches \c ParenType nodes.
///
/// Given
/// \code
///   int (*ptr_to_array)[4];
///   int *array_of_ptrs[4];
/// \endcode
///
/// The matcher \matcher{varDecl(hasType(pointsTo(parenType())))}
/// matches \match{int (*ptr_to_array)[4]}, but not
/// \nomatch{int *array_of_ptrs[4]}.
extern const AstTypeMatcher<ParenType> parenType;

/// Matches \c ParenType nodes where the inner type is a specific type.
///
/// Given
/// \code
///   int (*ptr_to_array)[4];
///   int (*ptr_to_func)(int);
/// \endcode
///
/// The matcher
/// \matcher{varDecl(hasType(pointsTo(parenType(innerType(functionType())))))}
/// matches \match{int (*ptr_to_func)(int)} but not
/// \nomatch{int (*ptr_to_array)[4]}.
///
/// Usable as: Matcher<ParenType>
AST_TYPE_TRAVERSE_MATCHER(innerType, getInnerType,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(ParenType));

/// Matches block pointer types, i.e. types syntactically represented as
/// "void (^)(int)".
///
/// The \c pointee is always required to be a \c FunctionType.
extern const AstTypeMatcher<BlockPointerType> blockPointerType;

/// Matches member pointer types.
/// Given
/// \code
///   struct A { int i; };
///   int A::* ptr = &A::i;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{memberPointerType()}
/// matches \match{type=typestr$int struct A::*}.
extern const AstTypeMatcher<MemberPointerType> memberPointerType;

/// Matches pointer types, but does not match Objective-C object pointer
/// types.
///
/// Given
/// \code
///   typedef int* int_ptr;
///   void foo(char *str,
///            int val,
///            int *val_ptr,
///            int_ptr not_a_ptr,
///            int_ptr *ptr);
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{parmVarDecl(hasType(pointerType()))}
/// matches \match{char *str}, \match{int *val_ptr} and
/// \match{int_ptr *ptr}.
///
/// \code
///   @interface Foo
///   @end
///   Foo *f;
/// \endcode
/// \compile_args{-ObjC}
extern const AstTypeMatcher<PointerType> pointerType;

/// Matches an Objective-C object pointer type, which is different from
/// a pointer type, despite being syntactically similar.
///
/// Given
/// \code
///   int *a;
///
///   @interface Foo
///   @end
///   Foo *f;
/// \endcode
/// \compile_args{-ObjC}
/// The matcher \matcher{pointerType()}
/// matches \match{type=typestr$Foo *}, but does not match
/// \nomatch{type=typestr$int *}.
extern const AstTypeMatcher<ObjCObjectPointerType> objcObjectPointerType;

/// Matches both lvalue and rvalue reference types.
///
/// Given
/// \code
///   int *a;
///   int &b = *a;
///   int &&c = 1;
///   auto &d = b;
///   auto &&e = c;
///   auto &&f = 2;
///   int g = 5;
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{referenceType()} matches the type
/// \match{type=typestr$int &} of \c b , the type \match{type=typestr$int &&} of
/// \c c, the type
/// \match{type=typestr$auto &} \c d, and the type
/// \match{type=typestr;count=2$auto &&} of \c e and \c f.
extern const AstTypeMatcher<ReferenceType> referenceType;

/// Matches lvalue reference types.
///
/// Given
/// \code
///   int *a;
///   int &b = *a;
///   int &&c = 1;
///   auto &d = b;
///   auto &&e = c;
///   auto &&f = 2;
///   int g = 5;
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{lValueReferenceType()} matches the type
/// \match{type=typestr$int &} of \c b and the type \match{type=typestr$auto &}
/// of \c d.
/// FIXME: figure out why auto changechange matches twice
extern const AstTypeMatcher<LValueReferenceType> lValueReferenceType;

/// Matches rvalue reference types.
///
/// Given
/// \code
///   int *a;
///   int &b = *a;
///   int &&c = 1;
///   auto &d = b;
///   auto &&e = c;
///   auto &&f = 2;
///   int g = 5;
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{rValueReferenceType()} matches the type
/// \match{type=typestr$int &&} of \c c and the type
/// \match{type=typestr;count=2$auto &&} of \c e and \c f.
extern const AstTypeMatcher<RValueReferenceType> rValueReferenceType;

/// Narrows PointerType (and similar) matchers to those where the
/// \c pointee matches a given matcher.
///
/// Given
/// \code
///   int *a;
///   const int *b;
///   int * const c = nullptr;
///   const float *f;
/// \endcode
/// \compile_args{-std=c++11-or-later,c23-or-later}
/// The matcher \matcher{pointerType(pointee(isConstQualified(), isInteger()))}
/// matches \match{type=typestr$const int *},
/// but does not match \nomatch{type=typestr$int * const}
/// or \nomatch{type=typestr$const float *}.
///
/// Usable as: Matcher<BlockPointerType>, Matcher<MemberPointerType>,
///   Matcher<PointerType>, Matcher<ReferenceType>
AST_TYPELOC_TRAVERSE_MATCHER_DECL(
    pointee, getPointee,
    AST_POLYMORPHIC_SUPPORTED_TYPES(BlockPointerType, MemberPointerType,
                                    PointerType, ReferenceType));

/// Matches typedef types.
///
/// Given
/// \code
///   typedef int X;
///   X x = 0;
/// \endcode
/// The matcher \matcher{typedefType()}
/// matches \match{type=typestr$X}.
extern const AstTypeMatcher<TypedefType> typedefType;

/// Matches qualified types when the qualifier is applied via a macro.
///
/// Given
/// \code
///   #define CDECL __attribute__((cdecl))
///   typedef void (CDECL *X)();
///   typedef void (__attribute__((cdecl)) *Y)();
/// \endcode
/// The matcher \matcher{macroQualifiedType()}
/// matches the type \match{type=typestr;std=c++,c23-or-later$CDECL void
/// (void)} of the typedef declaration of \c X , unless when in C98-C17, there
/// \match{type=typestr;std=c17-or-earlier$CDECL void ()},
/// but it does not match the type
/// \nomatch{type=typestr$__attribute((cdecl)) void ()} of \c Y .
extern const AstTypeMatcher<MacroQualifiedType> macroQualifiedType;

/// Matches enum types.
///
/// Given
/// \code
///   enum C { Green };
///   enum class S { Red };
///
///   C c;
///   S s;
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{enumType()} matches the type
/// \match{type=typestr$enum C} of \c c ,
/// and the type \match{type=typestr$enum S} of \c s .
extern const AstTypeMatcher<EnumType> enumType;

/// Matches template specialization types.
///
/// Given
/// \code
///   template <typename T>
///   class C { };
///
///   template class C<int>;
///   C<int> intvar;
///   C<char> charvar;
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
///
/// The matcher \matcher{templateSpecializationType()} matches the type
/// \match{type=typestr$C<int>} of the explicit instantiation in \c A and the
/// type \match{type=typestr$C<char>} of the variable declaration in
/// \c B.
extern const AstTypeMatcher<TemplateSpecializationType>
    templateSpecializationType;

/// Matches C++17 deduced template specialization types, e.g. deduced class
/// template types.
///
/// Given
/// \code
///   template <typename T>
///   class C { public: C(T); };
///
///   C c(123);
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++17-or-later}
/// The matcher \matcher{deducedTemplateSpecializationType()} matches the type
/// \match{type=typestr$C} of the declaration of the variable \c c.
extern const AstTypeMatcher<DeducedTemplateSpecializationType>
    deducedTemplateSpecializationType;

/// Matches types nodes representing unary type transformations.
///
/// Given
/// \code
///   template <typename T> struct A {
///     typedef __underlying_type(T) type;
///   };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{unaryTransformType()}
/// matches \match{type=typestr$__underlying_type(T)}
extern const AstTypeMatcher<UnaryTransformType> unaryTransformType;

/// Matches record types (e.g. structs, classes).
///
/// Given
/// \code
///   class C {};
///   struct S {};
///
///   C c;
///   S s;
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{recordType()} matches the type
/// \match{type=typestr;count=3$class C} of the variable declaration of \c c and
/// matches the type \match{type=typestr;count=3$struct S} of the variable
/// declaration of \c s.
/// Both of these types are matched three times, once for the type of the
/// variable, once for the definition of the class, and once for the type of the
/// implicit class declaration.
extern const AstTypeMatcher<RecordType> recordType;

/// Matches tag types (record and enum types).
///
/// Given
/// \code
///   enum E { Ok };
///   class C {};
///
///   E e;
///   C c;
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{tagType()} matches the type
/// \match{type=typestr$enum E} of variable \c e and the type
/// \match{type=typestr;count=3;std=c++$class C} three times, once for the type
/// of the variable \c c , once for the type of the class definition and once of
/// the type in the implicit class declaration.
extern const AstTypeMatcher<TagType> tagType;

/// Matches types specified with an elaborated type keyword or with a
/// qualified name.
///
/// Given
/// \code
///   namespace N {
///     namespace M {
///       class D {};
///     }
///   }
///   class C {};
///
///   C c;
///   N::M::D d;
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{elaboratedType()} matches the type
/// \match{type=typestr;count=3$C} three times. Once for the type of the
/// variable \c c, once for the type of the class definition and once for the
/// type in the implicit class declaration. For \c{class D}, it matches
/// \match{type=typestr$N::M::D} of variable \c d and its class definition and
/// implicit class declaration \match{type=typestr;count=2$D} one time
/// respectively.
extern const AstTypeMatcher<ElaboratedType> elaboratedType;

/// Matches ElaboratedTypes whose qualifier, a NestedNameSpecifier,
///   matches \c InnerMatcher if the qualifier exists.
///
/// Given
/// \code
///   namespace N {
///     namespace M {
///       class D {};
///     }
///   }
///   N::M::D d;
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher
/// \matcher{elaboratedType(hasQualifier(hasPrefix(specifiesNamespace(hasName("N")))))}
///   matches the type \match{type=typestr$N::M::D} of the variable declaration
///   of \c d.
AST_MATCHER_P(ElaboratedType, hasQualifier,
              internal::Matcher<NestedNameSpecifier>, InnerMatcher) {
  if (const NestedNameSpecifier *Qualifier = Node.getQualifier())
    return InnerMatcher.matches(*Qualifier, Finder, Builder);

  return false;
}

/// Matches ElaboratedTypes whose named type matches \c InnerMatcher.
///
/// Given
/// \code
///   namespace N {
///     namespace M {
///       enum E { Ok };
///     }
///   }
///   N::M::E e = N::M::Ok;
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{elaboratedType(namesType(enumType()))}
/// matches the type \match{type=typestr$N::M::E} of the declaration of \c e .
AST_MATCHER_P(ElaboratedType, namesType, internal::Matcher<QualType>,
              InnerMatcher) {
  return InnerMatcher.matches(Node.getNamedType(), Finder, Builder);
}

/// Matches types specified through a using declaration.
///
/// Given
/// \code
///   namespace a { struct S {}; }
///   using a::S;
///   S s;
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{usingType()} matches the type \match{type=typestr$a::S}
/// of the variable declaration of \c s.
extern const AstTypeMatcher<UsingType> usingType;

/// Matches types that represent the result of substituting a type for a
/// template type parameter.
///
/// Given
/// \code
///   template <typename T>
///   void F(T t) {
///     T local;
///     int i = 1 + t;
///   }
///   void f() {
///     F(0);
///   }
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
///
/// The matcher \matcher{varDecl(hasType(substTemplateTypeParmType()))}
/// matches \match{T t} and \match{T local} for the substituted template type
/// \c int in the instantiation of \c F .
extern const AstTypeMatcher<SubstTemplateTypeParmType>
    substTemplateTypeParmType;

/// Matches template type parameter substitutions that have a replacement
/// type that matches the provided matcher.
///
/// Given
/// \code
///   template <typename T>
///   double F(T t);
///   int i;
///   double j = F(i);
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
///
/// The matcher \matcher{substTemplateTypeParmType(hasReplacementType(type()))}
/// matches \match{type=typestr$int}.
AST_TYPE_TRAVERSE_MATCHER(
    hasReplacementType, getReplacementType,
    AST_POLYMORPHIC_SUPPORTED_TYPES(SubstTemplateTypeParmType));

/// Matches template type parameter types.
///
/// Given
/// \code
///   template <typename T> void f(int i);
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher \matcher{templateTypeParmType()} matches \match{type=typestr$T},
/// but does not match \nomatch{type=typestr$int}.
extern const AstTypeMatcher<TemplateTypeParmType> templateTypeParmType;

/// Matches injected class name types.
///
/// Given
/// \code
///   template <typename T> struct S {
///     void f(S s);
///     void g(S<T> s);
///   };
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++}
/// The matcher
/// \matcher{parmVarDecl(hasType(elaboratedType(namesType(injectedClassNameType()))))}
/// matches \match{S s}, but not \notmatch{S<T> s}
extern const AstTypeMatcher<InjectedClassNameType> injectedClassNameType;

/// Matches decayed type
/// \code
///   void f(int i[]) {
///     i[1] = 0;
///   }
/// \endcode
/// The matcher
/// \matcher{valueDecl(hasType(decayedType(hasDecayedType(pointerType()))))}
/// matches \match{int i[]} in declaration of \c{f}.
/// The matcher
/// \matcher{expr(hasType(decayedType(hasDecayedType(pointerType()))))}
/// matches \match{count=2$i} twice, once for the \c DeclRefExpr and oncde for
/// the cast from an l- to an r-value in \c{i[1]}.
///
extern const AstTypeMatcher<DecayedType> decayedType;

/// Matches the decayed type, whoes decayed type matches \c InnerMatcher
///
/// Given
/// \code
///   void f(int i[]) {
///     i[1] = 0;
///   }
/// \endcode
///
/// The matcher \matcher{parmVarDecl(hasType(decayedType()))}
/// matches \match{int i[]}.
AST_MATCHER_P(DecayedType, hasDecayedType, internal::Matcher<QualType>,
              InnerType) {
  return InnerType.matches(Node.getDecayedType(), Finder, Builder);
}

/// Matches declarations whose declaration context, interpreted as a
/// Decl, matches \c InnerMatcher.
///
/// Given
/// \code
///   namespace N {
///     namespace M {
///       class D {};
///     }
///   }
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{cxxRecordDecl(hasDeclContext(namedDecl(hasName("M"))))}
/// matches the definition of \match{class D {}}.
AST_MATCHER_P(Decl, hasDeclContext, internal::Matcher<Decl>, InnerMatcher) {
  const DeclContext *DC = Node.getDeclContext();
  if (!DC) return false;
  return InnerMatcher.matches(*Decl::castFromDeclContext(DC), Finder, Builder);
}

/// Matches nested name specifiers.
///
/// Given
/// \code
///   namespace ns {
///     struct A { static void f(); };
///     void A::f() {}
///     void g() { A::f(); }
///   }
///   ns::A a;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{nestedNameSpecifier()}
/// matches \match{type=name$ns} and both spellings of
/// \match{type=name;count=2$A} in \c A::f() and \c ns::A .
extern const internal::VariadicAllOfMatcher<NestedNameSpecifier>
    nestedNameSpecifier;

/// Same as \c nestedNameSpecifier but matches \c NestedNameSpecifierLoc.
///
/// Given
/// \code
///   namespace ns {
///     struct A { static void f(); };
///     void A::f() {}
///     void g() { A::f(); }
///   }
///   ns::A a;
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{nestedNameSpecifierLoc()} matches
/// \match{count=2$A::} twice for the spellings in \c A::f() and \c ns::A ,
/// and \match{ns::} once.
extern const internal::VariadicAllOfMatcher<NestedNameSpecifierLoc>
    nestedNameSpecifierLoc;

/// Matches \c NestedNameSpecifierLocs for which the given inner
/// NestedNameSpecifier-matcher matches.
///
/// Given
/// \code
///   namespace ns {
///     struct A { static void f(); };
///     void A::f() {}
///     void g() { A::f(); }
///   }
///   ns::A a;
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{nestedNameSpecifierLoc(loc(specifiesType(
/// hasDeclaration(namedDecl(hasName("A"))))))} matches \match{count=2$A::}
/// twice for the spellings in \c A::f() and \c ns::A .
AST_MATCHER_FUNCTION_P_OVERLOAD(
    internal::BindableMatcher<NestedNameSpecifierLoc>, loc,
    internal::Matcher<NestedNameSpecifier>, InnerMatcher, 1) {
  return internal::BindableMatcher<NestedNameSpecifierLoc>(
      new internal::LocMatcher<NestedNameSpecifierLoc, NestedNameSpecifier>(
          InnerMatcher));
}

/// Matches nested name specifiers that specify a type matching the
/// given \c QualType matcher without qualifiers.
///
/// Given
/// \code
///   struct A { struct B { struct C {}; }; };
///   A::B::C c;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{nestedNameSpecifier(specifiesType(
///   hasDeclaration(cxxRecordDecl(hasName("A")))
/// ))}
/// matches the spelling of \match{type=name$A} in \c A::B::C .
AST_MATCHER_P(NestedNameSpecifier, specifiesType,
              internal::Matcher<QualType>, InnerMatcher) {
  if (!Node.getAsType())
    return false;
  return InnerMatcher.matches(QualType(Node.getAsType(), 0), Finder, Builder);
}

/// Matches nested name specifier locs that specify a type matching the
/// given \c TypeLoc.
///
/// Given
/// \code
///   struct A { struct B { struct C {}; }; };
///   A::B::C c;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{nestedNameSpecifierLoc(specifiesTypeLoc(loc(qualType(
///   hasDeclaration(cxxRecordDecl(hasName("A")))))))}
/// matches \match{A::}
AST_MATCHER_P(NestedNameSpecifierLoc, specifiesTypeLoc,
              internal::Matcher<TypeLoc>, InnerMatcher) {
  return Node && Node.getNestedNameSpecifier()->getAsType() &&
         InnerMatcher.matches(Node.getTypeLoc(), Finder, Builder);
}

/// Matches on the prefix of a \c NestedNameSpecifier.
///
/// Given
/// \code
///   struct A { struct B { struct C {}; }; };
///   A::B::C c;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{nestedNameSpecifier(hasPrefix(specifiesType(asString(
/// "struct A"))))} matches \match{type=typestr$struct A::B}
AST_MATCHER_P_OVERLOAD(NestedNameSpecifier, hasPrefix,
                       internal::Matcher<NestedNameSpecifier>, InnerMatcher,
                       0) {
  const NestedNameSpecifier *NextNode = Node.getPrefix();
  if (!NextNode)
    return false;
  return InnerMatcher.matches(*NextNode, Finder, Builder);
}

/// Matches on the prefix of a \c NestedNameSpecifierLoc.
///
/// Given
/// \code
///   struct A { struct B { struct C {}; }; };
///   A::B::C c;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{nestedNameSpecifierLoc(hasPrefix(loc(specifiesType(asString(
/// "struct A")))))} matches \match{A::B::}.
AST_MATCHER_P_OVERLOAD(NestedNameSpecifierLoc, hasPrefix,
                       internal::Matcher<NestedNameSpecifierLoc>, InnerMatcher,
                       1) {
  NestedNameSpecifierLoc NextNode = Node.getPrefix();
  if (!NextNode)
    return false;
  return InnerMatcher.matches(NextNode, Finder, Builder);
}

/// Matches nested name specifiers that specify a namespace matching the
/// given namespace matcher.
///
/// Given
/// \code
///   namespace ns { struct A {}; }
///   ns::A a;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher
/// \matcher{nestedNameSpecifier(specifiesNamespace(hasName("ns")))} matches
/// the spelling of \match{type=name$ns} in \c ns::A .
AST_MATCHER_P(NestedNameSpecifier, specifiesNamespace,
              internal::Matcher<NamespaceDecl>, InnerMatcher) {
  if (!Node.getAsNamespace())
    return false;
  return InnerMatcher.matches(*Node.getAsNamespace(), Finder, Builder);
}

/// Matches attributes.
/// Attributes may be attached with a variety of different syntaxes (including
/// keywords, C++11 attributes, GNU ``__attribute``` and MSVC `__declspec``,
/// and ``#pragma``s). They may also be implicit.
///
/// Given
/// \code
///   struct [[nodiscard]] Foo{};
///   void bar(int * __attribute__((nonnull)) );
///   __declspec(noinline) void baz();
///
///   #pragma omp declare simd
///   int min();
/// \endcode
/// \compile_args{-fdeclspec;-fopenmp}
/// The matcher \matcher{attr()}
/// matches \match{nodiscard}, \match{nonnull}, \match{noinline}, and
/// \match{declare simd}.
extern const internal::VariadicAllOfMatcher<Attr> attr;

/// Overloads for the \c equalsNode matcher.
/// FIXME: Implement for other node types.
/// @{

/// Matches if a node equals another node.
///
/// \c Decl has pointer identity in the AST.
AST_MATCHER_P_OVERLOAD(Decl, equalsNode, const Decl*, Other, 0) {
  return &Node == Other;
}
/// Matches if a node equals another node.
///
/// \c Stmt has pointer identity in the AST.
AST_MATCHER_P_OVERLOAD(Stmt, equalsNode, const Stmt*, Other, 1) {
  return &Node == Other;
}
/// Matches if a node equals another node.
///
/// \c Type has pointer identity in the AST.
AST_MATCHER_P_OVERLOAD(Type, equalsNode, const Type*, Other, 2) {
    return &Node == Other;
}

/// @}

/// Matches each case or default statement belonging to the given switch
/// statement. This matcher may produce multiple matches.
///
/// Given
/// \code
///   void foo() {
///     switch (1) { case 1: case 2: default: switch (2) { case 3: case 4: ; } }
///   }
/// \endcode
/// The matcher
/// \matcher{switchStmt(forEachSwitchCase(caseStmt().bind("c")))}
/// matches four times, matching
/// \match{count=2$switch (1) { case 1: case 2: default: switch (2) { case 3:
/// case 4: ; } }} twice and
/// \match{count=2$switch (2) { case 3: case 4: ; }} twice, with
/// \matcher{type=sub$caseStmt()} matching each of \match{sub=c$case 1:},
/// \match{sub=c$case 2:}, \match{sub=c$case 3:}
/// and \match{sub=c$case 4:}.
AST_MATCHER_P(SwitchStmt, forEachSwitchCase, internal::Matcher<SwitchCase>,
              InnerMatcher) {
  BoundNodesTreeBuilder Result;
  // FIXME: getSwitchCaseList() does not necessarily guarantee a stable
  // iteration order. We should use the more general iterating matchers once
  // they are capable of expressing this matcher (for example, it should ignore
  // case statements belonging to nested switch statements).
  bool Matched = false;
  for (const SwitchCase *SC = Node.getSwitchCaseList(); SC;
       SC = SC->getNextSwitchCase()) {
    BoundNodesTreeBuilder CaseBuilder(*Builder);
    bool CaseMatched = InnerMatcher.matches(*SC, Finder, &CaseBuilder);
    if (CaseMatched) {
      Matched = true;
      Result.addMatch(CaseBuilder);
    }
  }
  *Builder = std::move(Result);
  return Matched;
}

/// Matches each constructor initializer in a constructor definition.
///
/// Given
/// \code
///   class A { A() : i(42), j(42) {} int i; int j; };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxConstructorDecl(forEachConstructorInitializer(
///   forField(fieldDecl().bind("x"))))}
/// matches the constructor of \match{count=2$A() : i(42), j(42) {}} twice, with
/// \matcher{type=sub$fieldDecl()} matching \match{sub=field$i} and
/// \match{sub=field$j} respectively.
AST_MATCHER_P(CXXConstructorDecl, forEachConstructorInitializer,
              internal::Matcher<CXXCtorInitializer>, InnerMatcher) {
  BoundNodesTreeBuilder Result;
  bool Matched = false;
  for (const auto *I : Node.inits()) {
    if (Finder->isTraversalIgnoringImplicitNodes() && !I->isWritten())
      continue;
    BoundNodesTreeBuilder InitBuilder(*Builder);
    if (InnerMatcher.matches(*I, Finder, &InitBuilder)) {
      Matched = true;
      Result.addMatch(InitBuilder);
    }
  }
  *Builder = std::move(Result);
  return Matched;
}

/// Matches constructor declarations that are copy constructors.
///
/// Given
/// \code
///   struct S {
///     S(); // #1
///     S(const S &); // #2
///     S(S &&); // #3
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxConstructorDecl(isCopyConstructor())}
/// matches \match{S(const S &)},
/// but does not match \nomatch{S()} or \nomatch{S(S &&)}.
AST_MATCHER(CXXConstructorDecl, isCopyConstructor) {
  return Node.isCopyConstructor();
}

/// Matches constructor declarations that are move constructors.
///
/// Given
/// \code
///   struct S {
///     S(); // #1
///     S(const S &); // #2
///     S(S &&); // #3
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxConstructorDecl(isMoveConstructor())}
/// matches \match{S(S &&)}
/// but does not match \nomatch{S();} or \nomatch{S(S &&);}
AST_MATCHER(CXXConstructorDecl, isMoveConstructor) {
  return Node.isMoveConstructor();
}

/// Matches constructor declarations that are default constructors.
///
/// Given
/// \code
///   struct S {
///     S(); // #1
///     S(const S &); // #2
///     S(S &&); // #3
///   };
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxConstructorDecl(isDefaultConstructor())}
/// matches \match{S()}
/// but does not match \nomatch{S(const S &);} or \nomatch{S(S &&);}.
AST_MATCHER(CXXConstructorDecl, isDefaultConstructor) {
  return Node.isDefaultConstructor();
}

/// Matches constructors that delegate to another constructor.
///
/// Given
/// \code
///   struct S {
///     S(); // #1
///     S(int) {} // #2
///     S(S &&) : S() {} // #3
///   };
///   S::S() : S(0) {} // #4
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{cxxConstructorDecl(isDelegatingConstructor())}
/// matches \match{S(S &&) : S() {}} and \match{S::S() : S(0) {}},
/// but does not match \nomatch{S()} or \nomatch{S(int)}.
AST_MATCHER(CXXConstructorDecl, isDelegatingConstructor) {
  return Node.isDelegatingConstructor();
}

/// Matches constructor, conversion function, and deduction guide declarations
/// that have an explicit specifier if this explicit specifier is resolved to
/// true.
///
/// Given
/// \code
///   template<bool b>
///   struct S {
///     S(int); // #1
///     explicit S(double); // #2
///     operator int(); // #3
///     explicit operator bool(); // #4
///     explicit(false) S(bool); // # 7
///     explicit(true) S(char); // # 8
///     explicit(b) S(float); // # 9
///   };
///   S(int) -> S<true>; // #5
///   explicit S(double) -> S<false>; // #6
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++20-or-later}
/// The matcher \matcher{cxxConstructorDecl(isExplicit())}
/// matches \match{explicit S(double)}
/// and \match{explicit(true) S(char)}
/// but does not match \nomatch{S(int);}, \nomatch{explicit(false) S(bool);} or
/// \nomatch{explicit(b) S(float)}
/// The matcher \matcher{cxxConversionDecl(isExplicit())}
/// matches \match{explicit operator bool()}
/// but does not match \nomatch{operator int()}.
/// The matcher \matcher{cxxDeductionGuideDecl(isExplicit())}
/// matches the deduction guide \match{explicit S(double) -> S<false>},
/// the implicit copy deduction candiate
/// \match{type=typestr$auto (double) -> S<b>} and
/// the implicitly generated deduction guide for \match{explicit(true) S(char)},
/// but does not match \nomatch{S(int) -> S<true>}.
AST_POLYMORPHIC_MATCHER(isExplicit, AST_POLYMORPHIC_SUPPORTED_TYPES(
                                        CXXConstructorDecl, CXXConversionDecl,
                                        CXXDeductionGuideDecl)) {
  return Node.isExplicit();
}

/// Matches the expression in an explicit specifier if present in the given
/// declaration.
///
/// Given
/// \code
///   template<bool b>
///   struct S {
///     S(int); // #1
///     explicit S(double); // #2
///     operator int(); // #3
///     explicit operator bool(); // #4
///     explicit(false) S(bool); // # 7
///     explicit(true) S(char); // # 8
///     explicit(b) S(float); // # 9
///   };
///   S(int) -> S<true>; // #5
///   explicit S(double) -> S<false>; // #6
/// \endcode
/// \compile_args{-fno-delayed-template-parsing;-std=c++20-or-later}
/// The matcher
/// \matcher{cxxConstructorDecl(hasExplicitSpecifier(constantExpr()))} matches
/// \match{explicit(false) S(bool)} and \match{explicit(true) S(char)},
/// but does not match \nomatch{explicit(b) S(float)}, \nomatch{S(int)} or
/// \nomatch{explicit S(double)}.
/// The matcher
/// \matcher{cxxConversionDecl(hasExplicitSpecifier(constantExpr()))} does not
/// match \nomatch{operator int()} or \nomatch{explicit operator bool()}.
/// Matcher
/// The matcher
/// \matcher{cxxDeductionGuideDecl(hasExplicitSpecifier(declRefExpr()))}
/// matches the implicitly generated deduction guide
/// \match{type=typestr$auto (float) -> S<b>} of the constructor
/// \c{explicit(b) S(float)}.
AST_MATCHER_P(FunctionDecl, hasExplicitSpecifier, internal::Matcher<Expr>,
              InnerMatcher) {
  ExplicitSpecifier ES = ExplicitSpecifier::getFromDecl(&Node);
  if (!ES.getExpr())
    return false;

  ASTChildrenNotSpelledInSourceScope RAII(Finder, false);

  return InnerMatcher.matches(*ES.getExpr(), Finder, Builder);
}

/// Matches functions, variables and namespace declarations that are marked with
/// the inline keyword.
///
/// Given
/// \code
///   inline void f();
///   void g();
///   namespace n {
///   inline namespace m {}
///   }
///   inline int Foo = 5;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{functionDecl(isInline())} matches
/// \match{inline void f()}.
/// The matcher \matcher{namespaceDecl(isInline())} matches
/// \match{inline namespace m {}}.
/// The matcher \matcher{varDecl(isInline())} matches
/// \match{inline int Foo = 5}.
AST_POLYMORPHIC_MATCHER(isInline, AST_POLYMORPHIC_SUPPORTED_TYPES(NamespaceDecl,
                                                                  FunctionDecl,
                                                                  VarDecl)) {
  // This is required because the spelling of the function used to determine
  // whether inline is specified or not differs between the polymorphic types.
  if (const auto *FD = dyn_cast<FunctionDecl>(&Node))
    return FD->isInlineSpecified();
  if (const auto *NSD = dyn_cast<NamespaceDecl>(&Node))
    return NSD->isInline();
  if (const auto *VD = dyn_cast<VarDecl>(&Node))
    return VD->isInline();
  llvm_unreachable("Not a valid polymorphic type");
}

/// Matches anonymous namespace declarations.
///
/// Given
/// \code
///   namespace n {
///   namespace {} // #1
///   }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{namespaceDecl(isAnonymous())}
/// matches \match{namespace {}}, but not \nomatch{namespace n}.
AST_MATCHER(NamespaceDecl, isAnonymous) {
  return Node.isAnonymousNamespace();
}

/// Matches declarations in the namespace `std`, but not in nested namespaces.
///
/// Given
/// \code
///   class vector {};
///   namespace foo {
///     class vector {};
///     namespace std {
///       class vector {};
///     }
///   }
///   namespace std {
///     inline namespace __1 {
///       class vector {}; // #1
///       namespace experimental {
///         class vector {};
///       }
///     }
///   }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxRecordDecl(hasName("vector"), isInStdNamespace())}
/// matches \match{class vector {}} inside of namespace std.
AST_MATCHER(Decl, isInStdNamespace) { return Node.isInStdNamespace(); }

/// Matches declarations in an anonymous namespace.
///
/// Given
/// \code
///   class vector {};
///   namespace foo {
///     class vector {};
///     namespace {
///       class vector {}; // #1
///     }
///   }
///   namespace {
///     class vector {}; // #2
///     namespace foo {
///       class vector {}; // #3
///     }
///   }
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxRecordDecl(hasName("vector"),
///                         isInAnonymousNamespace())}
/// matches \match{type=name;count=6$vector},
/// two times for the definition and the implicit class declaration
/// for each of the three definitions of \c vector .
AST_MATCHER(Decl, isInAnonymousNamespace) {
  return Node.isInAnonymousNamespace();
}

/// If the given case statement does not use the GNU case range
/// extension, matches the constant given in the statement.
///
/// Given
/// \code
///   void foo() {
///     switch (1) { case 1: break; case 1+1: break; case 3 ... 4: break; }
///   }
/// \endcode
/// The matcher
/// \matcher{caseStmt(hasCaseConstant(constantExpr(has(integerLiteral()))))}
/// matches \match{case 1: break}.
AST_MATCHER_P(CaseStmt, hasCaseConstant, internal::Matcher<Expr>,
              InnerMatcher) {
  if (Node.getRHS())
    return false;

  return InnerMatcher.matches(*Node.getLHS(), Finder, Builder);
}

/// Matches declaration that has a given attribute.
///
/// Given
/// \code
///   __attribute__((device)) void f() {}
/// \endcode
/// \compile_args{--cuda-gpu-arch=sm_70;-std=c++}
/// The matcher \matcher{decl(hasAttr(clang::attr::CUDADevice))}
/// matches \match{__attribute__((device)) void f() {}}.
/// If the matcher is used from clang-query, attr::Kind
/// parameter should be passed as a quoted string. e.g.,
/// \c hasAttr("attr::CUDADevice").
AST_MATCHER_P(Decl, hasAttr, attr::Kind, AttrKind) {
  for (const auto *Attr : Node.attrs()) {
    if (Attr->getKind() == AttrKind)
      return true;
  }
  return false;
}

/// Matches the return value expression of a return statement
///
/// Given
/// \code
///   int foo(int a, int b) {
///     return a + b;
///   }
/// \endcode
/// The matcher
/// \matcher{returnStmt(hasReturnValue(binaryOperator().bind("op")))} matches
/// \match{return a + b}, with \matcher{type=sub$binaryOperator()} matching
/// \match{sub=op$a + b}.
AST_MATCHER_P(ReturnStmt, hasReturnValue, internal::Matcher<Expr>,
              InnerMatcher) {
  if (const auto *RetValue = Node.getRetValue())
    return InnerMatcher.matches(*RetValue, Finder, Builder);
  return false;
}

/// Matches CUDA kernel call expression.
///
/// Given
/// \code
///   __global__ void kernel() {}
///   void f() {
///     kernel<<<32, 32>>>();
///   }
/// \endcode
/// \compile_args{--cuda-gpu-arch=sm_70;-std=c++}
/// The matcher \matcher{cudaKernelCallExpr()}
/// matches \match{kernel<<<32, 32>>>()}
extern const internal::VariadicDynCastAllOfMatcher<Stmt, CUDAKernelCallExpr>
    cudaKernelCallExpr;

/// Matches expressions that resolve to a null pointer constant, such as
/// GNU's __null, C++11's nullptr, or C's NULL macro.
///
/// Given
/// \code
///   #define NULL 0
///   void *v1 = NULL;
///   void *v2 = nullptr;
///   void *v3 = __null; // GNU extension
///   char *cp = (char *)0;
///   int *ip = 0;
///   int i = 0;
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{expr(nullPointerConstant())}
/// matches the initializer \match{NULL} of v1,
/// matches the initializer \match{nullptr} of v2,
/// matches the initializer \match{__null} of v3,
/// matches the initializer \match{0} of cp and
/// matches the initializer \match{0} of ip,
/// but does not match the initializer \nomatch{i} of i.
AST_MATCHER_FUNCTION(internal::Matcher<Expr>, nullPointerConstant) {
  return anyOf(
      gnuNullExpr(), cxxNullPtrLiteralExpr(),
      integerLiteral(equals(0), hasParent(expr(hasType(pointerType())))));
}

/// Matches the DecompositionDecl the binding belongs to.
///
/// Given
/// \code
/// void foo()
/// {
///     int arr[3];
///     auto &[f, s, t] = arr;
///
///     f = 42;
/// }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{bindingDecl(hasName("f"),
///                 forDecomposition(decompositionDecl()))}
/// matches \match{type=name$f} in \c{auto &[f, s, t]}.
AST_MATCHER_P(BindingDecl, forDecomposition, internal::Matcher<ValueDecl>,
              InnerMatcher) {
  if (const ValueDecl *VD = Node.getDecomposedDecl())
    return InnerMatcher.matches(*VD, Finder, Builder);
  return false;
}

/// Matches the Nth binding of a DecompositionDecl.
///
/// Given
/// \code
/// void foo()
/// {
///     int arr[3];
///     auto &[f, s, t] = arr;
///
///     f = 42;
/// }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{decompositionDecl(hasBinding(0,
///   bindingDecl(hasName("f")).bind("fBinding")))}
/// matches \match{auto &[f, s, t] = arr} with 'f' bound to "fBinding".
AST_MATCHER_P2(DecompositionDecl, hasBinding, unsigned, N,
               internal::Matcher<BindingDecl>, InnerMatcher) {
  if (Node.bindings().size() <= N)
    return false;
  return InnerMatcher.matches(*Node.bindings()[N], Finder, Builder);
}

/// Matches any binding of a DecompositionDecl.
///
/// For example, in:
/// \code
/// void foo()
/// {
///     int arr[3];
///     auto &[f, s, t] = arr;
///
///     f = 42;
/// }
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher
///  \matcher{decompositionDecl(hasAnyBinding(bindingDecl(hasName("f")).bind("fBinding")))}
/// matches \match{auto &[f, s, t] = arr} with 'f' bound to "fBinding".
AST_MATCHER_P(DecompositionDecl, hasAnyBinding, internal::Matcher<BindingDecl>,
              InnerMatcher) {
  return llvm::any_of(Node.bindings(), [&](const auto *Binding) {
    return InnerMatcher.matches(*Binding, Finder, Builder);
  });
}

/// Matches declaration of the function the statement belongs to.
///
/// Deprecated. Use forCallable() to correctly handle the situation when
/// the declaration is not a function (but a block or an Objective-C method).
/// The matcher \c forFunction() not only fails to take non-functions
/// into account but also may match the wrong declaration in their presence.
///
/// Given
/// \code
///   struct F {
///     F& operator=(const F& other) {
///       []() { return 42 == 42; };
///       return *this;
///     }
///   };
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{returnStmt(forFunction(hasName("operator=")))}
/// matches \match{return *this}
/// but does not match \nomatch{return 42 == 42}.
AST_MATCHER_P(Stmt, forFunction, internal::Matcher<FunctionDecl>,
              InnerMatcher) {
  const auto &Parents = Finder->getASTContext().getParents(Node);

  llvm::SmallVector<DynTypedNode, 8> Stack(Parents.begin(), Parents.end());
  while (!Stack.empty()) {
    const auto &CurNode = Stack.back();
    Stack.pop_back();
    if (const auto *FuncDeclNode = CurNode.get<FunctionDecl>()) {
      if (InnerMatcher.matches(*FuncDeclNode, Finder, Builder)) {
        return true;
      }
    } else if (const auto *LambdaExprNode = CurNode.get<LambdaExpr>()) {
      if (InnerMatcher.matches(*LambdaExprNode->getCallOperator(), Finder,
                               Builder)) {
        return true;
      }
    } else {
      llvm::append_range(Stack, Finder->getASTContext().getParents(CurNode));
    }
  }
  return false;
}

/// Matches declaration of the function, method, or block the statement
/// belongs to.
///
/// Given
/// \code
/// struct F {
///   F& operator=(const F& other) {
///     []() { return 42 == 42; };
///     return *this;
///   }
/// };
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{returnStmt(forFunction(hasName("operator=")))}
/// matches \match{return *this}
/// but does not match \nomatch{return 42 == 42}.
///
/// Given
/// \code
/// void foo {
///   int x = 1;
///   dispatch_sync(queue, ^{ int y = 2; });
/// }
/// \endcode
/// \compile_args{-ObjC}
/// The matcher \matcher{declStmt(forCallable(objcMethodDecl()))}
/// matches \match{int x = 1}
/// but does not match \nomatch{int y = 2}.
/// The matcher \matcher{declStmt(forCallable(blockDecl()))}
/// matches \match{int y = 2}
/// but does not match \nomatch{int x = 1}.
AST_MATCHER_P(Stmt, forCallable, internal::Matcher<Decl>, InnerMatcher) {
  const auto &Parents = Finder->getASTContext().getParents(Node);

  llvm::SmallVector<DynTypedNode, 8> Stack(Parents.begin(), Parents.end());
  while (!Stack.empty()) {
    const auto &CurNode = Stack.back();
    Stack.pop_back();
    if (const auto *FuncDeclNode = CurNode.get<FunctionDecl>()) {
      BoundNodesTreeBuilder B = *Builder;
      if (InnerMatcher.matches(*FuncDeclNode, Finder, &B)) {
        *Builder = std::move(B);
        return true;
      }
    } else if (const auto *LambdaExprNode = CurNode.get<LambdaExpr>()) {
      BoundNodesTreeBuilder B = *Builder;
      if (InnerMatcher.matches(*LambdaExprNode->getCallOperator(), Finder,
                               &B)) {
        *Builder = std::move(B);
        return true;
      }
    } else if (const auto *ObjCMethodDeclNode = CurNode.get<ObjCMethodDecl>()) {
      BoundNodesTreeBuilder B = *Builder;
      if (InnerMatcher.matches(*ObjCMethodDeclNode, Finder, &B)) {
        *Builder = std::move(B);
        return true;
      }
    } else if (const auto *BlockDeclNode = CurNode.get<BlockDecl>()) {
      BoundNodesTreeBuilder B = *Builder;
      if (InnerMatcher.matches(*BlockDeclNode, Finder, &B)) {
        *Builder = std::move(B);
        return true;
      }
    } else {
      llvm::append_range(Stack, Finder->getASTContext().getParents(CurNode));
    }
  }
  return false;
}

/// Matches a declaration that has external formal linkage.
///
/// Given
/// \code
/// void f() {
///   int a;
///   static int b;
/// }
/// int c;
/// static int d;
/// \endcode
/// The matcher \matcher{varDecl(hasExternalFormalLinkage())}
/// matches \match{int c},
/// but not \nomatch{int a}, \nomatch{static int b} or \nomatch{int d}.
///
/// Given
/// \code
///   namespace {
///     void f() {}
///   }
///   void g() {}
///   static void h() {}
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{functionDecl(hasExternalFormalLinkage())}
/// matches \match{void g() {}}, but not \nomatch{void f() {}} or
/// \nomatch{static void h() {}}.
AST_MATCHER(NamedDecl, hasExternalFormalLinkage) {
  return Node.hasExternalFormalLinkage();
}

/// Matches a declaration that has default arguments.
///
/// Given
/// \code
///   void x(int val) {}
///   void y(int val = 0) {}
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher \matcher{parmVarDecl(hasDefaultArgument())}
/// matches \match{int val = 0}.
///
/// Deprecated. Use hasInitializer() instead to be able to
/// match on the contents of the default argument.  For example:
///
/// Given
/// \code
///   void x(int val = 7) {}
///   void y(int val = 42) {}
/// \endcode
/// \compile_args{-std=c++}
///
/// The matcher
/// \matcher{parmVarDecl(hasInitializer(integerLiteral(equals(42))))},
/// matches \match{int val = 42}.
AST_MATCHER(ParmVarDecl, hasDefaultArgument) {
  return Node.hasDefaultArg();
}

/// Matches array new expressions.
///
/// Given
/// \code
///   struct MyClass { int x; };
///   MyClass *p1 = new MyClass[10];
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxNewExpr(isArray())}
/// matches \match{new MyClass[10]}.
AST_MATCHER(CXXNewExpr, isArray) {
  return Node.isArray();
}

/// Matches placement new expression arguments.
///
/// Given
/// \code
///   void *operator new(decltype(sizeof(void*)), int, void*);
///   struct MyClass { int x; };
///   unsigned char Storage[sizeof(MyClass) * 10];
///   MyClass *p1 = new (16, Storage) MyClass();
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{cxxNewExpr(hasPlacementArg(0,
///                       integerLiteral(equals(16))))}
/// matches \match{new (16, Storage) MyClass()}.
AST_MATCHER_P2(CXXNewExpr, hasPlacementArg, unsigned, Index,
               internal::Matcher<Expr>, InnerMatcher) {
  return Node.getNumPlacementArgs() > Index &&
         InnerMatcher.matches(*Node.getPlacementArg(Index), Finder, Builder);
}

/// Matches any placement new expression arguments.
///
/// Given
/// \code
///   void* operator new(decltype(sizeof(void*)), void*);
///   struct MyClass { int x; };
///   unsigned char Storage[sizeof(MyClass) * 10];
///   MyClass *p1 = new (Storage) MyClass();
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher \matcher{cxxNewExpr(hasAnyPlacementArg(anything()))}
/// matches \match{new (Storage) MyClass()}.
AST_MATCHER_P(CXXNewExpr, hasAnyPlacementArg, internal::Matcher<Expr>,
              InnerMatcher) {
  return llvm::any_of(Node.placement_arguments(), [&](const Expr *Arg) {
    return InnerMatcher.matches(*Arg, Finder, Builder);
  });
}

/// Matches array new expressions with a given array size.
///
/// Given
/// \code
///   void* operator new(decltype(sizeof(void*)));
///   struct MyClass { int x; };
///   MyClass *p1 = new MyClass[10];
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher
/// \matcher{cxxNewExpr(hasArraySize(
///             ignoringImplicit(integerLiteral(equals(10)))))}
/// matches \match{new MyClass[10]}.
AST_MATCHER_P(CXXNewExpr, hasArraySize, internal::Matcher<Expr>, InnerMatcher) {
  return Node.isArray() && *Node.getArraySize() &&
         InnerMatcher.matches(**Node.getArraySize(), Finder, Builder);
}

/// Matches a class declaration that is defined.
///
/// Given
/// \code
/// class x {};
/// class y;
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{cxxRecordDecl(hasDefinition())}
/// matches \match{class x {}}
AST_MATCHER(CXXRecordDecl, hasDefinition) {
  return Node.hasDefinition();
}

/// Matches C++11 scoped enum declaration.
///
/// Given
/// \code
/// enum X {};
/// enum class Y {};
/// \endcode
/// \compile_args{-std=c++}
/// The matcher \matcher{enumDecl(isScoped())}
/// matches \match{enum class Y {}}
AST_MATCHER(EnumDecl, isScoped) {
  return Node.isScoped();
}

/// Matches a function declared with a trailing return type.
///
/// Given
/// \code
/// int X() {}
/// auto Y() -> int {}
/// \endcode
/// \compile_args{-std=c++11-or-later}
/// The matcher \matcher{functionDecl(hasTrailingReturn())}
/// matches \match{auto Y() -> int {}}.
AST_MATCHER(FunctionDecl, hasTrailingReturn) {
  if (const auto *F = Node.getType()->getAs<FunctionProtoType>())
    return F->hasTrailingReturn();
  return false;
}

/// Matches expressions that match InnerMatcher that are possibly wrapped in an
/// elidable constructor and other corresponding bookkeeping nodes.
///
/// In C++17, elidable copy constructors are no longer being generated in the
/// AST as it is not permitted by the standard. They are, however, part of the
/// AST in C++14 and earlier. So, a matcher must abstract over these differences
/// to work in all language modes. This matcher skips elidable constructor-call
/// AST nodes, `ExprWithCleanups` nodes wrapping elidable constructor-calls and
/// various implicit nodes inside the constructor calls, all of which will not
/// appear in the C++17 AST.
///
/// Given
/// \code
/// struct H {};
/// H G();
/// void f() {
///   H D = G();
/// }
/// \endcode
/// \compile_args{-std=c++11-or-later}
///
/// The matcher
/// \matcher{varDecl(hasInitializer(ignoringElidableConstructorCall(callExpr())))}
/// matches \match{H D = G()}.
AST_MATCHER_P(Expr, ignoringElidableConstructorCall, internal::Matcher<Expr>,
              InnerMatcher) {
  // E tracks the node that we are examining.
  const Expr *E = &Node;
  // If present, remove an outer `ExprWithCleanups` corresponding to the
  // underlying `CXXConstructExpr`. This check won't cover all cases of added
  // `ExprWithCleanups` corresponding to `CXXConstructExpr` nodes (because the
  // EWC is placed on the outermost node of the expression, which this may not
  // be), but, it still improves the coverage of this matcher.
  if (const auto *CleanupsExpr = dyn_cast<ExprWithCleanups>(&Node))
    E = CleanupsExpr->getSubExpr();
  if (const auto *CtorExpr = dyn_cast<CXXConstructExpr>(E)) {
    if (CtorExpr->isElidable()) {
      if (const auto *MaterializeTemp =
              dyn_cast<MaterializeTemporaryExpr>(CtorExpr->getArg(0))) {
        return InnerMatcher.matches(*MaterializeTemp->getSubExpr(), Finder,
                                    Builder);
      }
    }
  }
  return InnerMatcher.matches(Node, Finder, Builder);
}

//----------------------------------------------------------------------------//
// OpenMP handling.
//----------------------------------------------------------------------------//

/// Matches any ``#pragma omp`` executable directive.
///
/// Given
/// \code
///   void foo() {
///     #pragma omp parallel
///       {}
///     #pragma omp parallel default(none)
///       {
///         #pragma omp taskyield
///       }
///   }
/// \endcode
/// \compile_args{-fopenmp}
/// The matcher \matcher{ompExecutableDirective()}
/// matches \match{#pragma omp parallel},
/// \match{#pragma omp parallel default(none)}
/// and \match{#pragma omp taskyield}.
extern const internal::VariadicDynCastAllOfMatcher<Stmt, OMPExecutableDirective>
    ompExecutableDirective;

/// Matches standalone OpenMP directives,
/// i.e., directives that can't have a structured block.
///
/// Given
/// \code
///   void foo() {
///     #pragma omp parallel
///     {
///       #pragma omp taskyield
///     }
///   }
/// \endcode
/// \compile_args{-fopenmp}
///
/// The matcher \matcher{ompExecutableDirective(isStandaloneDirective())}
/// matches \match{#pragma omp taskyield}.
AST_MATCHER(OMPExecutableDirective, isStandaloneDirective) {
  return Node.isStandaloneDirective();
}

/// Matches the structured-block of the OpenMP executable directive
///
/// Prerequisite: the executable directive must not be standalone directive.
/// If it is, it will never match.
///
/// Given
/// \code
///   void foo() {
///     #pragma omp parallel
///     ;
///     #pragma omp parallel
///     {}
///   }
/// \endcode
/// \compile_args{-fopenmp}
///
/// The matcher
/// \matcher{ompExecutableDirective(hasStructuredBlock(nullStmt().bind("stmt")))}
/// matches \match{#pragma omp parallel},
/// with \matcher{type=sub$stmtt()} matching \match{sub=stmt${}}.
AST_MATCHER_P(OMPExecutableDirective, hasStructuredBlock,
              internal::Matcher<Stmt>, InnerMatcher) {
  if (Node.isStandaloneDirective())
    return false; // Standalone directives have no structured blocks.
  return InnerMatcher.matches(*Node.getStructuredBlock(), Finder, Builder);
}

/// Matches any clause in an OpenMP directive.
///
/// Given
/// \code
///   void foo() {
///   #pragma omp parallel
///     ;
///   #pragma omp parallel default(none)
///     ;
///   }
/// \endcode
/// \compile_args{-fopenmp}
///
/// The matcher \matcher{ompExecutableDirective(hasAnyClause(anything()))}
/// matches \match{#pragma omp parallel default(none)}.
AST_MATCHER_P(OMPExecutableDirective, hasAnyClause,
              internal::Matcher<OMPClause>, InnerMatcher) {
  ArrayRef<OMPClause *> Clauses = Node.clauses();
  return matchesFirstInPointerRange(InnerMatcher, Clauses.begin(),
                                    Clauses.end(), Finder,
                                    Builder) != Clauses.end();
}

/// Matches OpenMP ``default`` clause.
///
/// Given
/// \code
///   void foo() {
///     #pragma omp parallel default(none)
///       ;
///     #pragma omp parallel default(shared)
///       ;
///     #pragma omp parallel default(private)
///       ;
///     #pragma omp parallel default(firstprivate)
///       ;
///     #pragma omp parallel
///       ;
///   }
/// \endcode
/// \compile_args{-fopenmp}
///
/// The matcher
/// \matcher{ompExecutableDirective(hasAnyClause(ompDefaultClause()))} matches
/// \match{#pragma omp parallel default(none)},
/// \match{#pragma omp parallel default(shared)},
/// \match{#pragma omp parallel default(private)} and
/// \match{#pragma omp parallel default(firstprivate)}.
extern const internal::VariadicDynCastAllOfMatcher<OMPClause, OMPDefaultClause>
    ompDefaultClause;

/// Matches if the OpenMP ``default`` clause has ``none`` kind specified.
///
/// Given
/// \code
///   void foo() {
///     #pragma omp parallel
///       ;
///     #pragma omp parallel default(none)
///       ;
///     #pragma omp parallel default(shared)
///       ;
///     #pragma omp parallel default(private)
///       ;
///     #pragma omp parallel default(firstprivate)
///       ;
///   }
/// \endcode
/// \compile_args{-fopenmp}
///
/// The matcher
/// \matcher{ompExecutableDirective(hasAnyClause(ompDefaultClause(isNoneKind())))}
/// matches only \match{#pragma omp parallel default(none)}.
AST_MATCHER(OMPDefaultClause, isNoneKind) {
  return Node.getDefaultKind() == llvm::omp::OMP_DEFAULT_none;
}

/// Matches if the OpenMP ``default`` clause has ``shared`` kind specified.
///
/// Given
/// \code
///   void foo() {
///     #pragma omp parallel
///       ;
///     #pragma omp parallel default(none)
///       ;
///   #pragma omp parallel default(shared)
///       ;
///   #pragma omp parallel default(private)
///       ;
///   #pragma omp parallel default(firstprivate)
///       ;
///   }
/// \endcode
/// \compile_args{-fopenmp}
///
/// The matcher
/// \matcher{ompExecutableDirective(hasAnyClause(ompDefaultClause(isSharedKind())))}
/// matches \match{#pragma omp parallel default(shared)}.
AST_MATCHER(OMPDefaultClause, isSharedKind) {
  return Node.getDefaultKind() == llvm::omp::OMP_DEFAULT_shared;
}

/// Matches if the OpenMP ``default`` clause has ``private`` kind
/// specified.
///
/// Given
/// \code
///   void foo() {
///     #pragma omp parallel
///       ;
///   #pragma omp parallel default(none)
///       ;
///   #pragma omp parallel default(shared)
///       ;
///   #pragma omp parallel default(private)
///       ;
///   #pragma omp parallel default(firstprivate)
///       ;
///   }
/// \endcode
/// \compile_args{-fopenmp}
///
/// The matcher
/// \matcher{ompExecutableDirective(hasAnyClause(ompDefaultClause(isPrivateKind())))}
/// matches \match{#pragma omp parallel default(private)}.
AST_MATCHER(OMPDefaultClause, isPrivateKind) {
  return Node.getDefaultKind() == llvm::omp::OMP_DEFAULT_private;
}

/// Matches if the OpenMP ``default`` clause has ``firstprivate`` kind
/// specified.
///
/// Given
/// \code
///   void foo() {
///     #pragma omp parallel
///       ;
///     #pragma omp parallel default(none)
///       ;
///     #pragma omp parallel default(shared)
///       ;
///     #pragma omp parallel default(private)
///       ;
///     #pragma omp parallel default(firstprivate)
///       ;
///   }
/// \endcode
/// \compile_args{-fopenmp}
///
/// The matcher
/// \matcher{ompExecutableDirective(hasAnyClause(ompDefaultClause(isFirstPrivateKind())))}
/// matches \match{#pragma omp parallel default(firstprivate)}.
AST_MATCHER(OMPDefaultClause, isFirstPrivateKind) {
  return Node.getDefaultKind() == llvm::omp::OMP_DEFAULT_firstprivate;
}

/// Matches if the OpenMP directive is allowed to contain the specified OpenMP
/// clause kind.
///
/// Given
/// \code
///   void foo() {
///     #pragma omp parallel
///       ;
///     #pragma omp parallel for
///       for (int i = 0; i < 10; ++i) {}
///     #pragma omp          for
///       for (int i = 0; i < 10; ++i) {}
///   }
/// \endcode
/// \compile_args{-fopenmp}
///
/// The matcher
/// \matcher{ompExecutableDirective(isAllowedToContainClauseKind(
/// OpenMPClauseKind::OMPC_default))}
/// matches \match{#pragma omp parallel}
/// and \match{#pragma omp parallel for}.
///
/// If the matcher is use from clang-query, ``OpenMPClauseKind`` parameter
/// should be passed as a quoted string. e.g.,
/// ``isAllowedToContainClauseKind("OMPC_default").``
AST_MATCHER_P(OMPExecutableDirective, isAllowedToContainClauseKind,
              OpenMPClauseKind, CKind) {
  return llvm::omp::isAllowedClauseForDirective(
      Node.getDirectiveKind(), CKind,
      Finder->getASTContext().getLangOpts().OpenMP);
}

//----------------------------------------------------------------------------//
// End OpenMP handling.
//----------------------------------------------------------------------------//

} // namespace ast_matchers
} // namespace clang

#endif // LLVM_CLANG_ASTMATCHERS_ASTMATCHERS_H
