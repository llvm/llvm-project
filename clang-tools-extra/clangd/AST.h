//===--- AST.h - Utility AST functions  -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various code that examines C++ source code using AST.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_AST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_AST_H

#include "Headers.h"
#include "index/Symbol.h"
#include "index/SymbolID.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/MacroInfo.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>
#include <vector>

namespace clang {
class SourceManager;
class Decl;
class DynTypedNode;

namespace clangd {

/// Returns true if the declaration is considered implementation detail based on
/// heuristics. For example, a declaration whose name is not explicitly spelled
/// in code is considered implementation detail.
bool isImplementationDetail(const Decl *D);

/// Find the source location of the identifier for \p D.
/// Transforms macro locations to locations spelled inside files. All code
/// that needs locations of declaration names (e.g. the index) should go through
/// this function.
SourceLocation nameLocation(const clang::Decl &D, const SourceManager &SM);

/// Returns the qualified name of ND. The scope doesn't contain unwritten scopes
/// like inline namespaces.
std::string printQualifiedName(const NamedDecl &ND);

/// Returns the first enclosing namespace scope starting from \p DC.
std::string printNamespaceScope(const DeclContext &DC);

/// Returns the name of the namespace inside the 'using namespace' directive, as
/// written in the code. E.g., passing 'using namespace ::std' will result in
/// '::std'.
std::string printUsingNamespaceName(const ASTContext &Ctx,
                                    const UsingDirectiveDecl &D);

/// Prints unqualified name of the decl for the purpose of displaying it to the
/// user. Anonymous decls return names of the form "(anonymous {kind})", e.g.
/// "(anonymous struct)" or "(anonymous namespace)".
std::string printName(const ASTContext &Ctx, const NamedDecl &ND);

/// Prints template arguments of a decl as written in the source code, including
/// enclosing '<' and '>', e.g for a partial specialization like: template
/// <typename U> struct Foo<int, U> will return '<int, U>'. Returns an empty
/// string if decl is not a template specialization.
std::string printTemplateSpecializationArgs(const NamedDecl &ND);

/// Print the Objective-C method name, including the full container name, e.g.
/// `-[MyClass(Category) method:]`
std::string printObjCMethod(const ObjCMethodDecl &Method);

/// Print the Objective-C container name including categories, e.g. `MyClass`,
// `MyClass()`, `MyClass(Category)`, and `MyProtocol`.
std::string printObjCContainer(const ObjCContainerDecl &C);

/// Returns true if this is a NamedDecl with a reserved name.
bool hasReservedName(const Decl &);
/// Returns true if this scope would be written with a reserved name.
/// This does not include unwritten scope elements like __1 in std::__1::vector.
bool hasReservedScope(const DeclContext &);

/// Gets the symbol ID for a declaration. Returned SymbolID might be null.
SymbolID getSymbolID(const Decl *D);

/// Gets the symbol ID for a macro. Returned SymbolID might be null.
/// Currently, this is an encoded USR of the macro, which incorporates macro
/// locations (e.g. file name, offset in file).
/// FIXME: the USR semantics might not be stable enough as the ID for index
/// macro (e.g. a change in definition offset can result in a different USR). We
/// could change these semantics in the future by reimplementing this funcure
/// (e.g. avoid USR for macros).
SymbolID getSymbolID(const llvm::StringRef MacroName, const MacroInfo *MI,
                     const SourceManager &SM);

/// Return the corresponding implementation/definition for the given ObjC
/// container if it has one, otherwise, return nullptr.
///
/// Objective-C classes can have three types of declarations:
///
/// - forward declaration: "@class MyClass;"
/// - true declaration (interface definition): "@interface MyClass ... @end"
/// - true definition (implementation): "@implementation MyClass ... @end"
///
/// Objective-C categories are extensions on classes:
///
/// - declaration: "@interface MyClass (Ext) ... @end"
/// - definition: "@implementation MyClass (Ext) ... @end"
///
/// With one special case, a class extension, which is normally used to keep
/// some declarations internal to a file without exposing them in a header.
///
/// - class extension declaration: "@interface MyClass () ... @end"
/// - which really links to class definition: "@implementation MyClass ... @end"
///
/// For Objective-C protocols, e.g. "@protocol MyProtocol ... @end" this will
/// return nullptr as protocols don't have an implementation.
const ObjCImplDecl *getCorrespondingObjCImpl(const ObjCContainerDecl *D);

/// Infer the include directive to use for the given \p FileName. It aims for
/// #import for ObjC files and #include for the rest.
///
/// - For source files we use LangOpts directly to infer ObjC-ness.
/// - For header files we also check for symbols declared by the file and
///   existing include directives, as the language can be set to ObjC++ as a
///   fallback in the absence of compile flags.
Symbol::IncludeDirective
preferredIncludeDirective(llvm::StringRef FileName, const LangOptions &LangOpts,
                          ArrayRef<Inclusion> MainFileIncludes,
                          ArrayRef<const Decl *> TopLevelDecls);

/// Returns a QualType as string. The result doesn't contain unwritten scopes
/// like anonymous/inline namespace.
std::string printType(const QualType QT, const DeclContext &CurContext,
                      llvm::StringRef Placeholder = "");

/// Indicates if \p D is a template instantiation implicitly generated by the
/// compiler, e.g.
///     template <class T> struct vector {};
///     vector<int> v; // 'vector<int>' is an implicit instantiation
bool isImplicitTemplateInstantiation(const NamedDecl *D);
/// Indicates if \p D is an explicit template specialization, e.g.
///   template <class T> struct vector {};
///   template <> struct vector<bool> {}; // <-- explicit specialization
///
/// Note that explicit instantiations are NOT explicit specializations, albeit
/// they look similar.
///   template struct vector<bool>; // <-- explicit instantiation, NOT an
///   explicit specialization.
bool isExplicitTemplateSpecialization(const NamedDecl *D);

/// Returns a nested name specifier loc of \p ND if it was present in the
/// source, e.g.
///     void ns::something::foo() -> returns 'ns::something'
///     void foo() -> returns null
NestedNameSpecifierLoc getQualifierLoc(const NamedDecl &ND);

// Returns a type corresponding to a declaration of that type.
// Unlike the method on ASTContext, attempts to preserve the type as-written
// (i.e. vector<T*> rather than vector<type-parameter-0-0 *>.
QualType declaredType(const TypeDecl *D);

/// Retrieves the deduced type at a given location (auto, decltype).
/// It will return the underlying type.
/// If the type is an undeduced auto, returns the type itself.
std::optional<QualType> getDeducedType(ASTContext &, SourceLocation Loc);

// Find the abbreviated-function-template `auto` within a type, or returns null.
// Similar to getContainedAutoTypeLoc, but these `auto`s are
// TemplateTypeParmTypes for implicit TTPs, instead of AutoTypes.
// Also we don't look very hard, just stripping const, references, pointers.
// FIXME: handle more type patterns.
TemplateTypeParmTypeLoc getContainedAutoParamType(TypeLoc TL);

// If TemplatedDecl is the generic body of a template, and the template has
// exactly one visible instantiation, return the instantiated body.
NamedDecl *getOnlyInstantiation(NamedDecl *TemplatedDecl);

/// Return attributes attached directly to a node.
std::vector<const Attr *> getAttributes(const DynTypedNode &);

/// Gets the nested name specifier necessary for spelling \p ND in \p
/// DestContext, at \p InsertionPoint. It selects the shortest suffix of \p ND
/// such that it is visible in \p DestContext.
/// Returns an empty string if no qualification is necessary. For example, if
/// you want to qualify clang::clangd::bar::foo in clang::clangd::x, this
/// function will return bar. Note that the result might be sub-optimal for
/// classes, e.g. when the \p ND is a member of the base class.
///
/// This version considers all the using namespace directives before \p
/// InsertionPoint. i.e, if you have `using namespace
/// clang::clangd::bar`, this function will return an empty string for the
/// example above since no qualification is necessary in that case.
/// FIXME: Also take using directives and namespace aliases inside function body
/// into account.
std::string getQualification(ASTContext &Context,
                             const DeclContext *DestContext,
                             SourceLocation InsertionPoint,
                             const NamedDecl *ND);

/// This function uses the \p VisibleNamespaces to figure out if a shorter
/// qualification is sufficient for \p ND, and ignores any using namespace
/// directives. It can be useful if there's no AST for the DestContext, but some
/// pseudo-parsing is done. i.e. if \p ND is ns1::ns2::X and \p DestContext is
/// ns1::, users can provide `ns2::` as visible to change the result to be
/// empty.
/// Elements in VisibleNamespaces should be in the form: `ns::`, with trailing
/// "::".
/// Note that this is just textual and might be incorrect. e.g. when there are
/// two namespaces ns1::a and ns2::a, the function will early exit if "a::" is
/// present in \p VisibleNamespaces, no matter whether it is from ns1:: or ns2::
std::string getQualification(ASTContext &Context,
                             const DeclContext *DestContext,
                             const NamedDecl *ND,
                             llvm::ArrayRef<std::string> VisibleNamespaces);

/// Whether we must avoid computing linkage for D during code completion.
/// Clang aggressively caches linkage computation, which is stable after the AST
/// is built. Unfortunately the AST is incomplete during code completion, so
/// linkage may still change.
///
/// Example: `auto x = []{^}` at file scope.
/// During code completion, the initializer for x hasn't been parsed yet.
/// x has type `undeduced auto`, and external linkage.
/// If we compute linkage at this point, the external linkage will be cached.
///
/// After code completion the initializer is attached, and x has a lambda type.
/// This means x has "unique external" linkage. If we computed linkage above,
/// the cached value is incorrect. (clang catches this with an assertion).
bool hasUnstableLinkage(const Decl *D);

/// Checks whether \p D is more than \p MaxDepth away from translation unit
/// scope.
/// This is useful for limiting traversals to keep operation latencies
/// reasonable.
bool isDeeplyNested(const Decl *D, unsigned MaxDepth = 10);

/// Recursively resolves the parameters of a FunctionDecl that forwards its
/// parameters to another function via variadic template parameters. This can
/// for example be used to retrieve the constructor parameter ParmVarDecl for a
/// make_unique or emplace_back call.
llvm::SmallVector<const ParmVarDecl *>
resolveForwardingParameters(const FunctionDecl *D, unsigned MaxDepth = 10);

/// Checks whether D is instantiated from a function parameter pack
/// whose type is a bare type parameter pack (e.g. `Args...`), or a
/// reference to one (e.g. `Args&...` or `Args&&...`).
bool isExpandedFromParameterPack(const ParmVarDecl *D);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_AST_H
