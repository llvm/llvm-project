// Addresses the dependent-argument case raised on the unknown-attribute review:
// what it takes for a plugin to make Clang "accept" an attribute whose (possibly
// dependent) expression argument participates in template instantiation.
//
// The key is that the plugin registers the attribute (ParsedAttrInfo), so Clang
// parses the argument as an expression in context at definition time, and the
// Attribute example lowers it onto an Expr-carrying attribute (AnnotateAttr).
// Clang's existing attribute-instantiation machinery then substitutes the
// dependent argument with no special support: retaining the tokens is not
// required, and re-parsing them after the fact is not either.
//
// Boundary: the substitution runs when the specialization is formed, after
// deduction, so a substitution failure is a hard error, not SFINAE. No attribute
// (enable_if included) does SFINAE on an argument's substitution failure, so that
// piece is a separate core feature, independent of plugins and of how the
// argument is stored.

// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++20 -load %llvmshlibdir/Attribute%pluginext \
// RUN:   -ast-dump %t/subst.cpp | FileCheck %s
// RUN: not %clang_cc1 -std=c++20 -load %llvmshlibdir/Attribute%pluginext \
// RUN:   -fsyntax-only %t/fail.cpp 2>&1 | FileCheck --check-prefix=HARD-ERROR %s

// REQUIRES: plugins, examples

//--- subst.cpp
template <class T> [[example("tag", T::value + 1)]] void f() {}
struct Foo { static constexpr int value = 41; };
template void f<Foo>();

// On the primary template the argument is a dependent expression.
// CHECK: FunctionTemplateDecl {{.*}} f
// CHECK: AnnotateAttr {{.*}} "example"
// CHECK: DependentScopeDeclRefExpr

// On the instantiation T::value + 1 has been substituted to 42, bound to
// Foo::value: the plugin attribute's argument is live.
// CHECK: FunctionDecl {{.*}} f 'void ()' explicit_instantiation_definition
// CHECK: AnnotateAttr {{.*}} "example"
// CHECK: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 42

//--- fail.cpp
template <class T> [[example("tag", T::value + 1)]] void f() {}
struct NoValue {};
template void f<NoValue>();

// The substitution failure is a hard error, not SFINAE.
// HARD-ERROR: error: no member named 'value' in 'NoValue'
