// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fvisibility=hidden -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,HIDDEN
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,DEFAULT

// Check the visibility of the alias that #pragma weak alias = target creates.
// As in GCC, the alias takes the visibility explicitly written on its own
// declaration, independently of the visibility of the target; with no such
// declaration it gets the -fvisibility default. The pragma synthesizes a fresh
// declaration for the alias rather than redeclaring an existing one, so the
// cases below cover the ways that visibility can reach it.

// visibility("default") on the alias, declared before the pragma.
// CHECK-DAG: @alias_declared_first = weak alias i32 (), ptr @target_declared_first
int target_declared_first(void) __attribute__((visibility("default")));
int alias_declared_first(void) __attribute__((visibility("default")));
#pragma weak alias_declared_first = target_declared_first
int target_declared_first(void) { return 42; }

// Same, but the pragma precedes both declarations, so the alias is created
// later from ProcessPragmaWeak() instead of ActOnPragmaWeakAlias().
// CHECK-DAG: @alias_pragma_first = weak alias i32 (), ptr @target_pragma_first
#pragma weak alias_pragma_first = target_pragma_first
int alias_pragma_first(void) __attribute__((visibility("default")));
int target_pragma_first(void) __attribute__((visibility("default")));
int target_pragma_first(void) { return 1; }

// The visibility may also come from #pragma GCC visibility. Here the target is
// hidden, which must not affect the alias.
// CHECK-DAG: @alias_from_gcc_pragma = weak alias i32 (), ptr @target_from_gcc_pragma
int target_from_gcc_pragma(void);
#pragma GCC visibility push(default)
int alias_from_gcc_pragma(void);
#pragma GCC visibility pop
#pragma weak alias_from_gcc_pragma = target_from_gcc_pragma
int target_from_gcc_pragma(void) { return 1; }

// Conversely, an explicit visibility on the target alone does not propagate to
// the alias.
// HIDDEN-DAG: @alias_without_visibility = weak hidden alias i32 (), ptr @target_with_visibility
// DEFAULT-DAG: @alias_without_visibility = weak alias i32 (), ptr @target_with_visibility
int target_with_visibility(void) __attribute__((visibility("default")));
int alias_without_visibility(void);
#pragma weak alias_without_visibility = target_with_visibility
int target_with_visibility(void) { return 1; }

// An explicit visibility("hidden") is honored too, even with -fvisibility left
// at its default.
// CHECK-DAG: @alias_explicitly_hidden = weak hidden alias i32 (), ptr @target_of_hidden_alias
int target_of_hidden_alias(void);
int alias_explicitly_hidden(void) __attribute__((visibility("hidden")));
#pragma weak alias_explicitly_hidden = target_of_hidden_alias
int target_of_hidden_alias(void) { return 1; }

// With no declaration of the alias there is no explicit visibility to carry
// over, so the -fvisibility default still applies.
// HIDDEN-DAG: @undeclared_alias = weak hidden alias i32 (), ptr @target_of_undeclared_alias
// DEFAULT-DAG: @undeclared_alias = weak alias i32 (), ptr @target_of_undeclared_alias
int target_of_undeclared_alias(void) __attribute__((visibility("default")));
#pragma weak undeclared_alias = target_of_undeclared_alias
int target_of_undeclared_alias(void) { return 1; }

// Variables take the same path through DeclClonePragmaWeak().
// CHECK-DAG: @alias_variable = weak alias i32, ptr @target_variable
extern int target_variable;
extern int alias_variable __attribute__((visibility("default")));
#pragma weak alias_variable = target_variable
int target_variable = 7;

// As in GCC, a later declaration may set the visibility too.
// CHECK-DAG: @alias_declared_late = weak alias i32 (), ptr @target_of_late_alias
int target_of_late_alias(void);
#pragma weak alias_declared_late = target_of_late_alias
int alias_declared_late(void) __attribute__((visibility("default")));
int target_of_late_alias(void) { return 1; }

// CHECK-DAG: @alias_late_gcc_pragma = weak alias i32 (), ptr @target_of_late_gcc_pragma
int target_of_late_gcc_pragma(void);
#pragma weak alias_late_gcc_pragma = target_of_late_gcc_pragma
#pragma GCC visibility push(default)
int alias_late_gcc_pragma(void);
#pragma GCC visibility pop
int target_of_late_gcc_pragma(void) { return 1; }

// CHECK-DAG: @alias_variable_late = weak alias i32, ptr @target_variable_late
int target_variable_late = 7;
#pragma weak alias_variable_late = target_variable_late
extern int alias_variable_late __attribute__((visibility("default")));
