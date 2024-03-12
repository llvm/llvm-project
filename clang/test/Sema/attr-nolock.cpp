// RUN: %clang_cc1 %s -ast-dump -fblocks | FileCheck %s
// expected-no-diagnostics

// Make sure that the attribute gets parsed and attached to the correct AST elements.
// Update 1 Mar 2024

#pragma clang diagnostic ignored "-Wunused-variable"

// =========================================================================================
// Square brackets, true

#define NOLOCK [[clang::nolock]]

// On the type of the FunctionDecl
void nl_function() NOLOCK;
// CHECK: FunctionDecl {{.*}} nl_function 'void () __attribute__((clang_nolock))'

// On the type of the VarDecl holding a function pointer
void (*nl_func_a)() NOLOCK;
// CHECK: VarDecl {{.*}} nl_func_a 'void (*)() __attribute__((clang_nolock))'

// Check alternate attribute type and placement
__attribute__((clang_nolock)) void (*nl_func_b)(void);
// CHECK: VarDecl {{.*}} nl_func_b 'void (*)() __attribute__((clang_nolock))'

// On the type of the ParmVarDecl of a function parameter
static void nlReceiver(void (*nl_func)() NOLOCK);
// CHECK: ParmVarDecl {{.*}} nl_func 'void (*)() __attribute__((clang_nolock))'

// As an AttributedType within the nested types of a typedef
typedef void (*nl_fp_type)() NOLOCK;
// CHECK: TypedefDecl {{.*}} nl_fp_type 'void (*)() __attribute__((clang_nolock))'
using nl_fp_talias = void (*)() NOLOCK;
// CHECK: TypeAliasDecl {{.*}} nl_fp_talias 'void (*)() __attribute__((clang_nolock))'

// From a typedef or typealias, on a VarDecl
nl_fp_type nl_fp_var1;
// CHECK: VarDecl {{.*}} nl_fp_var1 'nl_fp_type':'void (*)() __attribute__((clang_nolock))'
nl_fp_talias nl_fp_var2;
// CHECK: VarDecl {{.*}} nl_fp_var2 'nl_fp_talias':'void (*)() __attribute__((clang_nolock))'

// On type of a FieldDecl
struct Struct {
	void (*nl_func_field)() NOLOCK;
// CHECK: FieldDecl {{.*}} nl_func_field 'void (*)() __attribute__((clang_nolock))'
};

// noalloc should be subsumed into nolock
void nl1() [[clang::nolock]] [[clang::noalloc]];
// CHECK: FunctionDecl {{.*}} nl1 'void () __attribute__((clang_nolock))'

void nl2() [[clang::noalloc]] [[clang::nolock]];
// CHECK: FunctionDecl {{.*}} nl2 'void () __attribute__((clang_nolock))'

// --- Blocks ---

// On the type of the VarDecl holding a BlockDecl
void (^nl_block1)() NOLOCK = ^() NOLOCK {};
// CHECK: VarDecl {{.*}} nl_block1 'void (^)() __attribute__((clang_nolock))'

int (^nl_block2)() NOLOCK = ^() NOLOCK { return 0; };
// CHECK: VarDecl {{.*}} nl_block2 'int (^)() __attribute__((clang_nolock))'

// The operand of the CallExpr is an ImplicitCastExpr of a DeclRefExpr -> nl_block which hold the attribute
static void blockCaller() { nl_block1(); }
// CHECK: DeclRefExpr {{.*}} 'nl_block1' 'void (^)() __attribute__((clang_nolock))'

// $$$ TODO: There are still some loose ends in all the methods of the lambda
auto nl_lambda = []() NOLOCK {};

// =========================================================================================
// Square brackets, false

void nl_func_false() [[clang::nolock(false)]];
// CHECK: FunctionDecl {{.*}} nl_func_false 'void () __attribute__((clang_nolock(false)))'

// TODO: Duplicate the above for noalloc
// TODO: Duplicate the above for GNU-style attribute?
