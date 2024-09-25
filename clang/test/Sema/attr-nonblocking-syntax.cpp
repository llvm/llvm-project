// RUN: %clang_cc1 %s -ast-dump -fblocks | FileCheck %s

// Make sure that the attribute gets parsed and attached to the correct AST elements.

#pragma clang diagnostic ignored "-Wunused-variable"

// =========================================================================================
// Square brackets, true

namespace square_brackets {

// 1. On the type of the FunctionDecl
void nl_function() [[clang::nonblocking]];
// CHECK: FunctionDecl {{.*}} nl_function 'void () __attribute__((nonblocking))'

// 2. On the type of the VarDecl holding a function pointer
void (*nl_func_a)() [[clang::nonblocking]];
// CHECK: VarDecl {{.*}} nl_func_a 'void (*)() __attribute__((nonblocking))'

// 3. On the type of the ParmVarDecl of a function parameter
static void nlReceiver(void (*nl_func)() [[clang::nonblocking]]);
// CHECK: ParmVarDecl {{.*}} nl_func 'void (*)() __attribute__((nonblocking))'

// 4. As an AttributedType within the nested types of a typedef
typedef void (*nl_fp_type)() [[clang::nonblocking]];
// CHECK: TypedefDecl {{.*}} nl_fp_type 'void (*)() __attribute__((nonblocking))'
using nl_fp_talias = void (*)() [[clang::nonblocking]];
// CHECK: TypeAliasDecl {{.*}} nl_fp_talias 'void (*)() __attribute__((nonblocking))'

// 5. From a typedef or typealias, on a VarDecl
nl_fp_type nl_fp_var1;
// CHECK: VarDecl {{.*}} nl_fp_var1 'nl_fp_type':'void (*)() __attribute__((nonblocking))'
nl_fp_talias nl_fp_var2;
// CHECK: VarDecl {{.*}} nl_fp_var2 'nl_fp_talias':'void (*)() __attribute__((nonblocking))'

// 6. On type of a FieldDecl
struct Struct {
  void (*nl_func_field)() [[clang::nonblocking]];
// CHECK: FieldDecl {{.*}} nl_func_field 'void (*)() __attribute__((nonblocking))'
};

// nonallocating should NOT be subsumed into nonblocking
void nl1() [[clang::nonblocking]] [[clang::nonallocating]];
// CHECK: FunctionDecl {{.*}} nl1 'void () __attribute__((nonblocking)) __attribute__((nonallocating))'

void nl2() [[clang::nonallocating]] [[clang::nonblocking]];
// CHECK: FunctionDecl {{.*}} nl2 'void () __attribute__((nonblocking)) __attribute__((nonallocating))'

decltype(nl1) nl3;
// CHECK: FunctionDecl {{.*}} nl3 'decltype(nl1)':'void () __attribute__((nonblocking)) __attribute__((nonallocating))'

// Attribute propagates from base class virtual method to overrides.
struct Base {
  virtual void nb_method() [[clang::nonblocking]];
};
struct Derived : public Base {
  void nb_method() override;
  // CHECK: CXXMethodDecl {{.*}} nb_method 'void () __attribute__((nonblocking))'
};

// Dependent expression
template <bool V>
struct Dependent {
  void nb_method2() [[clang::nonblocking(V)]];
  // CHECK: CXXMethodDecl {{.*}} nb_method2 'void () __attribute__((nonblocking(V)))'
};

// --- Blocks ---

// On the type of the VarDecl holding a BlockDecl
void (^nl_block1)() [[clang::nonblocking]] = ^() [[clang::nonblocking]] {};
// CHECK: VarDecl {{.*}} nl_block1 'void (^)() __attribute__((nonblocking))'

int (^nl_block2)() [[clang::nonblocking]] = ^() [[clang::nonblocking]] { return 0; };
// CHECK: VarDecl {{.*}} nl_block2 'int (^)() __attribute__((nonblocking))'

// The operand of the CallExpr is an ImplicitCastExpr of a DeclRefExpr -> nl_block which hold the attribute
static void blockCaller() { nl_block1(); }
// CHECK: DeclRefExpr {{.*}} 'nl_block1' 'void (^)() __attribute__((nonblocking))'

// --- Lambdas ---

// On the operator() of a lambda's CXXMethodDecl
auto nl_lambda = []() [[clang::nonblocking]] {};
// CHECK: CXXMethodDecl {{.*}} operator() 'void () const __attribute__((nonblocking))' inline

// =========================================================================================
// Square brackets, false

void nl_func_false() [[clang::blocking]];
// CHECK: FunctionDecl {{.*}} nl_func_false 'void () __attribute__((blocking))'

auto nl_lambda_false = []() [[clang::blocking]] {};
// CHECK: CXXMethodDecl {{.*}} operator() 'void () const __attribute__((blocking))'

} // namespace square_brackets

// =========================================================================================
// GNU-style attribute, true

namespace gnu_style {

// 1. On the type of the FunctionDecl
void nl_function() __attribute__((nonblocking));
// CHECK: FunctionDecl {{.*}} nl_function 'void () __attribute__((nonblocking))'

// 1a. Alternate placement on the FunctionDecl
__attribute__((nonblocking)) void nl_function();
// CHECK: FunctionDecl {{.*}} nl_function 'void () __attribute__((nonblocking))'

// 2. On the type of the VarDecl holding a function pointer
void (*nl_func_a)() __attribute__((nonblocking));
// CHECK: VarDecl {{.*}} nl_func_a 'void (*)() __attribute__((nonblocking))'

// 2a. Alternate attribute placement on VarDecl
__attribute__((nonblocking)) void (*nl_func_b)();
// CHECK: VarDecl {{.*}} nl_func_b 'void (*)() __attribute__((nonblocking))'

// 3. On the type of the ParmVarDecl of a function parameter
static void nlReceiver(void (*nl_func)() __attribute__((nonblocking)));
// CHECK: ParmVarDecl {{.*}} nl_func 'void (*)() __attribute__((nonblocking))'

// 4. As an AttributedType within the nested types of a typedef
// Note different placement from square brackets for the typealias.
typedef void (*nl_fp_type)() __attribute__((nonblocking));
// CHECK: TypedefDecl {{.*}} nl_fp_type 'void (*)() __attribute__((nonblocking))'
using nl_fp_talias = __attribute__((nonblocking)) void (*)();
// CHECK: TypeAliasDecl {{.*}} nl_fp_talias 'void (*)() __attribute__((nonblocking))'

// 5. From a typedef or typealias, on a VarDecl
nl_fp_type nl_fp_var1;
// CHECK: VarDecl {{.*}} nl_fp_var1 'nl_fp_type':'void (*)() __attribute__((nonblocking))'
nl_fp_talias nl_fp_var2;
// CHECK: VarDecl {{.*}} nl_fp_var2 'nl_fp_talias':'void (*)() __attribute__((nonblocking))'

// 6. On type of a FieldDecl
struct Struct {
  void (*nl_func_field)() __attribute__((nonblocking));
// CHECK: FieldDecl {{.*}} nl_func_field 'void (*)() __attribute__((nonblocking))'
};

} // namespace gnu_style

// =========================================================================================
// nonallocating and allocating - quick checks because the code paths are generally
// identical after parsing.

void na_function() [[clang::nonallocating]];
// CHECK: FunctionDecl {{.*}} na_function 'void () __attribute__((nonallocating))'

void na_true_function() [[clang::nonallocating(true)]];
// CHECK: FunctionDecl {{.*}} na_true_function 'void () __attribute__((nonallocating))'

void na_false_function() [[clang::nonallocating(false)]];
// CHECK: FunctionDecl {{.*}} na_false_function 'void () __attribute__((allocating))'

void alloc_function() [[clang::allocating]];
// CHECK: FunctionDecl {{.*}} alloc_function 'void () __attribute__((allocating))'


// =========================================================================================
// Non-blocking with an expression parameter

void t0() [[clang::nonblocking(1 - 1)]];
// CHECK: FunctionDecl {{.*}} t0 'void () __attribute__((blocking))'
void t1() [[clang::nonblocking(1 + 1)]];
// CHECK: FunctionDecl {{.*}} t1 'void () __attribute__((nonblocking))'

template <bool V>
struct ValueDependent {
  void nb_method() [[clang::nonblocking(V)]];
};

void t3() [[clang::nonblocking]]
{
  ValueDependent<false> x1;
  x1.nb_method();
// CHECK: ClassTemplateSpecializationDecl {{.*}} ValueDependent
// CHECK: TemplateArgument integral 'false'
// CHECK: CXXMethodDecl {{.*}} nb_method 'void () __attribute__((blocking))'

   ValueDependent<true> x2;
   x2.nb_method();
// CHECK: ClassTemplateSpecializationDecl {{.*}} ValueDependent
// CHECK: TemplateArgument integral 'true'
// CHECK: CXXMethodDecl {{.*}} nb_method 'void () __attribute__((nonblocking))'
}

template <typename X>
struct TypeDependent {
  void td_method() [[clang::nonblocking(X::is_nb)]];
};

struct NBPolicyTrue {
  static constexpr bool is_nb = true;
};

struct NBPolicyFalse {
  static constexpr bool is_nb = false;
};

void t4()
{
  TypeDependent<NBPolicyFalse> x1;
  x1.td_method();
// CHECK: ClassTemplateSpecializationDecl {{.*}} TypeDependent
// CHECK: TemplateArgument type 'NBPolicyFalse'
// CHECK: CXXMethodDecl {{.*}} td_method 'void () __attribute__((blocking))'

  TypeDependent<NBPolicyTrue> x2;
  x2.td_method();
// CHECK: ClassTemplateSpecializationDecl {{.*}} TypeDependent
// CHECK: TemplateArgument type 'NBPolicyTrue'
// CHECK: CXXMethodDecl {{.*}} td_method 'void () __attribute__((nonblocking))'
}

