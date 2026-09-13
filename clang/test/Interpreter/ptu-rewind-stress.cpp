// REQUIRES: host-supports-jit
//
// Usage as a lit test (pipes every line below into clang-repl as a separate
// incremental input, the same way fail.cpp / code-undo.cpp do):
//   RUN: cat %s | clang-repl | FileCheck %s
//

extern "C" int printf(const char *, ...);

// =======================================================================
// Test 1: Sema::IdResolver keeps a dangling entry after a failed parse.
// =======================================================================
int dangling_name = 7; intentional_error_type garbage_after_dangling_name;

int dangling_name = 99;
auto t1 = printf("dangling_name = %d\n", dangling_name);
// CHECK: dangling_name = 99

// =======================================================================
// Test 2: Amplify test 1 by repeating the fail/redeclare cycle several
// times on the same identifier.
// =======================================================================
int stress_var = 0; intentional_error_type e0;
int stress_var = 1; intentional_error_type e1;
int stress_var = 2; intentional_error_type e2;
int stress_var = 3; intentional_error_type e3;
int stress_var = 4; intentional_error_type e4;
int stress_var = 42;
auto t2 = printf("stress_var = %d\n", stress_var);
// CHECK: stress_var = 42

// =======================================================================
// Test 3: Sema::PendingInstantiations does not survive a failed parse.
// =======================================================================
template <typename T> struct Boom { void trigger() { T v; (void)v; } };
Boom<int> boom_instance; boom_instance.trigger(); intentional_error_type boom_garbage;

// This chunk's own GlobalEagerInstantiationScope/LocalEagerInstantiationScope
// pass (constructed fresh in every ParseOrWrapTopLevelDecl call) is where a
// leftover PendingInstantiations entry from the failed chunk above would
// get processed against poisoned memory.
int after_boom_marker = 1;
auto t3 = printf("after_boom_marker = %d\n", after_boom_marker);
// CHECK: after_boom_marker = 1

// =======================================================================
// Test 4: ASTContextStateStash's Types.size() early-return guard skips
// cleanup of caches that don't depend on interning a new Type.
// =======================================================================
struct AlreadyKnown { int a; int b; };
unsigned long sz_known = sizeof(AlreadyKnown); intentional_error_type sizeof_garbage;

// Recomputing sizeof() on the same, still-live type must not dereference a
// stale ASTRecordLayout* left behind by the chunk above.
unsigned long sz_known2 = sizeof(AlreadyKnown);
auto t4 = printf("sizeof(AlreadyKnown) = %lu\n", sz_known2);
// CHECK: sizeof(AlreadyKnown) = 8

// =======================================================================
// Test 5: Virtual dispatch bookkeeping (Sema::VTableUses/VTablesUsed,
// ASTContext::KeyFunctions) is assert-only / guard-gated, same as above,
// but exercised through polymorphic classes and `new` instead of sizeof.
// =======================================================================
struct Base { virtual int val() { return 1; } virtual ~Base() {} };
struct Derived : Base { int val() override { return 2; } };
Base* bp = new Derived(); int vv = bp->val(); intentional_error_type vtable_garbage;

// A fresh Base*/Derived pair exercising the same key-function/vtable-use
// bookkeeping. If the discarded attempt above left stale KeyFunctions /
// VTablesUsed state pointing at poisoned memory, codegen for *this* vtable
// can be skipped, corrupted, or crash outright.
Base* bp2 = new Derived();
auto t5 = printf("val = %d\n", bp2->val());
// CHECK: val = 2

// =======================================================================
// Test 6: StringLiteralCache has no erase logic, so processing PRETTY_FUNCTION
// leaves an extra entry and can make the cache-size assertion fail deterministically.
// =======================================================================
void uses_predefined_expr() { const char* pf = __PRETTY_FUNCTION__; (void)pf; } intentional_error_type predefined_garbage;

void uses_predefined_expr_again() { const char* pf = __PRETTY_FUNCTION__; (void)pf; }
int pf_ran = 1;
auto t6 = printf("pf_ran = %d\n", pf_ran);
// CHECK: pf_ran = 1

// =======================================================================
// Test 7: Cross-declaration merging state can survive a failed using-declaration,
// leaving stale UsingShadowDecl bookkeeping that may affect a later successful using.
// =======================================================================
namespace NS { int helper() { return 10; } }
using NS::helper; intentional_error_type merge_garbage;

using NS::helper;
auto t7 = printf("helper() = %d\n", helper());
// CHECK: helper() = 10

%quit
