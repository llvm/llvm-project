// RUN: cat %s | clang-repl | FileCheck %s
// Tests for static const member materialization in clang-repl.
// See https://github.com/llvm/llvm-project/issues/146956

extern "C" int printf(const char*, ...);

struct Foo { static int const bar { 5 }; static int const baz { 10 }; };

// Test 1: Taking address of static const member
int const * p = &Foo::bar;
printf("Address test: %d\n", *p);
// CHECK: Address test: 5

// Test 2: Multiple static const members in same class
int const * q = &Foo::baz;
printf("Second member test: %d\n", *q);
// CHECK: Second member test: 10

// Test 3: static constexpr member (variant of in-class init)
struct Qux { static constexpr int val = 99; };
int const *p3 = &Qux::val;
printf("Constexpr test: %d\n", *p3);
// CHECK: Constexpr test: 99

// Test 4: Passing static const member by reference (exercises CGExpr.cpp path)
// NOTE: Uses a separate struct to ensure this is the first odr-use of RefOnly::val
struct RefOnly { static int const val { 77 }; };
void useRef(int const &x) { printf("Ref test: %d\n", x); }
useRef(RefOnly::val);
// CHECK: Ref test: 77

// ============================================================================
// Negative tests - cases that should NOT trigger materialization
// ============================================================================

// Test 5: Out-of-class definition (no need to materialize - already defined)
struct OutOfLine { static const int value; };
const int OutOfLine::value = 23;
int const *p5 = &OutOfLine::value;
printf("Out-of-line test: %d\n", *p5);
// CHECK: Out-of-line test: 23

// Test 6: Non-const static member (normal code gen path)
struct NonConst { static int value; };
int NonConst::value = 42;
int *p6 = &NonConst::value;
printf("Non-const test: %d\n", *p6);
// CHECK: Non-const test: 42

// ============================================================================
// Edge case tests
// ============================================================================

// Test 7: Repeated address-of reuses same definition
int const *p7 = &Foo::bar;
printf("Reuse test: %d\n", (*p == *p7) ? 1 : 0);
// CHECK: Reuse test: 1

%quit
