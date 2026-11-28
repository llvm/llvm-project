// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl | FileCheck %s

extern "C" int printf(const char *, ...);

// 1. Block-scope shadowing rolled back inside a failed function body.
int shadow_val = 1;
int use_shadow() { int shadow_val = 2; return shadow_val; }
auto r1 = printf("shadow_val=%d use_shadow=%d\n", shadow_val, use_shadow());
// CHECK: shadow_val=1 use_shadow=2

int use_shadow_bad() { int shadow_val = 99; { int shadow_val = 100; (void)shadow_val; } return shadow_val; } intentional_error_type shadow_garbage;
auto r2 = printf("shadow_val=%d use_shadow=%d\n", shadow_val, use_shadow());
// CHECK: shadow_val=1 use_shadow=2

// 2. Tag namespace vs ordinary namespace after rollback (IdResolver keeps
// separate chains per lookup id-namespace for the same spelling).
struct TagName { int x; };

int TagName = 5; struct TagName also_tag; intentional_error_type tag_garbage;
struct TagName t3; t3.x = 7;
int TagName2 = 42;
auto r3 = printf("t3.x=%d TagName2=%d\n", t3.x, TagName2);
// CHECK: t3.x=7 TagName2=42

// 3. ADL (argument-dependent lookup) across a rolled-back use site.
namespace adl_ns { struct Widget {}; int probe(Widget) { return 11; } }

int trigger_adl_bad() { adl_ns::Widget w; return probe(w); } intentional_error_type adl_garbage;
adl_ns::Widget w2;
auto r4 = printf("probe=%d\n", probe(w2));
// CHECK: probe=11

// 4. Enumerator (EnumConstantDecl) shadowing rolled back.
enum Color { Red, Green, Blue };

int enum_test_bad() { enum Local { Red, Yellow }; return Red; } intentional_error_type enum_garbage;
int check_global_red() { return Red; }
auto r5 = printf("Red=%d\n", check_global_red());
// CHECK: Red=0

// 5. Overload set growth rolled back.
int overload_fn(int x) { return x + 1; }

int overload_fn(double x) { return (int)(x + 2.5); } intentional_error_type overload_garbage;
auto r6 = printf("overload_fn(3)=%d\n", overload_fn(3));
// CHECK: overload_fn(3)=4

%quit
