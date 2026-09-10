// RUN: %clang_cc1 -fsyntax-only -fexperimental-overflow-behavior-types -verify %s

// An overflow behavior type is a type specifier, not a qualifier, so
// __typeof_unqual__ strips const/volatile off of it but keeps the overflow
// behavior itself. The type as written keeps its qualifiers on the underlying
// type, so they are stripped by OverflowBehaviorType::getSplitUnqualifiedType()
// rather than by desugaring, which cannot see through the node.
//
// The canonical type was always right here, so these tests check the printed
// type: an OverflowBehaviorType whose qualifiers cannot be stripped reports
// itself as qualified even though it is not.

typedef int __ob_trap tint;
typedef const int cint;
typedef int __attribute__((overflow_behavior(trap))) attr_tint;

// Every spelling of a qualified overflow behavior type strips down to the same
// unqualified overflow behavior type. Assigning to a 'char *' prints it.
void unqual_strips_qualifiers_only(void) {
  __typeof_unqual__(__ob_trap int) *a;
  char *p1 = a; // expected-error {{initializing 'char *' with an expression of type 'typeof_unqual(__ob_trap int) *' (aka '__ob_trap int *')}}

  __typeof_unqual__(const __ob_trap int) *b;
  char *p2 = b; // expected-error {{initializing 'char *' with an expression of type 'typeof_unqual(__ob_trap const int) *' (aka '__ob_trap int *')}}

  __typeof_unqual__(__ob_trap const int) *c;
  char *p3 = c; // expected-error {{initializing 'char *' with an expression of type 'typeof_unqual(__ob_trap const int) *' (aka '__ob_trap int *')}}

  __typeof_unqual__(volatile __ob_trap int) *d;
  char *p4 = d; // expected-error {{initializing 'char *' with an expression of type 'typeof_unqual(__ob_trap volatile int) *' (aka '__ob_trap int *')}}

  __typeof_unqual__(const volatile __ob_trap int) *e;
  char *p5 = e; // expected-error {{initializing 'char *' with an expression of type 'typeof_unqual(__ob_trap const volatile int) *' (aka '__ob_trap int *')}}

  __typeof_unqual__(const tint) *f;
  char *p6 = f; // expected-error {{initializing 'char *' with an expression of type 'typeof_unqual(const tint) *' (aka '__ob_trap int *')}}

  // The qualifier is reachable only through the typedef.
  __typeof_unqual__(__ob_trap cint) *g;
  char *p7 = g; // expected-error {{initializing 'char *' with an expression of type 'typeof_unqual(__ob_trap cint) *' (aka '__ob_trap int *')}}

  __typeof_unqual__(const attr_tint) *h;
  char *p8 = h; // expected-error {{initializing 'char *' with an expression of type 'typeof_unqual(const attr_tint) *' (aka '__ob_trap int *')}}
}

// The overflow behavior itself must survive __typeof_unqual__.
void overflow_behavior_survives(void) {
  __typeof_unqual__(const __ob_trap int) *a;
  int *p1 = a; // expected-warning {{discards overflow behavior}}
  __ob_wrap int *p2 = a; // expected-error {{with incompatible overflow behavior types ('__ob_wrap' and '__ob_trap')}}
  __ob_trap int *p3 = a;
}

// The stripped type really is writable, not just printed as writable.
void unqualified_is_writable(void) {
  __typeof_unqual__(const __ob_trap int) x;
  x = 1;
  __typeof_unqual__(__ob_trap cint) y;
  y = 1;
}

// Qualifiers on an overflow behavior type still apply, and still print, when
// they are not being stripped. Every spelling prints the same way.
void qualifiers_still_apply(void) {
  const __ob_trap int a = 0; // expected-note {{variable 'a' declared const here}}
  a = 1; // expected-error {{cannot assign to variable 'a' with const-qualified type '__ob_trap const int'}}
  __ob_trap const int b = 0; // expected-note {{variable 'b' declared const here}}
  b = 1; // expected-error {{cannot assign to variable 'b' with const-qualified type '__ob_trap const int'}}
  __ob_trap cint c = 0; // expected-note {{variable 'c' declared const here}}
  c = 1; // expected-error {{cannot assign to variable 'c' with const-qualified type '__ob_trap cint'}}
  const tint d = 0; // expected-note {{variable 'd' declared const here}}
  d = 1; // expected-error {{cannot assign to variable 'd' with const-qualified type 'const tint'}}
}

// Controls: stripping is unchanged for types with no overflow behavior.
extern int e;
extern __typeof_unqual__(const int) e;
extern __typeof_unqual__(cint) e;
extern __typeof_unqual__(_Atomic int) e;
