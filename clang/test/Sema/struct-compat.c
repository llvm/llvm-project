/* RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify
 */

extern struct {int a;} x; // expected-note {{previous definition is here}}
extern struct {int a;} x; // expected-error {{redefinition of 'x'}}

struct x;
int a(struct x* b) {
// Per C99 6.7.2.3, since the outer and inner "struct x"es have different
// scopes, they don't refer to the same type, and are therefore incompatible
struct x {int a;} *c = b; // expected-warning {{incompatible pointer types}}
}

struct x {int a;} r;
int b() {
struct x {char x;} s = r; // expected-error {{initializing 'struct x' with an expression of incompatible type 'struct x'}}
}
