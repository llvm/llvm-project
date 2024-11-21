// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify -Wall -pedantic %s

/* WG14 N3344: Yes
 * Slay Some Earthly Demons VI
 *
 * A 'void' parameter cannot have any qualifiers, storage class specifiers, or
 * be followed by an ellipsis.
 *
 * Note: Clang treats 'register void' as being a DR and rejects it in all
 * language modes; there's no evidence that this will break users and it's not
 * clear what the programmer intended if they wrote such code anyway. This
 * matches GCC's behavior.
 */

void baz(volatile void);         // expected-error {{'void' as parameter must not have type qualifiers}}
void bar(const void);            // expected-error {{'void' as parameter must not have type qualifiers}}
void foo(register void);         // expected-error {{invalid storage class specifier in function declarator}}
void foop(void register);        // expected-error {{invalid storage class specifier in function declarator}}
void quux(static void);          // expected-error {{invalid storage class specifier in function declarator}}
void quobble(auto void);         // expected-error {{invalid storage class specifier in function declarator}}
void quubble(extern void);       // expected-error {{invalid storage class specifier in function declarator}}
// FIXME: it's odd that these aren't diagnosed as storage class specifiers.
#if __STDC_VERSION__ >= 202311L
void quibble(constexpr void);    // expected-error {{function parameter cannot be constexpr}}
#endif
#if __STDC_VERSION__ >= 201112L
void quabble(_Thread_local void); // expected-error {{'_Thread_local' is only allowed on variable declarations}}
#endif
void bing(void, ...);            // expected-error {{'void' must be the first and only parameter if specified}}

// These declarations are fine.
void one(register void *);
void two(void register *);
void three(register void * (*)[4]);
