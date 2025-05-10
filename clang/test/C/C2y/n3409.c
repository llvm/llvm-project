// RUN: %clang_cc1 -verify -std=c2y -pedantic %s
// RUN: %clang_cc1 -verify=pre-c2y -std=c2y -Wpre-c2y-compat %s
// RUN: %clang_cc1 -verify=ext -std=c23 -pedantic %s
// expected-no-diagnostics

/* WG14 N3409: Clang 21
 * Slay Some Earthly Demons X
 *
 * Removes the requirement that an expression with type void cannot be used in
 * any way. This was making it UB to use a void expression in a _Generic
 * selection expression for no good reason, as well as making it UB to cast a
 * void expression to void, etc.
 */

extern void x;
void foo() {
  // FIXME: this is technically an extension before C2y and should be diagnosed
  // under -pedantic.
  (void)(void)1;
  // FIXME: same with this.
  x;
  _Generic(x, void: 1);      /* pre-c2y-warning {{use of incomplete type 'void' in a '_Generic' association is incompatible with C standards before C2y}}
                                ext-warning {{incomplete type 'void' in a '_Generic' association is a C2y extension}}
                              */
  _Generic(x, typeof(x): 1); /* pre-c2y-warning {{use of incomplete type 'typeof (x)' (aka 'void') in a '_Generic' association is incompatible with C standards before C2y}}
                                ext-warning {{incomplete type 'typeof (x)' (aka 'void') in a '_Generic' association is a C2y extension}}
                              */
  (void)_Generic(void, default : 1); /* pre-c2y-warning {{passing a type argument as the first operand to '_Generic' is incompatible with C standards before C2y}}
                                        ext-warning {{passing a type argument as the first operand to '_Generic' is a C2y extension}}
                                      */

  // This is not sufficiently important of an extension to warrant a "not
  // compatible with standards before C2y" warning, but it is an extension in
  // C23 and earlier.
  return x; // ext-warning {{void function 'foo' should not return void expression}}
}
