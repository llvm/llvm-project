// RUN: %clang_cc1 -std=c2y %s -verify -Wno-gnu-alignof-expression

/* WG14 N3244: Partial
 * Slay Some Earthly Demons I
 *
 * NB: the committee adopted:
 *   Annex J Item 21 (including additional change) -- no, we lack explicit documentation
 *   Annex J Item 56 -- yes
 *   Annex J Item 57 Option 1 -- yes
 *   Annex J Item 67 -- no
 *   Annex J Item 69 (alternative wording for semantics) -- no
 */

void reg_array(void) {
  // Decay of an array with the register storage class specifier has gone from
  // explicit undefined behavior to be implementation defined instead. Clang
  // does not support this.
  register int array[10];
  (void)sizeof(array); // okay
  int *vp = array;    // expected-error {{address of register variable requested}}
  int val = array[0]; // expected-error {{address of register variable requested}}
}

struct F;  // expected-note {{forward declaration of 'struct F'}}
void incomplete_no_linkage(struct F); // okay
void incomplete_no_linkage(struct F f) { // expected-error {{variable has incomplete type 'struct F'}}
  struct G g; // expected-error {{variable has incomplete type 'struct G'}} \
                 expected-note {{forward declaration of 'struct G'}}
  int i[];    // expected-error {{definition of variable with array type needs an explicit size or an initializer}}
}

void block_scope_non_extern_func_decl(void) {
  static void f(void); // expected-error {{function declared in block scope cannot have 'static' storage class}}
  extern void g(void); // okay
  __private_extern__ void h(void); // okay
}

// FIXME: this function should be diagnosed as it is never defined in the TU.
extern inline void never_defined_extern_inline(void);

// While this declaration is fine because the function is defined within the TU.
extern inline void is_defined_extern_inline(void);
extern inline void is_defined_extern_inline(void) {}

int NoAlignmentOnOriginalDecl;
// FIXME: the original declaration has no alignment specifier, so the
// declaration below should be diagnosed due to the incompatible alignment
// specifier.
_Alignas(8) int NoAlignmentOnOriginalDecl;
_Static_assert(_Alignof(NoAlignmentOnOriginalDecl) == 8, "");

_Alignas(8) int AlignmentOnOriginalDecl; // expected-note {{declared with '_Alignas' attribute here}}
// FIXME: this should be accepted because the redeclaration has no alignment
// specifier.
int AlignmentOnOriginalDecl; // expected-error {{'_Alignas' must be specified on definition if it is specified on any declaration}}
_Static_assert(_Alignof(AlignmentOnOriginalDecl) == 8, "");

long long CompatibleAlignment;
_Static_assert(_Alignof(__typeof__(CompatibleAlignment)) == _Alignof(long long), "");
_Alignas(_Alignof(long long)) long long CompatibleAlignment; // Okay, alignment is the same as the implied alignment

_Alignas(_Alignof(long long)) long long CompatibleAlignment2; // expected-note {{declared with '_Alignas' attribute here}}
// FIXME: this should be accepted because the redeclaration has no alignment
// specifier.
long long CompatibleAlignment2; // expected-error {{'_Alignas' must be specified on definition if it is specified on any declaration}}

// FIXME: this should be accepted because the definition specifies the
// alignment and a subsequent declaration does not specify any alignment.
_Alignas(8) long long DefnWithInit = 12; // expected-note {{declared with '_Alignas' attribute here}}
long long DefnWithInit; // expected-error {{'_Alignas' must be specified on definition if it is specified on any declaration}}

// This is accepted because the definition has an alignment specifier and the
// subsequent redeclaration does not specify an alignment.
_Alignas(8) long long DefnWithInit2 = 12;
extern long long DefnWithInit2;

// FIXME: this should be accepted because the definition specifies the
// alignment and a subsequent declaration specifies a compatible alignment.
long long DefnWithInit3 = 12; // expected-error {{'_Alignas' must be specified on definition if it is specified on any declaration}}
_Alignas(_Alignof(long long)) long long DefnWithInit3; // expected-note {{declared with '_Alignas' attribute here}}

_Alignas(8) int Mismatch;  // expected-note {{previous declaration is here}}
_Alignas(16) int Mismatch; // expected-error {{redeclaration has different alignment requirement (16 vs 8)}}
