// RUN: %clang_cc1 -verify=ped -std=c23 -Wall -pedantic %s
// RUN: %clang_cc1 -verify=yay -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify=pre -std=c2y -Wpre-c2y-compat -Wall -pedantic %s
// RUN: %clang_cc1 -verify=gnu -Wall -Wgnu -x c++ %s
// RUN: %clang_cc1 -verify=yay -Wall -Wgnu -Wno-gnu-imaginary-constant -x c++ %s


/* WG14 N3298: Yes
 * Introduce complex literals v. 2
 *
 * This introduces two suffixes for making complex literals: i and j (and I and
 * J), which can be combined in any order with the other floating literal
 * suffixes.
 *
 * We support these suffixes in older language modes as a conforming extension.
 * It used to be a GNU extension, but now it's a C2y extension.
 */

// yay-no-diagnostics

static_assert(_Generic(12.0i, _Complex double : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0fi, _Complex float : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0li, _Complex long double : 1, default : 0)); /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0if, _Complex float : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0il, _Complex long double : 1, default : 0)); /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */

static_assert(_Generic(12.0I, _Complex double : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0fI, _Complex float : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0lI, _Complex long double : 1, default : 0)); /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0If, _Complex float : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0Il, _Complex long double : 1, default : 0)); /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */

static_assert(_Generic(12.0j, _Complex double : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0fj, _Complex float : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0lj, _Complex long double : 1, default : 0)); /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0jf, _Complex float : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0jl, _Complex long double : 1, default : 0)); /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */

static_assert(_Generic(12.0J, _Complex double : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0fJ, _Complex float : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0lJ, _Complex long double : 1, default : 0)); /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0Jf, _Complex float : 1, default : 0));       /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */
static_assert(_Generic(12.0Jl, _Complex long double : 1, default : 0)); /* gnu-warning {{imaginary constants are a GNU extension}}
                                                                           ped-warning {{imaginary constants are a C2y extension}}
                                                                           pre-warning {{imaginary constants are incompatible with C standards before C2y}}
                                                                         */

