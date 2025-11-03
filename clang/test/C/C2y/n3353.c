// RUN: %clang_cc1 -verify=expected,c2y,c -pedantic -std=c2y %s
// RUN: %clang_cc1 -verify=expected,c2y,compat -Wpre-c2y-compat -std=c2y %s
// RUN: %clang_cc1 -verify=expected,ext,c -pedantic -std=c23 %s
// RUN: %clang_cc1 -verify=expected,cpp -pedantic -x c++ -Wno-c11-extensions %s


/* WG14 N3353: Clang 21
 * Obsolete implicitly octal literals and add delimited escape sequences
 */

constexpr int i = 0234;  // c2y-warning {{octal literals without a '0o' prefix are deprecated}}
constexpr int j = 0o234; /* ext-warning {{octal integer literals are a C2y extension}}
                            cpp-warning {{octal integer literals are a Clang extension}}
                            compat-warning {{octal integer literals are incompatible with standards before C2y}}
                          */

static_assert(i == 156);
static_assert(j == 156);

// Show that 0O is the same as Oo (tested above)
static_assert(0O1234 == 0o1234);  /* ext-warning 2 {{octal integer literals are a C2y extension}}
                                     cpp-warning 2 {{octal integer literals are a Clang extension}}
                                     compat-warning 2 {{octal integer literals are incompatible with standards before C2y}}
                                   */

// Show that you can use them with the usual integer literal suffixes.
static_assert(0o234ull == 156);  /* ext-warning {{octal integer literals are a C2y extension}}
                                    cpp-warning {{octal integer literals are a Clang extension}}
                                    compat-warning {{octal integer literals are incompatible with standards before C2y}}
                                  */

// And it's still a valid null pointer constant.
static const void *ptr = 0o0;  /* ext-warning {{octal integer literals are a C2y extension}}
                                  cpp-warning {{octal integer literals are a Clang extension}}
                                  compat-warning {{octal integer literals are incompatible with standards before C2y}}
                                */

// Demonstrate that it works fine in the preprocessor.
#if 0o123 != 0x53   /* ext-warning {{octal integer literals are a C2y extension}}
                       cpp-warning {{octal integer literals are a Clang extension}}
                       compat-warning {{octal integer literals are incompatible with standards before C2y}}
                     */
#error "oh no, math stopped working!"
#endif

// 0 by itself is not deprecated, of course.
int k1                = 0;
unsigned int k2       = 0u;
long k3               = 0l;
unsigned long k4      = 0ul;
long long k5          = 0ll;
unsigned long long k6 = 0ull;

// Test a preprocessor use of 0 by itself, which is also not deprecated.
#if 0
#endif

// Make sure there are no surprises with auto and type deduction. Promotion
// turns this into an 'int', and 'constexpr' implies 'const'.
constexpr auto l = 0o1234567; /* ext-warning {{octal integer literals are a C2y extension}}
                                 cpp-warning {{octal integer literals are a Clang extension}}
                                 compat-warning {{octal integer literals are incompatible with standards before C2y}}
                              */
static_assert(l == 0x53977);
static_assert(__extension__ _Generic(typeof(0o1), typeof(01) : 1, default : 0)); /* c2y-warning {{octal literals without a '0o' prefix are deprecated}}
                                                                                    compat-warning {{passing a type argument as the first operand to '_Generic' is incompatible with C standards before C2y}}
                                                                                    compat-warning {{octal integer literals are incompatible with standards before C2y}}
                                                                                  */
static_assert(__extension__ _Generic(typeof(l), const int : 1, default : 0)); // compat-warning {{passing a type argument as the first operand to '_Generic' is incompatible with C standards before C2y}}

// Note that 0o by itself is an invalid literal.
int m = 0o; /* expected-error {{invalid suffix 'o' on integer constant}}
             */

// Ensure negation works as expected.
static_assert(-0o1234 == -668); /* ext-warning {{octal integer literals are a C2y extension}}
                                   cpp-warning {{octal integer literals are a Clang extension}}
                                   compat-warning {{octal integer literals are incompatible with standards before C2y}}
                                 */

// FIXME: it would be better to not diagnose the compat and ext warnings when
// the octal literal is invalid.
// We expect diagnostics for non-octal digits.
int n = 0o18; /* expected-error {{invalid digit '8' in octal constant}}
                 compat-warning {{octal integer literals are incompatible with standards before C2y}}
                 ext-warning {{octal integer literals are a C2y extension}}
                 cpp-warning {{octal integer literals are a Clang extension}}
               */
int o1 = 0o8; /* expected-error {{invalid suffix 'o8' on integer constant}}
               */
// FIXME: however, it matches the behavior for hex literals in terms of the
// error reported. Unfortunately, we then go on to think 0 is an octal literal
// without a prefix, which is again a bit confusing.
int o2 = 0xG; /* expected-error {{invalid suffix 'xG' on integer constant}}
               */

// Show that floating-point suffixes on octal literals are rejected.
auto f1 = 0o0.;  /* expected-error {{invalid suffix '.' on integer constant}}
                    compat-warning {{octal integer literals are incompatible with standards before C2y}}
                    ext-warning {{octal integer literals are a C2y extension}}
                    cpp-warning {{octal integer literals are a Clang extension}}
                */
auto f2 = 0o0.1; /* expected-error {{invalid suffix '.1' on integer constant}}
                    compat-warning {{octal integer literals are incompatible with standards before C2y}}
                    ext-warning {{octal integer literals are a C2y extension}}
                    cpp-warning {{octal integer literals are a Clang extension}}
                */
auto f3 = 0o0e1; /* expected-error {{invalid suffix 'e1' on integer constant}}
                    compat-warning {{octal integer literals are incompatible with standards before C2y}}
                    ext-warning {{octal integer literals are a C2y extension}}
                    cpp-warning {{octal integer literals are a Clang extension}}
                 */
auto f4 = 0o0E1; /* expected-error {{invalid suffix 'E1' on integer constant}}
                    compat-warning {{octal integer literals are incompatible with standards before C2y}}
                    ext-warning {{octal integer literals are a C2y extension}}
                    cpp-warning {{octal integer literals are a Clang extension}}
                 */

// Show that valid floating-point literals with a leading 0 do not produce octal-related warnings.
auto f5 = 0.;
auto f7 = 00.;
auto f8 = 01.;
auto f9 = 0e1;
auto f10 = 0E1;
auto f11 = 00e1;
auto f12 = 00E1;

// Ensure digit separators work as expected.
constexpr int p = 0o0'1'2'3'4'5'6'7; /* compat-warning {{octal integer literals are incompatible with standards before C2y}}
                                        ext-warning {{octal integer literals are a C2y extension}}
                                        cpp-warning {{octal integer literals are a Clang extension}}
                                      */
static_assert(p == 01234567); // c2y-warning {{octal literals without a '0o' prefix are deprecated}}
int q = 0o'0'1; /* expected-error {{invalid suffix 'o'0'1' on integer constant}}
                 */

#define M 0o123
int r = M;  /* compat-warning {{octal integer literals are incompatible with standards before C2y}}
               ext-warning {{octal integer literals are a C2y extension}}
               cpp-warning {{octal integer literals are a Clang extension}}
             */

// Also, test delimited escape sequences. Note, this paper added a delimited
// escape sequence for octal *and* hex.
auto a = "\x{12}\o{12}\N{SPARKLES}";   /* compat-warning 2 {{delimited escape sequences are incompatible with C standards before C2y}}
                                          ext-warning 2 {{delimited escape sequences are a C2y extension}}
                                          cpp-warning 2 {{delimited escape sequences are a C++23 extension}}
                                          cpp-warning {{named escape sequences are a C++23 extension}}
                                          c-warning {{named escape sequences are a Clang extension}}
                                        */

#ifdef __cplusplus
template <unsigned N>
struct S {
  static_assert(N == 0o567); /* ext-warning {{octal integer literals are a C2y extension}}
                                cpp-warning {{octal integer literals are a Clang extension}}
                                compat-warning {{octal integer literals are incompatible with standards before C2y}}
                              */
};

void foo() {
  S<0o567> s; /* ext-warning {{octal integer literals are a C2y extension}}
                 cpp-warning {{octal integer literals are a Clang extension}}
                 compat-warning {{octal integer literals are incompatible with standards before C2y}}
               */
}
#endif

#line 0123  // expected-warning {{#line directive interprets number as decimal, not octal}}
#line 0o123 // expected-error {{#line directive requires a simple digit sequence}}
