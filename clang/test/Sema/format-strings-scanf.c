// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify -Wformat-nonliteral %s

// Test that -Wformat=0 works:
// RUN: %clang_cc1 -std=c11 -fsyntax-only -Werror -Wformat=0 %s

#include <stdarg.h>
typedef __SIZE_TYPE__ size_t;
#define __SSIZE_TYPE__                                                         \
  __typeof__(_Generic((__SIZE_TYPE__)0,                                        \
                      unsigned long long int : (long long int)0,               \
                      unsigned long int : (long int)0,                         \
                      unsigned int : (int)0,                                   \
                      unsigned short : (short)0,                               \
                      unsigned char : (signed char)0))
typedef __SSIZE_TYPE__ ssize_t;

typedef __PTRDIFF_TYPE__ ptrdiff_t;
#define __UNSIGNED_PTRDIFF_TYPE__                                              \
  __typeof__(_Generic((__PTRDIFF_TYPE__)0,                                     \
                      long long int : (unsigned long long int)0,               \
                      long int : (unsigned long int)0,                         \
                      int : (unsigned int)0,                                   \
                      short : (unsigned short)0,                               \
                      signed char : (unsigned char)0))

typedef struct _FILE FILE;
typedef __WCHAR_TYPE__ wchar_t;

int fscanf(FILE * restrict, const char * restrict, ...) ;
int scanf(const char * restrict, ...) ;
int sscanf(const char * restrict, const char * restrict, ...) ;
int my_scanf(const char * restrict, ...) __attribute__((__format__(__scanf__, 1, 2)));

int vscanf(const char * restrict, va_list);
int vfscanf(FILE * restrict, const char * restrict, va_list);
int vsscanf(const char * restrict, const char * restrict, va_list);

void test(const char *s, int *i) {
  scanf(s, i); // expected-warning{{format string is not a string literal}}
  scanf("%0d", i); // expected-warning{{zero field width in scanf format string is unused}}
  scanf("%00d", i); // expected-warning{{zero field width in scanf format string is unused}}
  scanf("%d%[asdfasdfd", i, s); // expected-warning{{no closing ']' for '%[' in scanf format string}}
  scanf("%B", i); // expected-warning{{invalid conversion specifier 'B'}}

  unsigned short s_x;
  scanf ("%" "hu" "\n", &s_x); // no-warning
  scanf("%hb", &s_x);
  scanf("%y", i); // expected-warning{{invalid conversion specifier 'y'}}
  scanf("%%"); // no-warning
  scanf("%%%1$d", i); // no-warning
  scanf("%1$d%%", i); // no-warning
  scanf("%d", i, i); // expected-warning{{data argument not used by format string}}
  scanf("%*d", i); // // expected-warning{{data argument not used by format string}}
  scanf("%*d", i); // // expected-warning{{data argument not used by format string}}
  scanf("%*d%1$d", i); // no-warning

  scanf("%s", (char*)0); // no-warning
  scanf("%s", (volatile char*)0); // no-warning
  scanf("%s", (signed char*)0); // no-warning
  scanf("%s", (unsigned char*)0); // no-warning
  scanf("%hhu", (signed char*)0); // no-warning
  scanf("%hhb", (signed char*)0); // no-warning
}

void bad_length_modifiers(char *s, void *p, wchar_t *ws, long double *ld) {
  scanf("%hhs", "foo"); // expected-warning{{length modifier 'hh' results in undefined behavior or no effect with 's' conversion specifier}}
  scanf("%1$zp", &p); // expected-warning{{length modifier 'z' results in undefined behavior or no effect with 'p' conversion specifier}}
  scanf("%ls", ws); // no-warning
  scanf("%#.2Lf", ld); // expected-warning{{invalid conversion specifier '#'}}
}

// Test that the scanf call site is where the warning is attached.  If the
// format string is somewhere else, point to it in a note.
void pr9751(void) {
  int *i;
  char str[100];
  const char kFormat1[] = "%00d"; // expected-note{{format string is defined here}}}
  scanf(kFormat1, i); // expected-warning{{zero field width in scanf format string is unused}}
  scanf("%00d", i); // expected-warning{{zero field width in scanf format string is unused}}
  const char kFormat2[] = "%["; // expected-note{{format string is defined here}}}
  scanf(kFormat2, str); // expected-warning{{no closing ']' for '%[' in scanf format string}}
  scanf("%[", str); // expected-warning{{no closing ']' for '%[' in scanf format string}}
  const char kFormat3[] = "%hu"; // expected-note{{format string is defined here}}}
  scanf(kFormat3, &i); // expected-warning {{format specifies type 'unsigned short *' but the argument}}
  const char kFormat4[] = "%lp"; // expected-note{{format string is defined here}}}
  scanf(kFormat4, &i); // expected-warning {{length modifier 'l' results in undefined behavior or no effect with 'p' conversion specifier}}
}

void test_variants(int *i, const char *s, ...) {
  FILE *f = 0;
  char buf[100];

  fscanf(f, "%ld", i); // expected-warning{{format specifies type 'long *' but the argument has type 'int *'}}
  sscanf(buf, "%ld", i); // expected-warning{{format specifies type 'long *' but the argument has type 'int *'}}
  my_scanf("%ld", i); // expected-warning{{format specifies type 'long *' but the argument has type 'int *'}}

  va_list ap;
  va_start(ap, s);

  vscanf("%[abc", ap); // expected-warning{{no closing ']' for '%[' in scanf format string}}
  vfscanf(f, "%[abc", ap); // expected-warning{{no closing ']' for '%[' in scanf format string}}
  vsscanf(buf, "%[abc", ap); // expected-warning{{no closing ']' for '%[' in scanf format string}}
}

void test_scanlist(int *ip, char *sp, wchar_t *ls) {
  scanf("%[abc]", ip); // expected-warning{{format specifies type 'char *' but the argument has type 'int *'}}
  scanf("%h[abc]", sp); // expected-warning{{length modifier 'h' results in undefined behavior or no effect with '[' conversion specifier}}
  scanf("%l[xyx]", ls); // no-warning
  scanf("%ll[xyx]", ls); // expected-warning {{length modifier 'll' results in undefined behavior or no effect with '[' conversion specifier}}

  // PR19559
  scanf("%[]% ]", sp); // no-warning
  scanf("%[^]% ]", sp); // no-warning
  scanf("%[a^]% ]", sp); // expected-warning {{invalid conversion specifier ' '}}
}

void test_alloc_extension(char **sp, wchar_t **lsp, float *fp) {
  /* Make sure "%a" gets parsed as a conversion specifier for float,
   * even when followed by an 's', 'S' or '[', which would cause it to be
   * parsed as a length modifier in C90. */
  scanf("%as", sp); // expected-warning{{format specifies type 'float *' but the argument has type 'char **'}}
  scanf("%aS", lsp); // expected-warning{{format specifies type 'float *' but the argument has type 'wchar_t **'}}
  scanf("%a[bcd]", sp); // expected-warning{{format specifies type 'float *' but the argument has type 'char **'}}

  // Test that the 'm' length modifier is only allowed with s, S, c, C or [.
  // TODO: Warn that 'm' is an extension.
  scanf("%ms", sp); // No warning.
  scanf("%mS", lsp); // No warning.
  scanf("%mc", sp); // No warning.
  scanf("%mC", lsp); // No warning.
  scanf("%m[abc]", sp); // No warning.
  scanf("%md", sp); // expected-warning{{length modifier 'm' results in undefined behavior or no effect with 'd' conversion specifier}}

  // Test argument type check for the 'm' length modifier.
  scanf("%ms", fp); // expected-warning{{format specifies type 'char **' but the argument has type 'float *'}}
  scanf("%mS", fp); // expected-warning-re{{format specifies type 'wchar_t **' (aka '{{[^']+}}') but the argument has type 'float *'}}
  scanf("%mc", fp); // expected-warning{{format specifies type 'char **' but the argument has type 'float *'}}
  scanf("%mC", fp); // expected-warning-re{{format specifies type 'wchar_t **' (aka '{{[^']+}}') but the argument has type 'float *'}}
  scanf("%m[abc]", fp); // expected-warning{{format specifies type 'char **' but the argument has type 'float *'}}
}

void test_quad(int *x, long long *llx) {
  scanf("%qd", x); // expected-warning{{format specifies type 'long long *' but the argument has type 'int *'}}
  scanf("%qd", llx); // no-warning
}

void test_writeback(int *x) {
  scanf("%n", (void*)0); // expected-warning{{format specifies type 'int *' but the argument has type 'void *'}}
  scanf("%n %c", x, x); // expected-warning{{format specifies type 'char *' but the argument has type 'int *'}}

  scanf("%hhn", (signed char*)0); // no-warning
  scanf("%hhn", (char*)0); // no-warning
  scanf("%hhn", (unsigned char*)0); // no-warning
  scanf("%hhn", (int*)0); // expected-warning{{format specifies type 'signed char *' but the argument has type 'int *'}}

  scanf("%hn", (short*)0); // no-warning
  scanf("%hn", (unsigned short*)0); // no-warning
  scanf("%hn", (int*)0); // expected-warning{{format specifies type 'short *' but the argument has type 'int *'}}

  scanf("%n", (int*)0); // no-warning
  scanf("%n", (unsigned int*)0); // no-warning
  scanf("%n", (char*)0); // expected-warning{{format specifies type 'int *' but the argument has type 'char *'}}

  scanf("%ln", (long*)0); // no-warning
  scanf("%ln", (unsigned long*)0); // no-warning
  scanf("%ln", (int*)0); // expected-warning{{format specifies type 'long *' but the argument has type 'int *'}}

  scanf("%lln", (long long*)0); // no-warning
  scanf("%lln", (unsigned long long*)0); // no-warning
  scanf("%lln", (int*)0); // expected-warning{{format specifies type 'long long *' but the argument has type 'int *'}}

  scanf("%qn", (long long*)0); // no-warning
  scanf("%qn", (unsigned long long*)0); // no-warning
  scanf("%qn", (int*)0); // expected-warning{{format specifies type 'long long *' but the argument has type 'int *'}}

}

void test_qualifiers(const int *cip, volatile int* vip,
                     const char *ccp, volatile char* vcp,
                     const volatile int *cvip) {
  scanf("%d", cip); // expected-warning{{format specifies type 'int *' but the argument has type 'const int *'}}
  scanf("%n", cip); // expected-warning{{format specifies type 'int *' but the argument has type 'const int *'}}
  scanf("%s", ccp); // expected-warning{{format specifies type 'char *' but the argument has type 'const char *'}}
  scanf("%d", cvip); // expected-warning{{format specifies type 'int *' but the argument has type 'const volatile int *'}}

  scanf("%d", vip); // No warning.
  scanf("%n", vip); // No warning.
  scanf("%c", vcp); // No warning.

  typedef int* ip_t;
  typedef const int* cip_t;
  scanf("%d", (ip_t)0); // No warning.
  scanf("%d", (cip_t)0); // expected-warning{{format specifies type 'int *' but the argument has type 'cip_t' (aka 'const int *')}}
}

void test_size_types(void) {
  size_t s = 0;
  scanf("%zu", &s); // No warning.
  scanf("%zb", &s);

  double d1 = 0.;
  scanf("%zu", &d1); // expected-warning-re{{format specifies type 'size_t *' (aka '{{.+}}') but the argument has type 'double *'}}

  ssize_t ss = 0;
  scanf("%zd", &s); // No warning.

  double d2 = 0.;
  scanf("%zd", &d2); // expected-warning-re{{format specifies type 'ssize_t *' (aka '{{.+}}') but the argument has type 'double *'}}

  ssize_t sn = 0;
  scanf("%zn", &sn); // No warning.

  double d3 = 0.;
  scanf("%zn", &d3); // expected-warning-re{{format specifies type 'ssize_t *' (aka '{{.+}}') but the argument has type 'double *'}}
}

void test_ptrdiff_t_types(void) {
  __UNSIGNED_PTRDIFF_TYPE__ p1 = 0;
  scanf("%tu", &p1); // No warning.
  scanf("%tb", &p1);

  double d1 = 0.;
  scanf("%tu", &d1); // expected-warning-re{{format specifies type 'unsigned ptrdiff_t *' (aka '{{.+}}') but the argument has type 'double *'}}

  ptrdiff_t p2 = 0;
  scanf("%td", &p2); // No warning.

  double d2 = 0.;
  scanf("%td", &d2); // expected-warning-re{{format specifies type 'ptrdiff_t *' (aka '{{.+}}') but the argument has type 'double *'}}

  ptrdiff_t p3 = 0;
  scanf("%tn", &p3); // No warning.

  double d3 = 0.;
  scanf("%tn", &d3); // expected-warning-re{{format specifies type 'ptrdiff_t *' (aka '{{.+}}') but the argument has type 'double *'}}
}

void check_conditional_literal(char *s, int *i) {
  scanf(0 ? "%s" : "%d", i); // no warning
  scanf(1 ? "%s" : "%d", i); // expected-warning{{format specifies type 'char *'}}
  scanf(0 ? "%d %d" : "%d", i); // no warning
  scanf(1 ? "%d %d" : "%d", i); // expected-warning{{more '%' conversions than data arguments}}
  scanf(0 ? "%d %d" : "%d", i, s); // expected-warning{{data argument not used}}
  scanf(1 ? "%d %s" : "%d", i, s); // no warning
  scanf(i ? "%d %s" : "%d", i, s); // no warning
  scanf(i ? "%d" : "%d", i, s); // expected-warning{{data argument not used}}
  scanf(i ? "%s" : "%d", s); // expected-warning{{format specifies type 'int *'}}
}

void test_promotion(void) {
  // No promotions for *scanf pointers clarified in N2562
  // https://github.com/llvm/llvm-project/issues/57102
  // N2562: https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2562.pdf
  int i;
  signed char sc;
  unsigned char uc;
  short ss;
  unsigned short us;

  // pointers could not be "promoted"
  scanf("%hhd", &i); // expected-warning{{format specifies type 'char *' but the argument has type 'int *'}}
  scanf("%hd", &i); // expected-warning{{format specifies type 'short *' but the argument has type 'int *'}}
  scanf("%d", &i); // no-warning
  // char & uchar
  scanf("%hhd", &sc); // no-warning
  scanf("%hhd", &uc); // no-warning
  scanf("%hd", &sc); // expected-warning{{format specifies type 'short *' but the argument has type 'signed char *'}}
  scanf("%hd", &uc); // expected-warning{{format specifies type 'short *' but the argument has type 'unsigned char *'}}
  scanf("%d", &sc); // expected-warning{{format specifies type 'int *' but the argument has type 'signed char *'}}
  scanf("%d", &uc); // expected-warning{{format specifies type 'int *' but the argument has type 'unsigned char *'}}
  // short & ushort
  scanf("%hhd", &ss); // expected-warning{{format specifies type 'char *' but the argument has type 'short *'}}
  scanf("%hhd", &us); // expected-warning{{format specifies type 'char *' but the argument has type 'unsigned short *'}}
  scanf("%hd", &ss); // no-warning
  scanf("%hd", &us); // no-warning
  scanf("%d", &ss); // expected-warning{{format specifies type 'int *' but the argument has type 'short *'}}
  scanf("%d", &us); // expected-warning{{format specifies type 'int *' but the argument has type 'unsigned short *'}}

  // long types
  scanf("%ld", &i); // expected-warning{{format specifies type 'long *' but the argument has type 'int *'}}
  scanf("%lld", &i); // expected-warning{{format specifies type 'long long *' but the argument has type 'int *'}}
  scanf("%ld", &sc); // expected-warning{{format specifies type 'long *' but the argument has type 'signed char *'}}
  scanf("%lld", &sc); // expected-warning{{format specifies type 'long long *' but the argument has type 'signed char *'}}
  scanf("%ld", &uc); // expected-warning{{format specifies type 'long *' but the argument has type 'unsigned char *'}}
  scanf("%lld", &uc); // expected-warning{{format specifies type 'long long *' but the argument has type 'unsigned char *'}}
  scanf("%llx", &i); // expected-warning{{format specifies type 'unsigned long long *' but the argument has type 'int *'}}

  // ill-formed floats
  scanf("%hf", // expected-warning{{length modifier 'h' results in undefined behavior or no effect with 'f' conversion specifier}}
  &sc);

  // pointers in scanf
  scanf("%s", i); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}

  // FIXME: does this match what the C committee allows or should it be pedantically warned on?
  char c;
  void *vp;
  scanf("%hhd", &c); // Pedantic warning?
  scanf("%hhd", vp); // expected-warning{{format specifies type 'char *' but the argument has type 'void *'}}
}
