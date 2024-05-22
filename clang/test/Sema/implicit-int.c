/* RUN: %clang_cc1 -fsyntax-only -std=c89 -Wimplicit-int %s -verify -Wno-strict-prototypes
   RUN: %clang_cc1 -fsyntax-only -std=c99 %s -verify=ext -Wno-strict-prototypes
   RUN: %clang_cc1 -fsyntax-only -std=c2x %s -verify=unsupported
*/

foo(void) { /* expected-warning {{type specifier missing, defaults to 'int'}} \
               ext-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} \
               unsupported-error {{a type specifier is required for all declarations}} */
  return 0;
}

y;  /* expected-warning {{type specifier missing, defaults to 'int'}} \
       ext-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} \
       unsupported-error {{a type specifier is required for all declarations}} */

void f((x));  /* expected-warning {{type specifier missing, defaults to 'int'}} \
                 ext-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} \
                 unsupported-error {{a type specifier is required for all declarations}} */

/* PR3702 */
#define PAD(ms10) { \
    register i;     \
}

#define ILPAD() PAD((NROW - tt.tt_row) * 10) /* 1 ms per char */

void
h19_insline(n)  /* ext-error {{parameter 'n' was not declared, defaults to 'int'; ISO C99 and later do not support implicit int}} \
                   unsupported-error {{unknown type name 'n'}} */
{
  ILPAD();  /* expected-warning {{type specifier missing, defaults to 'int'}} \
               ext-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} \
               unsupported-error {{a type specifier is required for all declarations}} */

}

struct foo {
 __extension__ __attribute__((packed)) x : 4; /* expected-warning {{type specifier missing, defaults to 'int'}} \
                                                 ext-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} \
                                                 unsupported-error {{unknown type name 'x'}} */

};
