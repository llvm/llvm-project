// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wbad-function-cast -ffixed-point -triple x86_64-unknown-unknown -verify

void vf(void);
int if1(void);
char if2(void);
long if3(void);
float rf1(void);
double rf2(void);
_Complex double cf(void);
enum e { E1 } ef(void);
_Bool bf(void);
char *pf1(void);
int *pf2(void);
_Fract ff1(void);

void
foo(void)
{

  /* default, no cast, should always be ok */
  ff1();
  /* Casts to void types are always OK.  */
  (void)vf();
  (void)if1();
  (void)cf();
  (const void)bf();
  (void)ff1();
  /* Casts to the same type or similar types are OK.  */
  (int)if1();
  (long)if2();
  (char)if3();
  (float)rf1();
  (long double)rf2();
  (_Complex float)cf();
  (enum f { F1 })ef();
  (_Bool)bf();
  (void *)pf1();
  (char *)pf2();
  (_Fract) ff1();
  /* All following casts issue warning */
  (float)if1(); /* expected-warning {{cast from function call of type 'int' to non-matching type 'float'}} */
  (double)if2(); /* expected-warning {{cast from function call of type 'char' to non-matching type 'double'}} */
  (_Bool)if3(); /* expected-warning {{cast from function call of type 'long' to non-matching type '_Bool'}} */
  (int)rf1(); /* expected-warning {{cast from function call of type 'float' to non-matching type 'int'}} */
  (long)rf2(); /* expected-warning {{cast from function call of type 'double' to non-matching type 'long'}} */
  (double)cf(); /* expected-warning {{cast from function call of type '_Complex double' to non-matching type 'double'}} */
  (int)ef(); /* expected-warning {{cast from function call of type 'enum e' to non-matching type 'int'}} */
  (int)bf(); /* expected-warning {{cast from function call of type '_Bool' to non-matching type 'int'}} */
  (__SIZE_TYPE__)pf1(); /* expected-warning {{cast from function call of type 'char *' to non-matching type 'unsigned long'}} */
  (__PTRDIFF_TYPE__)pf2(); /* expected-warning {{cast from function call of type 'int *' to non-matching type 'long'}} */
  (_Fract) if1();          /* expected-warning{{cast from function call of type 'int' to non-matching type '_Fract'}} */
  (int)ff1();              /* expected-warning{{cast from function call of type '_Fract' to non-matching type 'int'}} */
}

