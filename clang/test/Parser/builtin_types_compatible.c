// RUN: %clang_cc1 -fsyntax-only -verify %s

extern int funcInt(int);
extern float funcFloat(float);
extern double funcDouble(double);
// figure out why "char *" doesn't work (with gcc, nothing to do with clang)
//extern void funcCharPtr(char *);

#define func(expr) \
  do { \
    typeof(expr) tmp; \
    if (__builtin_types_compatible_p(typeof(expr), int)) funcInt(tmp); \
    else if (__builtin_types_compatible_p(typeof(expr), float)) funcFloat(tmp); \
    else if (__builtin_types_compatible_p(typeof(expr), double)) funcDouble(tmp); \
  } while (0)
#define func_choose(expr) \
  __builtin_choose_expr(__builtin_types_compatible_p(typeof(expr), int), funcInt(expr), \
    __builtin_choose_expr(__builtin_types_compatible_p(typeof(expr), float), funcFloat(expr), \
      __builtin_choose_expr(__builtin_types_compatible_p(typeof(expr), double), funcDouble(expr), (void)0)))

static void test()
{
  int a;
  float b;
  double d;

  func(a);
  func(b);
  func(d);
  a = func_choose(a);
  b = func_choose(b);
  d = func_choose(d);

  int c; 
  struct xx { int a; } x, y;
  
  c = __builtin_choose_expr(a+3-7, b, x); // expected-error{{'__builtin_choose_expr' requires a constant expression}}
  c = __builtin_choose_expr(0, b, x); // expected-error{{assigning to 'int' from incompatible type 'struct xx'}}
  c = __builtin_choose_expr(5+3-7, b, x);
  y = __builtin_choose_expr(4+3-7, b, x);

}

