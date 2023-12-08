// "System header" for testing that -Wunused-value is properly suppressed in
// certain cases.

#define POSSIBLY_BAD_MACRO(x) \
  { int i = x; \
    i; }

#define STATEMENT_EXPR_MACRO(x) \
  (__extension__ \
   ({int i = x; \
     i;}))

#define COMMA_MACRO_1(x, y) \
  {x, y;}

#define COMMA_MACRO_2(x, y) \
  if (x) { 1 == 2, y; }

#define COMMA_MACRO_3(x, y) \
  (x, y)

#define COMMA_MACRO_4(x, y) \
  ( 1 == 2, y )
