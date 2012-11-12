// Hammer the CFG with large numbers of overlapping variable scopes, which
// implicit destructors triggered at each edge.

#define EXPAND_BASIC_STRUCT(i) struct X##i { X##i(int); ~X##i(); };
#define EXPAND_NORET_STRUCT(i) struct X##i { X##i(int); ~X##i() __attribute__((noreturn)); };
EXPAND_BASIC_STRUCT(0000); EXPAND_NORET_STRUCT(0001);
EXPAND_BASIC_STRUCT(0010); EXPAND_BASIC_STRUCT(0011);
EXPAND_BASIC_STRUCT(0100); EXPAND_NORET_STRUCT(0101);
EXPAND_NORET_STRUCT(0110); EXPAND_BASIC_STRUCT(0111);
EXPAND_BASIC_STRUCT(1000); EXPAND_NORET_STRUCT(1001);
EXPAND_BASIC_STRUCT(1010); EXPAND_BASIC_STRUCT(1011);
EXPAND_NORET_STRUCT(1100); EXPAND_NORET_STRUCT(1101);
EXPAND_BASIC_STRUCT(1110); EXPAND_BASIC_STRUCT(1111);

#define EXPAND_2_VARS(c, i, x)  const X##i var_##c##_##i##0(x), &var_##c##_##i##1 = X##i(x)
#define EXPAND_4_VARS(c, i, x)  EXPAND_2_VARS(c, i##0, x);  EXPAND_2_VARS(c, i##1, x)
#define EXPAND_8_VARS(c, i, x)  EXPAND_4_VARS(c, i##0, x);  EXPAND_4_VARS(c, i##1, x)
#define EXPAND_16_VARS(c, i, x) EXPAND_8_VARS(c, i##0, x);  EXPAND_8_VARS(c, i##1, x)
#define EXPAND_32_VARS(c, x)    EXPAND_16_VARS(c, 0, x);    EXPAND_16_VARS(c, 1, x)

#define EXPAND_2_INNER_CASES(i, x, y)    INNER_CASE(i, x, y);             INNER_CASE(i + 1, x, y);
#define EXPAND_4_INNER_CASES(i, x, y)    EXPAND_2_INNER_CASES(i, x, y)    EXPAND_2_INNER_CASES(i + 2, x, y)
#define EXPAND_8_INNER_CASES(i, x, y)    EXPAND_4_INNER_CASES(i, x, y)    EXPAND_4_INNER_CASES(i + 4, x, y)
#define EXPAND_16_INNER_CASES(i, x, y)   EXPAND_8_INNER_CASES(i, x, y)    EXPAND_8_INNER_CASES(i + 8, x, y)
#define EXPAND_32_INNER_CASES(i, x, y)   EXPAND_16_INNER_CASES(i, x, y)   EXPAND_16_INNER_CASES(i + 16, x, y)

#define EXPAND_2_OUTER_CASES(i, x, y)    OUTER_CASE(i, x, y);             OUTER_CASE(i + 1, x, y);
#define EXPAND_4_OUTER_CASES(i, x, y)    EXPAND_2_OUTER_CASES(i, x, y)    EXPAND_2_OUTER_CASES(i + 2, x, y)
#define EXPAND_8_OUTER_CASES(i, x, y)    EXPAND_4_OUTER_CASES(i, x, y)    EXPAND_4_OUTER_CASES(i + 4, x, y)
#define EXPAND_16_OUTER_CASES(i, x, y)   EXPAND_8_OUTER_CASES(i, x, y)    EXPAND_8_OUTER_CASES(i + 8, x, y)
#define EXPAND_32_OUTER_CASES(i, x, y)   EXPAND_16_OUTER_CASES(i, x, y)   EXPAND_16_OUTER_CASES(i + 16, x, y)

unsigned cfg_nested_vars(int x) {
  int y = 0;
  while (x > 0) {
    EXPAND_32_VARS(a, x);
    switch (x) {
#define INNER_CASE(i, x, y) \
          case i: { \
            int case_var = 3*x + i; \
            EXPAND_32_VARS(c, case_var); \
            y += case_var - 1; \
            break; \
          }
#define OUTER_CASE(i, x, y) \
      case i: { \
        int case_var = y >> 8; \
        EXPAND_32_VARS(b, y); \
        switch (case_var) { \
          EXPAND_32_INNER_CASES(0, x, y); \
        } \
        break; \
      }
EXPAND_32_OUTER_CASES(0, x, y);
    }
    --x;
  }
  return y;
}
