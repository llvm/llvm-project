#define EXPAND_2_INNER_CASES(i, x, y)    INNER_CASE(i, x, y);             INNER_CASE(i + 1, x, y);
#define EXPAND_4_INNER_CASES(i, x, y)    EXPAND_2_INNER_CASES(i, x, y)    EXPAND_2_INNER_CASES(i + 2, x, y)
#define EXPAND_8_INNER_CASES(i, x, y)    EXPAND_4_INNER_CASES(i, x, y)    EXPAND_4_INNER_CASES(i + 4, x, y)
#define EXPAND_16_INNER_CASES(i, x, y)   EXPAND_8_INNER_CASES(i, x, y)    EXPAND_8_INNER_CASES(i + 8, x, y)
#define EXPAND_32_INNER_CASES(i, x, y)   EXPAND_16_INNER_CASES(i, x, y)   EXPAND_16_INNER_CASES(i + 16, x, y)
#define EXPAND_64_INNER_CASES(i, x, y)   EXPAND_32_INNER_CASES(i, x, y)   EXPAND_32_INNER_CASES(i + 32, x, y)

#define EXPAND_2_OUTER_CASES(i, x, y)    OUTER_CASE(i, x, y);             OUTER_CASE(i + 1, x, y);
#define EXPAND_4_OUTER_CASES(i, x, y)    EXPAND_2_OUTER_CASES(i, x, y)    EXPAND_2_OUTER_CASES(i + 2, x, y)
#define EXPAND_8_OUTER_CASES(i, x, y)    EXPAND_4_OUTER_CASES(i, x, y)    EXPAND_4_OUTER_CASES(i + 4, x, y)
#define EXPAND_16_OUTER_CASES(i, x, y)   EXPAND_8_OUTER_CASES(i, x, y)    EXPAND_8_OUTER_CASES(i + 8, x, y)
#define EXPAND_32_OUTER_CASES(i, x, y)   EXPAND_16_OUTER_CASES(i, x, y)   EXPAND_16_OUTER_CASES(i + 16, x, y)
#define EXPAND_64_OUTER_CASES(i, x, y)   EXPAND_32_OUTER_CASES(i, x, y)   EXPAND_32_OUTER_CASES(i + 32, x, y)

// Rather than a single monstrous fan-out, this fans out in smaller increments,
// but to a similar size.
unsigned cfg_nested_switch(int x) {
  unsigned y = 0;
  while (x > 0) {
    switch (x) {
#define INNER_CASE(i, x, y) \
          case i: { int case_var = 3*x + i; y += case_var - 1; break; }
#define OUTER_CASE(i, x, y) \
      case i: { \
        int case_var = y >> 8; \
        switch (case_var) { \
          EXPAND_64_INNER_CASES(0, x, y); \
        } \
        break; \
      }
EXPAND_64_OUTER_CASES(0, x, y);
    }
    --x;
  }
  return y;
}
