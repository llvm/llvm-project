#define EXPAND_2_CASES(i, x, y)    CASE(i, x, y);             CASE(i + 1, x, y);
#define EXPAND_4_CASES(i, x, y)    EXPAND_2_CASES(i, x, y)    EXPAND_2_CASES(i + 2, x, y)
#define EXPAND_8_CASES(i, x, y)    EXPAND_4_CASES(i, x, y)    EXPAND_4_CASES(i + 4, x, y)
#define EXPAND_16_CASES(i, x, y)   EXPAND_8_CASES(i, x, y)    EXPAND_8_CASES(i + 8, x, y)
#define EXPAND_32_CASES(i, x, y)   EXPAND_16_CASES(i, x, y)   EXPAND_16_CASES(i + 16, x, y)
#define EXPAND_64_CASES(i, x, y)   EXPAND_32_CASES(i, x, y)   EXPAND_32_CASES(i + 32, x, y)
#define EXPAND_128_CASES(i, x, y)  EXPAND_64_CASES(i, x, y)   EXPAND_64_CASES(i + 64, x, y)
#define EXPAND_256_CASES(i, x, y)  EXPAND_128_CASES(i, x, y)  EXPAND_128_CASES(i + 128, x, y)
#define EXPAND_512_CASES(i, x, y)  EXPAND_256_CASES(i, x, y)  EXPAND_256_CASES(i + 256, x, y)
#define EXPAND_1024_CASES(i, x, y) EXPAND_512_CASES(i, x, y)  EXPAND_512_CASES(i + 512, x, y)
#define EXPAND_2048_CASES(i, x, y) EXPAND_1024_CASES(i, x, y) EXPAND_1024_CASES(i + 1024, x, y)
#define EXPAND_4096_CASES(i, x, y) EXPAND_2048_CASES(i, x, y) EXPAND_2048_CASES(i + 2048, x, y)

// This has a *monstrous* single fan-out in the CFG, across 8000 blocks inside
// the while loop.
unsigned cfg_big_switch(int x) {
  unsigned y = 0;
  while (x > 0) {
    switch(x) {
#define CASE(i, x, y) \
      case i: { int case_var = 3*x + i; y += case_var - 1; break; }
EXPAND_4096_CASES(0, x, y);
    }
    --x;
  }
  return y;
}
