#define EXPAND_2_BRANCHES(i, x, y)    BRANCH(i, x, y);              BRANCH(i + 1, x, y);
#define EXPAND_4_BRANCHES(i, x, y)    EXPAND_2_BRANCHES(i, x, y)    EXPAND_2_BRANCHES(i + 2, x, y)
#define EXPAND_8_BRANCHES(i, x, y)    EXPAND_4_BRANCHES(i, x, y)    EXPAND_4_BRANCHES(i + 4, x, y)
#define EXPAND_16_BRANCHES(i, x, y)   EXPAND_8_BRANCHES(i, x, y)    EXPAND_8_BRANCHES(i + 8, x, y)
#define EXPAND_32_BRANCHES(i, x, y)   EXPAND_16_BRANCHES(i, x, y)   EXPAND_16_BRANCHES(i + 16, x, y)
#define EXPAND_64_BRANCHES(i, x, y)   EXPAND_32_BRANCHES(i, x, y)   EXPAND_32_BRANCHES(i + 32, x, y)
#define EXPAND_128_BRANCHES(i, x, y)  EXPAND_64_BRANCHES(i, x, y)   EXPAND_64_BRANCHES(i + 64, x, y)
#define EXPAND_256_BRANCHES(i, x, y)  EXPAND_128_BRANCHES(i, x, y)  EXPAND_128_BRANCHES(i + 128, x, y)
#define EXPAND_512_BRANCHES(i, x, y)  EXPAND_256_BRANCHES(i, x, y)  EXPAND_256_BRANCHES(i + 256, x, y)
#define EXPAND_1024_BRANCHES(i, x, y) EXPAND_512_BRANCHES(i, x, y)  EXPAND_512_BRANCHES(i + 512, x, y)
#define EXPAND_2048_BRANCHES(i, x, y) EXPAND_1024_BRANCHES(i, x, y) EXPAND_1024_BRANCHES(i + 1024, x, y)
#define EXPAND_4096_BRANCHES(i, x, y) EXPAND_2048_BRANCHES(i, x, y) EXPAND_2048_BRANCHES(i + 2048, x, y)

unsigned cfg_long_chain_multiple_exit(unsigned x) {
  unsigned y = 0;
#define BRANCH(i, x, y) if (((x % 13171) + ++y) < i) { int var = x / 13171 + y; return var; } 
  EXPAND_4096_BRANCHES(1, x, y);
#undef BRANCH
  return 42;
}
