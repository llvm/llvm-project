#define TEST __TEST

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = function)

int test(void);

#pragma clang attribute pop
