// RUN: %clang_builtins %s %librt -o %t && %run %t

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// bigx = sext x
static void sext(int x, unsigned int* bigx, unsigned words) {
    char sign = (x < 0) ? 0xFF : 0;
    memset(bigx, sign, words*sizeof(su_int));
#ifdef _YUGA_LITTLE_ENDIAN
    (void)words;
    memcpy(bigx, &x, sizeof(x));
#else
    memcpy(bigx + (sizeof(su_int)*words - sizeof(x)), &x, sizeof(x));
#endif
}

// x = trunc bigx
static void trunc(unsigned int* bigx, int *x, unsigned words) {
#ifdef _YUGA_LITTLE_ENDIAN
    (void)words;
    memcpy(x, bigx, sizeof(*x));
#else
    memcpy(x, bigx + (sizeof(su_int)*words - sizeof(x)), sizeof(x));
#endif
}

int test__divmodei5(int a, int b, int expected_quo, int expected_rem,
                    unsigned words, unsigned int *a_scratch,
                    unsigned int *b_scratch, unsigned int *quo_scratch,
                    unsigned int *rem_scratch) {
  // sign-extend a and b
  // 'a' needs to have an extra word, see documentation of __udivmodei5.
  sext(a, a_scratch, words+1);
  sext(b, b_scratch, words);

  memset(quo_scratch, 0, words * sizeof(su_int));
  memset(rem_scratch, 0, words * sizeof(su_int));

  __divmodei5(quo_scratch, rem_scratch, a_scratch, b_scratch, words);

  su_int sign_q = (expected_quo < 0) ? (su_int)-1 : 0;
  su_int sign_r = (expected_rem < 0) ? (su_int)-1 : 0;

  int q, r;
  trunc(quo_scratch, &q, words);
  trunc(rem_scratch, &r, words);

  if (q != expected_quo) {
    printf("error in __divmodei5: %d / %d = %d, expected %d\n", a, b, q,
           expected_quo);
    return 1;
  }
  if (r != expected_rem) {
    printf("error in __divmodei5: %d mod %d = %d, expected %d\n", a, b, r,
           expected_rem);
    return 1;
  }

  /// Expect upper bits of result to be sign-extended
#ifdef _YUGA_LITTLE_ENDIAN
    for(unsigned i= sizeof(q); i < words; ++i) {
#else
    for(unsigned i= words - 1; i >= sizeof(q); --i) {
#endif
    if (quo_scratch[i] != sign_q) {
      printf("error in __divmodei5: %d / %d = %d, R = %d, words=%d expected "
             "quo_scratch[%d] == %d but got %d\n",
             a, b, q, r, words, i, sign_q, quo_scratch[i]);
      return 1;
    }
    if (rem_scratch[i] != sign_r) {
      printf("error in __divmodei5: %d / %d = %d, R = %d, words=%d expected "
             "rem_scratch[%d] == %d  but got %d\n",
             a, b, q, r, words, i, sign_r, rem_scratch[i]);
      return 1;
    }
  }

  return 0;
}

// Multiples of sizeof(int) are allowed for _divmodei5.
int words[] = {1, 2, 4, 5, 6, 8};

int main() {
  for (unsigned iwords = 0; iwords < sizeof(words) / sizeof(words[0]); ++iwords) {
    int nwords = words[iwords];
    // 'a' needs to have an extra word, see documentation of __udivmodei5.
    unsigned int *a_scratch = malloc((nwords + 1) * sizeof(su_int));
    unsigned int *b_scratch = malloc(nwords * sizeof(su_int));
    unsigned int *quo_scratch = malloc(nwords * sizeof(su_int));
    unsigned int *rem_scratch = malloc(nwords * sizeof(su_int));

    if (test__divmodei5(0, 1, 0, 0, nwords, a_scratch, b_scratch, quo_scratch,
                        rem_scratch))
      return 1;
    if (test__divmodei5(0, -1, 0, 0, nwords, a_scratch, b_scratch, quo_scratch,
                        rem_scratch))
      return 1;
    if (test__divmodei5(1, 1, 1, 0, nwords, a_scratch, b_scratch, quo_scratch,
                        rem_scratch))
      return 1;

    if (test__divmodei5(2, 1, 2, 0, nwords, a_scratch, b_scratch, quo_scratch,
                        rem_scratch))
      return 1;
    if (test__divmodei5(2, -1, -2, 0, nwords, a_scratch, b_scratch, quo_scratch,
                        rem_scratch))
      return 1;
    if (test__divmodei5(-2, 1, -2, 0, nwords, a_scratch, b_scratch, quo_scratch,
                        rem_scratch))
      return 1;
    if (test__divmodei5(-2, -1, 2, 0, nwords, a_scratch, b_scratch, quo_scratch,
                        rem_scratch))
      return 1;

    if (test__divmodei5(7, 5, 1, 2, nwords, a_scratch, b_scratch, quo_scratch,
                        rem_scratch))
      return 1;
    if (test__divmodei5(-7, 5, -1, -2, nwords, a_scratch, b_scratch, quo_scratch,
                        rem_scratch))
      return 1;
    if (test__divmodei5(19, 5, 3, 4, nwords, a_scratch, b_scratch, quo_scratch,
                        rem_scratch))
      return 1;
    if (test__divmodei5(19, -5, -3, 4, nwords, a_scratch, b_scratch, quo_scratch,
                        rem_scratch))
      return 1;

    if (test__divmodei5(0x80000000, 8, 0xf0000000, 0, nwords, a_scratch,
                        b_scratch, quo_scratch, rem_scratch))
      return 1;
    if (test__divmodei5(0x80000007, 8, 0xf0000001, -1, nwords, a_scratch,
                        b_scratch, quo_scratch, rem_scratch))
      return 1;
  }

  return 0;
}
