// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floattitf

#define QUAD_PRECISION
#include "fp_lib.h"
#include "int_lib.h"
#include <float.h>
#include <stdio.h>

// The testcase currently assumes IEEE TF format, once that has been
// fixed the defined(CRT_HAS_IEEE_TF) guard can be removed to enable it for
// IBM 128 floats as well.
#if defined(CRT_HAS_IEEE_TF)

#  include "fp_test.h"

/* Returns: convert a ti_int to a fp_t, rounding toward even. */

/* Assumption: fp_t is a IEEE 128 bit floating point type
 *             ti_int is a 128 bit integral type
 */

/* seee eeee eeee eeee mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm |
 * mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm
 */

COMPILER_RT_ABI fp_t __floattitf(ti_int a);

int _test__floattitf(int line, ti_int a, fp_t expected) {
  fp_t x = __floattitf(a);
  if (x != expected) {
    twords at;
    at.all = a;
    __uint128_t expected_rep = toRep128(expected);
    __uint128_t res_rep = toRep128(x);
    printf("%s:%d: error in __floattitf(0x%.16llX%.16llX) = "
           "0x%.16llX%.16llX, expected 0x%.16llX%.16llX\n",
           __FILE__, line, at.s.high, at.s.low, (uint64_t)(res_rep >> 64),
           (uint64_t)res_rep, (uint64_t)(expected_rep >> 64),
           (uint64_t)expected_rep);
  }
  return x != expected;
}

#  define test__floattitf(a, e) _test__floattitf(__LINE__, a, e)

char assumption_1[sizeof(ti_int) == 2 * sizeof(di_int)] = {0};
char assumption_2[sizeof(ti_int) * CHAR_BIT == 128] = {0};
char assumption_3[sizeof(fp_t) * CHAR_BIT == 128] = {0};

int main() {
  if (test__floattitf(0, TF_C(0.0)))
    return 1;

  if (test__floattitf(1, TF_C(1.0)))
    return 1;
  if (test__floattitf(2, TF_C(2.0)))
    return 1;
  if (test__floattitf(20, TF_C(20.0)))
    return 1;
  if (test__floattitf(-1, -TF_C(1.0)))
    return 1;
  if (test__floattitf(-2, -TF_C(2.0)))
    return 1;
  if (test__floattitf(-20, -TF_C(20.0)))
    return 1;

  if (test__floattitf(0x7FFFFF8000000000LL, TF_C(0x1.FFFFFEp+62)))
    return 1;
  if (test__floattitf(0x7FFFFFFFFFFFF800LL, TF_C(0x1.FFFFFFFFFFFFEp+62)))
    return 1;
  if (test__floattitf(0x7FFFFF0000000000LL, TF_C(0x1.FFFFFCp+62)))
    return 1;
  if (test__floattitf(0x7FFFFFFFFFFFF000LL, TF_C(0x1.FFFFFFFFFFFFCp+62)))
    return 1;

  if (test__floattitf(make_ti(0x8000008000000000LL, 0), -TF_C(0x1.FFFFFEp+126)))
    return 1;
  if (test__floattitf(make_ti(0x8000000000000800LL, 0),
                      -TF_C(0x1.FFFFFFFFFFFFEp+126)))
    return 1;
  if (test__floattitf(make_ti(0x8000010000000000LL, 0), -TF_C(0x1.FFFFFCp+126)))
    return 1;
  if (test__floattitf(make_ti(0x8000000000001000LL, 0),
                      -TF_C(0x1.FFFFFFFFFFFFCp+126)))
    return 1;

  if (test__floattitf(make_ti(0x8000000000000000LL, 0), -TF_C(0x1.000000p+127)))
    return 1;
  if (test__floattitf(make_ti(0x8000000000000001LL, 0),
                      -TF_C(0x1.FFFFFFFFFFFFFFFCp+126)))
    return 1;

  if (test__floattitf(0x0007FB72E8000000LL, TF_C(0x1.FEDCBAp+50)))
    return 1;

  if (test__floattitf(0x0007FB72EA000000LL, TF_C(0x1.FEDCBA8p+50)))
    return 1;
  if (test__floattitf(0x0007FB72EB000000LL, TF_C(0x1.FEDCBACp+50)))
    return 1;
  if (test__floattitf(0x0007FB72EBFFFFFFLL, TF_C(0x1.FEDCBAFFFFFFCp+50)))
    return 1;
  if (test__floattitf(0x0007FB72EC000000LL, TF_C(0x1.FEDCBBp+50)))
    return 1;
  if (test__floattitf(0x0007FB72E8000001LL, TF_C(0x1.FEDCBA0000004p+50)))
    return 1;

  if (test__floattitf(0x0007FB72E6000000LL, TF_C(0x1.FEDCB98p+50)))
    return 1;
  if (test__floattitf(0x0007FB72E7000000LL, TF_C(0x1.FEDCB9Cp+50)))
    return 1;
  if (test__floattitf(0x0007FB72E7FFFFFFLL, TF_C(0x1.FEDCB9FFFFFFCp+50)))
    return 1;
  if (test__floattitf(0x0007FB72E4000001LL, TF_C(0x1.FEDCB90000004p+50)))
    return 1;
  if (test__floattitf(0x0007FB72E4000000LL, TF_C(0x1.FEDCB9p+50)))
    return 1;

  if (test__floattitf(0x023479FD0E092DC0LL, TF_C(0x1.1A3CFE870496Ep+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DA1LL, TF_C(0x1.1A3CFE870496D08p+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DB0LL, TF_C(0x1.1A3CFE870496D8p+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DB8LL, TF_C(0x1.1A3CFE870496DCp+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DB6LL, TF_C(0x1.1A3CFE870496DBp+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DBFLL, TF_C(0x1.1A3CFE870496DF8p+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DC1LL, TF_C(0x1.1A3CFE870496E08p+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DC7LL, TF_C(0x1.1A3CFE870496E38p+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DC8LL, TF_C(0x1.1A3CFE870496E4p+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DCFLL, TF_C(0x1.1A3CFE870496E78p+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DD0LL, TF_C(0x1.1A3CFE870496E8p+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DD1LL, TF_C(0x1.1A3CFE870496E88p+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DD8LL, TF_C(0x1.1A3CFE870496ECp+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DDFLL, TF_C(0x1.1A3CFE870496EF8p+57)))
    return 1;
  if (test__floattitf(0x023479FD0E092DE0LL, TF_C(0x1.1A3CFE870496Fp+57)))
    return 1;

  if (test__floattitf(make_ti(0x023479FD0E092DC0LL, 0),
                      TF_C(0x1.1A3CFE870496Ep+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DA1LL, 1),
                      TF_C(0x1.1A3CFE870496D08p+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DB0LL, 2),
                      TF_C(0x1.1A3CFE870496D8p+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DB8LL, 3),
                      TF_C(0x1.1A3CFE870496DCp+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DB6LL, 4),
                      TF_C(0x1.1A3CFE870496DBp+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DBFLL, 5),
                      TF_C(0x1.1A3CFE870496DF8p+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DC1LL, 6),
                      TF_C(0x1.1A3CFE870496E08p+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DC7LL, 7),
                      TF_C(0x1.1A3CFE870496E38p+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DC8LL, 8),
                      TF_C(0x1.1A3CFE870496E4p+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DCFLL, 9),
                      TF_C(0x1.1A3CFE870496E78p+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DD0LL, 0),
                      TF_C(0x1.1A3CFE870496E8p+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DD1LL, 11),
                      TF_C(0x1.1A3CFE870496E88p+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DD8LL, 12),
                      TF_C(0x1.1A3CFE870496ECp+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DDFLL, 13),
                      TF_C(0x1.1A3CFE870496EF8p+121)))
    return 1;
  if (test__floattitf(make_ti(0x023479FD0E092DE0LL, 14),
                      TF_C(0x1.1A3CFE870496Fp+121)))
    return 1;

  if (test__floattitf(make_ti(0, 0xFFFFFFFFFFFFFFFFLL),
                      TF_C(0x1.FFFFFFFFFFFFFFFEp+63)))
    return 1;

  if (test__floattitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC2801LL),
                      TF_C(0x1.23456789ABCDEF0123456789ABC3p+124)))
    return 1;
  if (test__floattitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC3000LL),
                      TF_C(0x1.23456789ABCDEF0123456789ABC3p+124)))
    return 1;
  if (test__floattitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC37FFLL),
                      TF_C(0x1.23456789ABCDEF0123456789ABC3p+124)))
    return 1;
  if (test__floattitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC3800LL),
                      TF_C(0x1.23456789ABCDEF0123456789ABC4p+124)))
    return 1;
  if (test__floattitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC4000LL),
                      TF_C(0x1.23456789ABCDEF0123456789ABC4p+124)))
    return 1;
  if (test__floattitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC47FFLL),
                      TF_C(0x1.23456789ABCDEF0123456789ABC4p+124)))
    return 1;
  if (test__floattitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC4800LL),
                      TF_C(0x1.23456789ABCDEF0123456789ABC4p+124)))
    return 1;
  if (test__floattitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC4801LL),
                      TF_C(0x1.23456789ABCDEF0123456789ABC5p+124)))
    return 1;
  if (test__floattitf(make_ti(0x123456789ABCDEF0LL, 0x123456789ABC57FFLL),
                      TF_C(0x1.23456789ABCDEF0123456789ABC5p+124)))
    return 1;
  return 0;
}

#else
int main() {
  printf("skipped\n");
  return 0;
}
#endif
