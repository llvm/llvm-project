// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_multf3

#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "int_lib.h"
#include "fp_test.h"

// Returns: a * b
COMPILER_RT_ABI long double __multf3(long double a, long double b);

int test__multf3(long double a, long double b,
                 uint64_t expectedHi, uint64_t expectedLo)
{
    long double x = __multf3(a, b);
    int ret = compareResultF128(x, expectedHi, expectedLo);

    if (ret){
        printf("error in test__multf3(%.20Lf, %.20Lf) = %.20Lf, "
               "expected %.20Lf\n", a, b, x,
               fromRep128(expectedHi, expectedLo));
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    // qNaN * any = qNaN
    if (test__multf3(makeQNaN128(),
                     0x1.23456789abcdefp+5L,
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;
    // NaN * any = NaN
    if (test__multf3(makeNaN128(UINT64_C(0x800030000000)),
                     0x1.23456789abcdefp+5L,
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;
    // inf * any = inf
    if (test__multf3(makeInf128(),
                     0x1.23456789abcdefp+5L,
                     UINT64_C(0x7fff000000000000),
                     UINT64_C(0x0)))
        return 1;
    // any * any
    if (test__multf3(0x1.2eab345678439abcdefea56782346p+5L,
                     0x1.edcb34a235253948765432134674fp-1L,
                     UINT64_C(0x400423e7f9e3c9fc),
                     UINT64_C(0xd906c2c2a85777c4)))
        return 1;
    if (test__multf3(0x1.353e45674d89abacc3a2ebf3ff4ffp-50L,
                     0x1.ed8764648369535adf4be3214567fp-9L,
                     UINT64_C(0x3fc52a163c6223fc),
                     UINT64_C(0xc94c4bf0430768b4)))
        return 1;
    if (test__multf3(0x1.234425696abcad34a35eeffefdcbap+456L,
                     0x451.ed98d76e5d46e5f24323dff21ffp+600L,
                     UINT64_C(0x44293a91de5e0e94),
                     UINT64_C(0xe8ed17cc2cdf64ac)))
        return 1;
    if (test__multf3(0x1.4356473c82a9fabf2d22ace345defp-234L,
                     0x1.eda98765476743ab21da23d45678fp-455L,
                     UINT64_C(0x3d4f37c1a3137cae),
                     UINT64_C(0xfc6807048bc2836a)))
        return 1;
    // underflow
    if (test__multf3(0x1.23456734245345p-10000L,
                     0x1.edcba524498724p-6497L,
                     UINT64_C(0x0),
                     UINT64_C(0x0)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}
