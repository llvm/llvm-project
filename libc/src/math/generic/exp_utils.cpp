//===-- Implemention of exp and friends' utils ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exp_utils.h"

namespace LIBC_NAMESPACE {

const Exp2fDataTable exp2f_data = {
    // :tab[i] = uint(2^(i/N)) - (i << 52-BITS)
    //    used for computing 2^(k/N) for an int |k| < 150 N as
    //    double(tab[k%N] + (k << 52-BITS))
    {
// tab
#if N == 8
        0x3ff0000000000000,
        0x3fef72b83c7d517b,
        0x3fef06fe0a31b715,
        0x3feebfdad5362a27,
        0x3feea09e667f3bcd,
        0x3feeace5422aa0db,
        0x3feee89f995ad3ad,
        0x3fef5818dcfba487,
#elif N == 16
        0x3ff0000000000000,
        0x3fefb5586cf9890f,
        0x3fef72b83c7d517b,
        0x3fef387a6e756238,
        0x3fef06fe0a31b715,
        0x3feedea64c123422,
        0x3feebfdad5362a27,
        0x3feeab07dd485429,
        0x3feea09e667f3bcd,
        0x3feea11473eb0187,
        0x3feeace5422aa0db,
        0x3feec49182a3f090,
        0x3feee89f995ad3ad,
        0x3fef199bdd85529c,
        0x3fef5818dcfba487,
        0x3fefa4afa2a490da,
#elif N == 32
        0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f,
        0x3fef9301d0125b51, 0x3fef72b83c7d517b, 0x3fef54873168b9aa,
        0x3fef387a6e756238, 0x3fef1e9df51fdee1, 0x3fef06fe0a31b715,
        0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
        0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429,
        0x3feea47eb03a5585, 0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74,
        0x3feea11473eb0187, 0x3feea589994cce13, 0x3feeace5422aa0db,
        0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
        0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c,
        0x3fef3720dcef9069, 0x3fef5818dcfba487, 0x3fef7c97337b9b5f,
        0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
#elif N == 64
        0x3ff0000000000000, 0x3fefec9a3e778061, 0x3fefd9b0d3158574,
        0x3fefc74518759bc8, 0x3fefb5586cf9890f, 0x3fefa3ec32d3d1a2,
        0x3fef9301d0125b51, 0x3fef829aaea92de0, 0x3fef72b83c7d517b,
        0x3fef635beb6fcb75, 0x3fef54873168b9aa, 0x3fef463b88628cd6,
        0x3fef387a6e756238, 0x3fef2b4565e27cdd, 0x3fef1e9df51fdee1,
        0x3fef1285a6e4030b, 0x3fef06fe0a31b715, 0x3feefc08b26416ff,
        0x3feef1a7373aa9cb, 0x3feee7db34e59ff7, 0x3feedea64c123422,
        0x3feed60a21f72e2a, 0x3feece086061892d, 0x3feec6a2b5c13cd0,
        0x3feebfdad5362a27, 0x3feeb9b2769d2ca7, 0x3feeb42b569d4f82,
        0x3feeaf4736b527da, 0x3feeab07dd485429, 0x3feea76f15ad2148,
        0x3feea47eb03a5585, 0x3feea23882552225, 0x3feea09e667f3bcd,
        0x3fee9fb23c651a2f, 0x3fee9f75e8ec5f74, 0x3fee9feb564267c9,
        0x3feea11473eb0187, 0x3feea2f336cf4e62, 0x3feea589994cce13,
        0x3feea8d99b4492ed, 0x3feeace5422aa0db, 0x3feeb1ae99157736,
        0x3feeb737b0cdc5e5, 0x3feebd829fde4e50, 0x3feec49182a3f090,
        0x3feecc667b5de565, 0x3feed503b23e255d, 0x3feede6b5579fdbf,
        0x3feee89f995ad3ad, 0x3feef3a2b84f15fb, 0x3feeff76f2fb5e47,
        0x3fef0c1e904bc1d2, 0x3fef199bdd85529c, 0x3fef27f12e57d14b,
        0x3fef3720dcef9069, 0x3fef472d4a07897c, 0x3fef5818dcfba487,
        0x3fef69e603db3285, 0x3fef7c97337b9b5f, 0x3fef902ee78b3ff6,
        0x3fefa4afa2a490da, 0x3fefba1bee615a27, 0x3fefd0765b6e4540,
        0x3fefe7c1819e90d8,
#endif
    },
    as_double(0x4338000000000000) / N, // shift_scaled
    {
// poly
#if N == 8
        as_double(0x3fac6a00335106e2),
        as_double(0x3fcec0c313449f55),
        as_double(0x3fe62e431111f69f),
#elif N == 16
        as_double(0x3fac6ac6aa313963),
        as_double(0x3fcebfff4532d9ba),
        as_double(0x3fe62e43001bc49f),
#elif N == 32
        as_double(0x3fac6af84b912394),
        as_double(0x3fcebfce50fac4f3),
        as_double(0x3fe62e42ff0c52d6),
#elif N == 64
        as_double(0x3fac6b04b4221b2a),
        as_double(0x3fcebfc213e184d7),
        as_double(0x3fe62e42fefb5b7f),
#endif
    },
    as_double(0x4338000000000000),     // shift
    as_double(0x3ff71547652b82fe) * N, // invln2_scaled
    {
// poly_scaled
#if N == 8
        as_double(0x3fac6a00335106e2) / N / N / N,
        as_double(0x3fcec0c313449f55) / N / N,
        as_double(0x3fe62e431111f69f) / N,
#elif N == 16
        as_double(0x3fac6ac6aa313963) / N / N / N,
        as_double(0x3fcebfff4532d9ba) / N / N,
        as_double(0x3fe62e43001bc49f) / N,
#elif N == 32
        as_double(0x3fac6af84b912394) / N / N / N,
        as_double(0x3fcebfce50fac4f3) / N / N,
        as_double(0x3fe62e42ff0c52d6) / N,
#elif N == 64
        as_double(0x3fac6b04b4221b2a) / N / N / N,
        as_double(0x3fcebfc213e184d7) / N / N,
        as_double(0x3fe62e42fefb5b7f) / N,
#endif
    },
};

} // namespace LIBC_NAMESPACE
