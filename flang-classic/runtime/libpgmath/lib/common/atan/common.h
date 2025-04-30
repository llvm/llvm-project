
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

    // P = fpminimax(atan(x),
    // [|1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39|],
    // [|double...|],[0.000000000000001;1.0], floating, relative);
    // const vdouble C1 = vcast_vd_d(0x1p0);
    const double C2 = -0x1.555555555543fp-2;
    const double C3 = 0x1.999999998357fp-3;
    const double C4 = -0x1.2492491f82b1ep-3;
    const double C5 = 0x1.c71c70986d997p-4;
    const double C6 = -0x1.745d01dfeedccp-4;
    const double C7 = 0x1.3b12afded14e7p-4;
    const double C8 = -0x1.1108885ecb366p-4;
    const double C9 = 0x1.e17749a95ee9fp-5;
    const double C10 = -0x1.ad2fb9d1c3fc2p-5;
    const double C11 = 0x1.7edb66d1f72d7p-5;
    const double C12 = -0x1.4f32588ce844dp-5;
    const double C13 = 0x1.16f6061fc7091p-5;
    const double C14 = -0x1.a6d39bcd1c5d7p-6;
    const double C15 = 0x1.15d9a141937d7p-6;
    const double C16 = -0x1.2c7c74714ff5p-7;
    const double C17 = 0x1.f863b451c4fffp-9;
    const double C18 = -0x1.3066efb84f247p-10;
    const double C19 = 0x1.d25b20dafefb2p-13;
    const double C20 = -0x1.52c1661292134p-16;

    #define PI_2 1.57079632679489655799898173427

