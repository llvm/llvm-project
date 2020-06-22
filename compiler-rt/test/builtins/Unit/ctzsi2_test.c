// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_ctzsi2

#include "int_lib.h"
#include <stdio.h>

// Returns: the number of trailing 0-bits

// Precondition: a != 0

COMPILER_RT_ABI si_int __ctzsi2(si_int a);

int test__ctzsi2(si_int a, si_int expected)
{
    si_int x = __ctzsi2(a);
    if (x != expected)
        printf("error in __ctzsi2(0x%X) = %d, expected %d\n", a, x, expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};
char assumption_2[sizeof(si_int)*CHAR_BIT == 32] = {0};

int main()
{
//     if (test__ctzsi2(0x00000000, 32))  // undefined
//         return 1;
    if (test__ctzsi2(0x00000001, 0))
        return 1;
    if (test__ctzsi2(0x00000002, 1))
        return 1;
    if (test__ctzsi2(0x00000003, 0))
        return 1;
    if (test__ctzsi2(0x00000004, 2))
        return 1;
    if (test__ctzsi2(0x00000005, 0))
        return 1;
    if (test__ctzsi2(0x00000006, 1))
        return 1;
    if (test__ctzsi2(0x00000007, 0))
        return 1;
    if (test__ctzsi2(0x00000008, 3))
        return 1;
    if (test__ctzsi2(0x00000009, 0))
        return 1;
    if (test__ctzsi2(0x0000000A, 1))
        return 1;
    if (test__ctzsi2(0x0000000B, 0))
        return 1;
    if (test__ctzsi2(0x0000000C, 2))
        return 1;
    if (test__ctzsi2(0x0000000D, 0))
        return 1;
    if (test__ctzsi2(0x0000000E, 1))
        return 1;
    if (test__ctzsi2(0x0000000F, 0))
        return 1;
    if (test__ctzsi2(0x00000010, 4))
        return 1;
    if (test__ctzsi2(0x00000012, 1))
        return 1;
    if (test__ctzsi2(0x00000013, 0))
        return 1;
    if (test__ctzsi2(0x00000014, 2))
        return 1;
    if (test__ctzsi2(0x00000015, 0))
        return 1;
    if (test__ctzsi2(0x00000016, 1))
        return 1;
    if (test__ctzsi2(0x00000017, 0))
        return 1;
    if (test__ctzsi2(0x00000018, 3))
        return 1;
    if (test__ctzsi2(0x00000019, 0))
        return 1;
    if (test__ctzsi2(0x0000001A, 1))
        return 1;
    if (test__ctzsi2(0x0000001B, 0))
        return 1;
    if (test__ctzsi2(0x0000001C, 2))
        return 1;
    if (test__ctzsi2(0x0000001D, 0))
        return 1;
    if (test__ctzsi2(0x0000001E, 1))
        return 1;
    if (test__ctzsi2(0x0000001F, 0))
        return 1;
    if (test__ctzsi2(0x00000020, 5))
        return 1;
    if (test__ctzsi2(0x00000022, 1))
        return 1;
    if (test__ctzsi2(0x00000023, 0))
        return 1;
    if (test__ctzsi2(0x00000024, 2))
        return 1;
    if (test__ctzsi2(0x00000025, 0))
        return 1;
    if (test__ctzsi2(0x00000026, 1))
        return 1;
    if (test__ctzsi2(0x00000027, 0))
        return 1;
    if (test__ctzsi2(0x00000028, 3))
        return 1;
    if (test__ctzsi2(0x00000029, 0))
        return 1;
    if (test__ctzsi2(0x0000002A, 1))
        return 1;
    if (test__ctzsi2(0x0000002B, 0))
        return 1;
    if (test__ctzsi2(0x0000002C, 2))
        return 1;
    if (test__ctzsi2(0x0000002D, 0))
        return 1;
    if (test__ctzsi2(0x0000002E, 1))
        return 1;
    if (test__ctzsi2(0x0000002F, 0))
        return 1;
    if (test__ctzsi2(0x00000030, 4))
        return 1;
    if (test__ctzsi2(0x00000032, 1))
        return 1;
    if (test__ctzsi2(0x00000033, 0))
        return 1;
    if (test__ctzsi2(0x00000034, 2))
        return 1;
    if (test__ctzsi2(0x00000035, 0))
        return 1;
    if (test__ctzsi2(0x00000036, 1))
        return 1;
    if (test__ctzsi2(0x00000037, 0))
        return 1;
    if (test__ctzsi2(0x00000038, 3))
        return 1;
    if (test__ctzsi2(0x00000039, 0))
        return 1;
    if (test__ctzsi2(0x0000003A, 1))
        return 1;
    if (test__ctzsi2(0x0000003B, 0))
        return 1;
    if (test__ctzsi2(0x0000003C, 2))
        return 1;
    if (test__ctzsi2(0x0000003D, 0))
        return 1;
    if (test__ctzsi2(0x0000003E, 1))
        return 1;
    if (test__ctzsi2(0x0000003F, 0))
        return 1;
    if (test__ctzsi2(0x00000040, 6))
        return 1;
    if (test__ctzsi2(0x00000042, 1))
        return 1;
    if (test__ctzsi2(0x00000043, 0))
        return 1;
    if (test__ctzsi2(0x00000044, 2))
        return 1;
    if (test__ctzsi2(0x00000045, 0))
        return 1;
    if (test__ctzsi2(0x00000046, 1))
        return 1;
    if (test__ctzsi2(0x00000047, 0))
        return 1;
    if (test__ctzsi2(0x00000048, 3))
        return 1;
    if (test__ctzsi2(0x00000049, 0))
        return 1;
    if (test__ctzsi2(0x0000004A, 1))
        return 1;
    if (test__ctzsi2(0x0000004B, 0))
        return 1;
    if (test__ctzsi2(0x0000004C, 2))
        return 1;
    if (test__ctzsi2(0x0000004D, 0))
        return 1;
    if (test__ctzsi2(0x0000004E, 1))
        return 1;
    if (test__ctzsi2(0x0000004F, 0))
        return 1;
    if (test__ctzsi2(0x00000050, 4))
        return 1;
    if (test__ctzsi2(0x00000052, 1))
        return 1;
    if (test__ctzsi2(0x00000053, 0))
        return 1;
    if (test__ctzsi2(0x00000054, 2))
        return 1;
    if (test__ctzsi2(0x00000055, 0))
        return 1;
    if (test__ctzsi2(0x00000056, 1))
        return 1;
    if (test__ctzsi2(0x00000057, 0))
        return 1;
    if (test__ctzsi2(0x00000058, 3))
        return 1;
    if (test__ctzsi2(0x00000059, 0))
        return 1;
    if (test__ctzsi2(0x0000005A, 1))
        return 1;
    if (test__ctzsi2(0x0000005B, 0))
        return 1;
    if (test__ctzsi2(0x0000005C, 2))
        return 1;
    if (test__ctzsi2(0x0000005D, 0))
        return 1;
    if (test__ctzsi2(0x0000005E, 1))
        return 1;
    if (test__ctzsi2(0x0000005F, 0))
        return 1;
    if (test__ctzsi2(0x00000060, 5))
        return 1;
    if (test__ctzsi2(0x00000062, 1))
        return 1;
    if (test__ctzsi2(0x00000063, 0))
        return 1;
    if (test__ctzsi2(0x00000064, 2))
        return 1;
    if (test__ctzsi2(0x00000065, 0))
        return 1;
    if (test__ctzsi2(0x00000066, 1))
        return 1;
    if (test__ctzsi2(0x00000067, 0))
        return 1;
    if (test__ctzsi2(0x00000068, 3))
        return 1;
    if (test__ctzsi2(0x00000069, 0))
        return 1;
    if (test__ctzsi2(0x0000006A, 1))
        return 1;
    if (test__ctzsi2(0x0000006B, 0))
        return 1;
    if (test__ctzsi2(0x0000006C, 2))
        return 1;
    if (test__ctzsi2(0x0000006D, 0))
        return 1;
    if (test__ctzsi2(0x0000006E, 1))
        return 1;
    if (test__ctzsi2(0x0000006F, 0))
        return 1;
    if (test__ctzsi2(0x00000070, 4))
        return 1;
    if (test__ctzsi2(0x00000072, 1))
        return 1;
    if (test__ctzsi2(0x00000073, 0))
        return 1;
    if (test__ctzsi2(0x00000074, 2))
        return 1;
    if (test__ctzsi2(0x00000075, 0))
        return 1;
    if (test__ctzsi2(0x00000076, 1))
        return 1;
    if (test__ctzsi2(0x00000077, 0))
        return 1;
    if (test__ctzsi2(0x00000078, 3))
        return 1;
    if (test__ctzsi2(0x00000079, 0))
        return 1;
    if (test__ctzsi2(0x0000007A, 1))
        return 1;
    if (test__ctzsi2(0x0000007B, 0))
        return 1;
    if (test__ctzsi2(0x0000007C, 2))
        return 1;
    if (test__ctzsi2(0x0000007D, 0))
        return 1;
    if (test__ctzsi2(0x0000007E, 1))
        return 1;
    if (test__ctzsi2(0x0000007F, 0))
        return 1;
    if (test__ctzsi2(0x00000080, 7))
        return 1;
    if (test__ctzsi2(0x00000082, 1))
        return 1;
    if (test__ctzsi2(0x00000083, 0))
        return 1;
    if (test__ctzsi2(0x00000084, 2))
        return 1;
    if (test__ctzsi2(0x00000085, 0))
        return 1;
    if (test__ctzsi2(0x00000086, 1))
        return 1;
    if (test__ctzsi2(0x00000087, 0))
        return 1;
    if (test__ctzsi2(0x00000088, 3))
        return 1;
    if (test__ctzsi2(0x00000089, 0))
        return 1;
    if (test__ctzsi2(0x0000008A, 1))
        return 1;
    if (test__ctzsi2(0x0000008B, 0))
        return 1;
    if (test__ctzsi2(0x0000008C, 2))
        return 1;
    if (test__ctzsi2(0x0000008D, 0))
        return 1;
    if (test__ctzsi2(0x0000008E, 1))
        return 1;
    if (test__ctzsi2(0x0000008F, 0))
        return 1;
    if (test__ctzsi2(0x00000090, 4))
        return 1;
    if (test__ctzsi2(0x00000092, 1))
        return 1;
    if (test__ctzsi2(0x00000093, 0))
        return 1;
    if (test__ctzsi2(0x00000094, 2))
        return 1;
    if (test__ctzsi2(0x00000095, 0))
        return 1;
    if (test__ctzsi2(0x00000096, 1))
        return 1;
    if (test__ctzsi2(0x00000097, 0))
        return 1;
    if (test__ctzsi2(0x00000098, 3))
        return 1;
    if (test__ctzsi2(0x00000099, 0))
        return 1;
    if (test__ctzsi2(0x0000009A, 1))
        return 1;
    if (test__ctzsi2(0x0000009B, 0))
        return 1;
    if (test__ctzsi2(0x0000009C, 2))
        return 1;
    if (test__ctzsi2(0x0000009D, 0))
        return 1;
    if (test__ctzsi2(0x0000009E, 1))
        return 1;
    if (test__ctzsi2(0x0000009F, 0))
        return 1;
    if (test__ctzsi2(0x000000A0, 5))
        return 1;
    if (test__ctzsi2(0x000000A2, 1))
        return 1;
    if (test__ctzsi2(0x000000A3, 0))
        return 1;
    if (test__ctzsi2(0x000000A4, 2))
        return 1;
    if (test__ctzsi2(0x000000A5, 0))
        return 1;
    if (test__ctzsi2(0x000000A6, 1))
        return 1;
    if (test__ctzsi2(0x000000A7, 0))
        return 1;
    if (test__ctzsi2(0x000000A8, 3))
        return 1;
    if (test__ctzsi2(0x000000A9, 0))
        return 1;
    if (test__ctzsi2(0x000000AA, 1))
        return 1;
    if (test__ctzsi2(0x000000AB, 0))
        return 1;
    if (test__ctzsi2(0x000000AC, 2))
        return 1;
    if (test__ctzsi2(0x000000AD, 0))
        return 1;
    if (test__ctzsi2(0x000000AE, 1))
        return 1;
    if (test__ctzsi2(0x000000AF, 0))
        return 1;
    if (test__ctzsi2(0x000000B0, 4))
        return 1;
    if (test__ctzsi2(0x000000B2, 1))
        return 1;
    if (test__ctzsi2(0x000000B3, 0))
        return 1;
    if (test__ctzsi2(0x000000B4, 2))
        return 1;
    if (test__ctzsi2(0x000000B5, 0))
        return 1;
    if (test__ctzsi2(0x000000B6, 1))
        return 1;
    if (test__ctzsi2(0x000000B7, 0))
        return 1;
    if (test__ctzsi2(0x000000B8, 3))
        return 1;
    if (test__ctzsi2(0x000000B9, 0))
        return 1;
    if (test__ctzsi2(0x000000BA, 1))
        return 1;
    if (test__ctzsi2(0x000000BB, 0))
        return 1;
    if (test__ctzsi2(0x000000BC, 2))
        return 1;
    if (test__ctzsi2(0x000000BD, 0))
        return 1;
    if (test__ctzsi2(0x000000BE, 1))
        return 1;
    if (test__ctzsi2(0x000000BF, 0))
        return 1;
    if (test__ctzsi2(0x000000C0, 6))
        return 1;
    if (test__ctzsi2(0x000000C2, 1))
        return 1;
    if (test__ctzsi2(0x000000C3, 0))
        return 1;
    if (test__ctzsi2(0x000000C4, 2))
        return 1;
    if (test__ctzsi2(0x000000C5, 0))
        return 1;
    if (test__ctzsi2(0x000000C6, 1))
        return 1;
    if (test__ctzsi2(0x000000C7, 0))
        return 1;
    if (test__ctzsi2(0x000000C8, 3))
        return 1;
    if (test__ctzsi2(0x000000C9, 0))
        return 1;
    if (test__ctzsi2(0x000000CA, 1))
        return 1;
    if (test__ctzsi2(0x000000CB, 0))
        return 1;
    if (test__ctzsi2(0x000000CC, 2))
        return 1;
    if (test__ctzsi2(0x000000CD, 0))
        return 1;
    if (test__ctzsi2(0x000000CE, 1))
        return 1;
    if (test__ctzsi2(0x000000CF, 0))
        return 1;
    if (test__ctzsi2(0x000000D0, 4))
        return 1;
    if (test__ctzsi2(0x000000D2, 1))
        return 1;
    if (test__ctzsi2(0x000000D3, 0))
        return 1;
    if (test__ctzsi2(0x000000D4, 2))
        return 1;
    if (test__ctzsi2(0x000000D5, 0))
        return 1;
    if (test__ctzsi2(0x000000D6, 1))
        return 1;
    if (test__ctzsi2(0x000000D7, 0))
        return 1;
    if (test__ctzsi2(0x000000D8, 3))
        return 1;
    if (test__ctzsi2(0x000000D9, 0))
        return 1;
    if (test__ctzsi2(0x000000DA, 1))
        return 1;
    if (test__ctzsi2(0x000000DB, 0))
        return 1;
    if (test__ctzsi2(0x000000DC, 2))
        return 1;
    if (test__ctzsi2(0x000000DD, 0))
        return 1;
    if (test__ctzsi2(0x000000DE, 1))
        return 1;
    if (test__ctzsi2(0x000000DF, 0))
        return 1;
    if (test__ctzsi2(0x000000E0, 5))
        return 1;
    if (test__ctzsi2(0x000000E2, 1))
        return 1;
    if (test__ctzsi2(0x000000E3, 0))
        return 1;
    if (test__ctzsi2(0x000000E4, 2))
        return 1;
    if (test__ctzsi2(0x000000E5, 0))
        return 1;
    if (test__ctzsi2(0x000000E6, 1))
        return 1;
    if (test__ctzsi2(0x000000E7, 0))
        return 1;
    if (test__ctzsi2(0x000000E8, 3))
        return 1;
    if (test__ctzsi2(0x000000E9, 0))
        return 1;
    if (test__ctzsi2(0x000000EA, 1))
        return 1;
    if (test__ctzsi2(0x000000EB, 0))
        return 1;
    if (test__ctzsi2(0x000000EC, 2))
        return 1;
    if (test__ctzsi2(0x000000ED, 0))
        return 1;
    if (test__ctzsi2(0x000000EE, 1))
        return 1;
    if (test__ctzsi2(0x000000EF, 0))
        return 1;
    if (test__ctzsi2(0x000000F0, 4))
        return 1;
    if (test__ctzsi2(0x000000F2, 1))
        return 1;
    if (test__ctzsi2(0x000000F3, 0))
        return 1;
    if (test__ctzsi2(0x000000F4, 2))
        return 1;
    if (test__ctzsi2(0x000000F5, 0))
        return 1;
    if (test__ctzsi2(0x000000F6, 1))
        return 1;
    if (test__ctzsi2(0x000000F7, 0))
        return 1;
    if (test__ctzsi2(0x000000F8, 3))
        return 1;
    if (test__ctzsi2(0x000000F9, 0))
        return 1;
    if (test__ctzsi2(0x000000FA, 1))
        return 1;
    if (test__ctzsi2(0x000000FB, 0))
        return 1;
    if (test__ctzsi2(0x000000FC, 2))
        return 1;
    if (test__ctzsi2(0x000000FD, 0))
        return 1;
    if (test__ctzsi2(0x000000FE, 1))
        return 1;
    if (test__ctzsi2(0x000000FF, 0))
        return 1;

    if (test__ctzsi2(0x00000100, 8))
        return 1;
    if (test__ctzsi2(0x00000200, 9))
        return 1;
    if (test__ctzsi2(0x00000400, 10))
        return 1;
    if (test__ctzsi2(0x00000800, 11))
        return 1;
    if (test__ctzsi2(0x00001000, 12))
        return 1;
    if (test__ctzsi2(0x00002000, 13))
        return 1;
    if (test__ctzsi2(0x00004000, 14))
        return 1;
    if (test__ctzsi2(0x00008000, 15))
        return 1;
    if (test__ctzsi2(0x00010000, 16))
        return 1;
    if (test__ctzsi2(0x00020000, 17))
        return 1;
    if (test__ctzsi2(0x00040000, 18))
        return 1;
    if (test__ctzsi2(0x00080000, 19))
        return 1;
    if (test__ctzsi2(0x00100000, 20))
        return 1;
    if (test__ctzsi2(0x00200000, 21))
        return 1;
    if (test__ctzsi2(0x00400000, 22))
        return 1;
    if (test__ctzsi2(0x00800000, 23))
        return 1;
    if (test__ctzsi2(0x01000000, 24))
        return 1;
    if (test__ctzsi2(0x02000000, 25))
        return 1;
    if (test__ctzsi2(0x04000000, 26))
        return 1;
    if (test__ctzsi2(0x08000000, 27))
        return 1;
    if (test__ctzsi2(0x10000000, 28))
        return 1;
    if (test__ctzsi2(0x20000000, 29))
        return 1;
    if (test__ctzsi2(0x40000000, 30))
        return 1;
    if (test__ctzsi2(0x80000000, 31))
        return 1;

   return 0;
}
