// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/Inputs -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/Inputs -fsyntax-only %s -fexperimental-new-constant-interpreter

#include <stdbit.h>

constexpr __UINT8_TYPE__ u8_0xAB = stdc_memreverse8u8((__UINT8_TYPE__)0xAB);
_Static_assert(u8_0xAB == (__UINT8_TYPE__)0xAB, "");
constexpr __UINT8_TYPE__ u8_0x00 = stdc_memreverse8u8((__UINT8_TYPE__)0x00);
_Static_assert(u8_0x00 == (__UINT8_TYPE__)0x00, "");
constexpr __UINT8_TYPE__ u8_0xFF = stdc_memreverse8u8((__UINT8_TYPE__)0xFF);
_Static_assert(u8_0xFF == (__UINT8_TYPE__)0xFF, "");

constexpr __UINT16_TYPE__ u16_0x1234 = stdc_memreverse8u16((__UINT16_TYPE__)0x1234);
_Static_assert(u16_0x1234 == (__UINT16_TYPE__)0x3412, "");
constexpr __UINT16_TYPE__ u16_0x0000 = stdc_memreverse8u16((__UINT16_TYPE__)0x0000);
_Static_assert(u16_0x0000 == (__UINT16_TYPE__)0x0000, "");
constexpr __UINT16_TYPE__ u16_0xAAAA = stdc_memreverse8u16((__UINT16_TYPE__)0xAAAA);
_Static_assert(u16_0xAAAA == (__UINT16_TYPE__)0xAAAA, "");

constexpr __UINT32_TYPE__ u32_0x12345678 = stdc_memreverse8u32((__UINT32_TYPE__)0x12345678);
_Static_assert(u32_0x12345678 == (__UINT32_TYPE__)0x78563412, "");
constexpr __UINT32_TYPE__ u32_0x00000000 = stdc_memreverse8u32((__UINT32_TYPE__)0x00000000);
_Static_assert(u32_0x00000000 == (__UINT32_TYPE__)0x00000000, "");
constexpr __UINT32_TYPE__ u32_0xDEADBEEF = stdc_memreverse8u32((__UINT32_TYPE__)0xDEADBEEF);
_Static_assert(u32_0xDEADBEEF == (__UINT32_TYPE__)0xEFBEADDE, "");

constexpr __UINT64_TYPE__ u64_0x0102030405060708 = stdc_memreverse8u64((__UINT64_TYPE__)0x0102030405060708ULL);
_Static_assert(u64_0x0102030405060708 == (__UINT64_TYPE__)0x0807060504030201ULL, "");
constexpr __UINT64_TYPE__ u64_0x0000000000000000 = stdc_memreverse8u64((__UINT64_TYPE__)0x0000000000000000ULL);
_Static_assert(u64_0x0000000000000000 == (__UINT64_TYPE__)0x0000000000000000ULL, "");

constexpr __UINT16_TYPE__ u16_rt = stdc_memreverse8u16(stdc_memreverse8u16((__UINT16_TYPE__)0xABCD));
_Static_assert(u16_rt == (__UINT16_TYPE__)0xABCD, "");
constexpr __UINT32_TYPE__ u32_rt = stdc_memreverse8u32(stdc_memreverse8u32((__UINT32_TYPE__)0xDEADBEEF));
_Static_assert(u32_rt == (__UINT32_TYPE__)0xDEADBEEF, "");
constexpr __UINT64_TYPE__ u64_rt = stdc_memreverse8u64(stdc_memreverse8u64((__UINT64_TYPE__)0xCAFEBABEDEADBEEFULL));
_Static_assert(u64_rt == (__UINT64_TYPE__)0xCAFEBABEDEADBEEFULL, "");
