// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -ffreestanding -verify %s
// RUN: %clang_cc1 -std=c++11 -triple i686-pc-linux-gnu -ffreestanding -verify %s
// RUN: %clang_cc1 -std=c++11 -triple avr-unknown-unknown -ffreestanding -verify %s

// expected-no-diagnostics

// Test that checks that the builtins return the same type as the stdint.h type
// withe the same witdh. This is done by first declaring a variable of a stdint
// type of the correct width and the redeclaring the variable with the type that
// the builting return. If the types are different you should the an error from
// clang of the form:
// "redefinition of '<varname>' with a different type: '<type1>' vs '<type2>'
// (with gcc you get an error message
// "conflicting declaration '<type> <varname>'").

#include <stdint.h>

extern uint16_t bswap16;
decltype(__builtin_bswap16(0)) bswap16 = 42;
extern uint32_t bswap32;
decltype(__builtin_bswap32(0)) bswap32 = 42;
extern uint64_t bswap64;
decltype(__builtin_bswap64(0)) bswap64 = 42;

#ifdef __clang__
extern uint8_t bitrev8;
decltype(__builtin_bitreverse8(0)) bitrev8 = 42;
extern uint16_t bitrev16;
decltype(__builtin_bitreverse16(0)) bitrev16 = 42;
extern uint32_t bitrev32;
decltype(__builtin_bitreverse32(0)) bitrev32 = 42;
extern uint64_t bitrev64;
decltype(__builtin_bitreverse64(0)) bitrev64 = 42;

extern uint8_t rotl8;
decltype(__builtin_rotateleft8(0,0)) rotl8 = 42;
extern uint16_t rotl16;
decltype(__builtin_rotateleft16(0,0)) rotl16 = 42;
extern uint32_t rotl32;
decltype(__builtin_rotateleft32(0,0)) rotl32 = 42;
extern uint64_t rotl64;
decltype(__builtin_rotateleft64(0,0)) rotl64 = 42;

extern uint8_t rotr8;
decltype(__builtin_rotateright8(0,0)) rotr8 = 42;
extern uint16_t rotr16;
decltype(__builtin_rotateright16(0,0)) rotr16 = 42;
extern uint32_t rotr32;
decltype(__builtin_rotateright32(0,0)) rotr32 = 42;
extern uint64_t rotr64;
decltype(__builtin_rotateright64(0,0)) rotr64 = 42;
#endif
