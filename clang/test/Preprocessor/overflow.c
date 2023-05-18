// RUN: %clang_cc1 -Eonly %s -verify -triple i686-pc-linux-gnu

// Multiply signed overflow
#if 0x7FFFFFFFFFFFFFFF*2 // expected-warning {{overflow}}
#endif

// Multiply unsigned overflow
#if 0xFFFFFFFFFFFFFFFF*2
#endif

// Add signed overflow
#if 0x7FFFFFFFFFFFFFFF+1 // expected-warning {{overflow}}
#endif

// Add unsigned overflow
#if 0xFFFFFFFFFFFFFFFF+1
#endif

// Subtract signed overflow
#if 0x7FFFFFFFFFFFFFFF- -1 // expected-warning {{overflow}}
#endif

// Subtract unsigned overflow
#if 0xFFFFFFFFFFFFFFFF- -1 // expected-warning {{converted from negative value}}
#endif
