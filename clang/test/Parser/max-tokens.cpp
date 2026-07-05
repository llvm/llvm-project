// RUN: %clang_cc1 -fsyntax-only -verify %s -Wmax-tokens
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wmax-tokens -DMAX_TOKENS          -fmax-tokens=2
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wmax-tokens -DMAX_TOKENS_OVERRIDE -fmax-tokens=9

int x, y, z;

#pragma clang max_tokens_here         // expected-error  {{missing argument to '#pragma clang max_tokens_here'; expected integer}}
#pragma clang max_tokens_here foo     // expected-error  {{expected an integer argument in '#pragma clang max_tokens_here'}}
#pragma clang max_tokens_here 123 456 // expected-warning{{extra tokens at end of '#pragma clang max_tokens_here' - ignored}}

#pragma clang max_tokens_here 1 // expected-warning{{the number of preprocessor source tokens (7) exceeds this token limit (1)}}


#pragma clang max_tokens_total // expected-error{{missing argument to '#pragma clang max_tokens_total'; expected integer}}
#pragma clang max_tokens_total foo // expected-error{{expected an integer argument in '#pragma clang max_tokens_total'}}
#pragma clang max_tokens_total 123 456 // expected-warning{{extra tokens at end of '#pragma clang max_tokens_total' - ignored}}

#ifdef MAX_TOKENS_OVERRIDE
#pragma clang max_tokens_total 3 // expected-warning@+4{{the total number of preprocessor source tokens (8) exceeds the token limit (3)}}
                                // expected-note@-1{{total token limit set here}}
#elif MAX_TOKENS
// expected-warning@+1{{the total number of preprocessor source tokens (8) exceeds the token limit (2)}}
#endif
