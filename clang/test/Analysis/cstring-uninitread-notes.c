// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core,alpha.unix.cstring \
// RUN:   -analyzer-output=text

#include "Inputs/system-header-simulator.h"

// Inspired by a report on ffmpeg, libavcodec/tiertexseqv.c, seq_decode_op1().
int coin();

void maybeWrite(const char *src, unsigned size, int *dst) {
  if (coin()) // expected-note{{Assuming the condition is false}}
              // expected-note@-1{{Taking false branch}}
    memcpy(dst, src, size);
} // expected-note{{Returning without writing to '*dst'}}

void returning_without_writing_to_memcpy(const char *src, unsigned size) {
  int block[8 * 8]; // expected-note{{'block' initialized here}}
                                // expected-note@+1{{Calling 'maybeWrite'}}
  maybeWrite(src, size, block); // expected-note{{Returning from 'maybeWrite'}}

  int buf[8 * 8];
  memcpy(buf, &block[0], 8); // expected-warning{{The first element of the 2nd argument is undefined [alpha.unix.cstring.UninitializedRead]}}
                             // expected-note@-1{{The first element of the 2nd argument is undefined}}
                             // expected-note@-2{{Other elements might also be undefined}}
}
