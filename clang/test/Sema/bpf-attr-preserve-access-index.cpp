// RUN: %clang_cc1 -x c++ -triple bpf-pc-linux-gnu -dwarf-version=4 -fsyntax-only -verify %s

#define __reloc__ __attribute__((preserve_access_index))

struct t1 {
  int a;
  int b[4];
  int c:1;
} __reloc__; // expected-warning {{'preserve_access_index' attribute ignored}}
