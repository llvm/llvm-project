// RUN: %clang_cc1 -x c -triple bpf-pc-linux-gnu -dwarf-version=4 -fsyntax-only -verify %s

struct {
  char f1[100];
  int f2;
} tmp = {};

unsigned invalid1() { return __builtin_btf_type_id(1, tmp); } // expected-error {{__builtin_btf_type_id argument 2 not a constant}}
unsigned invalid2() { return __builtin_btf_type_id(1, 1, 1); } // expected-error {{too many arguments to function call, expected 2, have 3}}

int valid1() { return __builtin_btf_type_id(tmp, 0); }
int valid2() { return __builtin_btf_type_id(&tmp, 1); }
int valid3() { return __builtin_btf_type_id(tmp.f1[4], 10); }
