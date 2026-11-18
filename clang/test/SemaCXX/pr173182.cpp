// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-unknown-linux-gnu %s

enum E2 { // expected-warning {{enumeration values exceed range of largest integer}}
  V2 = ((__int128)0x1000000000000000 << 64) + 1
};

__int128 get_val() {
  return (enum E2)__int128(V2 >> 4);
}