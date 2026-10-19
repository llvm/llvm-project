// RUN: %clang_cc1 -x c++ -std=c++23

unsigned h = U'\N{INVALID€}'; \
//expected-error {{'INVALID<0x80>' is not a valid Unicode character name}}
//expected-note  {{character <0x80> cannot appear in a Unicode character name}}
