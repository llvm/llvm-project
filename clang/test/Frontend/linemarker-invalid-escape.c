// RUN: %clang_cc1 -emit-llvm -o - -verify %s

# 1 "original\x12source.c" // expected-error {{invalid escape sequence '\x12' in an unevaluated string literal}}

int x = 0;
