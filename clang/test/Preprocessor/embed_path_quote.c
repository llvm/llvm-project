// RUN: %clang_cc1 -std=c23 %s -fsyntax-only --embed-dir=%S/Inputs -verify
// expected-no-diagnostics

const char data[] = {
#embed "single_byte.txt"
};
static_assert(sizeof(data) == 1);
static_assert('a' == data[0]);
