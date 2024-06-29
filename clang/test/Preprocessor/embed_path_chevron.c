// RUN: %clang_cc1 %s -std=c23 -fsyntax-only --embed-dir=%S/Inputs -verify
// expected-no-diagnostics

const char data[] = {
#embed <single_byte.txt>
};
static_assert(sizeof(data) == 1);
static_assert('b' == data[0]);
