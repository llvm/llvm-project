// RUN: %clang_cc1 -std=c23 %s -embed-dir=%S/Inputs -fsyntax-only -verify
// expected-no-diagnostics

const char data[] = {
#embed <jk.txt>
};
const char offset_data[] = {
#embed <jk.txt> limit(1)
};
static_assert(sizeof(data) == 2);
static_assert('j' == data[0]);
static_assert('k' == data[1]);
static_assert(sizeof(offset_data) == 1);
static_assert('j' == offset_data[0]);
static_assert(offset_data[0] == data[0]);
