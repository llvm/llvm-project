// RUN: %clang_cc1 -std=c23 %s --embed-dir=%S/Inputs -fsyntax-only -verify

const char data[] = {
#embed <media/empty> if_empty(123, 124, 125)
};
const char non_empty_data[] = {
#embed <jk.txt> if_empty(123, 124, 125)
};
static_assert(sizeof(data) == 3);
static_assert(123 == data[0]);
static_assert(124 == data[1]);
static_assert(125 == data[2]);
static_assert(sizeof(non_empty_data) == 2);
static_assert('j' == non_empty_data[0]);
static_assert('k' == non_empty_data[1]);

// Ensure we diagnose duplicate parameters even if they're the same value.
const unsigned char a[] = {
#embed <jk.txt> if_empty(1) prefix() if_empty(2)
// expected-error@-1 {{cannot specify parameter 'if_empty' twice in the same '#embed' directive}}
,
#embed <jk.txt> if_empty(1) suffix() if_empty(2)
// expected-error@-1 {{cannot specify parameter 'if_empty' twice in the same '#embed' directive}}
};
