// RUN: %clang_cc1 -std=c23 %s --embed-dir=%S/Inputs -fsyntax-only -verify

const char data[] = {
#embed <single_byte.txt> suffix(, '\xA')
};
const char empty_data[] = {
#embed <media/empty> suffix(, '\xA')
1
};
static_assert(sizeof(data) == 2);
static_assert('b' == data[0]);
static_assert('\xA' == data[1]);
static_assert(sizeof(empty_data) == 1);
static_assert(1 == empty_data[0]);

struct S {
  int x, y, z;
};

const struct S s = {
#embed <single_byte.txt> suffix( , .y = 100, .z = 10 )
};

static_assert(s.x == 'b');
static_assert(s.y == 100);
static_assert(s.z == 10);

// Ensure that an empty file does not produce any suffix tokens. If it did,
// there would be random tokens here that the parser would trip on.
#embed <media/empty> suffix(0)

// Ensure we diagnose duplicate parameters even if they're the same value.
const unsigned char a[] = {
#embed <jk.txt> suffix(,1) prefix() suffix(,1)
// expected-error@-1 {{cannot specify parameter 'suffix' twice in the same '#embed' directive}}
,
#embed <jk.txt> suffix(,1) if_empty() suffix(,2)
// expected-error@-1 {{cannot specify parameter 'suffix' twice in the same '#embed' directive}}
};
