// RUN: %clang_cc1 -std=c23 %s --embed-dir=%S/Inputs -fsyntax-only -verify

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

// Cannot have a negative limit.
#embed <jk.txt> limit(-1)
// expected-error@-1 {{invalid value '-1'; must be positive}}

// It can have a limit of 0, in which case the __has_embed should return false.
#if __has_embed(<jk.txt> limit(0)) != __STDC_EMBED_EMPTY__
#error "__has_embed should return false when there's no data"
#endif

// When the limit is zero, the resource is empty, so if_empty kicks in.
const unsigned char buffer[] = {
#embed <jk.txt> limit(0) if_empty(1)
};
static_assert(sizeof(buffer) == 1);
static_assert(buffer[0] == 1);

// However, prefix and suffix do not kick in.
const unsigned char other_buffer[] = {
  1,
#embed <jk.txt> limit(0) prefix(2,) suffix(3)
};
static_assert(sizeof(other_buffer) == 1);
static_assert(other_buffer[0] == 1);

// Ensure we can limit to something larger than the file size as well.
const unsigned char third_buffer[] = {
#embed <jk.txt> limit(100)
};
static_assert(sizeof(third_buffer) == 2);
static_assert('j' == third_buffer[0]);
static_assert('k' == third_buffer[1]);

// Test the limits of a file with more than one character in it.
const unsigned char fourth_buffer[] = {
#embed <media/art.txt> limit(10)
};
static_assert(sizeof(fourth_buffer) == 10);
static_assert(' ' == fourth_buffer[0]);
static_assert(' ' == fourth_buffer[1]);
static_assert(' ' == fourth_buffer[2]);
static_assert(' ' == fourth_buffer[3]);
static_assert(' ' == fourth_buffer[4]);
static_assert(' ' == fourth_buffer[5]);
static_assert(' ' == fourth_buffer[6]);
static_assert(' ' == fourth_buffer[7]);
static_assert(' ' == fourth_buffer[8]);
static_assert(' ' == fourth_buffer[9]);

// Ensure that a limit larger than what can fit into a 64-bit value is
// rejected. This limit is fine because it fits in a 64-bit value.
const unsigned char fifth_buffer[] = {
#embed <jk.txt> limit(0xFFFF'FFFF'FFFF'FFFF)
};
static_assert(sizeof(fifth_buffer) == 2);
static_assert('j' == fifth_buffer[0]);
static_assert('k' == fifth_buffer[1]);

// But this one is not fine because it does not fit into a 64-bit value.
const unsigned char sixth_buffer[] = {
#embed <jk.txt> limit(0xFFFF'FFFF'FFFF'FFFF'1)
};
// expected-error@-2 {{integer literal is too large to be represented in any integer type}}
// Note: the preprocessor will continue with the truncated value, so the parser
// will treat this case and the previous one identically in terms of what
// contents are retained from the embedded resource (which is the entire file).

// Ensure we diagnose duplicate parameters even if they're the same value.
const unsigned char a[] = {
#embed <jk.txt> limit(1) prefix() limit(1)
// expected-error@-1 {{cannot specify parameter 'limit' twice in the same '#embed' directive}}
,
#embed <jk.txt> limit(1) if_empty() limit(2)
// expected-error@-1 {{cannot specify parameter 'limit' twice in the same '#embed' directive}}
};

// C23 6.10.3.2p2
static_assert(
#embed <jk.txt> limit(defined(FOO)) // expected-error {{'defined' cannot appear within this context}}
  == 0); // expected-error {{expected expression}}
