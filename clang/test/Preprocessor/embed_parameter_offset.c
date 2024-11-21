// RUN: %clang_cc1 -std=c23 %s --embed-dir=%S/Inputs -fsyntax-only -verify

const char data[] = {
#embed <jk.txt>
};
const char offset_data[] = {
#embed <jk.txt> clang::offset(1)
};
static_assert(sizeof(data) == 2);
static_assert('j' == data[0]);
static_assert('k' == data[1]);
static_assert(sizeof(offset_data) == 1);
static_assert('k' == offset_data[0]);
static_assert(offset_data[0] == data[1]);

// Cannot have a negative offset.
#embed <jk.txt> clang::offset(-1)
// expected-error@-1 {{invalid value '-1'; must be positive}}

// If the offset is past the end of the file, the file should be considered
// empty.
#if __has_embed(<jk.txt> clang::offset(3)) != __STDC_EMBED_EMPTY__
#error "__has_embed should return false when there's no data"
#endif

// When the offset is past the end of the file, the resource is empty, so if_empty kicks in.
const unsigned char buffer[] = {
#embed <jk.txt> clang::offset(3) if_empty(1)
};
static_assert(sizeof(buffer) == 1);
static_assert(buffer[0] == 1);

// However, prefix and suffix do not kick in.
const unsigned char other_buffer[] = {
  1,
#embed <jk.txt> clang::offset(3) prefix(2,) suffix(3)
};
static_assert(sizeof(other_buffer) == 1);
static_assert(other_buffer[0] == 1);

// Ensure we can offset to zero (that's the default behavior)
const unsigned char third_buffer[] = {
#embed <jk.txt> clang::offset(0)
};
static_assert(sizeof(third_buffer) == 2);
static_assert('j' == third_buffer[0]);
static_assert('k' == third_buffer[1]);

// Test the offsets of a file with more than one character in it.
const unsigned char fourth_buffer[] = {
#embed <media/art.txt> clang::offset(24) limit(4)
};
static_assert(sizeof(fourth_buffer) == 4);
static_assert('.' == fourth_buffer[0]);
static_assert('-' == fourth_buffer[1]);
static_assert('.' == fourth_buffer[2]);
static_assert('\'' == fourth_buffer[3]);

// Ensure that an offset larger than what can fit into a 64-bit value is
// rejected. This offset is fine because it fits in a 64-bit value.
const unsigned char fifth_buffer[] = {
  1,
#embed <jk.txt> clang::offset(0xFFFF'FFFF'FFFF'FFFF)
};
static_assert(sizeof(fifth_buffer) == 1);
static_assert(1 == fifth_buffer[0]);

// But this one is not fine because it does not fit into a 64-bit value.
const unsigned char sixth_buffer[] = {
#embed <jk.txt> clang::offset(0xFFFF'FFFF'FFFF'FFFF'1)
};
// expected-error@-2 {{integer literal is too large to be represented in any integer type}}

// Ensure we diagnose duplicate parameters even if they're the same value.
const unsigned char a[] = {
#embed <jk.txt> clang::offset(1) prefix() clang::offset(1)
// expected-error@-1 {{cannot specify parameter 'clang::offset' twice in the same '#embed' directive}}
,
#embed <jk.txt> clang::offset(1) if_empty() clang::offset(2)
// expected-error@-1 {{cannot specify parameter 'clang::offset' twice in the same '#embed' directive}}
};

// Matches with C23 6.10.3.2p2, is documented as part of our extension.
static_assert(
#embed <jk.txt> clang::offset(defined(FOO))
  == 0); // expected-error {{expected expression}}
 /* expected-error@-2 {{'defined' cannot appear within this context}}
    pedantic-warning@-2 {{'clang::offset' is a Clang extension}}
  */
