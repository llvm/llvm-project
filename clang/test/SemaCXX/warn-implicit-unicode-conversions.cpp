// RUN: %clang_cc1 -verify -fsyntax-only -std=c++20 -Wconversion %s

void c8(char8_t);
void c16(char16_t);
void c32(char32_t);

void test(char8_t u8, char16_t u16, char32_t u32) {
    c8(u8);
    c8(u16); // expected-warning {{implicit conversion from 'char16_t' to 'char8_t' may lose precision and change the meaning of the represented code unit}}
    c8(u32); // expected-warning {{implicit conversion from 'char32_t' to 'char8_t' may lose precision and change the meaning of the represented code unit}}

    c16(u8);  // expected-warning {{implicit conversion from 'char8_t' to 'char16_t' may change the meaning of the represented code unit}}
    c16(u16);
    c16(u32); // expected-warning {{implicit conversion from 'char32_t' to 'char16_t' may lose precision and change the meaning of the represented code unit}}

    c32(u8);  // expected-warning {{implicit conversion from 'char8_t' to 'char32_t' may change the meaning of the represented code unit}}
    c32(u16);
    c32(u32);


    c8(char32_t(0x7f));
    c8(char32_t(0x80));   // expected-warning {{implicit conversion from 'char32_t' to 'char8_t' changes the meaning of the code point '<U+0080>'}}

    c8(char16_t(0x7f));
    c8(char16_t(0x80));   // expected-warning {{implicit conversion from 'char16_t' to 'char8_t' changes the meaning of the code point '<U+0080>'}}
    c8(char16_t(0xD800)); // expected-warning {{implicit conversion from 'char16_t' to 'char8_t' changes the meaning of the code unit '<0xD800>'}}
    c8(char16_t(0xE000)); // expected-warning {{implicit conversion from 'char16_t' to 'char8_t' changes the meaning of the code point '<U+E000>'}}


    c16(char32_t(0x7f));
    c16(char32_t(0x80));
    c16(char32_t(0xD7FF));
    c16(char32_t(0xD800));
    c16(char32_t(0xE000));
    c16(char32_t(U'🐉')); // expected-warning {{implicit conversion from 'char32_t' to 'char16_t' changes the meaning of the code point '🐉'}}


    c32(char8_t(0x7f));
    c32(char8_t(0x80)); // expected-warning {{implicit conversion from 'char8_t' to 'char32_t' changes the meaning of the code unit '<0x80>'}}
    c32(char8_t(0xFF)); // expected-warning {{implicit conversion from 'char8_t' to 'char32_t' changes the meaning of the code unit '<0xFF>'}}


    c32(char16_t(0x7f));
    c32(char16_t(0x80));

    c32(char16_t(0xD7FF));
    c32(char16_t(0xD800));
    c32(char16_t(0xDFFF));
    c32(char16_t(0xE000));
    c32(char16_t(u'☕'));

    (void)static_cast<char32_t>(char8_t(0x80)); //no warnings for explicit conversions.

    using Char8 = char8_t;
    Char8 c81 = u16; // expected-warning {{implicit conversion from 'char16_t' to 'Char8' (aka 'char8_t') may lose precision and change the meaning of the represented code unit}}

    [[maybe_unused]] char c = u16; // expected-warning {{implicit conversion loses integer precision: 'char16_t' to 'char'}}

    // FIXME: We should apply the same logic to wchar
    [[maybe_unused]] wchar_t wc = u16;
    [[maybe_unused]] wchar_t wc2 = u8;
}

void test_comp(char8_t u8, char16_t u16, char32_t u32) {
    (void)(u8 == u8' ');
    (void)(u8 == u' ');
    (void)(u8 == U' ');

    (void)(u16 == u8' ');
    (void)(u16 == U' ');

    (void)(u32 == u8' ');
    (void)(u32 == u' ');
    (void)(u32 == U' ');

    (void)(u8 == u'\u00FF'); // expected-warning{{comparing values of different Unicode code unit types 'char8_t' and 'char16_t' may compare different code points}}
    (void)(u8 == U'\u00FF'); // expected-warning{{comparing values of different Unicode code unit types 'char8_t' and 'char32_t' may compare different code points}}

    (void)(u16 == u8'\xFF'); // expected-warning{{comparing values of different Unicode code unit types 'char16_t' and 'char8_t' may compare different code points}}
    (void)(u16 == u'\u00FF');
    (void)(u16 == U'\u00FF');
    (void)(u16 == U'\xD800'); // expected-warning{{comparing values of different Unicode code unit types 'char16_t' and 'char32_t' may compare different code points}}

    (void)(u32 == u8'\xFF');  // expected-warning{{comparing values of different Unicode code unit types 'char32_t' and 'char8_t' may compare different code points}}
    (void)(u32 == u'\u00FF');
    (void)(u32 == u'\xD800'); // expected-warning{{comparing values of different Unicode code unit types 'char32_t' and 'char16_t' may compare different code points}}

    (void)(char8_t(0x7f) == char8_t(0x7f));
    (void)(char8_t(0x7f) == char16_t(0x7f));
    (void)(char8_t(0x7f) == char32_t(0x7f));

    (void)(char8_t(0x80) == char8_t(0x80));
    (void)(char8_t(0x80) == char16_t(0x80)); // expected-warning{{comparing values of different Unicode code unit types 'char8_t' and 'char16_t' compares unrelated code units '<0x80>' and '<U+0080>}}
    (void)(char8_t(0x80) == char32_t(0x80)); // expected-warning{{comparing values of different Unicode code unit types 'char8_t' and 'char32_t' compares unrelated code units '<0x80>' and '<U+0080>}}

    (void)(char8_t(0x80) == char8_t(0x7f));
    (void)(char8_t(0x80) == char16_t(0x7f)); // expected-warning{{comparing values of different Unicode code unit types 'char8_t' and 'char16_t' compares unrelated code units '<0x80>' and '<U+007F>'}}
    (void)(char8_t(0x80) == char32_t(0x7f)); // expected-warning{{comparing values of different Unicode code unit types 'char8_t' and 'char32_t' compares unrelated code units '<0x80>' and '<U+007F>'}}


    (void)(char16_t(0x7f) < char8_t(0x7f));
    (void)(char16_t(0x7f) < char16_t(0x7f));
    (void)(char16_t(0x7f) < char32_t(0x7f));

    (void)(char16_t(0x80) < char8_t(0x80)); // expected-warning{{comparing values of different Unicode code unit types 'char16_t' and 'char8_t' compares unrelated code units '<U+0080>' and '<0x80>'}}
    (void)(char16_t(0x80) < char16_t(0x80));
    (void)(char16_t(0x80) < char32_t(0x80));

    (void)(char16_t(0x80) == char8_t(0x7f));
    (void)(char16_t(0x80) < char16_t(0x7f));
    (void)(char16_t(0x80) < char32_t(0x7f));


    (void)(char32_t(0x7f) < char8_t(0x7f));
    (void)(char32_t(0x7f) < char16_t(0x7f));
    (void)(char32_t(0x7f) < char32_t(0x7f));

    (void)(char32_t(0x80) < char8_t(0x80)); // expected-warning{{comparing values of different Unicode code unit types 'char32_t' and 'char8_t' compares unrelated code units '<U+0080>' and '<0x80>'}}
    (void)(char32_t(0x80) < char16_t(0x80));
    (void)(char32_t(0x80) < char32_t(0x80));

    (void)(char32_t(0x80) == char8_t(0x7f));
    (void)(char32_t(0x80) < char16_t(0x7f));
    (void)(char32_t(0x80) < char32_t(0x7f));


    (void)(char32_t(U'🐉') <= char16_t(0xD800)); // expected-warning{{comparing values of different Unicode code unit types 'char32_t' and 'char16_t' compares unrelated code units '🐉' and '<0xD800>'}}
    (void)(char32_t(U'🐉') <= char16_t(0xD7FF));

    (void)(char16_t(0xD800) >= char32_t(U'🐉')); // expected-warning{{comparing values of different Unicode code unit types 'char16_t' and 'char32_t' compares unrelated code units '<0xD800>' and '🐉'}}
    (void)(char16_t(0xD7FF) >= char32_t(U'🐉'));
}

void check_arithmetic(char8_t u8, char16_t u16, char32_t u32) {

    (void)(u8 + u8);
    (void)(u16 += u16);
    (void)(u32 & u32);
    (void)(1 ? u16 : u16);

    (void)(u8 + u16);  // expected-warning {{arithmetic between different Unicode character types 'char8_t' and 'char16_t'}}
    (void)(u8 += u16); // expected-warning {{compound assignment of different Unicode character types 'char8_t' and 'char16_t'}}
    (void)(u8 & u16);  // expected-warning {{bitwise operation between different Unicode character types 'char8_t' and 'char16_t'}}
    (void)(1 ? u8 : u16);  // expected-warning {{conditional expression between different Unicode character types 'char8_t' and 'char16_t'}}


    (void)(u16 * u32);  // expected-warning {{arithmetic between different Unicode character types 'char16_t' and 'char32_t'}}
    (void)(u16 -= u32); // expected-warning {{compound assignment of different Unicode character types 'char16_t' and 'char32_t'}}
    (void)(u16 | u32);  // expected-warning {{bitwise operation between different Unicode character types 'char16_t' and 'char32_t'}}
    (void)(1 ? u32 : u16);  // expected-warning {{conditional expression between different Unicode character types 'char32_t' and 'char16_t'}}
}
