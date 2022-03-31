// RUN: %clang_cc1 -std=c++20 -fsyntax-only -ast-print %s | FileCheck %s

using size_t = __SIZE_TYPE__;
static_assert(__has_builtin(__make_integer_seq));

template <class T, T... I> class idx_seq {};
template <size_t N> using make_idx_seq = __make_integer_seq<idx_seq, size_t, N>;

template <class CharT, size_t N>
struct Str {
  constexpr Str(CharT const (&s)[N]) : Str(s, make_idx_seq<N>()) {}
  CharT value[N];

private:
  template <size_t... I>
  constexpr Str(CharT const (&s)[N], idx_seq<size_t, I...>) : value{s[I]...} {}
};

template <Str> class ASCII {};

void not_string() {
  // CHECK{LITERAL}: ASCII<{{9, -1, 42}}>
  new ASCII<(int[]){9, -1, 42}>;
  // CHECK{LITERAL}: ASCII<{{3.140000e+00, 0.000000e+00, 4.200000e+01}}>
  new ASCII<(double[]){3.14, 0., 42.}>;
}

void narrow() {
  // CHECK{LITERAL}: ASCII<{""}>
  new ASCII<"">;
  // CHECK{LITERAL}: ASCII<{"the quick brown fox jumps"}>
  new ASCII<"the quick brown fox jumps">;
  // CHECK{LITERAL}: ASCII<{"OVER THE LAZY DOG 0123456789"}>
  new ASCII<"OVER THE LAZY DOG 0123456789">;
  // CHECK{LITERAL}: ASCII<{"\\`~!@#$%^&*()_+-={}[]|\'\";:,.<>?/"}>
  new ASCII<R"(\`~!@#$%^&*()_+-={}[]|'";:,.<>?/)">;
  // CHECK{LITERAL}: ASCII<{{101, 115, 99, 97, 112, 101, 0, 0}}>
  new ASCII<"escape\0">;
  // CHECK{LITERAL}: ASCII<{"escape\r\n"}>
  new ASCII<"escape\r\n">;
  // CHECK{LITERAL}: ASCII<{"escape\\\t\f\v"}>
  new ASCII<"escape\\\t\f\v">;
  // CHECK{LITERAL}: ASCII<{"escape\a\bc"}>
  new ASCII<"escape\a\b\c">;
  // CHECK{LITERAL}: ASCII<{{110, 111, 116, 17, 0}}>
  new ASCII<"not\x11">;
  // CHECK{LITERAL}: ASCII<{{18, 20, 127, 16, 1, 32, 97, 98, 99, 0}}>
  new ASCII<"\x12\x14\x7f\x10\x01 abc">;
  // CHECK{LITERAL}: ASCII<{{18, 20, 127, 16, 1, 32, 97, 98, 99, 100, 0}}>
  new ASCII<"\x12\x14\x7f\x10\x01 abcd">;
  // CHECK{LITERAL}: ASCII<{"print more characters as string"}>
  new ASCII<"print more characters as string">;
  // CHECK{LITERAL}: ASCII<{"print more characters as string, no uplimit"}>
  new ASCII<"print more characters as string, no uplimit">;
}

void wide() {
  // CHECK{LITERAL}: ASCII<{L""}>
  new ASCII<L"">;
  // CHECK{LITERAL}: ASCII<{L"the quick brown fox jumps"}>
  new ASCII<L"the quick brown fox jumps">;
  // CHECK{LITERAL}: ASCII<{L"OVER THE LAZY DOG 0123456789"}>
  new ASCII<L"OVER THE LAZY DOG 0123456789">;
  // CHECK{LITERAL}: ASCII<{L"\\`~!@#$%^&*()_+-={}[]|\'\";:,.<>?/"}>
  new ASCII<LR"(\`~!@#$%^&*()_+-={}[]|'";:,.<>?/)">;
  // CHECK{LITERAL}: ASCII<{{101, 115, 99, 97, 112, 101, 0, 0}}>
  new ASCII<L"escape\0">;
  // CHECK{LITERAL}: ASCII<{L"escape\r\n"}>
  new ASCII<L"escape\r\n">;
  // CHECK{LITERAL}: ASCII<{L"escape\\\t\f\v"}>
  new ASCII<L"escape\\\t\f\v">;
  // CHECK{LITERAL}: ASCII<{L"escape\a\bc"}>
  new ASCII<L"escape\a\b\c">;
  // CHECK{LITERAL}: ASCII<{{110, 111, 116, 17, 0}}>
  new ASCII<L"not\x11">;
  // CHECK{LITERAL}: ASCII<{{18, 20, 255, 22909, 136, 32, 97, 98, 99, 0}}>
  new ASCII<L"\x12\x14\xff\x597d\x88 abc">;
  // CHECK{LITERAL}: ASCII<{{18, 20, 255, 22909, 136, 32, 97, 98, 99, 100, 0}}>
  new ASCII<L"\x12\x14\xff\x597d\x88 abcd">;
  // CHECK{LITERAL}: ASCII<{L"print more characters as string"}>
  new ASCII<L"print more characters as string">;
  // CHECK{LITERAL}: ASCII<{L"print more characters as string, no uplimit"}>
  new ASCII<L"print more characters as string, no uplimit">;
}

void utf8() {
  // CHECK{LITERAL}: ASCII<{u8""}>
  new ASCII<u8"">;
  // CHECK{LITERAL}: ASCII<{u8"\\`~!@#$%^&*()_+-={}[]|\'\";:,.<>?/"}>
  new ASCII<u8R"(\`~!@#$%^&*()_+-={}[]|'";:,.<>?/)">;
  // CHECK{LITERAL}: ASCII<{{101, 115, 99, 97, 112, 101, 0, 0}}>
  new ASCII<u8"escape\0">;
  // CHECK{LITERAL}: ASCII<{u8"escape\r\n"}>
  new ASCII<u8"escape\r\n">;
  // CHECK{LITERAL}: ASCII<{{229, 165, 189, 239, 191, 189, 0}}>
  new ASCII<u8"\u597d\ufffd">;
  // CHECK{LITERAL}: ASCII<{u8"print more characters as string, no uplimit"}>
  new ASCII<u8"print more characters as string, no uplimit">;
}

void utf16() {
  // CHECK{LITERAL}: ASCII<{u""}>
  new ASCII<u"">;
  // CHECK{LITERAL}: ASCII<{u"\\`~!@#$%^&*()_+-={}[]|\'\";:,.<>?/"}>
  new ASCII<uR"(\`~!@#$%^&*()_+-={}[]|'";:,.<>?/)">;
  // CHECK{LITERAL}: ASCII<{{101, 115, 99, 97, 112, 101, 0, 0}}>
  new ASCII<u"escape\0">;
  // CHECK{LITERAL}: ASCII<{u"escape\r\n"}>
  new ASCII<u"escape\r\n">;
  // CHECK{LITERAL}: ASCII<{{22909, 65533, 0}}>
  new ASCII<u"\u597d\ufffd">;
  // CHECK{LITERAL}: ASCII<{u"print more characters as string, no uplimit"}>
  new ASCII<u"print more characters as string, no uplimit">;
}

void utf32() {
  // CHECK{LITERAL}: ASCII<{U""}>
  new ASCII<U"">;
  // CHECK{LITERAL}: ASCII<{U"\\`~!@#$%^&*()_+-={}[]|\'\";:,.<>?/"}>
  new ASCII<UR"(\`~!@#$%^&*()_+-={}[]|'";:,.<>?/)">;
  // CHECK{LITERAL}: ASCII<{{101, 115, 99, 97, 112, 101, 0, 0}}>
  new ASCII<U"escape\0">;
  // CHECK{LITERAL}: ASCII<{U"escape\r\n"}>
  new ASCII<U"escape\r\n">;
  // CHECK{LITERAL}: ASCII<{{22909, 131358, 0}}>
  new ASCII<U"\u597d\U0002011E">;
  // CHECK{LITERAL}: ASCII<{U"print more characters as string, no uplimit"}>
  new ASCII<U"print more characters as string, no uplimit">;
}
