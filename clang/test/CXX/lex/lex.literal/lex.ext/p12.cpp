// RUN: %clang_cc1 -std=gnu++11 -verify %s

template<typename T, T... cs> struct check; // expected-note {{template is declared here}} expected-note {{template is declared here}}
template<>
struct check<char, 34, -47, -126, -48, -75, -47, -127, -47, -126, 32, -16, -112, -128, -128>{};
template<>
struct check<char16_t, 34, 1090, 1077, 1089, 1090, 32, 55296, 56320>{};
template<>
struct check<char32_t, 34, 1090, 1077, 1089, 1090, 32, 65536>{};
template<typename T, T... str> int operator""_x() { // #1 expected-warning {{string literal operator templates are a GNU extension}}
    check<T, str...> chars; // expected-error {{implicit instantiation of undefined template 'check<char, 't', 'e', 's', 't'>'}} \
                            // expected-error {{implicit instantiation of undefined template 'check<char32_t, U'"', U'\u0442', U'\u0435', U'\u0441', U'\u0442', U'_', U'\U00010000'>'}}
    return 1;
}
void *operator""_x(const char*); // #2
void *a = 123_x; // ok, calls #2
int b = u8"\"тест 𐀀"_x; // ok, calls #1
int c = u8R"("тест 𐀀)"_x; // ok, calls #1
int d = "test"_x; // expected-note {{in instantiation of function template specialization 'operator""_x<char, 't', 'e', 's', 't'>' requested here}}
int e = uR"("тест 𐀀)"_x;
int f = UR"("тест 𐀀)"_x;
int g = UR"("тест_𐀀)"_x; // expected-note {{in instantiation of function template specialization 'operator""_x<char32_t, U'"', U'\u0442', U'\u0435', U'\u0441', U'\u0442', U'_', U'\U00010000'>' requested here}}
