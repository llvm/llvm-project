// RUN: %clang_cc1 -verify -std=c++11 %s

char c00 = '\u0000'; // ok
char c01 = '\u0001'; // ok
char c1f = '\u001f'; // ok
char c20 = '\u0020'; // ' ', ok
char c22 = '\u0022'; // ", ok
char c23 = '\u0023'; // #, ok
char c24 = '\u0024'; // $, ok
char c25 = '\u0025'; // %, ok
char c27 = '\u0027'; // ', ok
char c3f = '\u003f'; // ?, ok
char c40 = '\u0040'; // @, ok
char c41 = '\u0041'; // A, ok
char c5f = '\u005f'; // _, ok
char c60 = '\u0060'; // `, ok
char c7e = '\u007e'; // ~, ok
char c7f = '\u007f'; // ok

wchar_t w007f = L'\u007f';
wchar_t w0080 = L'\u0080';
wchar_t w009f = L'\u009f';
wchar_t w00a0 = L'\u00a0';

wchar_t wd799 = L'\ud799';
wchar_t wd800 = L'\ud800'; // expected-error {{invalid universal character}}
wchar_t wdfff = L'\udfff'; // expected-error {{invalid universal character}}
wchar_t we000 = L'\ue000';

char32_t w10fffe = U'\U0010fffe';
char32_t w10ffff = U'\U0010ffff';
char32_t w110000 = U'\U00110000'; // expected-error {{invalid universal character}}

const char *p1 = "\u0000\u0001\u001f\u0020\u0022\u0023\u0024\u0025\u0027\u003f\u0040\u0041\u005f\u0060\u007e\u007f";
const wchar_t *p2 = L"\u0000\u0012\u004e\u007f\u0080\u009f\u00a0\ud799\ue000";
const char *p3 = u8"\u0000\u0012\u004e\u007f\u0080\u009f\u00a0\ud799\ue000";
const char16_t *p4 = u"\u0000\u0012\u004e\u007f\u0080\u009f\u00a0\ud799\ue000";
const char32_t *p5 = U"\u0000\u0012\u004e\u007f\u0080\u009f\u00a0\ud799\ue000";
const wchar_t *p6 = L"foo \U00110000 bar"; // expected-error {{invalid universal character}}
const char *p7 = u8"foo \U0000d800 bar"; // expected-error {{invalid universal character}}
const char16_t *p8 = u"foo \U0000dfff bar"; // expected-error {{invalid universal character}}
const char32_t *p9 = U"foo \U0010ffff bar"; // ok
