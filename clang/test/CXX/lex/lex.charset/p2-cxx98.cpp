// RUN: %clang_cc1 -verify -std=c++98 %s

char c00 = '\u0000'; // expected-error {{universal character name refers to a control character}}
char c01 = '\u0001'; // expected-error {{universal character name refers to a control character}}
char c1f = '\u001f'; // expected-error {{universal character name refers to a control character}}
char c20 = '\u0020'; // ' ', expected-error {{character ' ' cannot be specified by a universal character name}}
char c22 = '\u0022'; // ", expected-error {{character '"' cannot be specified by a universal character name}}
char c23 = '\u0023'; // #, expected-error {{character '#' cannot be specified by a universal character name}}
char c24 = '\u0024'; // $, ok
char c25 = '\u0025'; // %, expected-error {{character '%' cannot be specified by a universal character name}}
char c27 = '\u0027'; // ', expected-error {{character ''' cannot be specified by a universal character name}}
char c3f = '\u003f'; // ?, expected-error {{character '?' cannot be specified by a universal character name}}
char c40 = '\u0040'; // @, ok
char c41 = '\u0041'; // A, expected-error {{character 'A' cannot be specified by a universal character name}}
char c5f = '\u005f'; // _, expected-error {{character '_' cannot be specified by a universal character name}}
char c60 = '\u0060'; // `, ok
char c7e = '\u007e'; // ~, expected-error {{character '~' cannot be specified by a universal character name}}
char c7f = '\u007f'; // expected-error {{universal character name refers to a control character}}

wchar_t w007f = L'\u007f'; // expected-error {{universal character name refers to a control character}}
wchar_t w0080 = L'\u0080'; // expected-error {{universal character name refers to a control character}}
wchar_t w009f = L'\u009f'; // expected-error {{universal character name refers to a control character}}
wchar_t w00a0 = L'\u00a0';

wchar_t wd799 = L'\ud799';
wchar_t wd800 = L'\ud800'; // expected-error {{invalid universal character}}
wchar_t wdfff = L'\udfff'; // expected-error {{invalid universal character}}
wchar_t we000 = L'\ue000';

const char *s00 = "\u0000"; // expected-error {{universal character name refers to a control character}}
const char *s01 = "\u0001"; // expected-error {{universal character name refers to a control character}}
const char *s1f = "\u001f"; // expected-error {{universal character name refers to a control character}}
const char *s20 = "\u0020"; // ' ', expected-error {{character ' ' cannot be specified by a universal character name}}
const char *s22 = "\u0022"; // ", expected-error {{character '"' cannot be specified by a universal character name}}
const char *s23 = "\u0023"; // #, expected-error {{character '#' cannot be specified by a universal character name}}
const char *s24 = "\u0024"; // $, ok
const char *s25 = "\u0025"; // %, expected-error {{character '%' cannot be specified by a universal character name}}
const char *s27 = "\u0027"; // ', expected-error {{character ''' cannot be specified by a universal character name}}
const char *s3f = "\u003f"; // ?, expected-error {{character '?' cannot be specified by a universal character name}}
const char *s40 = "\u0040"; // @, ok
const char *s41 = "\u0041"; // A, expected-error {{character 'A' cannot be specified by a universal character name}}
const char *s5f = "\u005f"; // _, expected-error {{character '_' cannot be specified by a universal character name}}
const char *s60 = "\u0060"; // `, ok
const char *s7e = "\u007e"; // ~, expected-error {{character '~' cannot be specified by a universal character name}}
const char *s7f = "\u007f"; // expected-error {{universal character name refers to a control character}}

const wchar_t *ws007f = L"\u007f"; // expected-error {{universal character name refers to a control character}}
const wchar_t *ws0080 = L"\u0080"; // expected-error {{universal character name refers to a control character}}
const wchar_t *ws009f = L"\u009f"; // expected-error {{universal character name refers to a control character}}
const wchar_t *ws00a0 = L"\u00a0";

const wchar_t *wsd799 = L"\ud799";
const wchar_t *wsd800 = L"\ud800"; // expected-error {{invalid universal character}}
const wchar_t *wsdfff = L"\udfff"; // expected-error {{invalid universal character}}
const wchar_t *wse000 = L"\ue000";
