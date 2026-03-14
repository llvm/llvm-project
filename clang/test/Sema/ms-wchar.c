// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions -triple i386-pc-win32 %s

// C++ mode with -fno-wchar works the same as C mode for wchar_t.
// RUN: %clang_cc1 -x c++ -fno-wchar -fsyntax-only -verify -fms-extensions -triple i386-pc-win32 %s

wchar_t f(void); // expected-error{{unknown type name 'wchar_t'}}

// __wchar_t is available as an MS extension.
__wchar_t g = L'a'; // expected-note {{previous}}

// __wchar_t is a distinct type, separate from the target's integer type for wide chars.
unsigned short g; // expected-error {{redefinition of 'g' with a different type: 'unsigned short' vs '__wchar_t'}}

// The type of a wide string literal is actually not __wchar_t.
__wchar_t s[] = L"Hello world!"; // expected-error-re {{array initializer must be an initializer list{{$}}}}

// Do not suggest initializing with a string here, because it would not work.
__wchar_t t[] = 1; // expected-error-re {{array initializer must be an initializer list{{$}}}}
