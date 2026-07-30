// RUN: %clang_cc1 -std=c++11 -verify %s -triple x86_64-linux-gnu

alignas(double) void f(); // expected-error {{'alignas' attribute only applies to variables, data members and tag types}}
alignas(double) unsigned char c[sizeof(double)]; // expected-note {{previous}}
extern unsigned char c[sizeof(double)];
alignas(float) extern unsigned char c[sizeof(double)]; // expected-error {{different alignment}}
