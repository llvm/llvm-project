// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -fsyntax-only -verify %s

int array0[u'ñ' == u'\xf1'? 1 : -1];
int array1['\xF1' !=  u'\xf1'? 1 : -1];
int array1['ñ' !=  u'\xf1'? 1 : -1]; // expected-error {{character too large for enclosing character literal type}}
