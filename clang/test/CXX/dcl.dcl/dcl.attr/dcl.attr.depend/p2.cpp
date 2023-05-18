// RUN: %clang_cc1 -verify -std=c++11 %s

int f(int); // expected-note 2{{declaration missing '[[carries_dependency]]' attribute is here}}
[[carries_dependency]] int f(int); // expected-error {{function declared '[[carries_dependency]]' after its first declaration}}
int f(int n [[carries_dependency]]); // expected-error {{parameter declared '[[carries_dependency]]' after its first declaration}}

int g([[carries_dependency]] int n); // expected-note {{declaration missing '[[carries_dependency]]' attribute is here}}
int g(int);
[[carries_dependency]] int g(int); // expected-error {{function declared '[[carries_dependency]]' after its first declaration}}
int g(int n [[carries_dependency]]);

int h [[carries_dependency]]();
int h();
[[carries_dependency]] int h();
