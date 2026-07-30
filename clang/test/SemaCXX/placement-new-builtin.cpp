// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

struct S {
  void* operator new(__SIZE_TYPE__, char*);
  void* operator new(__SIZE_TYPE__, __SIZE_TYPE__);
};

int main() {
  new (__builtin_strchr) S; // expected-error {{builtin functions must be directly called}}
  new ((__builtin_strlen)) S; // expected-error {{builtin functions must be directly called}}
}
