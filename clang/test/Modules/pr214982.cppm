// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/part.cppm -o %t/part.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/repro.cppm \
// RUN:   -fmodule-file=repro:part=%t/part.pcm -o %t/repro.pcm
//
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/valid.cpp \
// RUN:   -fmodule-file=repro=%t/repro.pcm \
// RUN:   -fmodule-file=repro:part=%t/part.pcm 2>&1 -verify
//
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/main.cpp \
// RUN:   -fmodule-file=repro=%t/repro.pcm \
// RUN:   -fmodule-file=repro:part=%t/part.pcm 2>&1 -verify


//--- part.cppm
export module repro:part;

namespace {
struct guard_t {
  guard_t(int &x) : x{x} { ++this->x; }
  ~guard_t() { --this->x; }
  int &x;
};
}

export template <typename T>
struct widget_t {
  void bump() const { guard_t g{this->x}; }
  void fine() const { }
  mutable int x = 0;
};

//--- repro.cppm
export module repro;
export import :part;

//--- valid.cpp
// expected-no-diagnostics
import repro;

void valid() {
  widget_t<int> w;
  w.fine();
}

//--- main.cpp
import repro;

int main() {
  widget_t<int> w;
            // expected-error@part.cppm:13 {{no matching constructor for initialization of 'guard_t'}}
  w.bump(); // expected-warning {{instantiation of 'bump' triggers reference to TU-local entity 'guard_t' from other TU 'repro:part'}}
            // expected-note@-1 {{in instantiation of member function 'widget_t<int>::bump' requested here}}
}
