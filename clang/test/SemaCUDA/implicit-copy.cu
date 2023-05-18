// RUN: %clang_cc1 -std=gnu++11 -triple nvptx64-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=gnu++11 -triple nvptx64-unknown-unknown -fcuda-is-device -fsyntax-only -verify %s

struct CopyableH {
  const CopyableH& operator=(const CopyableH& x) { return *this; }
};
struct CopyableD {
  __attribute__((device)) const CopyableD& operator=(const CopyableD x) { return *this; }
};

struct SimpleH {
  CopyableH b;
};
// expected-note@-3 2 {{candidate function (the implicit copy assignment operator) not viable: call to __host__ function from __device__ function}}
// expected-note@-4 2 {{candidate function (the implicit move assignment operator) not viable: call to __host__ function from __device__ function}}

struct SimpleD {
  CopyableD b;
};
// expected-note@-3 2 {{candidate function (the implicit copy assignment operator) not viable: call to __device__ function from __host__ function}}
// expected-note@-4 2 {{candidate function (the implicit move assignment operator) not viable: call to __device__ function from __host__ function}}

void foo1hh() {
  SimpleH a, b;
  a = b;
}
__attribute__((device)) void foo1hd() {
  SimpleH a, b;
  a = b; // expected-error {{no viable overloaded}}
}
void foo1dh() {
  SimpleD a, b;
  a = b; // expected-error {{no viable overloaded}}
}
__attribute__((device)) void foo1dd() {
  SimpleD a, b;
  a = b;
}

void foo2hh(SimpleH &a, SimpleH &b) {
  a = b;
}
__attribute__((device)) void foo2hd(SimpleH &a, SimpleH &b) {
  a = b; // expected-error {{no viable overloaded}}
}
void foo2dh(SimpleD &a, SimpleD &b) {
  a = b; // expected-error {{no viable overloaded}}
}
__attribute__((device)) void foo2dd(SimpleD &a, SimpleD &b) {
  a = b;
}
