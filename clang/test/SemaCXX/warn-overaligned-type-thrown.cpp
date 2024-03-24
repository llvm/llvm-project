// RUN: %clang_cc1 -triple x86_64-apple-macosx10.13.99 -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions -DUNDERALIGNED %s
// RUN: %clang_cc1 -triple arm64-apple-ios10 -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions -DUNDERALIGNED %s
// RUN: %clang_cc1 -triple arm64-apple-tvos10 -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions -DUNDERALIGNED %s
// RUN: %clang_cc1 -triple arm64-apple-watchos4 -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions -DUNDERALIGNED %s
// RUN: %clang_cc1 -triple arm-linux-gnueabi -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions  -DUNDERALIGNED %s
// RUN: %clang_cc1 -triple thumbv7em-apple-unknown-macho -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions -DUNDERALIGNED %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14 -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions %s
// RUN: %clang_cc1 -triple arm64-apple-ios12 -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions %s
// RUN: %clang_cc1 -triple arm64-apple-tvos12 -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions %s
// RUN: %clang_cc1 -triple arm64-apple-watchos5 -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions %s
// RUN: %clang_cc1 -triple arm-linux-androideabi -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions %s
// RUN: %clang_cc1 -triple aarch64 -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions %s
// RUN: %clang_cc1 -triple mipsel-linux-gnu -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions %s
// RUN: %clang_cc1 -triple mips64el-linux-gnu -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions %s
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions %s
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14 -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions -Wno-underaligned-exception-object -DNODIAG %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -verify -fsyntax-only -std=c++11 -fcxx-exceptions -fexceptions -DNODIAG %s

struct S0 {
  S0();
  int m;
};

struct Overaligned1 {
  Overaligned1();
  int __attribute__((aligned(16))) m;
};

struct __attribute__((aligned(16))) Overaligned2 {
  Overaligned2();
  int m;
};

struct Overaligned3 {
  Overaligned3();
  int __attribute__((aligned(64))) m;
};

void test0() {
  throw S0();
}

void test1() {
  throw Overaligned1();
}

void test2() {
  throw Overaligned2();
}

void test3() {
  throw Overaligned3();
}

#if defined(NODIAG)
// expected-no-diagnostics
#elif defined(UNDERALIGNED)
// expected-warning@-14 {{underaligned exception object thrown}}
// expected-note@-15 {{(16 bytes) is larger than the supported alignment of C++ exception objects on this target (8 bytes)}}
// expected-warning@-12 {{underaligned exception object thrown}}
// expected-note@-13 {{(16 bytes) is larger than the supported alignment of C++ exception objects on this target (8 bytes)}}
// expected-warning@-10 {{underaligned exception object thrown}}
// expected-note@-11 {{(64 bytes) is larger than the supported alignment of C++ exception objects on this target (8 bytes)}}
#else
// expected-warning@-13 {{underaligned exception object thrown}}
// expected-note@-14 {{(64 bytes) is larger than the supported alignment of C++ exception objects on this target (16 bytes)}}
#endif
