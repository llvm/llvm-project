// RUN: %clang_cc1 -DATTR=noreturn -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=always_inline -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=cdecl -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=const -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=fastcall -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=malloc -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=nothrow -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=stdcall -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=used -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=unused -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=weak -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s

#define ATTR_DECL(a) __attribute__((ATTR(a)))

int a;

inline ATTR_DECL(a) void* foo(); // expected-error{{attribute takes no arguments}}



// RUN: %clang_cc1 -DATTR=noreturn -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=always_inline -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=cdecl -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=const -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=fastcall -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=malloc -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=nothrow -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=stdcall -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=used -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=unused -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s
// RUN: %clang_cc1 -DATTR=weak -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s

#define ATTR_DECL(a) __attribute__((ATTR(a)))

int a;

inline ATTR_DECL(a) void* foo(); // expected-error{{attribute takes no arguments}}



