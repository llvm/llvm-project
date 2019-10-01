// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-linux-pc %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple spir-unknown-unknown %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple armv7a-linux-gnu %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple aarch64-linux-gnu %s -DHAVE

#ifdef HAVE
// expected-no-diagnostics
#else
// expected-error@+2{{_Float16 is not supported on this target}}
#endif // HAVE
_Float16 f;
