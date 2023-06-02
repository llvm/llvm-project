// RUN: %clang_cc1 -triple=i386-none-none -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple=x86_64-none-none -fsyntax-only -verify %s

// Check that we can undefine triple-specific defines without warning
// expected-no-diagnostics
#undef __i386
#undef __i386__
#undef i386
#undef __amd64
#undef __amd64__
#undef __x86_64
#undef __x86_64__
