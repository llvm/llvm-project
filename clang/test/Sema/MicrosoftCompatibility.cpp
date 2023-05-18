// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify -fms-compatibility

// PR15845
int foo(xxx); // expected-error{{unknown type name}}

struct cls {
  char *m;
};

char * cls::* __uptr wrong2 = &cls::m; // expected-error {{'__uptr' attribute cannot be used with pointers to members}}
