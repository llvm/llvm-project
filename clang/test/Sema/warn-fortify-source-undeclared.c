// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c99 %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -verify

// Empty builtin prototypes must not provide implicit library declarations.
// getcwd has a complete prototype and provides the usual header hint in C.
void call_undeclared(void) {
  char buf[4];
  read(0, buf, 4); // expected-error {{undeclared}}
  write(0, buf, 4); // expected-error {{undeclared}}
  pread(0, buf, 4, 0); // expected-error {{undeclared}}
  pread64(0, buf, 4, 0); // expected-error {{undeclared}}
  pwrite(0, buf, 4, 0); // expected-error {{undeclared}}
  pwrite64(0, buf, 4, 0); // expected-error {{undeclared}}
  readlink("/", buf, 4); // expected-error {{undeclared}}
  readlinkat(0, "/", buf, 4); // expected-error {{undeclared}}
  getcwd(buf, 4); // expected-error {{undeclared}}
#ifndef __cplusplus
  // expected-note@-2 {{include the header <unistd.h> or explicitly provide a declaration for 'getcwd'}}
#endif
}
