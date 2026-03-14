// RUN:  %clang_cc1 -std=c++2a -verify %s -code-completion-at=%s:6:16
// expected-no-diagnostics

template <typename T> concept C = true;
void bar(C auto foo);
int y = bar(