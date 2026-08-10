// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify %s
// expected-no-diagnostics

typedef
struct __attribute__((packed)) S1 {
  char c0;
  int x;
  char c1;
} S1;

void bar(__unaligned int *);

void foo(__unaligned S1* s1)
{
    bar(&s1->x);
}
