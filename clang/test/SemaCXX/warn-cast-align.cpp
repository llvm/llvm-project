// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -Wcast-align -verify %s

// Simple casts.
void test0(char *P) {
  char *a; short *b; int *c;

  a = (char*) P;
  a = static_cast<char*>(P);
  a = reinterpret_cast<char*>(P);
  typedef char *CharPtr;
  a = CharPtr(P);

  b = (short*) P; // expected-warning {{cast from 'char *' to 'short *' increases required alignment from 1 to 2}}
  b = reinterpret_cast<short*>(P);
  typedef short *ShortPtr;
  b = ShortPtr(P); // expected-warning {{cast from 'char *' to 'ShortPtr' (aka 'short *') increases required alignment from 1 to 2}}

  c = (int*) P; // expected-warning {{cast from 'char *' to 'int *' increases required alignment from 1 to 4}}
  c = reinterpret_cast<int*>(P);
  typedef int *IntPtr;
  c = IntPtr(P); // expected-warning {{cast from 'char *' to 'IntPtr' (aka 'int *') increases required alignment from 1 to 4}}
}

// Casts from void* are a special case.
void test1(void *P) {
  char *a; short *b; int *c;

  a = (char*) P;
  a = static_cast<char*>(P);
  a = reinterpret_cast<char*>(P);
  typedef char *CharPtr;
  a = CharPtr(P);

  b = (short*) P;
  b = static_cast<short*>(P);
  b = reinterpret_cast<short*>(P);
  typedef short *ShortPtr;
  b = ShortPtr(P);

  c = (int*) P;
  c = static_cast<int*>(P);
  c = reinterpret_cast<int*>(P);
  typedef int *IntPtr;
  c = IntPtr(P);
}
