// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -Wno-unused-value -Wcast-align -verify %s

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

struct __attribute__((aligned(16))) AlignedS {
  char m[16];
};

struct __attribute__((aligned(16))) A {
  char m0[16];
  char m1[16];
  AlignedS *getAlignedS() {
    return (AlignedS *)m1;
  }
};

struct B0 {
  char m0[16];
};

struct B1 {
  char m0[16];
};

struct C {
  A &m0;
  B0 &m1;
  A m2;
};

struct __attribute__((aligned(16))) D0 : B0, B1 {
};

struct __attribute__((aligned(16))) D1 : virtual B0 {
};

struct B2 {
  char m0[8];
};

struct B3 {
  char m0[8];
};

struct B4 {
  char m0[8];
};

struct D2 : B2, B3 {
};

struct __attribute__((aligned(16))) D3 : B4, D2 {
};

struct __attribute__((aligned(16))) D4 : virtual D2 {
};

struct D5 : virtual D0 {
  char m0[16];
  AlignedS *get() {
    return (AlignedS *)m0; // expected-warning {{cast from 'char *' to 'AlignedS *'}}
  }
};

struct D6 : virtual D5 {
};

struct D7 : virtual D3 {
};

void test2(int n, A *a2) {
  __attribute__((aligned(16))) char m[sizeof(A) * 2];
  char(&m_ref)[sizeof(A) * 2] = m;
  extern char(&m_ref_noinit)[sizeof(A) * 2];
  __attribute__((aligned(16))) char vararray[10][n];
  A t0;
  B0 t1;
  C t2 = {.m0 = t0, .m1 = t1};
  __attribute__((aligned(16))) char t3[5][5][5];
  __attribute__((aligned(16))) char t4[4][16];
  D0 t5;
  D1 t6;
  D3 t7;
  D4 t8;
  D6 t9;
  __attribute__((aligned(1))) D7 t10;

  A *a;
  a = (A *)&m;
  a = (A *)(m + sizeof(A));
  a = (A *)(sizeof(A) + m);
  a = (A *)((sizeof(A) * 2 + m) - sizeof(A));
  a = (A *)((sizeof(A) * 2 + m) - 1); // expected-warning {{cast from 'char *' to 'A *'}}
  a = (A *)(m + 1);                   // expected-warning {{cast from 'char *' to 'A *'}}
  a = (A *)(1 + m);                   // expected-warning {{cast from 'char *' to 'A *'}}
  a = (A *)(m + n);                   // expected-warning {{cast from 'char *' to 'A *'}}
  a = (A *)&*&m[sizeof(A)];
  a = (A *)(0, 0, &m[sizeof(A)]);
  a = (A *)&(0, 0, *&m[sizeof(A)]);
  a = (A *)&m[n]; // expected-warning {{cast from 'char *' to 'A *'}}
  a = (A *)&m_ref;
  a = (A *)&m_ref_noinit;        // expected-warning {{cast from 'char (*)[64]' to 'A *'}}
  a = (A *)(&vararray[4][0]);    // expected-warning {{cast from 'char *' to 'A *'}}
  a = (A *)(a2->m0 + sizeof(A)); // expected-warning {{cast from 'char *' to 'A *'}}
  a = (A *)(&t2.m0);
  a = (A *)(&t2.m1); // expected-warning {{cast from 'B0 *' to 'A *'}}
  a = (A *)(&t2.m2);
  a = (A *)(t2.m2.m1);
  a = (A *)(&t3[3][3][0]); // expected-warning {{cast from 'char *' to 'A *'}}
  a = (A *)(&t3[2][2][4]);
  a = (A *)(&t3[0][n][0]); // expected-warning {{cast from 'char *' to 'A *'}}
  a = (A *)&t4[n][0];
  a = (A *)&t4[n][1]; // expected-warning {{cast from 'char *' to 'A *'}}
  a = (A *)(t4 + 1);
  a = (A *)(t4 + n);
  a = (A *)(static_cast<B1 *>(&t5));
  a = (A *)(&(static_cast<B1 &>(t5)));
  a = (A *)(static_cast<B0 *>(&t6)); // expected-warning {{cast from 'B0 *' to 'A *'}}
  a = (A *)(static_cast<B2 *>(&t7)); // expected-warning {{cast from 'B2 *' to 'A *'}}
  a = (A *)(static_cast<B3 *>(&t7));
  a = (A *)(static_cast<B2 *>(&t8));  // expected-warning {{cast from 'B2 *' to 'A *'}}
  a = (A *)(static_cast<B3 *>(&t8));  // expected-warning {{cast from 'B3 *' to 'A *'}}
  a = (A *)(static_cast<D5 *>(&t9));  // expected-warning {{cast from 'D5 *' to 'A *'}}
  a = (A *)(static_cast<D3 *>(&t10)); // expected-warning {{cast from 'D3 *' to 'A *'}}
}
