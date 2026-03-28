// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wno-strict-prototypes -Wno-zero-length-array

int array_acceptor_case1(unsigned long *par1) { // #case1
  return par1 != (unsigned long *)0;
}

int array_acceptor_case2(unsigned long *par1) { // #case2
  return par1 != (unsigned long *)0;
}

int array_acceptor_case3(unsigned long *par1) { // #case3
  return par1 != (unsigned long *)0;
}

struct S {
  int a;
};

int array_acceptor_case4(struct S *par1) { // #case4
  return par1 != (struct S *)0;
}

int array_acceptor_case5(struct S *par1) { // #case5
  return par1 != (struct S *)0;
}

int array_tester() {
  unsigned long mdarr[5][6];
  double mddarr[5][6];
  unsigned long sdarr[30];
  unsigned long mdarr3d[5][6][2];
  unsigned long mdarr4d[5][6][2][1];
  unsigned long mdarrz4d[5][6][0][1];
  struct S mdsarr[5][6][2];

  array_acceptor_case1(sdarr);
  array_acceptor_case1(mdarr); // expected-error {{incompatible pointer types passing 'unsigned long[5][6]' to parameter of type 'unsigned long *'}} \
                               // expected-note@#case1 {{passing argument to parameter 'par1' here}}
  array_acceptor_case1(mdarr3d); // expected-error {{incompatible pointer types passing 'unsigned long[5][6][2]' to parameter of type 'unsigned long *'}} \
                                 // expected-note@#case1 {{passing argument to parameter 'par1' here}}
  array_acceptor_case1(mdarr4d); // expected-error {{incompatible pointer types passing 'unsigned long[5][6][2][1]' to parameter of type 'unsigned long *'}} \
                                 // expected-note@#case1 {{passing argument to parameter 'par1' here}}
  array_acceptor_case2(mddarr); // expected-error {{incompatible pointer types passing 'double[5][6]' to parameter of type 'unsigned long *'}} \
                                // expected-note@#case2 {{passing argument to parameter 'par1' here}}
  array_acceptor_case4(mddarr); // expected-error {{incompatible pointer types passing 'double[5][6]' to parameter of type 'struct S *'}} \
                                // expected-note@#case4 {{passing argument to parameter 'par1' here}}
  array_acceptor_case3(mdarrz4d); // expected-error {{incompatible pointer types passing 'unsigned long[5][6][0][1]' to parameter of type 'unsigned long *'}} \
                                  // expected-note@#case3 {{passing argument to parameter 'par1' here}}
  array_acceptor_case5(mdsarr); // expected-error {{incompatible pointer types passing 'struct S[5][6][2]' to parameter of type 'struct S *'}} \
                                // expected-note@#case5 {{passing argument to parameter 'par1' here}}
}
