// RUN: %clang_cc1 %s -fsyntax-only -verify -verify=c2x -pedantic -Wno-strict-prototypes -Wno-zero-length-array

int array_acceptor_good(unsigned long * par1)
{
  return par1 != (unsigned long *)0;
}

int array_acceptor_bad(unsigned long * par1) // expected-note {{passing argument to parameter 'par1' here}}
{
  return par1 != (unsigned long *)0;
}

int array_acceptor_bad2(unsigned long * par1) // expected-note {{passing argument to parameter 'par1' here}}
{
  return par1 != (unsigned long *)0;
}

struct S
{
  int a;
};

int array_acceptor_bad1(struct S * par1) // expected-note {{passing argument to parameter 'par1' here}}
{
  return par1 != (struct S *)0;
}

int array_struct_acceptor(struct S * par1)
{
  return par1 != (struct S *)0;
}


int array_tester()
{
  unsigned long mdarr[5][6];
  double mddarr[5][6];
  unsigned long sdarr[30];
  unsigned long mdarr3d[5][6][2];
  unsigned long mdarr4d[5][6][2][1];
  unsigned long mdarrz4d[5][6][0][1];
  struct S mdsarr[5][6][2];

  array_acceptor_good(sdarr);
  array_acceptor_good(mdarr);
  array_acceptor_good(mdarr3d);
  array_acceptor_good(mdarr4d);
  array_acceptor_bad(mddarr); // expected-error {{incompatible pointer types passing 'double[5][6]' to parameter of type 'unsigned long *'}}
  array_acceptor_bad1(mddarr); // expected-error {{incompatible pointer types passing 'double[5][6]' to parameter of type 'struct S *'}}
  array_acceptor_bad2(mdarrz4d); // expected-error {{incompatible pointer types passing 'unsigned long[5][6][0][1]' to parameter of type 'unsigned long *'}}
  array_struct_acceptor(mdsarr);
}