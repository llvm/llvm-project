// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -Warray-parameter -verify %s
void f0(int a[]);
void f0(int *a); // no warning

void f1(int a[]);  // expected-note {{previously declared as 'int[]' here}}
void f1(int a[2]); // expected-warning {{argument 'a' of type 'int[2]' with mismatched bound}}

void f2(int a[3]); // expected-note {{previously declared as 'int[3]' here}}
void f2(int a[2]); // expected-warning {{argument 'a' of type 'int[2]' with mismatched bound}}

void f3(int a[const 2]);
void f3(int a[2]); // no warning

void f4(int a[static 2]);
void f4(int a[2]); // no warning

void f5(int a[restrict 2]);
void f5(int a[2]); // no warning

void f6(int a[volatile 2]);
void f6(int a[2]); // no warning

void f7(int a[*]);
void f7(int a[]); // no warning

void f8(int n, int a[*]); // expected-note {{previously declared as 'int[*]' here}}
void f8(int n, int a[n]); // expected-warning {{argument 'a' of type 'int[n]' with mismatched bound}}

void f9(int *a);
void f9(int a[2]);
void f9(int a[]); // expected-warning {{argument 'a' of type 'int[]' with mismatched bound}}
                  // expected-note@-2 {{previously declared as 'int[2]' here}}
void f9(int a[2]) // expected-warning {{argument 'a' of type 'int[2]' with mismatched bound}}
                  // expected-note@-3 {{previously declared as 'int[]' here}}
{}
